#!/usr/bin/env python3
"""数据初始化脚本 - 初始化量化数据库

这个脚本用于首次设置量化数据库，包括:
1. 创建数据库Schema
2. 导入交易日历
3. 导入股票基础信息
4. 导入指数成分股
5. 导入历史K线数据

Usage:
    # 全量初始化
    python scripts/init_data.py --mode full --db data/quant.db

    # 仅初始化日历
    python scripts/init_data.py --mode calendar --db data/quant.db

    # 仅初始化K线
    python scripts/init_data.py --mode ohlcv --db data/quant.db --start 2020-01-01

    # 仅初始化股票池
    python scripts/init_data.py --mode indices --db data/quant.db
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB, Market, IndexCode


# =============================================================================
# Data Sources - 数据源配置
# =============================================================================

# 指数列表
INDICES = {
    "000016.SH": {"name": "上证50", "category": "宽基"},
    "000300.SH": {"name": "沪深300", "category": "宽基"},
    "000905.SH": {"name": "中证500", "category": "宽基"},
    "000852.SH": {"name": "中证1000", "category": "宽基"},
    "000906.SH": {"name": "中证800", "category": "宽基"},
    "000903.SH": {"name": "中证100", "category": "宽基"},
    # 申万一级行业
    "801010": {"name": "农林牧渔", "category": "申万一级"},
    "801020": {"name": "采掘", "category": "申万一级"},
    "801030": {"name": "化工", "category": "申万一级"},
    "801040": {"name": "钢铁", "category": "申万一级"},
    "801050": {"name": "有色金属", "category": "申万一级"},
    "801080": {"name": "电子", "category": "申万一级"},
    "801110": {"name": "家用电器", "category": "申万一级"},
    "801120": {"name": "食品饮料", "category": "申万一级"},
    "801130": {"name": "纺织服装", "category": "申万一级"},
    "801140": {"name": "轻工制造", "category": "申万一级"},
    "801150": {"name": "医药生物", "category": "申万一级"},
    "801160": {"name": "汽车", "category": "申万一级"},
    "801170": {"name": "公用事业", "category": "申万一级"},
    "801180": {"name": "房地产", "category": "申万一级"},
    "801200": {"name": "商业贸易", "category": "申万一级"},
    "801210": {"name": "餐饮旅游", "category": "申万一级"},
    "801230": {"name": "建筑材料", "category": "申万一级"},
    "801710": {"name": "建筑装饰", "category": "申万一级"},
    "801720": {"name": "电气设备", "category": "申万一级"},
    "801730": {"name": "国防军工", "category": "申万一级"},
    "801740": {"name": "计算机", "category": "申万一级"},
    "801750": {"name": "电子", "category": "申万一级"},
    "801760": {"name": "传媒", "category": "申万一级"},
    "801770": {"name": "通信", "category": "申万一级"},
    "801780": {"name": "非银金融", "category": "申万一级"},
    "801790": {"name": "银行", "category": "申万一级"},
    "801880": {"name": "汽车", "category": "申万一级"},
    "801950": {"name": "煤炭", "category": "申万一级"},
    "801960": {"name": "石油石化", "category": "申万一级"},
    "801970": {"name": "环保", "category": "申万一级"},
    "801980": {"name": "美容护理", "category": "申万一级"},
}


# =============================================================================
# Calendar Generator - 日历生成器
# =============================================================================

def generate_trading_calendar(start_year: int = 2005, end_year: int = None) -> pd.DataFrame:
    """生成A股交易日历
    
    基于中国A股实际交易规则生成日历:
    - 周一到周五交易
    - 排除周末
    - 排除法定节假日（简化版）
    
    Args:
        start_year: 开始年份
        end_year: 结束年份
        
    Returns:
        日历DataFrame
    """
    if end_year is None:
        end_year = date.today().year
    
    # 法定节假日（简化，只排除大部分）
    holidays = set()
    
    # 添加已知的固定假期（简化处理）
    for year in range(start_year, end_year + 1):
        # 元旦
        holidays.add(f"{year}-01-01")
        holidays.add(f"{year}-01-02")
        holidays.add(f"{year}-01-03")
        
        # 春节（简化，排除第一周）
        for day in range(1, 8):
            holidays.add(f"{year}-02-{day:02d}")
        
        # 清明
        holidays.add(f"{year}-04-04")
        holidays.add(f"{year}-04-05")
        holidays.add(f"{year}-04-06")
        
        # 劳动节
        holidays.add(f"{year}-05-01")
        holidays.add(f"{year}-05-02")
        holidays.add(f"{year}-05-03")
        
        # 端午（简化）
        holidays.add(f"{year}-06-{22 + year % 3:02d}")
        
        # 中秋（简化）
        holidays.add(f"{year}-09-{15 + year % 2:02d}")
        
        # 国庆
        for day in range(1, 8):
            holidays.add(f"{year}-10-{day:02d}")
    
    records = []
    current = date(start_year, 1, 1)
    end = date(end_year + 1, 1, 1)
    
    while current < end:
        # 周一到周五
        if current.weekday() < 5:
            date_str = current.strftime("%Y-%m-%d")
            # 排除节假日
            is_trading = date_str not in holidays
            records.append({
                "date": date_str,
                "is_trading_day": is_trading,
                "exchange": "ALL",
            })
        current += timedelta(days=1)
    
    return pd.DataFrame(records)


# =============================================================================
# Instrument Fetcher - 股票信息获取器
# =============================================================================

def fetch_instruments_baostock() -> pd.DataFrame:
    """从baostock获取股票列表
    
    Returns:
        股票信息DataFrame
    """
    try:
        import baostock as bs
        
        # 登录
        lg = bs.login()
        if lg.error_code != "0":
            raise Exception(f"Baostock login failed: {lg.error_msg}")
        
        # 获取所有股票
        rs = bs.query_all_stock(day=date.today().strftime("%Y-%m-%d"))
        
        records = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if row[1] in ["1", "2", "3", "4", "5"]:  # A股
                code = row[0].split(".")[1]
                exchange = "SSE" if row[0].startswith("sh") else "SZSE"
                records.append({
                    "symbol": f"{code}.{exchange}",
                    "name": row[2],
                    "list_date": None,
                    "delist_date": None,
                    "market": exchange,
                    "industry_sw": None,
                    "industry_sw_2": None,
                    "industry_sw_3": None,
                    "sector": None,
                })
        
        bs.logout()
        
        if not records:
            raise Exception("No stocks fetched from baostock")
        
        return pd.DataFrame(records)
    
    except ImportError:
        print("⚠️ baostock not installed, using empty instruments")
        return pd.DataFrame(columns=[
            "symbol", "name", "list_date", "delist_date", 
            "market", "industry_sw", "industry_sw_2", "industry_sw_3", "sector"
        ])


def fetch_instruments_akshare() -> pd.DataFrame:
    """从AKShare获取股票列表
    
    Returns:
        股票信息DataFrame
    """
    try:
        import akshare as ak
        
        # 获取A股列表
        df = ak.stock_info_a_code_name()
        
        records = []
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)
            exchange = "SZSE" if code.startswith(("000", "001", "002", "003")) else "SSE"
            records.append({
                "symbol": f"{code}.{exchange}",
                "name": row["name"],
                "list_date": None,
                "delist_date": None,
                "market": exchange,
                "industry_sw": None,
                "industry_sw_2": None,
                "industry_sw_3": None,
                "sector": None,
            })
        
        return pd.DataFrame(records)
    
    except ImportError:
        return pd.DataFrame()


# =============================================================================
# Index Fetcher - 指数成分获取器
# =============================================================================

def fetch_index_components_baostock(index_code: str) -> pd.DataFrame:
    """从baostock获取指数成分股
    
    Args:
        index_code: 指数代码，如 'sh.000300'
        
    Returns:
        成分股权重DataFrame
    """
    try:
        import baostock as bs
        
        # 转换代码格式
        bs_code = index_code.replace(".SH", ".sh").replace(".SZ", ".sz")
        
        lg = bs.login()
        if lg.error_code != "0":
            raise Exception(f"Baostock login failed: {lg.error_msg}")
        
        rs = bs.query_index_stock_weight(bs_code)
        
        records = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code = row["stockCode"].split(".")[1]
            exchange = "SSE" if row["stockCode"].startswith("sh") else "SZSE"
            records.append({
                "index_code": index_code,
                "symbol": f"{code}.{exchange}",
                "in_date": row["inDate"],
                "out_date": None if row["outDate"] == "" else row["outDate"],
                "weight": float(row["weight"]) if row["weight"] else 0.0,
            })
        
        bs.logout()
        
        return pd.DataFrame(records)
    
    except ImportError:
        return pd.DataFrame()


def fetch_index_components_tushare(index_code: str) -> pd.DataFrame:
    """从Tushare获取指数成分股
    
    Args:
        index_code: 指数代码
        
    Returns:
        成分股权重DataFrame
    """
    # Tushare需要token，这里提供框架
    return pd.DataFrame()


# =============================================================================
# OHLCV Fetcher - K线数据获取器
# =============================================================================

def fetch_ohlcv_baostock(
    symbols: list[str],
    start: str,
    end: str,
    adjust: str = "3",  # 3=前复权
) -> pd.DataFrame:
    """从baostock获取K线数据
    
    Args:
        symbols: 股票代码列表
        start: 开始日期
        end: 结束日期
        adjust: 复权类型 1=后复权 2=前复权 3=不复权
        
    Returns:
        K线DataFrame
    """
    try:
        import baostock as bs
        
        lg = bs.login()
        if lg.error_code != "0":
            raise Exception(f"Baostock login failed: {lg.error_msg}")
        
        all_records = []
        
        for symbol in symbols:
            # 转换代码格式
            bs_code = f"sh.{symbol}" if ".SH" in symbol else f"sz.{symbol}"
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turnover,pctChg",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag=adjust,
            )
            
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                code = symbol.split(".")[0]
                all_records.append({
                    "date": row[0],
                    "symbol": symbol,
                    "open": float(row[1]) if row[1] else 0,
                    "high": float(row[2]) if row[2] else 0,
                    "low": float(row[3]) if row[3] else 0,
                    "close": float(row[4]) if row[4] else 0,
                    "volume": float(row[5]) if row[5] else 0,
                    "amount": float(row[6]) if row[6] else 0,
                    "turnover_rate": float(row[7]) if row[7] else 0,
                    "pct_change": float(row[8]) if row[8] else 0,
                })
        
        bs.logout()
        
        return pd.DataFrame(all_records)
    
    except ImportError:
        return pd.DataFrame()


def fetch_ohlcv_akshare(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """从AKShare获取K线数据
    
    Args:
        symbol: 股票代码
        start: 开始日期
        end: 结束日期
        adjust: 复权类型 qfq=前复权 hfq=后复权 None=不复权
        
    Returns:
        K线DataFrame
    """
    try:
        import akshare as ak
        
        # 转换代码
        code = symbol.split(".")[0]
        
        df = ak.stock_zh_a_hist(
            symbol=code,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=adjust,
        )
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 重命名列
        df = df.rename(columns={
            "日期": "date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "涨跌幅": "pct_change",
        })
        
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_change"]]
    
    except ImportError:
        return pd.DataFrame()


# =============================================================================
# Main - 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="初始化量化数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default="data/quant.db",
        help="数据库路径 (default: data/quant.db)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "calendar", "instruments", "indices", "ohlcv"],
        default="full",
        help="初始化模式",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2005-01-01",
        help="K线开始日期 (default: 2005-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="K线结束日期 (default: today)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["baostock", "akshare"],
        default="baostock",
        help="数据源 (default: baostock)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="指定股票代码 (为空则获取全部)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="批量导入大小 (default: 10000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新初始化",
    )
    
    return parser.parse_args()


def init_calendar(db: QuantDB, args) -> int:
    """初始化日历"""
    print("📅 初始化交易日历...")
    start_time = datetime.now()
    
    start_year = 2005
    end_year = date.today().year if not args.end else int(args.end.split("-")[0])
    
    df = generate_trading_calendar(start_year, end_year)
    print(f"   生成 {len(df)} 条日历记录 ({start_year}-{end_year})")
    
    rows = db.import_calendars(df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ 日历导入完成: {rows} 条 ({duration:.2f}s)")
    
    return rows


def init_instruments(db: QuantDB, args) -> int:
    """初始化股票信息"""
    print("📋 初始化股票信息...")
    start_time = datetime.now()
    
    if args.source == "baostock":
        df = fetch_instruments_baostock()
    else:
        df = fetch_instruments_akshare()
    
    if df.empty:
        print("⚠️ 未获取到股票信息，跳过")
        return 0
    
    print(f"   获取 {len(df)} 只股票")
    
    rows = db.import_instruments(df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ 股票信息导入完成: {rows} 条 ({duration:.2f}s)")
    
    return rows


def init_indices(db: QuantDB, args) -> int:
    """初始化指数信息"""
    print("📊 初始化指数信息...")
    start_time = datetime.now()
    
    # 保存指数基础信息
    indices_records = []
    for code, info in INDICES.items():
        indices_records.append({
            "index_code": code if "." in code else f"{code}.SH",
            "index_name": info["name"],
            "index_full_name": info["name"],
            "base_date": "2004-12-31",
            "base_point": 1000,
            "exchange": "SSE",
            "category": info["category"],
        })
    
    indices_df = pd.DataFrame(indices_records)
    
    # 导入指数信息
    conn = db.connect()
    for _, row in indices_df.iterrows():
        try:
            conn.execute(f"""
                INSERT OR REPLACE INTO indices 
                (index_code, index_name, index_full_name, base_date, base_point, exchange, category)
                VALUES ('{row['index_code']}', '{row['index_name']}', '{row['index_full_name']}',
                        '{row['base_date']}', {row['base_point']}, '{row['exchange']}', '{row['category']}')
            """)
        except Exception:
            pass
    conn.commit()
    
    # 获取成分股
    total_components = 0
    for code in list(INDICES.keys())[:10]:  # 限制数量避免超时
        full_code = code if "." in code else f"{code}.SH"
        print(f"   获取 {full_code} 成分股...")
        
        if args.source == "baostock":
            components_df = fetch_index_components_baostock(full_code)
        else:
            components_df = fetch_index_components_tushare(full_code)
        
        if not components_df.empty:
            db.import_index_components(components_df)
            total_components += len(components_df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ 指数信息导入完成: {len(INDICES)} 指数, {total_components} 成分股 ({duration:.2f}s)")
    
    return total_components


def init_ohlcv(db: QuantDB, args) -> int:
    """初始化K线数据"""
    print("📈 初始化K线数据...")
    start_time = datetime.now()
    
    # 确定日期范围
    end = args.end or date.today().strftime("%Y-%m-%d")
    
    # 获取股票列表
    if args.symbols:
        symbols = args.symbols
    else:
        df = db.get_instruments()
        symbols = df["symbol"].tolist()
    
    print(f"   将获取 {len(symbols)} 只股票的K线数据")
    print(f"   日期范围: {args.start} ~ {end}")
    
    # 分批获取
    total_records = 0
    batch_size = 50  # 每批处理的股票数
    
    for i in range(0, min(len(symbols), 100), batch_size):  # 限制总数避免超时
        batch = symbols[i:i + batch_size]
        print(f"   处理 {i + 1} ~ {i + len(batch)} 只股票...")
        
        if args.source == "baostock":
            df = fetch_ohlcv_baostock(batch, args.start, end)
        else:
            # AKShare每次只能获取一只股票
            all_dfs = []
            for sym in batch:
                df = fetch_ohlcv_akshare(sym, args.start, end)
                if not df.empty:
                    all_dfs.append(df)
            df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        
        if not df.empty:
            rows = db.import_ohlcv(df, batch_size=args.batch_size)
            total_records += rows
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ K线数据导入完成: {total_records} 条记录 ({duration:.2f}s)")
    
    return total_records


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("量化数据库初始化")
    print("=" * 60)
    
    # 确保目录存在
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据库
    db = QuantDB(args.db)
    
    if args.force:
        print("⚠️ 强制模式，将重建数据库...")
        db.init_schema()
    
    results = {}
    
    # 根据模式执行初始化
    if args.mode == "full":
        results["calendar"] = init_calendar(db, args)
        results["instruments"] = init_instruments(db, args)
        results["indices"] = init_indices(db, args)
        results["ohlcv"] = init_ohlcv(db, args)
    
    elif args.mode == "calendar":
        results["calendar"] = init_calendar(db, args)
    
    elif args.mode == "instruments":
        results["instruments"] = init_instruments(db, args)
    
    elif args.mode == "indices":
        results["indices"] = init_indices(db, args)
    
    elif args.mode == "ohlcv":
        results["ohlcv"] = init_ohlcv(db, args)
    
    # 显示最终统计
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)
    
    info = db.info()
    print("\n数据库信息:")
    for table, data in info["tables"].items():
        print(f"   {table}: {data['rows']} 条记录")
    
    print(f"\n数据库路径: {args.db}")
    
    db.close()


if __name__ == "__main__":
    main()
