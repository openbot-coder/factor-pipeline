#!/usr/bin/env python3
"""数据初始化脚本 - 初始化量化数据库 (分层架构)

采用数仓分层设计:
- ODS (Operational Data Store): 原始数据层
- DWD (Data Warehouse Detail): 明细数据层
- DWS (Data Warehouse Summary): 汇总数据层
- ADS (Application Data Service): 应用数据层
- Factors: 因子数据层

Usage:
    # 全量初始化 (ODS + DWD + DWS)
    python scripts/init_data.py --mode full --db data/quant.db

    # 仅拉取原始数据到ODS
    python scripts/init_data.py --mode ods --db data/quant.db

    # 仅清洗转换到DWD
    python scripts/init_data.py --mode dwd --db data/quant.db

    # 仅初始化K线
    python scripts/init_data.py --mode ohlcv --db data/quant.db --start 2020-01-01
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

from factor_pipeline.data.quantdb import QuantDB, IndexCode


# =============================================================================
# Configuration - 配置
# =============================================================================

# 指数配置
INDICES = {
    "000016.SH": {"name": "上证50", "category": "宽基"},
    "000300.SH": {"name": "沪深300", "category": "宽基"},
    "000905.SH": {"name": "中证500", "category": "宽基"},
    "000852.SH": {"name": "中证1000", "category": "宽基"},
    "000906.SH": {"name": "中证800", "category": "宽基"},
    "000903.SH": {"name": "中证100", "category": "宽基"},
}

# 申万一级行业
SW_INDUSTRIES = [
    ("801010", "农林牧渔"),
    ("801020", "采掘"),
    ("801030", "化工"),
    ("801040", "钢铁"),
    ("801050", "有色金属"),
    ("801080", "电子"),
    ("801110", "家用电器"),
    ("801120", "食品饮料"),
    ("801130", "纺织服装"),
    ("801140", "轻工制造"),
    ("801150", "医药生物"),
    ("801160", "汽车"),
    ("801170", "公用事业"),
    ("801180", "房地产"),
    ("801200", "商业贸易"),
    ("801210", "餐饮旅游"),
    ("801230", "建筑材料"),
    ("801710", "建筑装饰"),
    ("801720", "电气设备"),
    ("801730", "国防军工"),
    ("801740", "计算机"),
    ("801750", "电子"),
    ("801760", "传媒"),
    ("801770", "通信"),
    ("801780", "非银金融"),
    ("801790", "银行"),
    ("801880", "汽车"),
    ("801950", "煤炭"),
    ("801960", "石油石化"),
    ("801970", "环保"),
    ("801980", "美容护理"),
]


# =============================================================================
# ODS Fetcher - 原始数据拉取
# =============================================================================

def generate_trading_calendar(start_year: int = 2005, end_year: int = None) -> pd.DataFrame:
    """生成A股交易日历"""
    if end_year is None:
        end_year = date.today().year
    
    # 法定节假日
    holidays = set()
    
    for year in range(start_year, end_year + 1):
        holidays.add(f"{year}-01-01")
        holidays.add(f"{year}-01-02")
        holidays.add(f"{year}-01-03")
        for day in range(1, 8):
            holidays.add(f"{year}-02-{day:02d}")
        holidays.add(f"{year}-04-04")
        holidays.add(f"{year}-04-05")
        holidays.add(f"{year}-04-06")
        holidays.add(f"{year}-05-01")
        holidays.add(f"{year}-05-02")
        holidays.add(f"{year}-05-03")
        holidays.add(f"{year}-06-{22 + year % 3:02d}")
        holidays.add(f"{year}-09-{15 + year % 2:02d}")
        for day in range(1, 8):
            holidays.add(f"{year}-10-{day:02d}")
    
    records = []
    current = date(start_year, 1, 1)
    end = date(end_year + 1, 1, 1)
    
    while current < end:
        if current.weekday() < 5:
            date_str = current.strftime("%Y-%m-%d")
            records.append({
                "date": date_str,
                "exchange": "ALL",
                "is_trading_day": date_str not in holidays,
                "fetched_at": datetime.now(),
            })
        current += timedelta(days=1)
    
    return pd.DataFrame(records)


def fetch_instruments_baostock() -> pd.DataFrame:
    """从baostock获取股票列表"""
    try:
        import baostock as bs
        
        lg = bs.login()
        rs = bs.query_all_stock(day=date.today().strftime("%Y-%m-%d"))
        
        records = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if row[1] in ["1", "2", "3", "4", "5"]:
                code = row[0].split(".")[1]
                exchange = "SSE" if row[0].startswith("sh") else "SZSE"
                records.append({
                    "symbol": f"{code}.{exchange}",
                    "name": row[2],
                    "list_date": None,
                    "delist_date": None,
                    "market": exchange,
                    "fetched_at": datetime.now(),
                })
        
        bs.logout()
        return pd.DataFrame(records)
    
    except ImportError:
        return pd.DataFrame()


def fetch_instruments_akshare() -> pd.DataFrame:
    """从AKShare获取股票列表"""
    try:
        import akshare as ak
        
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
                "fetched_at": datetime.now(),
            })
        
        return pd.DataFrame(records)
    
    except ImportError:
        return pd.DataFrame()


def fetch_index_components_baostock(index_code: str) -> pd.DataFrame:
    """从baostock获取指数成分股"""
    try:
        import baostock as bs
        
        bs_code = index_code.replace(".SH", ".sh").replace(".SZ", ".sz")
        lg = bs.login()
        
        rs = bs.query_index_stock_weight(bs_code)
        
        records = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code = row["stockCode"].split(".")[1]
            exchange = "SSE" if row["stockCode"].startswith("sh") else "SZSE"
            index_name = INDICES.get(index_code, {}).get("name", index_code)
            records.append({
                "index_code": index_code,
                "index_name": index_name,
                "symbol": f"{code}.{exchange}",
                "in_date": row["inDate"],
                "out_date": None if row["outDate"] == "" else row["outDate"],
                "weight": float(row["weight"]) if row["weight"] else 0.0,
                "source": "baostock",
                "fetched_at": datetime.now(),
            })
        
        bs.logout()
        return pd.DataFrame(records)
    
    except ImportError:
        return pd.DataFrame()


def fetch_ohlcv_baostock(
    symbols: list[str],
    start: str,
    end: str,
    adjust: str = "2",  # 2=前复权
) -> pd.DataFrame:
    """从baostock获取K线数据"""
    try:
        import baostock as bs
        
        lg = bs.login()
        all_records = []
        
        for symbol in symbols:
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
                    "adjust_flag": adjust,
                    "source": "baostock",
                    "fetched_at": datetime.now(),
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
    """从AKShare获取K线数据"""
    try:
        import akshare as ak
        
        code = symbol.split(".")[0]
        
        df = ak.stock_zh_a_hist(
            symbol=code,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=adjust,
        )
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.rename(columns={
            "日期": "date",
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
        df["adjust_flag"] = "2" if adjust == "qfq" else "1"
        df["source"] = "akshare"
        df["fetched_at"] = datetime.now()
        
        return df[["date", "symbol", "open", "high", "low", "close", "volume", 
                   "amount", "turnover_rate", "pct_change", "adjust_flag", "source", "fetched_at"]]
    
    except ImportError:
        return pd.DataFrame()


# =============================================================================
# Init Functions - 初始化函数
# =============================================================================

def init_ods_calendars(db: QuantDB, args) -> int:
    """初始化ODS日历"""
    print("📅 [ODS] 拉取交易日历...")
    start_time = datetime.now()
    
    df = generate_trading_calendar(2005, date.today().year)
    print(f"   生成 {len(df)} 条日历记录")
    
    rows = db.import_ods_calendars(df, source="generated")
    
    end_time = datetime.now()
    print(f"✅ [ODS] 日历导入完成: {rows} 条 ({(end_time - start_time).total_seconds():.2f}s)")
    return rows


def init_ods_instruments(db: QuantDB, args) -> int:
    """初始化ODS股票信息"""
    print("📋 [ODS] 拉取股票列表...")
    start_time = datetime.now()
    
    if args.source == "baostock":
        df = fetch_instruments_baostock()
    else:
        df = fetch_instruments_akshare()
    
    if df.empty:
        print("⚠️ 未获取到股票信息")
        return 0
    
    print(f"   获取 {len(df)} 只股票")
    rows = db.import_ods_instruments(df, source=args.source)
    
    print(f"✅ [ODS] 股票信息导入完成: {rows} 条 ({(datetime.now() - start_time).total_seconds():.2f}s)")
    return rows


def init_ods_index_components(db: QuantDB, args) -> int:
    """初始化ODS指数成分"""
    print("📊 [ODS] 拉取指数成分...")
    start_time = datetime.now()
    
    total = 0
    for code in INDICES.keys():
        print(f"   获取 {code} ({INDICES[code]['name']})...")
        df = fetch_index_components_baostock(code)
        if not df.empty:
            db.import_ods_index_components(df, source="baostock")
            total += len(df)
    
    print(f"✅ [ODS] 指数成分导入完成: {total} 条 ({(datetime.now() - start_time).total_seconds():.2f}s)")
    return total


def init_ods_ohlcv(db: QuantDB, args) -> int:
    """初始化ODS K线数据"""
    print("📈 [ODS] 拉取K线数据...")
    start_time = datetime.now()
    
    end = args.end or date.today().strftime("%Y-%m-%d")
    
    df = db.get_instruments()
    symbols = df["symbol"].tolist() if not df.empty else []
    
    print(f"   股票数: {len(symbols)}, 日期范围: {args.start} ~ {end}")
    
    total = 0
    batch_size = 50
    
    for i in range(0, min(len(symbols), 100), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"   处理 {i + 1} ~ {i + len(batch)}...")
        
        if args.source == "baostock":
            ohlcv_df = fetch_ohlcv_baostock(batch, args.start, end, adjust="2")
        else:
            all_dfs = []
            for sym in batch:
                ohlcv_df = fetch_ohlcv_akshare(sym, args.start, end)
                if not ohlcv_df.empty:
                    all_dfs.append(ohlcv_df)
            ohlcv_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        
        if not ohlcv_df.empty:
            db.import_ods_ohlcv(ohlcv_df, source=args.source)
            total += len(ohlcv_df)
    
    print(f"✅ [ODS] K线导入完成: {total} 条 ({(datetime.now() - start_time).total_seconds():.2f}s)")
    return total


def init_dwd(db: QuantDB, args) -> dict:
    """初始化DWD明细数据层"""
    print("\n" + "=" * 50)
    print("🔄 [DWD] 清洗转换数据...")
    print("=" * 50)
    start_time = datetime.now()
    
    results = {}
    
    print("📅 转换日历...")
    results["calendars"] = db.transform_calendars_ods_to_dwd()
    print(f"   ✅ {results['calendars']} 条")
    
    print("📋 转换股票信息...")
    results["instruments"] = db.transform_instruments_ods_to_dwd()
    print(f"   ✅ {results['instruments']} 条")
    
    print("📊 转换指数成分...")
    results["index_components"] = db.transform_index_components_ods_to_dwd()
    print(f"   ✅ {results['index_components']} 条")
    
    print("📈 转换K线数据 (前复权)...")
    results["ohlcv"] = db.transform_ohlcv_ods_to_dwd()
    print(f"   ✅ {results['ohlcv']} 条")
    
    print(f"\n✅ [DWD] 转换完成 ({(datetime.now() - start_time).total_seconds():.2f}s)")
    return results


def init_dws(db: QuantDB, args) -> dict:
    """初始化DWS汇总数据层"""
    print("\n" + "=" * 50)
    print("🔄 [DWS] 聚合汇总数据...")
    print("=" * 50)
    start_time = datetime.now()
    
    results = {}
    
    print("📊 聚合月度统计...")
    results["monthly"] = db.aggregate_monthly_stats()
    print(f"   ✅ {results['monthly']} 条")
    
    print(f"\n✅ [DWS] 汇总完成 ({(datetime.now() - start_time).total_seconds():.2f}s)")
    return results


def validate_data(db: QuantDB) -> dict:
    """校验数据"""
    print("\n" + "=" * 50)
    print("🔍 [校验] 数据质量检查...")
    print("=" * 50)
    
    results = db.validate_all()
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print(f"\n校验结果: {passed} 通过, {failed} 失败")
    
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"   {status} {r.rule.name}: {r.actual}")
    
    return {"total": len(results), "passed": passed, "failed": failed}


# =============================================================================
# Main - 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="初始化量化数据库 (分层架构)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "ods", "dwd", "dws", "validate", "calendar", "instruments", "indices", "ohlcv"],
        default="full",
        help="初始化模式",
    )
    parser.add_argument("--start", type=str, default="2005-01-01", help="K线开始日期")
    parser.add_argument("--end", type=str, default=None, help="K线结束日期")
    parser.add_argument("--source", type=str, choices=["baostock", "akshare"], default="baostock", help="数据源")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--batch-size", type=int, default=10000, help="批量导入大小")
    parser.add_argument("--force", action="store_true", help="强制重新初始化")
    parser.add_argument("--skip-validation", action="store_true", help="跳过校验")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("量化数据库初始化 (分层架构)")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"模式: {args.mode}")
    print(f"数据源: {args.source}")
    print("=" * 60)
    
    # 确保目录存在
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据库
    db = QuantDB(args.db)
    
    if args.force:
        print("⚠️ 强制模式，将重建数据库...")
        db.init_schema()
    
    results = {}
    
    # ODS层
    if args.mode in ["full", "ods"]:
        results["ods_calendars"] = init_ods_calendars(db, args)
        results["ods_instruments"] = init_ods_instruments(db, args)
        results["ods_index_components"] = init_ods_index_components(db, args)
        results["ods_ohlcv"] = init_ods_ohlcv(db, args)
    
    elif args.mode == "calendar":
        results["ods_calendars"] = init_ods_calendars(db, args)
    
    elif args.mode == "instruments":
        results["ods_instruments"] = init_ods_instruments(db, args)
    
    elif args.mode == "indices":
        results["ods_index_components"] = init_ods_index_components(db, args)
    
    elif args.mode == "ohlcv":
        results["ods_ohlcv"] = init_ods_ohlcv(db, args)
    
    # DWD层
    if args.mode in ["full", "dwd"]:
        results["dwd"] = init_dwd(db, args)
    
    # DWS层
    if args.mode in ["full", "dws"]:
        results["dws"] = init_dws(db, args)
    
    # 校验
    if args.mode == "validate":
        results["validation"] = validate_data(db)
    elif args.mode == "full" and not args.skip_validation:
        results["validation"] = validate_data(db)
    
    # 显示最终统计
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)
    
    info = db.info()
    print("\n数据库信息:")
    for layer, tables in info.items():
        print(f"\n  [{layer}]")
        for table, data in tables.items():
            print(f"    {table}: {data['rows']} 条")
    
    print(f"\n数据库路径: {args.db}")
    
    db.close()


if __name__ == "__main__":
    main()
