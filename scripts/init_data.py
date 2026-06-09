#!/usr/bin/env python3
"""数据初始化脚本 - 初始化量化数据库 (3层架构 + ETL分离)

分层设计:
- ODS (原始数据层): 按数据源分表 (ods_xxx_baostock)
- DWD (明细数据层): 标准化数据
- APP (应用数据层): 聚合统计、因子数据
- ETL (数据迁移): 独立的迁移脚本

Usage:
    # 全量初始化 (ODS + ETL + APP)
    python scripts/init_data.py --mode full --db data/quant.db

    # 仅拉取原始数据到ODS
    python scripts/init_data.py --mode ods --db data/quant.db

    # 仅执行ETL迁移
    python scripts/init_data.py --mode etl --db data/quant.db

    # 仅初始化K线 (不限量全市场)
    python scripts/init_data.py --mode ohlcv --db data/quant.db --start 2005-01-01
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB
from factor_pipeline.data.etl import ETLPipeline
from factor_pipeline.data.utils import norm_code


# =============================================================================
# Configuration - 配置
# =============================================================================

INDICES = {
    "000016": {"name": "上证50", "category": "宽基", "pool_name": "sse50"},
    "000300": {"name": "沪深300", "category": "宽基", "pool_name": "csi300"},
    "000905": {"name": "中证500", "category": "宽基", "pool_name": "csi500"},
    "000852": {"name": "中证1000", "category": "宽基", "pool_name": "csi1000"},
}


# =============================================================================
# Data Fetchers - 数据拉取 (不关心ODS表名)
# =============================================================================

def generate_trading_calendar(start_year: int = 2005, end_year: int = None) -> pd.DataFrame:
    """生成A股交易日历"""
    if end_year is None:
        end_year = date.today().year
    
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
            })
        current += timedelta(days=1)
    
    return pd.DataFrame(records)


# Fallback list of well-known A-share stocks (used when bs.query_all_stock
# returns 0 rows, which is the case with the current baostock server).
# These are CSI 300 top constituents + a few popular mid-caps. The K-line API
# (bs.query_history_k_data_plus) has been verified to work for all of these.
_FALLBACK_INSTRUMENTS = [
    # 上证 50 / 沪深 300 头部
    ("600000", "浦发银行"), ("600036", "招商银行"), ("600519", "贵州茅台"),
    ("601318", "中国平安"), ("601398", "工商银行"), ("601988", "中国银行"),
    ("600028", "中国石化"), ("601857", "中国石油"), ("600050", "中国联通"),
    ("601628", "中国人寿"), ("600030", "中信证券"), ("600276", "恒瑞医药"),
    ("600887", "伊利股份"), ("601888", "中国中免"), ("600585", "海螺水泥"),
    ("601012", "隆基绿能"), ("600900", "长江电力"), ("601166", "兴业银行"),
    ("600196", "复星医药"), ("601088", "中国神华"), ("601668", "中国建筑"),
    ("600690", "海尔智家"), ("600104", "上汽集团"), ("601728", "中国电信"),
    ("601800", "中国交建"), ("600048", "保利发展"), ("601138", "工业富联"),
    ("600436", "片仔癀"), ("600745", "闻泰科技"), ("600438", "通威股份"),
    ("601899", "紫金矿业"), ("601318", "中国平安"), ("600905", "三峡能源"),
    ("601658", "邮储银行"), ("601288", "农业银行"), ("600016", "民生银行"),
    ("600000", "浦发银行"), ("601169", "北京银行"), ("600015", "华夏银行"),
    ("600837", "海通证券"), ("601066", "中信建投"), ("601995", "中金公司"),
    # 深证 100 / 沪深 300 头部
    ("000001", "平安银行"), ("000002", "万科A"), ("000063", "中兴通讯"),
    ("000333", "美的集团"), ("000651", "格力电器"), ("000858", "五粮液"),
    ("000725", "京东方A"), ("000768", "中航西飞"), ("000876", "新希望"),
    ("000938", "紫光股份"), ("000963", "华东医药"), ("000977", "浪潮信息"),
    ("002230", "科大讯飞"), ("002415", "海康威视"), ("002475", "立讯精密"),
    ("002594", "比亚迪"), ("002714", "牧原股份"), ("002812", "恩捷股份"),
    ("300750", "宁德时代"), ("300059", "东方财富"), ("300015", "爱尔眼科"),
    ("300122", "智飞生物"), ("300124", "汇川技术"), ("300142", "沃森生物"),
    ("300144", "宋城演艺"), ("300347", "泰格医药"), ("300408", "三环集团"),
    ("300413", "芒果超媒"), ("300498", "温氏股份"), ("300601", "康泰生物"),
    ("300760", "迈瑞医疗"), ("300782", "卓胜微"), ("300888", "稳健医疗"),
    ("300999", "金龙鱼"),
]


def _get_fallback_instruments() -> pd.DataFrame:
    """Build instruments DataFrame from hardcoded fallback list.

    Symbols are de-duplicated and assigned to SSE/SZSE/BSE based on prefix.
    """
    seen = set()
    records = []
    for code, name in _FALLBACK_INSTRUMENTS:
        if code in seen:
            continue
        seen.add(code)
        if code.startswith(("60", "68", "90")):
            exchange = "SSE"
        elif code.startswith(("00", "30", "20")):
            exchange = "SZSE"
        elif code.startswith(("43", "83", "87", "92")):
            exchange = "BSE"
        else:
            exchange = "SZSE"
        records.append({
            "symbol": f"{code}.{exchange}",
            "name": name,
            "market": exchange,
        })
    return pd.DataFrame(records)


def fetch_instruments_akshare() -> pd.DataFrame:
    """从 akshare 获取全 A 股列表 (stock_info_a_code_name)"""
    import akshare as ak

    print("  [akshare] 获取全 A 股列表 ...", end=" ", flush=True)
    df = ak.stock_info_a_code_name()
    print(f"{len(df)} 只")

    # stock_info_a_code_name 只返回 code/name，需要自己推导 market/list_date
    # 用 stock_info_sh_name_code / stock_info_sz_name_code 补充 list_date
    records = []
    for _, row in df.iterrows():
        code = str(row["code"]).strip().zfill(6)
        sym = norm_code(code)
        if not sym:
            continue
        records.append({
            "symbol": sym,
            "name": str(row.get("name", "")),
            "list_date": None,
            "delist_date": None,
            "market": sym.split(".")[1],
        })

    # 补充 list_date: 尝试用 akshare stock_info_sh_name_code / stock_info_sz_name_code
    try:
        sh_df = ak.stock_info_sh_name_code()
        sh_map = {}
        for _, r in sh_df.iterrows():
            c = str(r.get("证券代码", "")).zfill(6)
            ld = r.get("上市日期")
            if ld and str(ld) != "nan" and str(ld) != "NaT":
                sh_map[c] = str(ld)
        for rec in records:
            code = rec["symbol"].split(".")[0]
            if code in sh_map:
                rec["list_date"] = sh_map[code]
    except Exception:
        pass

    try:
        sz_df = ak.stock_info_sz_name_code()
        sz_map = {}
        for _, r in sz_df.iterrows():
            c = str(r.get("A股代码", r.get("代码", ""))).zfill(6)
            ld = r.get("上市日期", r.get("A股上市日期", ""))
            if ld and str(ld) != "nan" and str(ld) != "NaT":
                sz_map[c] = str(ld)
        for rec in records:
            code = rec["symbol"].split(".")[0]
            if code in sz_map:
                rec["list_date"] = sz_map[code]
    except Exception:
        pass

    return pd.DataFrame(records)


def fetch_index_components_akshare(index_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 akshare 获取指数成分股 (权重 + 纳入日期)

    Returns:
        (weight_df, cons_df) — 权重数据 + 成分股列表(含纳入日期)
    """
    import akshare as ak

    # 主数据源: CSIndex — 成分权重
    df_weight = ak.index_stock_cons_weight_csindex(symbol=index_code)
    # 辅助: Sina — 纳入日期
    df_cons = ak.index_stock_cons(symbol=index_code)

    return df_weight, df_cons


def fetch_shenwan_industries_bs() -> pd.DataFrame:
    """从 baostock 获取证监会行业分类 (CSRC 标准)

    baostock query_stock_industry 返回字段:
        [update_date, code, stock_name, industry_code_name, classification_type]
    字段3: 如 'J66货币金融服务', 'G56航空运输业' — 是证监会行业分类

    Returns:
        DataFrame with columns: [symbol, industry_name, industry_code]
    """
    import baostock as bs

    bs.login()
    rs = bs.query_stock_industry()
    records = []
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        code = str(row[1]).split(".")[-1]
        sym = norm_code(code)
        if sym:
            ind = str(row[3]) if len(row) > 3 and row[3] else ""
            records.append({
                "symbol": sym,
                "industry_name": ind,
                "industry_code": ind,
            })
    bs.logout()
    df = pd.DataFrame(records)
    # 过滤掉空行业
    df = df[df["industry_name"].str.len() > 0].reset_index(drop=True)
    print(f"  [baostock] 证监会行业分类: {len(df)} 条, {df['industry_name'].nunique() if not df.empty else 0} 个行业")
    return df


def _build_index_constituents(index_code: str, pool_name: str, df_weight: pd.DataFrame, df_cons: pd.DataFrame,
                               latest_only: bool = True) -> pd.DataFrame:
    """将 CSIndex 权重 + Sina 纳入日期合并为 pool_registration 格式"""
    import akshare as ak

    # 取最新日期
    if latest_only:
        latest = df_weight["日期"].max()
        df_weight = df_weight[df_weight["日期"] == latest]

    # Sina in_dates
    in_dates = {}
    for _, row in df_cons.iterrows():
        sym = norm_code(str(row["品种代码"]))
        if sym:
            in_dates[sym] = str(row["纳入日期"])

    latest_date = df_weight["日期"].max()
    fallback_date = str(latest_date) if not hasattr(latest_date, "strftime") else latest_date.strftime("%Y-%m-%d")
    index_name = INDICES[index_code]["name"]

    records = []
    for _, row in df_weight.iterrows():
        sym = norm_code(str(row["成分券代码"]))
        if not sym:
            continue
        w = float(row.get("权重", 0) or 0)
        records.append({
            "index_code": index_code,
            "index_name": index_name,
            "pool_name": pool_name,
            "symbol": sym,
            "in_date": in_dates.get(sym, fallback_date),
            "out_date": None,
            "weight": w,
            "source": "akshare",
        })
    return pd.DataFrame(records)


def _to_bs_code(symbol: str) -> str:
    """Convert '600000.SSE' / '000001.SZSE' / '830001.BSE' to 'sh.600000' etc."""
    parts = symbol.split(".")
    if len(parts) == 2:
        code, market = parts
        if market == "SSE":
            return f"sh.{code}"
        elif market == "SZSE":
            return f"sz.{code}"
        elif market == "BSE":
            return f"bj.{code}"
    # Fallback: try prefix matching
    code = parts[0]
    if code.startswith(("60", "68", "90")):
        return f"sh.{code}"
    elif code.startswith(("43", "83", "87", "92")):
        return f"bj.{code}"
    return f"sz.{code}"


def fetch_ohlcv_baostock(
    symbols: list[str],
    start: str,
    end: str,
    adjust: str = "2",  # 默认前复权
) -> pd.DataFrame:
    """从baostock获取K线数据"""
    try:
        import baostock as bs

        lg = bs.login()
        all_records = []

        for symbol in symbols:
            bs_code = _to_bs_code(symbol)
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
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
                })
        
        bs.logout()
        return pd.DataFrame(all_records)
    except ImportError:
        return pd.DataFrame()


# =============================================================================
# Init Functions - 初始化函数
# =============================================================================

def init_ods(db: QuantDB, args) -> dict:
    """初始化ODS原始数据层"""
    print("\n" + "=" * 50)
    print("📦 [ODS] 拉取原始数据")
    print("=" * 50)
    
    source = args.source
    results = {}
    
    # 创建ODS表
    db.create_ods_tables(source)
    
    # ============================
    # 1. 日历
    # ============================
    print("\n📅 拉取日历...")
    df = generate_trading_calendar(2005, date.today().year)
    rows = db.import_ods(source, "calendars", df)
    results["calendars"] = rows
    print(f"   ✅ {rows} 条")
    
    # ============================
    # 2. 全量股票列表
    # ============================
    print("\n📋 拉取全量A股列表...")
    df_instr = pd.DataFrame()
    try:
        df_instr = fetch_instruments_akshare()
        if df_instr.empty:
            raise ValueError("空结果")
    except Exception as e:
        print(f"   ⚠️ akshare 失败: {e}")
        print("   ⚠️ 回退到 baostock + 内置列表...")
        try:
            import baostock as bs
            bs.login()
            rs = bs.query_all_stock(day=date.today().strftime("%Y-%m-%d"))
            records = []
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if row[1] in ["1", "2", "3", "4", "5"]:
                    code = row[0].split(".")[1]
                    exchange = "SSE" if row[0].startswith("sh") else "SZSE"
                    records.append({"symbol": f"{code}.{exchange}", "name": row[2], "market": exchange})
            bs.logout()
            if records:
                df_instr = pd.DataFrame(records)
        except Exception:
            pass
        if df_instr.empty:
            df_instr = _get_fallback_instruments()
    
    if not df_instr.empty:
        rows = db.import_ods(source, "instruments", df_instr)
        results["instruments"] = rows
        print(f"   ✅ {rows} 只股票")
    else:
        results["instruments"] = 0
        print("   ⚠️ 未获取到股票")
    
    # ============================
    # 3. 指数成分股 + 申万行业
    # ============================
    print("\n📊 拉取指数成分 + 行业分类...")
    
    total_components = 0
    total_industries = 0
    
    for code, info in INDICES.items():
        try:
            df_w, df_c = fetch_index_components_akshare(code)
            df_combined = _build_index_constituents(code, info["pool_name"], df_w, df_c, latest_only=True)
            if not df_combined.empty:
                db.import_ods(source, "index_components", df_combined)
                total_components += len(df_combined)
                print(f"   ✅ {info['name']} ({code}): {len(df_combined)} 只")
        except Exception as e:
            print(f"   ⚠️ {info['name']} ({code}) 获取失败: {e}")
    
    # 申万行业分类 (写入 dwd_instruments_pool_registration 直接用 pool_name=sw_xxx)
    if args.shenwan:
        print("\n🏭 拉取申万行业分类...")
        try:
            df_sw = fetch_shenwan_industries_bs()
            if not df_sw.empty:
                conn = db.connect()
                written = 0
                for ind_name in df_sw["industry_name"].unique():
                    pool = f"csrc_{ind_name}"
                    subset = df_sw[df_sw["industry_name"] == ind_name]
                    for _, r in subset.iterrows():
                        try:
                            conn.execute("""
                                INSERT INTO dwd_instruments_pool_registration
                                (pool_name, symbol, in_date, out_date, weight, source, updated_at)
                                VALUES (?, ?, ?, NULL, 0, 'baostock', now())
                                ON CONFLICT(pool_name, symbol, in_date) DO UPDATE SET
                                    out_date = NULL, updated_at = now()
                            """, [pool, r["symbol"], date.today().strftime("%Y-%m-%d")])
                            written += 1
                        except Exception:
                            pass
                conn.commit()
                total_industries = written
                print(f"   ✅ 申万行业: {df_sw['industry_name'].nunique()} 个行业, {written} 条记录")
        except Exception as e:
            print(f"   ⚠️ 申万行业获取失败: {e}")
    
    results["index_components"] = total_components
    results["shenwan_industries"] = total_industries
    
    # ============================
    # 4. K线数据 (不限量)
    # ============================
    if args.mode == "ohlcv" or args.mode == "full":
        print("\n📈 拉取K线数据...")
        end = args.end or date.today().strftime("%Y-%m-%d")
        
        df_syms = db.query(f"SELECT symbol FROM ods_instruments_{source}")
        symbols = df_syms["symbol"].tolist()
        
        if args.limit and args.limit < len(symbols):
            symbols = symbols[:args.limit]
        
        print(f"   股票数: {len(symbols)}, 日期: {args.start} ~ {end}")
        
        total = 0
        batch_size = 50
        start_time = time.time()
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            print(f"   处理 {i + 1} ~ {i + len(batch)} / {len(symbols)}...", end=" ", flush=True)
            df = fetch_ohlcv_baostock(batch, args.start, end, adjust=args.adjust)
            if not df.empty:
                rows = db.import_ods(source, "daily_ohlcv", df)
                total += rows
                print(f"+{rows} 条", flush=True)
            else:
                print("(空)", flush=True)
        
        elapsed = time.time() - start_time
        results["daily_ohlcv"] = total
        print(f"   ✅ {total} 条K线 (耗时 {elapsed:.0f}s)")
    
    return results


def init_etl(db: QuantDB, args) -> dict:
    """执行ETL迁移"""
    print("\n" + "=" * 50)
    print("🔄 [ETL] 数据迁移")
    print("=" * 50)
    
    etl = ETLPipeline(db)
    
    if args.source:
        results = etl.run(source=args.source)
    else:
        results = etl.run()
    
    return results


def init_app(db: QuantDB, args) -> dict:
    """初始化APP应用数据层"""
    print("\n" + "=" * 50)
    print("📊 [APP] 聚合汇总")
    print("=" * 50)
    
    results = {}
    
    print("📅 月度统计...")
    rows = db.aggregate_monthly_stats()
    results["monthly"] = rows
    print(f"   ✅ {rows} 条")
    
    return results


def validate_data(db: QuantDB) -> dict:
    """校验数据"""
    print("\n" + "=" * 50)
    print("🔍 [校验] 数据质量检查")
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
        description="初始化量化数据库 (3层架构 + ETL分离)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "ods", "etl", "app", "validate", "calendar", "ohlcv", "shenwan"],
        default="full",
        help="初始化模式",
    )
    parser.add_argument("--start", type=str, default="2005-01-01", help="K线开始日期")
    parser.add_argument("--end", type=str, default=None, help="K线结束日期")
    parser.add_argument("--source", type=str, default="baostock", help="数据源")
    parser.add_argument("--force", action="store_true", help="强制重新初始化")
    parser.add_argument("--skip-validation", action="store_true", help="跳过校验")
    parser.add_argument("--shenwan", action="store_true", help="拉取申万行业分类")
    parser.add_argument("--limit", type=int, default=0, help="K线股票数量限制 (0=不限量)")
    parser.add_argument("--adjust", type=str, default="2", help="复权方式 (2=前复权, 3=后复权)")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("量化数据库初始化 (3层架构 + ETL分离)")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"模式: {args.mode}")
    print(f"数据源: {args.source}")
    print("=" * 60)
    
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    
    db = QuantDB(args.db)
    
    if args.force:
        print("⚠️ 强制模式，将重建数据库...")
        db.init_schema()
    
    # 注册数据源
    db.register_source(args.source, priority=1)
    
    results = {}
    
    # 证监会行业专用模式 (baostock 提供的是 CSRC 分类)
    if args.mode == "shenwan":
        print("\n" + "=" * 50)
        print("[ODS] 仅拉取证监会行业分类")
        print("=" * 50)
        df_sw = fetch_shenwan_industries_bs()
        if not df_sw.empty:
            conn = db.connect()
            written = 0
            for ind_name in df_sw["industry_name"].unique():
                pool = f"csrc_{ind_name}"
                subset = df_sw[df_sw["industry_name"] == ind_name]
                for _, r in subset.iterrows():
                    try:
                        conn.execute("""
                            INSERT INTO dwd_instruments_pool_registration
                            (pool_name, symbol, in_date, out_date, weight, source, updated_at)
                            VALUES (?, ?, ?, NULL, 0, 'baostock', now())
                            ON CONFLICT(pool_name, symbol, in_date) DO UPDATE SET
                                out_date = NULL, updated_at = now()
                        """, [pool, r["symbol"], date.today().strftime("%Y-%m-%d")])
                        written += 1
                    except Exception:
                        pass
            conn.commit()
            print(f"\n[OK] 证监会行业: {df_sw['industry_name'].nunique()} 个行业, {written} 条记录")
        db.close()
        return
    
    # ODS层
    if args.mode in ["full", "ods", "calendar", "ohlcv"]:
        results["ods"] = init_ods(db, args)
    
    # ETL迁移
    if args.mode in ["full", "etl"]:
        results["etl"] = init_etl(db, args)
    
    # APP层
    if args.mode in ["full", "app"]:
        results["app"] = init_app(db, args)
    
    # 校验
    if args.mode == "validate":
        results["validation"] = validate_data(db)
    elif args.mode == "full" and not args.skip_validation:
        results["validation"] = validate_data(db)
    
    # 显示最终统计
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)
    
    # 显示参数表
    print("\n" + "-" * 60)
    print("📋 参数表状态 (meta_table_params)")
    print("-" * 60)
    params = db.get_all_table_status()
    if not params.empty:
        for _, row in params.iterrows():
            source_str = f" [{row['source']}]" if row['source'] else ""
            print(f"  [{row['layer']}] {row['table_name']}{source_str}: {row['freshness']} ({row['last_update_records']} 条)")
    else:
        print("  (无记录)")
    
    info = db.info()
    print(f"\n数据库路径: {args.db}")
    print(f"数据源: {', '.join(info['sources'])}")
    print("\n表统计:")
    
    for layer, data in info["tables"].items():
        print(f"\n  [{layer}]")
        if isinstance(data, dict):
            for name, info_data in data.items():
                if isinstance(info_data, dict):
                    print(f"    {name}: {info_data.get('rows', 0)} 条")
                elif isinstance(info_data, list):
                    print(f"    {name}: {len(info_data)} 个表")
    
    db.close()


if __name__ == "__main__":
    main()
