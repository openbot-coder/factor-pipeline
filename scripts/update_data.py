#!/usr/bin/env python3
"""日更新脚本 - 增量更新量化数据库 (3层架构 + ETL分离)

分层更新策略:
- ODS: 拉取最新原始数据
- ETL: 独立的迁移脚本
- APP: 更新汇总统计

Usage:
    # 更新所有层
    python scripts/update_data.py --db data/quant.db

    # 仅拉取K线到ODS
    python scripts/update_data.py --db data/quant.db --ods-ohlcv

    # 仅执行ETL迁移
    python scripts/update_data.py --db data/quant.db --etl

    # 仅执行校验
    python scripts/update_data.py --db data/quant.db --check
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB
from factor_pipeline.data.etl import ETLPipeline


# =============================================================================
# Helpers
# =============================================================================

from factor_pipeline.data.utils import norm_code as _norm_code


# =============================================================================
# Logger - 日志配置
# =============================================================================

def setup_logger(name: str = "update_data") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# =============================================================================
# Data Fetchers - 数据拉取
# =============================================================================

def fetch_recent_ohlcv_baostock(
    symbols: list[str],
    days: int = 5,
) -> pd.DataFrame:
    """从baostock获取最近K线数据"""
    try:
        import baostock as bs
        
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        lg = bs.login()
        all_records = []
        
        for symbol in symbols:
            bs_code = f"sh.{symbol}" if ".SH" in symbol else f"sz.{symbol}"
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turnover,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
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
                    "adjust_flag": "2",
                })
        
        bs.logout()
        return pd.DataFrame(all_records)
    except ImportError:
        return pd.DataFrame()


def check_calendar_needs_update(db: QuantDB) -> tuple[bool, str | None]:
    """检查日历是否需要增量更新。

    规则: dwd_calendars 中 date > today 的交易日 < 10 条时触发更新.

    Returns:
        (need_update, max_date_str) — 是否需要更新, DWD 中最大日期.
    """
    try:
        row = db.query("""
            SELECT COUNT(*) AS cnt, MAX(date) AS max_dt
            FROM dwd_calendars
            WHERE date > CURRENT_DATE AND (sse OR szse)
        """)
    except Exception:
        return True, None  # 表不存在 → 需要全量

    remaining = row.iloc[0]["cnt"]
    max_dt = row.iloc[0]["max_dt"]

    need = remaining < 10
    if need:
        print(f"   ⚠️ 剩余未来交易日: {remaining} 条 (< 10), 触发更新")
    else:
        print(f"   ✅ 剩余未来交易日: {remaining} 条 (充足)")
    return need, str(max_dt) if max_dt and not pd.isna(max_dt) else None


def generate_calendar_range(start_date: str, end_date: str) -> pd.DataFrame:
    """生成指定日期范围内的 A 股交易日历数据 (ODS 格式)。

    仅周一至周五为潜在交易日，周末自动标记为 is_trading_day=False.
    调用方可通过 ETL 将 ODS 数据展开为按交易所分列的 DWD 格式.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # 基本法定假日集合（简化版，实际应接入交易所官方日历）
    holidays = set()
    for y in range(start.year, end.year + 2):
        holidays.add(f"{y}-01-01")
        for d in range(1, 8):
            holidays.add(f"{y}-02-{d:02d}")
        holidays.add(f"{y}-04-04")
        holidays.add(f"{y}-04-05")
        holidays.add(f"{y}-04-06")
        for d in range(1, 4):
            holidays.add(f"{y}-05-{d:02d}")
        holidays.add(f"{y}-09-{15 + y % 2:02d}")
        for d in range(1, 8):
            holidays.add(f"{y}-10-{d:02d}")

    records = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        is_weekday = current.weekday() < 5
        # 非周末且非节假日 = 交易日
        is_trading = is_weekday and date_str not in holidays

        records.append({
            "date": date_str,
            "exchange": "ALL",
            "is_trading_day": is_trading,
        })
        current += timedelta(days=1)

    return pd.DataFrame(records)


# =============================================================================
# Update Functions - 更新函数
# =============================================================================

def update_ods(db: QuantDB, source: str, args) -> dict:
    """更新ODS原始数据层"""
    print("\n" + "=" * 50)
    print(f"📦 [ODS] 更新原始数据 ({source})")
    print("=" * 50)
    
    results = {}
    
    # 确保ODS表存在
    db.create_ods_tables(source)
    
    # 更新日历
    print("\n📅 检查日历更新...")
    need_update, max_dt = check_calendar_needs_update(db)
    if need_update:
        # 确定补充范围: 从 DWD 最大日期到 +1 年
        if max_dt:
            start = max_dt
        else:
            start = "2005-01-01"
        end = (date.today() + timedelta(days=365)).strftime("%Y-%m-%d")
        print(f"   📥 补充日历: {start} → {end}")
        df = generate_calendar_range(start, end)
        if not df.empty:
            rows = db.import_ods(source, "calendars", df)
            results["calendars"] = rows
            print(f"   ✅ 写入 ODS: +{rows} 条")
            # 触发 ETL
            print("   🔄 触发日历 ETL...")
            from factor_pipeline.data.etl import ETLPipeline
            etl = ETLPipeline(db)
            etl._transform_calendars(source)
            print("   ✅ ETL 完成")
        else:
            results["calendars"] = 0
            print("   ⚠️ 无新数据可生成")
    else:
        results["calendars"] = 0
        print("   ✅ 日历已是最新")
    
    # 更新K线
    if args.ohlcv:
        print("\n📈 更新K线数据...")
        
        df = db.get_instruments(active_only=True)
        symbols = df["symbol"].tolist()[:100]  # 限制数量
        
        print(f"   股票数: {len(symbols)}")
        
        df = fetch_recent_ohlcv_baostock(symbols, days=args.days)
        if not df.empty:
            rows = db.import_ods(source, "daily_ohlcv", df)
            results["daily_ohlcv"] = rows
            print(f"   ✅ +{rows} 条")
        else:
            results["daily_ohlcv"] = 0
            print("   ⚠️ 无新数据")
    
    return results


def run_etl(db: QuantDB, source: str) -> dict:
    """执行ETL迁移"""
    print("\n" + "=" * 50)
    print(f"🔄 [ETL] 数据迁移 ({source})")
    print("=" * 50)
    
    etl = ETLPipeline(db)
    results = etl.run(source=source)
    
    return results


def update_app(db: QuantDB, args) -> dict:
    """更新APP应用数据层"""
    print("\n" + "=" * 50)
    print("📊 [APP] 更新汇总统计")
    print("=" * 50)
    
    results = {}
    
    today = date.today()
    start = f"{today.year}-{today.month:02d}-01"
    end = today.strftime("%Y-%m-%d")
    
    print(f"📅 更新月度统计: {start} ~ {end}")
    rows = db.aggregate_monthly_stats(start=start, end=end)
    results["monthly"] = rows
    print(f"   ✅ {rows} 条")
    
    return results


def update_pool_registration(db: QuantDB, args) -> dict:
    """检查并更新股票池进出数据"""
    import akshare as ak

    print("\n" + "=" * 50)
    print("📊 [POOL] 更新股票池进出")
    print("=" * 50)

    results = {}

    INDICES = [
        ("000016", "sse50",   "上证50"),
        ("000300", "csi300",  "沪深300"),
        ("000905", "csi500",  "中证500"),
        ("000852", "csi1000", "中证1000"),
    ]

    conn = db.connect()
    today_str = date.today().strftime("%Y-%m-%d")

    for idx_code, pool_name, idx_name in INDICES:
        print(f"\n  📈 {idx_name} ({pool_name})...", end=" ", flush=True)

        try:
            # 1. 获取当前最新成分（CSIndex）
            df_csindex = ak.index_stock_cons_weight_csindex(symbol=idx_code)
            # 2. 获取纳入日期（Sina）
            df_sina = ak.index_stock_cons(symbol=idx_code)
            sina_dates = {}
            for _, row in df_sina.iterrows():
                sym = _norm_code(str(row["品种代码"]))
                if sym:
                    sina_dates[sym] = str(row["纳入日期"])

            # 3. 构建当前成分集合
            current_symbols = set()
            for _, row in df_csindex.iterrows():
                sym = _norm_code(str(row["成分券代码"]))
                if sym:
                    current_symbols.add(sym)

            if not current_symbols:
                print("⚠️ 无数据")
                continue

            # 4. 查询数据库中的最新记录
            existing = set()
            for r in conn.execute(
                f"SELECT symbol FROM dwd_instruments_pool_registration WHERE pool_name='{pool_name}' AND out_date IS NULL"
            ).fetchall():
                existing.add(r[0])

            # 5. 检测剔除的股票 → 设置 out_date
            removed = existing - current_symbols
            for sym in removed:
                conn.execute(f"""
                    UPDATE dwd_instruments_pool_registration
                    SET out_date = '{today_str}', updated_at = now()
                    WHERE pool_name='{pool_name}' AND symbol='{sym}' AND out_date IS NULL
                """)

            # 6. 检测新纳入的股票 → 插入新记录
            added = current_symbols - existing
            for sym in added:
                w = 0.0
                for _, row in df_csindex.iterrows():
                    if _norm_code(str(row["成分券代码"])) == sym:
                        w = float(row.get("权重", 0) or 0)
                        break
                in_date = sina_dates.get(sym, today_str)
                conn.execute(f"""
                    INSERT INTO dwd_instruments_pool_registration
                    (pool_name, symbol, in_date, out_date, weight, source, updated_at)
                    VALUES ('{pool_name}', '{sym}', '{in_date}', NULL, {w}, 'akshare', now())
                """)

            # 7. 更新权重（现有成分权重可能变化）
            weight_changes = 0
            weight_map = {}
            for _, row in df_csindex.iterrows():
                sym = _norm_code(str(row["成分券代码"]))
                if sym:
                    weight_map[sym] = float(row.get("权重", 0) or 0)
            for sym in current_symbols & existing:
                w = weight_map.get(sym, 0)
                conn.execute(f"""
                    UPDATE dwd_instruments_pool_registration
                    SET weight = {w}, updated_at = now()
                    WHERE pool_name='{pool_name}' AND symbol='{sym}' AND out_date IS NULL
                    AND weight != {w}
                """)
                weight_changes += 1

            conn.commit()
            print(f"在池:{len(current_symbols)} 新增:{len(added)} 剔除:{len(removed)}", flush=True)
            results[pool_name] = {"in": len(current_symbols), "new": len(added), "removed": len(removed)}

        except Exception as e:
            print(f"❌ {e}", flush=True)
            results[pool_name] = {"error": str(e)}

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


def health_check(db: QuantDB) -> dict:
    """健康检查"""
    print("\n" + "=" * 50)
    print("🏥 健康检查")
    print("=" * 50)
    
    info = db.info()
    
    print(f"\n数据源: {', '.join(info['sources'])}")
    
    for layer, data in info["tables"].items():
        print(f"\n  [{layer}]")
        if isinstance(data, dict):
            for name, info_data in data.items():
                if isinstance(info_data, dict):
                    print(f"    {name}: {info_data.get('rows', 0)} 条")
                elif isinstance(info_data, list):
                    print(f"    {name}: {len(info_data)} 个表")
    
    return {"status": "ok"}


# =============================================================================
# Main - 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="增量更新量化数据库 (3层架构 + ETL分离)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整更新 (ODS + ETL + APP)
    python scripts/update_data.py --db data/quant.db

    # 仅ODS拉取
    python scripts/update_data.py --db data/quant.db --ods-ohlcv

    # 仅ETL迁移
    python scripts/update_data.py --db data/quant.db --etl

    # 仅校验
    python scripts/update_data.py --db data/quant.db --check
        """
    )
    
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument("--source", type=str, default="baostock", help="数据源")
    parser.add_argument("--ods-calendar", action="store_true", help="更新日历")
    parser.add_argument("--ods-ohlcv", action="store_true", help="更新K线")
    parser.add_argument("--pool", action="store_true", help="更新股票池进出")
    parser.add_argument("--etl", action="store_true", help="执行ETL迁移")
    parser.add_argument("--validate", action="store_true", help="执行校验")
    parser.add_argument("--check", action="store_true", help="仅健康检查")
    parser.add_argument("--days", type=int, default=5, help="K线更新天数")
    parser.add_argument("--dry-run", action="store_true", help="仅检查")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("量化数据库日更新 (3层架构 + ETL分离)")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"数据源: {args.source}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not Path(args.db).exists():
        print(f"❌ 数据库不存在: {args.db}")
        print("   请先运行初始化:")
        print(f"   python scripts/init_data.py --mode full --db {args.db}")
        sys.exit(1)
    
    db = QuantDB(args.db)
    
    # 仅健康检查
    if args.check:
        health_check(db)
        db.close()
        return
    
    # 仅干运行
    if args.dry_run:
        print("🔍 干运行模式...")
        health_check(db)
        db.close()
        return
    
    start_time = datetime.now()
    results = {}
    
    # ODS更新
    update_ods_flag = args.ods_calendar or args.ods_ohlcv
    if update_ods_flag or (not args.etl and not args.validate and not args.pool):
        results["ods"] = update_ods(db, args.source, args)
    
    # ETL迁移
    if args.etl or (not args.ods_calendar and not args.ods_ohlcv and not args.validate and not args.pool):
        results["etl"] = run_etl(db, args.source)
    
    # 股票池更新
    if args.pool or (not args.ods_calendar and not args.ods_ohlcv and not args.etl and not args.validate):
        results["pool"] = update_pool_registration(db, args)
    
    # APP更新
    if not args.ods_calendar and not args.ods_ohlcv and not args.validate and not args.pool:
        results["app"] = update_app(db, args)
    
    # 校验
    if args.validate:
        results["validation"] = validate_data(db)
    
    # 汇总
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("更新完成!")
    print("=" * 60)
    print(f"\n耗时: {duration:.2f} 秒")
    
    # 健康检查
    print("\n" + "-" * 60)
    health_check(db)
    
    # 参数表
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
    
    print("=" * 60)
    
    db.close()


if __name__ == "__main__":
    main()
