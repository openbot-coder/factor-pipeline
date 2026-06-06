#!/usr/bin/env python3
"""日更新脚本 - 增量更新量化数据库 (分层架构)

采用数仓分层更新策略:
- ODS: 拉取最新原始数据
- DWD: 增量转换清洗数据
- DWS: 更新汇总统计
- 校验: 数据质量检查

Usage:
    # 更新所有层
    python scripts/update_data.py --db data/quant.db

    # 仅更新ODS原始数据
    python scripts/update_data.py --db data/quant.db --ods-only

    # 仅更新K线并校验
    python scripts/update_data.py --db data/quant.db --ohlcv --validate

    # 仅执行校验
    python scripts/update_data.py --db data/quant.db --check

    # 定时任务示例 (crontab)
    # 每天 16:30 执行 (A股收盘后)
    # 30 16 * * 1-5 python /path/to/update_data.py --db /data/quant.db --ohlcv
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB, ValidationResult


# =============================================================================
# Logger - 日志配置
# =============================================================================

def setup_logger(name: str = "update_data") -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# =============================================================================
# ODS Updater - 原始数据层更新
# =============================================================================

def update_ods_calendars(db: QuantDB, days: int = 7) -> int:
    """更新日历 (ODS)"""
    logger.info("📅 [ODS] 检查交易日历更新...")
    
    # 获取最新日历日期
    trading_days = db.get_trading_days()
    if trading_days:
        latest_date = max(trading_days)
    else:
        latest_date = "2005-01-01"
    
    logger.info(f"   最新日历日期: {latest_date}")
    
    today = date.today()
    latest = datetime.strptime(latest_date, "%Y-%m-%d").date()
    
    if latest >= today:
        logger.info("   日历已是最新")
        return 0
    
    # 生成新日期
    new_dates = []
    current = latest + timedelta(days=1)
    while current <= today:
        if current.weekday() < 5:
            new_dates.append({
                "date": current.strftime("%Y-%m-%d"),
                "exchange": "ALL",
                "is_trading_day": True,
                "fetched_at": datetime.now(),
            })
        current += timedelta(days=1)
    
    if not new_dates:
        return 0
    
    df = pd.DataFrame(new_dates)
    rows = db.import_ods_calendars(df, source="generated")
    logger.info(f"✅ [ODS] 日历更新: +{rows} 条")
    return rows


def update_ods_ohlcv(db: QuantDB, source: str = "baostock", days: int = 5) -> int:
    """更新K线数据 (ODS)"""
    logger.info(f"📈 [ODS] 从 {source} 更新K线数据...")
    
    # 确定日期范围
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # 获取活跃股票
    df = db.get_instruments(active_only=True)
    symbols = df["symbol"].tolist()
    logger.info(f"   活跃股票: {len(symbols)}")
    
    try:
        if source == "baostock":
            import baostock as bs
            
            lg = bs.login()
            total = 0
            updated = 0
            
            for symbol in symbols[:100]:  # 限制数量
                try:
                    bs_code = f"sh.{symbol}" if ".SH" in symbol else f"sz.{symbol}"
                    
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,open,high,low,close,volume,amount,turnover,pctChg",
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag="2",  # 前复权
                    )
                    
                    records = []
                    while rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        records.append({
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
                            "source": "baostock",
                            "fetched_at": datetime.now(),
                        })
                    
                    if records:
                        db.import_ods_ohlcv(pd.DataFrame(records), source="baostock")
                        total += len(records)
                        updated += 1
                        
                except Exception as e:
                    logger.debug(f"   {symbol} 更新失败: {e}")
            
            bs.logout()
            logger.info(f"✅ [ODS] K线更新: {updated} 只股票, {total} 条记录")
            return total
            
        else:  # akshare
            import akshare as ak
            
            total = 0
            updated = 0
            
            for symbol in symbols[:50]:
                try:
                    code = symbol.split(".")[0]
                    
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq",
                    )
                    
                    if df is not None and not df.empty:
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
                        df["adjust_flag"] = "2"
                        df["source"] = "akshare"
                        df["fetched_at"] = datetime.now()
                        
                        db.import_ods_ohlcv(df, source="akshare")
                        total += len(df)
                        updated += 1
                        
                except Exception as e:
                    logger.debug(f"   {symbol} 更新失败: {e}")
            
            logger.info(f"✅ [ODS] K线更新: {updated} 只股票, {total} 条记录")
            return total
            
    except ImportError as e:
        logger.warning(f"⚠️ {source} 未安装: {e}")
        return 0
    except Exception as e:
        logger.error(f"❌ K线更新失败: {e}")
        return 0


def update_ods_instruments(db: QuantDB, source: str = "baostock") -> int:
    """更新股票列表 (ODS)"""
    logger.info("📋 [ODS] 检查股票列表更新...")
    # TODO: 实现增量检查新上市/退市股票
    logger.info("   股票列表更新待实现")
    return 0


def update_ods_index_components(db: QuantDB, source: str = "baostock") -> int:
    """更新指数成分 (ODS)"""
    logger.info("📊 [ODS] 检查指数成分更新...")
    # TODO: 实现增量检查成分变动
    logger.info("   指数成分更新待实现")
    return 0


# =============================================================================
# DWD Updater - 明细数据层更新
# =============================================================================

def update_dwd(db: QuantDB) -> dict:
    """更新DWD明细数据层"""
    logger.info("\n" + "=" * 50)
    logger.info("🔄 [DWD] 增量转换数据...")
    logger.info("=" * 50)
    
    results = {}
    
    logger.info("📅 转换日历...")
    results["calendars"] = db.transform_calendars_ods_to_dwd()
    logger.info(f"   ✅ {results['calendars']} 条")
    
    logger.info("📈 转换K线数据...")
    results["ohlcv"] = db.transform_ohlcv_ods_to_dwd()
    logger.info(f"   ✅ {results['ohlcv']} 条")
    
    return results


def update_dws(db: QuantDB) -> dict:
    """更新DWS汇总数据层"""
    logger.info("\n" + "=" * 50)
    logger.info("🔄 [DWS] 更新汇总统计...")
    logger.info("=" * 50)
    
    results = {}
    
    # 只更新当月统计
    today = date.today()
    start = f"{today.year}-{today.month:02d}-01"
    end = today.strftime("%Y-%m-%d")
    
    logger.info(f"📊 更新月度统计: {start} ~ {end}")
    results["monthly"] = db.aggregate_monthly_stats(start=start, end=end)
    logger.info(f"   ✅ {results['monthly']} 条")
    
    return results


# =============================================================================
# Validation - 数据校验
# =============================================================================

def validate_data(db: QuantDB, layer: Optional[str] = None) -> dict:
    """执行数据校验"""
    logger.info("\n" + "=" * 50)
    logger.info(f"🔍 [校验] 数据质量检查{layer if layer else ''}...")
    logger.info("=" * 50)
    
    results = db.validate_all(layer=layer)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    warnings = sum(1 for r in results if not r.passed and r.rule.severity == "WARNING")
    errors = sum(1 for r in results if not r.passed and r.rule.severity == "ERROR")
    
    logger.info(f"\n校验结果:")
    logger.info(f"   ✅ 通过: {passed}")
    logger.info(f"   ⚠️  警告: {warnings}")
    logger.info(f"   ❌ 错误: {errors}")
    
    # 显示失败项
    failed_results = [r for r in results if not r.passed]
    if failed_results:
        logger.info(f"\n失败详情:")
        for r in failed_results[:10]:  # 最多显示10条
            severity_icon = "⚠️" if r.rule.severity == "WARNING" else "❌"
            logger.info(f"   {severity_icon} {r.rule.name}")
            logger.info(f"      预期: {r.expected}")
            logger.info(f"      实际: {r.actual}")
    
    return {
        "total": len(results),
        "passed": passed,
        "warnings": warnings,
        "errors": errors,
        "results": [r.to_dict() for r in results]
    }


# =============================================================================
# Health Check - 健康检查
# =============================================================================

def health_check(db: QuantDB) -> dict:
    """数据库健康检查"""
    logger.info("🏥 执行数据库健康检查...")
    
    results = {
        "status": "healthy",
        "issues": [],
        "checks": {}
    }
    
    # 检查各层数据
    info = db.info()
    
    for layer, tables in info.items():
        results["checks"][layer] = {}
        for table, data in tables.items():
            count = data.get("rows", 0)
            results["checks"][layer][table] = count
            
            if layer == "DWD":
                if table == "dwd_daily_ohlcv" and count == 0:
                    results["issues"].append("K线数据为空")
                elif table == "dwd_calendars" and count == 0:
                    results["issues"].append("日历数据为空")
    
    # 检查最近更新时间
    history = db.get_update_history(limit=1)
    if history.empty:
        results["issues"].append("从未执行过更新")
    else:
        results["checks"]["last_update"] = str(history.iloc[0]["started_at"])
    
    # 检查是否有未完成的更新
    running = db.query("SELECT COUNT(*) FROM meta_update_log WHERE status = 'RUNNING'")
    if running.iloc[0][0] > 0:
        results["issues"].append("存在未完成的更新任务")
    
    if results["issues"]:
        results["status"] = "warning"
        for issue in results["issues"]:
            logger.warning(f"   ⚠️ {issue}")
    
    return results


# =============================================================================
# Main - 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="增量更新量化数据库 (分层架构)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 更新所有数据
    python scripts/update_data.py --db data/quant.db

    # 仅更新K线并校验
    python scripts/update_data.py --db data/quant.db --ohlcv --validate

    # 仅执行校验
    python scripts/update_data.py --db data/quant.db --check

    # 定时任务 (crontab)
    30 16 * * 1-5 python /path/to/update_data.py --db /data/quant.db --ohlcv
        """
    )
    
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument("--ods-only", action="store_true", help="仅更新ODS原始数据")
    parser.add_argument("--dwd-only", action="store_true", help="仅更新DWD明细数据")
    parser.add_argument("--calendar", action="store_true", help="更新日历")
    parser.add_argument("--instruments", action="store_true", help="更新股票列表")
    parser.add_argument("--indices", action="store_true", help="更新指数成分")
    parser.add_argument("--ohlcv", action="store_true", help="更新K线数据")
    parser.add_argument("--validate", action="store_true", help="执行校验")
    parser.add_argument("--check", action="store_true", help="仅健康检查")
    parser.add_argument("--source", type=str, choices=["baostock", "akshare"], default="baostock", help="数据源")
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
    print("量化数据库日更新 (分层架构)")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查数据库
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
        print("🔍 干运行模式，仅检查...")
        health_check(db)
        db.close()
        return
    
    start_time = datetime.now()
    results = {}
    
    try:
        # ODS层更新
        update_ods = args.ods_only or not any([args.dwd_only, args.validate])
        
        if update_ods:
            if args.ods_only or args.calendar:
                results["ods_calendars"] = update_ods_calendars(db)
            if args.ods_only or args.instruments:
                results["ods_instruments"] = update_ods_instruments(db, args.source)
            if args.ods_only or args.indices:
                results["ods_index_components"] = update_ods_index_components(db, args.source)
            if args.ods_only or args.ohlcv:
                results["ods_ohlcv"] = update_ods_ohlcv(db, args.source, args.days)
        
        # DWD层更新
        if args.dwd_only or (not args.ods_only and not any([args.validate, args.check])):
            results["dwd"] = update_dwd(db)
        
        # DWS层更新
        if not args.ods_only and not args.validate:
            results["dws"] = update_dws(db)
        
        # 校验
        if args.validate:
            layer = "DWD" if args.dwd_only else None
            results["validation"] = validate_data(db, layer)
        
    except Exception as e:
        logger.error(f"❌ 更新失败: {e}")
        raise
    finally:
        db.close()
    
    # 显示结果
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("更新完成!")
    print("=" * 60)
    print(f"\n耗时: {duration:.2f} 秒")
    
    if results:
        print("\n更新统计:")
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"   {sub_key}: +{sub_value}")
            else:
                print(f"   {key}: +{value}")
    
    # 健康检查
    print("\n" + "-" * 60)
    print("健康检查:")
    db = QuantDB(args.db)
    health = health_check(db)
    db.close()
    
    if health["status"] == "healthy":
        print("   ✅ 数据库状态正常")
    else:
        print("   ⚠️ 数据库存在一些问题")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
