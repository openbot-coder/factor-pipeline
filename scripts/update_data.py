#!/usr/bin/env python3
"""日更新脚本 - 增量更新量化数据库

这个脚本用于每日增量更新数据库:
1. 更新交易日历
2. 更新股票基础信息
3. 更新指数成分股变动
4. 更新最新K线数据

Usage:
    # 更新所有数据
    python scripts/update_data.py --db data/quant.db

    # 仅更新日历
    python scripts/update_data.py --db data/quant.db --calendar

    # 仅更新K线
    python scripts/update_data.py --db data/quant.db --ohlcv

    # 仅更新指数
    python scripts/update_data.py --db data/quant.db --indices

    # 定时任务示例 (crontab)
    # 每天 16:30 执行 (A股收盘后)
    # 30 16 * * 1-5 python /path/to/update_data.py --db /data/quant.db
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

from factor_pipeline.data.quantdb import QuantDB


# =============================================================================
# Logger - 日志配置
# =============================================================================

def setup_logger(name: str = "update_data") -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 控制台输出
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# =============================================================================
# Calendar Updater - 日历更新
# =============================================================================

def update_calendar(db: QuantDB, days: int = 7) -> int:
    """更新日历
    
    Args:
        db: 数据库连接
        days: 向前更新的天数
        
    Returns:
        新增记录数
    """
    logger.info("📅 检查交易日历更新...")
    
    # 获取最新日历日期
    trading_days = db.get_trading_days()
    if trading_days:
        latest_date = max(trading_days)
    else:
        latest_date = "2005-01-01"
    
    logger.info(f"   最新日历日期: {latest_date}")
    
    # 生成新日期
    today = date.today()
    latest = datetime.strptime(latest_date, "%Y-%m-%d").date()
    
    # 如果最新日期已超过今天，不需要更新
    if latest >= today:
        logger.info("   日历已是最新，无需更新")
        return 0
    
    # 生成缺失的日期
    new_dates = []
    current = latest + timedelta(days=1)
    while current <= today:
        # 简单判断：周一到周五
        if current.weekday() < 5:
            new_dates.append({
                "date": current.strftime("%Y-%m-%d"),
                "is_trading_day": True,  # 简化，实际需要对接交易所
                "exchange": "ALL",
            })
        current += timedelta(days=1)
    
    if not new_dates:
        logger.info("   没有新的交易日")
        return 0
    
    # 导入新日期
    df = pd.DataFrame(new_dates)
    rows = db.import_calendars(df)
    
    logger.info(f"✅ 日历更新完成: 新增 {rows} 条")
    return rows


# =============================================================================
# Instrument Updater - 股票信息更新
# =============================================================================

def update_instruments(db: QuantDB) -> int:
    """更新股票信息
    
    检查新上市和退市股票
    
    Returns:
        更新记录数
    """
    logger.info("📋 检查股票信息更新...")
    
    # TODO: 实现股票信息增量更新
    # 1. 查询新上市股票
    # 2. 查询退市股票
    # 3. 更新行业分类
    
    logger.info("   股票信息更新功能待实现")
    return 0


# =============================================================================
# Index Updater - 指数成分更新
# =============================================================================

def update_index_components(db: QuantDB) -> dict:
    """更新指数成分股
    
    检查指数成分变动
    
    Returns:
        各指数更新统计
    """
    logger.info("📊 检查指数成分更新...")
    
    results = {}
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    
    # 获取已跟踪的指数
    indices = db.query("SELECT index_code, index_name FROM indices")
    
    for _, row in indices.iterrows():
        index_code = row["index_code"]
        index_name = row["index_name"]
        
        logger.info(f"   检查 {index_name} ({index_code})...")
        
        # TODO: 对接数据源获取最新成分股
        # 目前仅记录待实现
        
        results[index_code] = {"added": 0, "removed": 0}
    
    logger.info(f"✅ 指数更新完成")
    return results


# =============================================================================
# OHLCV Updater - K线数据更新
# =============================================================================

def update_ohlcv_baostock(db: QuantDB, days: int = 5) -> int:
    """从baostock更新K线数据
    
    Args:
        db: 数据库连接
        days: 向前更新的天数
        
    Returns:
        新增记录数
    """
    logger.info("📈 从 baostock 更新K线数据...")
    
    try:
        import baostock as bs
        
        # 登录
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"Baostock登录失败: {lg.error_msg}")
            return 0
        
        # 获取需要更新的股票
        symbols = db.get_symbols(active_only=True)
        logger.info(f"   活跃股票数: {len(symbols)}")
        
        # 确定日期范围
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # 检查每个股票的最近更新日期
        total_records = 0
        updated_stocks = 0
        
        for symbol in symbols[:100]:  # 限制数量避免超时
            try:
                # 转换代码格式
                bs_code = f"sh.{symbol}" if ".SH" in symbol else f"sz.{symbol}"
                
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3",
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
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    db.import_ohlcv(df)
                    total_records += len(records)
                    updated_stocks += 1
                
            except Exception as e:
                logger.debug(f"   {symbol} 更新失败: {e}")
        
        bs.logout()
        
        logger.info(f"✅ K线更新完成: {updated_stocks} 只股票, {total_records} 条记录")
        return total_records
        
    except ImportError:
        logger.warning("⚠️ baostock 未安装，跳过K线更新")
        logger.info("   安装: pip install baostock")
        return 0
    except Exception as e:
        logger.error(f"❌ K线更新失败: {e}")
        return 0


def update_ohlcv_akshare(db: QuantDB, days: int = 5) -> int:
    """从AKShare更新K线数据
    
    Args:
        db: 数据库连接
        days: 向前更新的天数
        
    Returns:
        新增记录数
    """
    logger.info("📈 从 AKShare 更新K线数据...")
    
    try:
        import akshare as ak
        
        symbols = db.get_symbols(active_only=True)
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
        
        total_records = 0
        updated_stocks = 0
        
        for symbol in symbols[:50]:  # 限制数量
            try:
                code = symbol.split(".")[0]
                
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    start_date=start_date,
                    end_date=end_date,
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
                    })
                    df["symbol"] = symbol
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    
                    db.import_ohlcv(df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]])
                    total_records += len(df)
                    updated_stocks += 1
                    
            except Exception as e:
                logger.debug(f"   {symbol} 更新失败: {e}")
        
        logger.info(f"✅ K线更新完成: {updated_stocks} 只股票, {total_records} 条记录")
        return total_records
        
    except ImportError:
        logger.warning("⚠️ akshare 未安装，跳过K线更新")
        return 0
    except Exception as e:
        logger.error(f"❌ K线更新失败: {e}")
        return 0


def update_ohlcv(db: QuantDB, source: str = "baostock", days: int = 5) -> int:
    """更新K线数据
    
    Args:
        db: 数据库连接
        source: 数据源
        days: 更新天数
        
    Returns:
        新增记录数
    """
    if source == "baostock":
        return update_ohlcv_baostock(db, days)
    else:
        return update_ohlcv_akshare(db, days)


# =============================================================================
# Health Check - 健康检查
# =============================================================================

def health_check(db: QuantDB) -> dict:
    """数据库健康检查
    
    Returns:
        健康检查结果
    """
    logger.info("🏥 执行数据库健康检查...")
    
    results = {
        "status": "healthy",
        "issues": [],
        "stats": {},
    }
    
    # 检查各表记录数
    info = db.info()
    for table, data in info["tables"].items():
        count = data.get("rows", 0)
        results["stats"][table] = count
        
        if table == "daily_ohlcv" and count == 0:
            results["issues"].append("K线数据为空")
        elif table == "calendars" and count == 0:
            results["issues"].append("日历数据为空")
        elif table == "instruments" and count == 0:
            results["issues"].append("股票信息为空")
    
    # 检查最近更新时间
    history = db.get_update_history(limit=1)
    if history.empty:
        results["issues"].append("从未执行过更新")
    else:
        last_update = history.iloc[0]
        results["stats"]["last_update"] = str(last_update["started_at"])
    
    # 检查是否有未完成的更新
    running = db.query("SELECT COUNT(*) FROM data_update_log WHERE status = 'RUNNING'")
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
        description="增量更新量化数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 更新所有数据
    python scripts/update_data.py --db data/quant.db

    # 仅更新日历
    python scripts/update_data.py --db data/quant.db --calendar

    # 仅更新K线 (使用AKShare)
    python scripts/update_data.py --db data/quant.db --ohlcv --source akshare

    # 定时任务 (crontab)
    30 16 * * 1-5 python /path/to/update_data.py --db /data/quant.db --ohlcv
        """
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default="data/quant.db",
        help="数据库路径 (default: data/quant.db)",
    )
    parser.add_argument(
        "--calendar",
        action="store_true",
        help="仅更新日历",
    )
    parser.add_argument(
        "--instruments",
        action="store_true",
        help="仅更新股票信息",
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="仅更新指数成分",
    )
    parser.add_argument(
        "--ohlcv",
        action="store_true",
        help="仅更新K线数据",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["baostock", "akshare"],
        default="baostock",
        help="K线数据源 (default: baostock)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="K线更新天数 (default: 5)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="仅执行健康检查",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查，不实际更新",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("量化数据库日更新")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查数据库是否存在
    if not Path(args.db).exists():
        print(f"❌ 数据库不存在: {args.db}")
        print("   请先运行初始化脚本:")
        print(f"   python scripts/init_data.py --db {args.db} --mode full")
        sys.exit(1)
    
    # 连接数据库
    db = QuantDB(args.db)
    
    # 健康检查
    if args.health:
        health_check(db)
        db.close()
        return
    
    if args.dry_run:
        print("🔍 干运行模式，仅检查...")
        health_check(db)
        db.close()
        return
    
    # 记录开始时间
    start_time = datetime.now()
    
    results = {
        "calendar": 0,
        "instruments": 0,
        "indices": 0,
        "ohlcv": 0,
    }
    
    # 确定更新模式
    update_all = not any([args.calendar, args.instruments, args.indices, args.ohlcv])
    
    # 执行更新
    try:
        # 日历更新
        if update_all or args.calendar:
            results["calendar"] = update_calendar(db)
        
        # 股票信息更新
        if update_all or args.instruments:
            results["instruments"] = update_instruments(db)
        
        # 指数更新
        if update_all or args.indices:
            index_results = update_index_components(db)
            results["indices"] = sum(r["added"] + r["removed"] for r in index_results.values())
        
        # K线更新
        if update_all or args.ohlcv:
            results["ohlcv"] = update_ohlcv(db, source=args.source, days=args.days)
        
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
    print("\n更新统计:")
    print(f"   日历:   +{results['calendar']} 条")
    print(f"   股票:   +{results['instruments']} 条")
    print(f"   指数:   +{results['indices']} 条")
    print(f"   K线:    +{results['ohlcv']} 条")
    
    # 健康检查
    print("\n" + "-" * 60)
    print("健康检查:")
    db = QuantDB(args.db)
    health = health_check(db)
    db.close()
    
    if health["status"] == "healthy":
        print("   ✅ 数据库状态正常")
    else:
        print("   ⚠️ 数据库存在一些问题，请检查")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
