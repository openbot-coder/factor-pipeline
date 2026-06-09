#!/usr/bin/env python3
"""补充股票池进出数据

akshare 指数 API 限制：只返回最新快照，不支持历史查询。

策略：
1. 用 index_stock_cons() 获取当前成分 + 真实 in_date（纳入日期）
2. 用 index_stock_cons_weight_csindex() 获取当前权重
3. 当前成分 out_date = NULL
4. 历史数据需从今天起每日快照积累

Usage:
    python scripts/update_pool_history.py --db examples/data/ohlcv.duckdb
    python scripts/update_pool_history.py --db data/quant.db --dry-run
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB

logger = logging.getLogger(__name__)

INDICES = [
    ("000016", "sse50",   "上证50"),
    ("000300", "csi300",  "沪深300"),
    ("000905", "csi500",  "中证500"),
    ("000852", "csi1000", "中证1000"),
]


from factor_pipeline.data.utils import norm_code as _norm_code


def fetch_all_constituents(index_code: str, pool_name: str) -> list[dict]:
    """获取完整成分（CSIndex 为全量数据源，Sina 补充 in_date）"""
    import akshare as ak

    # 主数据源：CSIndex — 成分权重（全量 1000 只）
    df_weight = ak.index_stock_cons_weight_csindex(symbol=index_code)

    # 辅助数据源：Sina — 纳入日期（可能有缺失）
    df_sina = ak.index_stock_cons(symbol=index_code)
    in_dates = {}
    for _, row in df_sina.iterrows():
        sym = _norm_code(str(row["品种代码"]))
        if sym:
            in_dates[sym] = str(row["纳入日期"])

    # 当前最新日期（用于 Sina 缺失的 fallback）
    latest_date = df_weight["日期"].max()
    if hasattr(latest_date, "strftime"):
        fallback_date = latest_date.strftime("%Y-%m-%d")
    else:
        fallback_date = str(latest_date)

    records = []
    for _, row in df_weight.iterrows():
        sym = _norm_code(str(row["成分券代码"]))
        if not sym:
            continue
        records.append({
            "pool_name": pool_name,
            "symbol": sym,
            "in_date": in_dates.get(sym, fallback_date),
            "out_date": None,
            "weight": float(row.get("权重", 0) or 0),
            "source": "akshare",
        })

    return records


def update_pool_history(db: QuantDB, args):
    """补充当前成分的 in_date + weight"""
    print("=" * 60)
    print("📊 补充股票池进出数据")
    print("=" * 60)
    print("数据源: CSIndex(全量) + Sina(in_date)")

    # 先清理旧数据（fetch_instruments.py 写入的 csi016/csi852/csi905）
    conn = db.connect()
    old_pools = conn.execute(
        "SELECT DISTINCT pool_name FROM dwd_instruments_pool_registration WHERE pool_name NOT IN ('sse50','csi300','csi500','csi1000')"
    ).fetchall()
    for row in old_pools:
        p = row[0]
        conn.execute(f"DELETE FROM dwd_instruments_pool_registration WHERE pool_name = '{p}'")
        print(f"  🗑️ 清理旧 pool_name: {p}")
    conn.commit()

    for idx_code, pool_name, idx_name in INDICES:
        print(f"\n{'='*50}")
        print(f"📈 {idx_name} ({idx_code}) → {pool_name}")
        print(f"{'='*50}")

        try:
            records = fetch_all_constituents(idx_code, pool_name)
            print(f"  成分: {len(records)} 只")

            if records:
                fallback_date = records[0]["in_date"]
                n_with_in_date = sum(1 for r in records if r["in_date"] != fallback_date)
                n_fallback = len(records) - n_with_in_date
            else:
                n_with_in_date = 0
                n_fallback = 0
            print(f"  有真实 in_date: {n_with_in_date} 只")
            print(f"  使用 fallback 日期: {n_fallback} 只")

            if args.dry_run:
                print(f"  🔍 干运行，跳过写入")
                continue

            inserted = 0
            for r in records:
                try:
                    conn.execute(
                        """
                        INSERT INTO dwd_instruments_pool_registration
                        (pool_name, symbol, in_date, out_date, weight, source, updated_at)
                        VALUES (?, ?, ?, NULL, ?, 'akshare', now())
                        ON CONFLICT(pool_name, symbol, in_date) DO UPDATE SET
                            out_date = NULL, weight = excluded.weight, updated_at = now()
                        """,
                        [pool_name, r['symbol'], r['in_date'], r['weight']],
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning("写入失败 pool=%s symbol=%s: %s", pool_name, r['symbol'], e)
            conn.commit()
            print(f"  ✅ 写入 {inserted} 条")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    # 汇总
    print("\n" + "=" * 50)
    print("📊 汇总")
    print("=" * 50)
    counts = conn.execute(
        "SELECT pool_name, COUNT(*) FROM dwd_instruments_pool_registration GROUP BY pool_name ORDER BY pool_name"
    ).fetchall()
    for pool, cnt in counts:
        print(f"  {pool}: {cnt} 条")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="补充股票池进出数据")
    parser.add_argument("--db", default="data/quant.db", help="数据库路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不写入")
    args = parser.parse_args()

    print(f"数据库: {args.db}")
    db = QuantDB(args.db)
    update_pool_history(db, args)
    db.close()
    print("\n✅ 完成")


if __name__ == "__main__":
    main()
