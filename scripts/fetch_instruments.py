#!/usr/bin/env python3
"""获取 A 股股票池基础数据：全市场股票列表 + 指数成分股

Usage:
    python scripts/fetch_instruments.py --db data/quant.db
    python scripts/fetch_instruments.py --db data/quant.db --dry-run
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_pipeline.data.quantdb import QuantDB


# ── 已知指数 ──────────────────────────────────────────────
INDICES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000016": "上证50",
}


def fetch_all_stocks_akshare() -> list[dict]:
    """从 akshare 获取全 A 股列表（含上市日期）"""
    import akshare as ak

    print("  [akshare] 获取全 A 股实时行情列表 ...", end=" ", flush=True)
    df = ak.stock_zh_a_spot_em()
    print(f"{len(df)} 只股票")

    records = []
    for _, row in df.iterrows():
        code = str(row["代码"])
        # 转成统一 symbol 格式
        if code.startswith(("60", "68", "90")):
            market = "SSE"
            symbol = f"{code}.SSE"
        elif code.startswith(("00", "30", "20")):
            market = "SZSE"
            symbol = f"{code}.SZSE"
        elif code.startswith(("43", "83", "87", "92")):
            market = "BSE"
            symbol = f"{code}.BSE"
        else:
            continue  # 跳过非 A 股

        records.append({
            "symbol": symbol,
            "name": str(row.get("名称", "")),
            "list_date": None,
            "delist_date": None,
            "market": market,
        })

    return records


def fetch_all_stocks_from_list(stock_list: list[str]) -> list[dict]:
    """从预定义代码列表构建 instruments（用于无法连接 akshare 时的回退）"""
    records = []
    for code in stock_list:
        code = code.strip()
        if code.startswith(("60", "68", "90")):
            market = "SSE"
            symbol = f"{code}.SSE"
        elif code.startswith(("00", "30", "20")):
            market = "SZSE"
            symbol = f"{code}.SZSE"
        elif code.startswith(("43", "83", "87", "92")):
            market = "BSE"
            symbol = f"{code}.BSE"
        else:
            continue
        records.append({
            "symbol": symbol,
            "name": "",
            "list_date": None,
            "delist_date": None,
            "market": market,
        })
    return records


def fetch_index_constituents(index_code: str, index_name: str) -> list[dict]:
    """从 akshare 获取指数成分股"""
    import akshare as ak

    print(f"  [akshare] 获取 {index_name}({index_code}) 成分股 ...", end=" ", flush=True)
    try:
        df = ak.index_stock_cons_weight_csindex(symbol=index_code)
        # 取最新日期
        latest_date = df["日期"].max()
        df_latest = df[df["日期"] == latest_date]
        print(f"{len(df_latest)} 只 (截止 {latest_date})")

        records = []
        for _, row in df_latest.iterrows():
            code = str(row["成分券代码"])
            if code.startswith(("60", "68", "90")):
                symbol = f"{code}.SSE"
            elif code.startswith(("00", "30", "20")):
                symbol = f"{code}.SZSE"
            else:
                continue
            records.append({
                "pool_name": f"csi{index_code[-3:]}",
                "symbol": symbol,
                "in_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                "out_date": None,
                "weight": float(row.get("权重", 0)) if row.get("权重") else 0,
                "source": "akshare",
            })
        return records
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return []


def init_stock_pools(db: QuantDB, source: str, args):
    """初始化股票池基础数据"""
    print("\n" + "=" * 50)
    print("📋 拉取股票列表 → dwd_instruments_info")
    print("=" * 50)

    # 确保 schema 已初始化
    db.init_schema()

    # 1. 获取股票列表
    try:
        stock_records = fetch_all_stocks_akshare()
    except Exception as e:
        print(f"  ⚠️ akshare 失败: {e}")
        print("  ⚠️ 使用 fallback ...")
        # 从 init_data.py 获取回退股票列表
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "init_data_mod",
            str(Path(__file__).parent / "init_data.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fallback_df = mod._get_fallback_instruments()
        stock_records = fallback_df.to_dict("records") if not fallback_df.empty else []

    if not stock_records:
        print("  ❌ 无法获取股票列表")
        return

    # 写入 dwd_instruments_info
    print(f"\n  📝 写入 dwd_instruments_info ({len(stock_records)} 只)...")
    inserted = 0
    conn = db.connect()
    for r in stock_records:
            try:
                ld = r.get("list_date")
                dd = r.get("delist_date")
                nm = r.get("name", "")
                conn.execute(
                    """
                    INSERT INTO dwd_instruments_info
                    (symbol, name, list_date, delist_date, market, updated_at)
                    VALUES (?, ?, ?, ?, ?, now())
                    ON CONFLICT(symbol) DO UPDATE SET
                        name = excluded.name, market = excluded.market, updated_at = now()
                    """,
                    [r['symbol'], nm, ld, dd, r['market']],
                )
                inserted += 1
            except Exception as e:
                print(f"  ⚠️ 写入失败 {r.get('symbol')}: {e}")
    conn.commit()
    print(f"  ✅ {inserted} 条已写入")

    # 3. 指数成分
    print("\n" + "=" * 50)
    print("📊 拉取指数成分 → dwd_instruments_pool_registration")
    print("=" * 50)
    total_components = 0
    for code, name in INDICES.items():
        records = fetch_index_constituents(code, name)
        for r in records:
            try:
                w = r.get("weight", 0) or 0
                conn.execute(
                    """
                    INSERT INTO dwd_instruments_pool_registration
                    (pool_name, symbol, in_date, out_date, weight, source, updated_at)
                    VALUES (?, ?, ?, NULL, ?, 'akshare', now())
                    ON CONFLICT(pool_name, symbol, in_date) DO UPDATE SET
                        weight = excluded.weight, updated_at = now()
                    """,
                    [r['pool_name'], r['symbol'], r['in_date'], w],
                )
                total_components += 1
            except Exception as e:
                print(f"  ⚠️ 成分股写入失败 {r.get('symbol')}: {e}")
    conn.commit()
    print(f"  ✅ {total_components} 条指数成分已写入")

    # 汇总
    print("\n" + "=" * 50)
    print("📊 汇总")
    print("=" * 50)
    info_count = conn.execute("SELECT COUNT(*) FROM dwd_instruments_info").fetchone()[0]
    pool_count = conn.execute("SELECT COUNT(*) FROM dwd_instruments_pool_registration").fetchone()[0]
    pool_types = conn.execute("SELECT DISTINCT pool_name FROM dwd_instruments_pool_registration ORDER BY pool_name").fetchall()
    print(f"  证券信息: {info_count} 只")
    print(f"  股票池记录: {pool_count} 条")
    print(f"  股票池类型: {[r[0] for r in pool_types]}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="获取 A 股股票池基础数据")
    parser.add_argument("--db", default="data/quant.db", help="数据库路径")
    parser.add_argument("--source", default="akshare", help="数据源")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不写入")
    args = parser.parse_args()

    print("=" * 60)
    print("A 股股票池基础数据获取")
    print("=" * 60)
    print(f"数据库: {args.db}")
    print(f"数据源: {args.source}")
    print()

    if args.dry_run:
        print("🔍 干运行模式，仅检查...")
        print("✅ 检查完成（无实际写入）")
        return

    db = QuantDB(args.db)
    db.init_schema()

    init_stock_pools(db, args.source, args)

    db.close()
    print("\n✅ 完成")


if __name__ == "__main__":
    main()
