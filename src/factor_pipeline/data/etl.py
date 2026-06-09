"""ETL Pipeline - 数据迁移脚本

ODS → DWD 独立迁移脚本:
- 支持多数据源
- 数据源切换只需修改此脚本
- 不影响数据源抽取逻辑

Usage:
    from factor_pipeline.data.etl import ETLPipeline

    db = QuantDB("data/quant.db")
    etl = ETLPipeline(db)

    # 迁移所有数据源
    etl.run()

    # 仅迁移 baostock
    etl.run(source="baostock")

    # 仅迁移 K线数据
    etl.run(source="baostock", tables=["daily_ohlcv"])
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class ETLPipeline:
    """ETL数据迁移管道

    负责将ODS层数据迁移到DWD层:
    - 清洗数据
    - 标准化格式
    - 合并多数据源
    - 记录迁移日志
    """

    def __init__(self, db: QuantDB):
        """初始化ETL管道

        Args:
            db: QuantDB实例
        """
        self.db = db
        self.start_time: datetime | None = None

    def run(
        self,
        source: str | None = None,
        tables: list[str] | None = None,
    ) -> dict:
        """运行ETL迁移

        Args:
            source: 指定数据源 (None表示所有数据源)
            tables: 指定迁移哪些表 (默认全部)

        Returns:
            迁移结果统计
        """
        self.start_time = datetime.now()

        if tables is None:
            tables = ["calendars", "instruments", "index_components", "daily_ohlcv"]

        if source:
            sources = [source]
        else:
            sources = self.db.get_active_sources()

        results = {}

        for src in sources:
            results[src] = {}
            print(f"\n{'='*50}")
            print(f"🔄 ETL: {src}")
            print(f"{'='*50}")

            for table in tables:
                try:
                    print(f"\n📦 迁移 ods_{table}_{src} → dwd_{table}")
                    count = self._transform_table(src, table)
                    results[src][table] = {"success": True, "records": count}
                    print(f"   ✅ {count} 条记录")
                except Exception as e:
                    print(f"   ❌ 失败: {e}")
                    results[src][table] = {"success": False, "error": str(e)}

        self._print_summary(results)
        return results

    def _transform_table(self, source: str, table_type: str) -> int:
        """转换单个表

        Args:
            source: 数据源
            table_type: 表类型

        Returns:
            迁移记录数
        """
        if table_type == "calendars":
            return self._transform_calendars(source)
        elif table_type == "instruments":
            return self._transform_instruments(source)
        elif table_type == "index_components":
            return self._transform_index_components(source)
        elif table_type == "daily_ohlcv":
            return self._transform_ohlcv(source)
        else:
            raise ValueError(f"Unknown table type: {table_type}")

    def _transform_calendars(self, source: str) -> int:
        """转换日历数据"""
        # 获取ODS数据
        ods_table = f"ods_calendars_{source}"

        try:
            df = self.db.query(f"SELECT * FROM {ods_table}")
        except Exception:
            print(f"   ⚠️ 表 {ods_table} 不存在")
            return 0

        if df.empty:
            return 0

        # 转换并丰富字段
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end
        df["is_quarter_end"] = pd.to_datetime(df["date"]).dt.is_quarter_end
        df["is_year_end"] = pd.to_datetime(df["date"]).dt.is_year_end
        df["is_week_end"] = pd.to_datetime(df["date"]).dt.dayofweek == 4  # 周五
        df["updated_at"] = datetime.now()

        # 写入DWD
        conn = self.db.connect()

        for _, row in df.iterrows():
            try:
                conn.execute(f"""
                    INSERT INTO dwd_calendars
                    (date, is_trading_day, year, quarter, month, week_of_year, day_of_week,
                     is_month_end, is_quarter_end, is_year_end, is_week_end, exchange, updated_at)
                    VALUES ('{row['date']}', {row['is_trading_day']}, {row['year']}, {row['quarter']},
                            {row['month']}, {row['week_of_year']}, {row['day_of_week']},
                            {row['is_month_end']}, {row['is_quarter_end']}, {row['is_year_end']},
                            {row['is_week_end']}, '{row['exchange']}', '{row['updated_at']}')
                    ON CONFLICT(date) DO UPDATE SET
                        is_trading_day = {row['is_trading_day']},
                        year = {row['year']},
                        quarter = {row['quarter']},
                        month = {row['month']},
                        week_of_year = {row['week_of_year']},
                        day_of_week = {row['day_of_week']},
                        is_month_end = {row['is_month_end']},
                        is_quarter_end = {row['is_quarter_end']},
                        is_year_end = {row['is_year_end']},
                        is_week_end = {row['is_week_end']},
                        updated_at = '{row['updated_at']}'
                """)
            except Exception:
                pass

        conn.commit()

        # 记录日志
        # 更新参数表
        self.db.update_table_params("DWD", "dwd_calendars", len(df), source)
        self.db.log_update(
            layer="ETL",
            table="dwd_calendars",
            source=source,
            update_type="TRANSFORM",
            records=len(df),
            status="SUCCESS",
        )

        return len(df)

    def _transform_instruments(self, source: str) -> int:
        """转换股票信息"""
        ods_table = f"ods_instruments_{source}"

        try:
            df = self.db.query(f"SELECT * FROM {ods_table}")
        except Exception:
            print(f"   ⚠️ 表 {ods_table} 不存在")
            return 0

        if df.empty:
            return 0

        # 标准化字段
        def get_board_type(symbol):
            if symbol.startswith("688"):
                return "科创板"
            elif symbol.startswith(("002", "003")):
                return "创业板"
            elif symbol.startswith(("430", "830")):
                return "北交所"
            else:
                return "主板"

        df["board_type"] = df["symbol"].apply(get_board_type)
        df["status"] = df["delist_date"].apply(lambda x: "DELISTED" if x else "ACTIVE")
        df["updated_at"] = datetime.now()

        conn = self.db.connect()

        for _, row in df.iterrows():
            try:
                conn.execute(f"""
                    INSERT INTO dwd_instruments
                    (symbol, name, list_date, delist_date, market, board_type, status, updated_at)
                    VALUES ('{row['symbol']}', '{row['name']}',
                            {'NULL' if pd.isna(row.get('list_date')) else f"'{row['list_date']}'"},
                            {'NULL' if pd.isna(row.get('delist_date')) else f"'{row['delist_date']}'"},
                            '{row['market']}', '{row['board_type']}', '{row['status']}', '{row['updated_at']}')
                    ON CONFLICT(symbol) DO UPDATE SET
                        name = '{row['name']}',
                        delist_date = {'NULL' if pd.isna(row.get('delist_date')) else f"'{row['delist_date']}'"},
                        status = '{row['status']}',
                        updated_at = '{row['updated_at']}'
                """)
            except Exception:
                pass

        conn.commit()

        # 更新参数表
        self.db.update_table_params("DWD", "dwd_instruments", len(df), source)
        self.db.log_update(
            layer="ETL",
            table="dwd_instruments",
            source=source,
            update_type="TRANSFORM",
            records=len(df),
            status="SUCCESS",
        )

        return len(df)

    def _transform_index_components(self, source: str) -> int:
        """转换指数成分"""
        ods_table = f"ods_index_components_{source}"

        try:
            df = self.db.query(f"SELECT * FROM {ods_table}")
        except Exception:
            print(f"   ⚠️ 表 {ods_table} 不存在")
            return 0

        if df.empty:
            return 0

        # 先取消当前标记
        self.db.execute(
            "UPDATE dwd_index_components SET is_current = FALSE WHERE is_current = TRUE"
        )

        df["is_current"] = True
        df["source"] = source
        df["updated_at"] = datetime.now()

        conn = self.db.connect()

        # 获取当前最大ID
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM dwd_index_components").fetchone()[0]

        records_processed = 0
        for _, row in df.iterrows():
            max_id += 1
            try:
                in_date = f"'{row['in_date']}'" if not pd.isna(row.get("in_date")) else "NULL"
                out_date = f"'{row['out_date']}'" if not pd.isna(row.get("out_date")) else "NULL"

                conn.execute(f"""
                    INSERT INTO dwd_index_components
                    (id, index_code, index_name, symbol, in_date, out_date, weight, is_current, source, updated_at)
                    VALUES ({max_id}, '{row['index_code']}', '{row['index_name']}', '{row['symbol']}',
                            {in_date}, {out_date}, {row.get('weight', 0)}, TRUE, '{source}', '{row['updated_at']}')
                    ON CONFLICT(index_code, symbol, in_date) DO UPDATE SET
                        out_date = {out_date},
                        weight = {row.get('weight', 0)},
                        is_current = TRUE,
                        updated_at = '{row['updated_at']}'
                """)
                records_processed += 1
            except Exception:
                pass

        conn.commit()

        # 更新参数表
        self.db.update_table_params("DWD", "dwd_index_components", records_processed, source)

        return records_processed

    def _transform_ohlcv(self, source: str) -> int:
        """转换K线数据 (前复权)"""
        ods_table = f"ods_daily_ohlcv_{source}"

        # Try unadjusted first ('3'), then forward-adjusted ('2')
        try:
            df = self.db.query(f"SELECT * FROM {ods_table} WHERE adjust_flag = '3'")
        except Exception:
            print(f"   ⚠️ 表 {ods_table} 不存在")
            return 0

        if df.empty:
            try:
                df = self.db.query(f"SELECT * FROM {ods_table} WHERE adjust_flag = '2'")
            except Exception:
                df = self.db.query(f"SELECT * FROM {ods_table}")

        if df.empty:
            print(f"   ⚠️ 表 {ods_table} 无数据")
            return 0

        df["factor"] = 1.0
        df["raw_close"] = df["close"]
        df["source"] = source
        now = datetime.now()

        conn = self.db.connect()
        target_cols = [
            "date", "symbol", "open", "high", "low", "close",
            "volume", "amount", "turnover_rate", "pct_change",
            "factor", "raw_close", "source", "updated_at"
        ]
        df_out = df[[c for c in target_cols if c != "updated_at"]].assign(updated_at=now)

        # DuckDB Arrow native append：比 executemany 快 50 倍
        # DuckDB.execute() 支持直接接收 pyarrow.Table 作为参数
        import pyarrow as pa
        tbl = pa.Table.from_pandas(df_out, preserve_index=False)
        conn.execute("DELETE FROM dwd_daily_ohlcv")
        conn.execute(
            "INSERT INTO dwd_daily_ohlcv BY NAME SELECT * FROM table_ref",
            params=[tbl],
        )
        total = conn.execute("SELECT COUNT(*) FROM dwd_daily_ohlcv").fetchone()[0]
        conn.commit()

        # 更新参数表
        self.db.update_table_params("DWD", "dwd_daily_ohlcv", total, source)
        self.db.log_update(
            layer="ETL",
            table="dwd_daily_ohlcv",
            source=source,
            update_type="TRANSFORM",
            records=total,
            status="SUCCESS",
        )

        return total

    def _print_summary(self, results: dict) -> None:
        """打印迁移汇总"""
        print(f"\n{'='*50}")
        print("📊 ETL 迁移汇总")
        print(f"{'='*50}")

        total_records = 0
        total_failed = 0

        for source, tables in results.items():
            print(f"\n  [{source}]")
            for table, result in tables.items():
                if result["success"]:
                    print(f"    ✅ {table}: {result['records']} 条")
                    total_records += result["records"]
                else:
                    print(f"    ❌ {table}: {result['error']}")
                    total_failed += 1

        print(f"\n总计: {total_records} 条记录, {total_failed} 个失败")

        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"耗时: {duration:.2f} 秒")


# =============================================================================
# CLI - 命令行接口
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import argparse

    from factor_pipeline.data.quantdb import QuantDB

    parser = argparse.ArgumentParser(description="ETL数据迁移脚本")
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument("--source", type=str, default=None, help="指定数据源")
    parser.add_argument("--tables", type=str, nargs="+", default=None, help="指定表")

    args = parser.parse_args()

    print(f"连接数据库: {args.db}")
    db = QuantDB(args.db)

    etl = ETLPipeline(db)
    etl.run(source=args.source, tables=args.tables)

    db.close()
