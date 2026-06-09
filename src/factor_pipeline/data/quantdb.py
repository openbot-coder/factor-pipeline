"""Quantitative Database - 量化数据库核心模块 (3层架构 + ETL分离)

架构设计:
- ODS (原始数据层): 按数据源分表命名
- DWD (明细数据层): 清洗、标准化后的数据
- APP (应用数据层): 聚合统计、因子数据
- ETL (数据迁移): 独立的迁移脚本
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import duckdb
import pandas as pd

# =============================================================================
# Constants
# =============================================================================


class Market(Enum):
    SSE = "SSE"
    SZSE = "SZSE"
    BSE = "BSE"


class IndexCode(Enum):
    SSE50 = ("000016", "SH", "上证50")
    CSI300 = ("000300", "SH", "沪深300")
    CSI500 = ("000905", "SH", "中证500")
    CSI1000 = ("000852", "SH", "中证1000")

    def __init__(self, code: str, exchange: str, name_zh: str):
        self.code = code
        self.exchange = exchange
        self.name_zh = name_zh

    @property
    def full_code(self) -> str:
        return f"{self.code}.{self.exchange}"


# =============================================================================
# Data Dictionary - 数据字典
# =============================================================================

DATA_DICTIONARY = {
    "dwd_calendars": {
        "name": "交易日历",
        "layer": "DWD",
        "description": "标准化后的A股交易日历",
        "fields": {
            "date": {"type": "DATE", "description": "日期"},
            "is_trading_day": {"type": "BOOLEAN", "description": "是否交易日"},
            "year": {"type": "INTEGER", "description": "年份"},
            "quarter": {"type": "INTEGER", "description": "季度 (1-4)"},
            "month": {"type": "INTEGER", "description": "月份 (1-12)"},
            "week_of_year": {"type": "INTEGER", "description": "年内第几周"},
            "day_of_week": {"type": "INTEGER", "description": "周几 (0=周一, 6=周日)"},
            "is_month_end": {"type": "BOOLEAN", "description": "是否月末"},
            "is_quarter_end": {"type": "BOOLEAN", "description": "是否季末"},
            "is_year_end": {"type": "BOOLEAN", "description": "是否年末"},
            "is_week_end": {"type": "BOOLEAN", "description": "是否周末"},
            "exchange": {"type": "VARCHAR", "description": "交易所 (ALL/SSE/SZSE)"},
            "updated_at": {"type": "TIMESTAMP", "description": "更新时间"},
        },
    },
    "dwd_instruments": {
        "name": "股票基础信息",
        "layer": "DWD",
        "description": "A股股票基本信息",
        "fields": {
            "symbol": {"type": "VARCHAR", "description": "股票代码 (如 000001.SZ)"},
            "name": {"type": "VARCHAR", "description": "股票名称"},
            "list_date": {"type": "DATE", "description": "上市日期"},
            "delist_date": {"type": "DATE", "description": "退市日期 (NULL表示仍在交易)"},
            "market": {"type": "VARCHAR", "description": "所属市场 (SSE/SZSE/BSE)"},
            "board_type": {
                "type": "VARCHAR",
                "description": "板块类型 (主板/创业板/科创板/北交所)",
            },
            "industry_sw_l1": {"type": "VARCHAR", "description": "申万一级行业"},
            "status": {"type": "VARCHAR", "description": "状态 (ACTIVE/DELISTED)"},
            "updated_at": {"type": "TIMESTAMP", "description": "更新时间"},
        },
    },
    "dwd_index_components": {
        "name": "指数成分股快照",
        "layer": "DWD",
        "description": "各指数成分股及其纳入/剔除日期",
        "fields": {
            "id": {"type": "INTEGER", "description": "主键ID"},
            "index_code": {"type": "VARCHAR", "description": "指数代码 (如 000300.SH)"},
            "index_name": {"type": "VARCHAR", "description": "指数名称"},
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "in_date": {"type": "DATE", "description": "纳入日期"},
            "out_date": {"type": "DATE", "description": "剔除日期 (NULL表示仍在池中)"},
            "weight": {"type": "DOUBLE", "description": "权重 (%)"},
            "is_current": {"type": "BOOLEAN", "description": "是否为当前成分股"},
            "source": {"type": "VARCHAR", "description": "数据来源"},
            "updated_at": {"type": "TIMESTAMP", "description": "更新时间"},
        },
    },
    "dwd_daily_ohlcv": {
        "name": "日K线数据 (前复权)",
        "layer": "DWD",
        "description": "前复权处理的日线行情数据",
        "fields": {
            "date": {"type": "DATE", "description": "交易日期"},
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "open": {"type": "DOUBLE", "description": "开盘价 (前复权)"},
            "high": {"type": "DOUBLE", "description": "最高价 (前复权)"},
            "low": {"type": "DOUBLE", "description": "最低价 (前复权)"},
            "close": {"type": "DOUBLE", "description": "收盘价 (前复权)"},
            "volume": {"type": "DOUBLE", "description": "成交量 (股数)"},
            "amount": {"type": "DOUBLE", "description": "成交额 (元)"},
            "turnover_rate": {"type": "DOUBLE", "description": "换手率 (%)"},
            "pct_change": {"type": "DOUBLE", "description": "涨跌幅 (%)"},
            "factor": {"type": "DOUBLE", "description": "复权因子"},
            "raw_close": {"type": "DOUBLE", "description": "原始收盘价"},
            "source": {"type": "VARCHAR", "description": "数据来源"},
            "updated_at": {"type": "TIMESTAMP", "description": "更新时间"},
        },
    },
    "app_monthly_stats": {
        "name": "月度汇总统计",
        "layer": "APP",
        "description": "按月聚合的股票统计指标",
        "fields": {
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "year": {"type": "INTEGER", "description": "年份"},
            "month": {"type": "INTEGER", "description": "月份"},
            "start_date": {"type": "DATE", "description": "月首交易日"},
            "end_date": {"type": "DATE", "description": "月末交易日"},
            "open_first": {"type": "DOUBLE", "description": "月首开盘价"},
            "close_last": {"type": "DOUBLE", "description": "月末收盘价"},
            "high_max": {"type": "DOUBLE", "description": "月最高价"},
            "low_min": {"type": "DOUBLE", "description": "月最低价"},
            "volume_sum": {"type": "DOUBLE", "description": "月度总成交量"},
            "amount_sum": {"type": "DOUBLE", "description": "月度总成交额"},
            "avg_turnover_rate": {"type": "DOUBLE", "description": "月均换手率 (%)"},
            "pct_change_monthly": {"type": "DOUBLE", "description": "月度涨跌幅 (%)"},
        },
    },
    "app_yearly_stats": {
        "name": "年度汇总统计",
        "layer": "APP",
        "description": "按年聚合的股票统计指标",
        "fields": {
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "year": {"type": "INTEGER", "description": "年份"},
            "start_date": {"type": "DATE", "description": "年首交易日"},
            "end_date": {"type": "DATE", "description": "年末交易日"},
            "open_first": {"type": "DOUBLE", "description": "年首开盘价"},
            "close_last": {"type": "DOUBLE", "description": "年末收盘价"},
            "high_max": {"type": "DOUBLE", "description": "年最高价"},
            "low_min": {"type": "DOUBLE", "description": "年最低价"},
            "volume_sum": {"type": "DOUBLE", "description": "年度总成交量"},
            "amount_sum": {"type": "DOUBLE", "description": "年度总成交额"},
            "avg_turnover_rate": {"type": "DOUBLE", "description": "年均换手率 (%)"},
            "pct_change_yearly": {"type": "DOUBLE", "description": "年度涨跌幅 (%)"},
        },
    },
    "app_index_members": {
        "name": "当前股票池成员",
        "layer": "APP",
        "description": "当前有效的指数成分股",
        "fields": {
            "index_code": {"type": "VARCHAR", "description": "指数代码"},
            "index_name": {"type": "VARCHAR", "description": "指数名称"},
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "name": {"type": "VARCHAR", "description": "股票名称"},
            "in_date": {"type": "DATE", "description": "纳入日期"},
            "weight": {"type": "DOUBLE", "description": "指数权重 (%)"},
            "market_cap": {"type": "DOUBLE", "description": "总市值 (元)"},
        },
    },
    "app_limit_up_down": {
        "name": "涨跌停股票",
        "layer": "APP",
        "description": "涨跌停股票记录",
        "fields": {
            "date": {"type": "DATE", "description": "交易日期"},
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "limit_type": {"type": "VARCHAR", "description": "涨停/跌停 (LIMIT_UP/LIMIT_DOWN)"},
            "prev_close": {"type": "DOUBLE", "description": "前收盘价"},
            "open": {"type": "DOUBLE", "description": "开盘价"},
            "high": {"type": "DOUBLE", "description": "最高价"},
            "low": {"type": "DOUBLE", "description": "最低价"},
            "close": {"type": "DOUBLE", "description": "收盘价"},
            "amplitude": {"type": "DOUBLE", "description": "振幅 (%)"},
            "turnover_rate": {"type": "DOUBLE", "description": "换手率 (%)"},
            "reason": {"type": "VARCHAR", "description": "涨停原因"},
        },
    },
    "app_factors_registry": {
        "name": "因子定义表",
        "layer": "APP",
        "description": "因子注册表",
        "fields": {
            "id": {"type": "INTEGER", "description": "主键ID"},
            "name": {"type": "VARCHAR", "description": "因子名称"},
            "category": {
                "type": "VARCHAR",
                "description": "因子类别 (technical/fundamental/alpha)",
            },
            "description": {"type": "TEXT", "description": "因子描述"},
            "expression": {"type": "TEXT", "description": "计算表达式"},
            "parameters": {"type": "JSON", "description": "参数字典"},
            "created_at": {"type": "TIMESTAMP", "description": "创建时间"},
            "updated_at": {"type": "TIMESTAMP", "description": "更新时间"},
        },
    },
    "app_factors_values": {
        "name": "因子值缓存表",
        "layer": "APP",
        "description": "计算好的因子值",
        "fields": {
            "date": {"type": "DATE", "description": "日期"},
            "symbol": {"type": "VARCHAR", "description": "股票代码"},
            "factor_name": {"type": "VARCHAR", "description": "因子名称"},
            "value": {"type": "DOUBLE", "description": "因子值"},
            "source": {"type": "VARCHAR", "description": "计算来源"},
        },
    },
    "app_factors_ic": {
        "name": "因子IC分析结果",
        "layer": "APP",
        "description": "因子IC回测分析",
        "fields": {
            "id": {"type": "INTEGER", "description": "主键ID"},
            "factor_name": {"type": "VARCHAR", "description": "因子名称"},
            "index_code": {"type": "VARCHAR", "description": "股票池代码"},
            "start_date": {"type": "DATE", "description": "回测开始日期"},
            "end_date": {"type": "DATE", "description": "回测结束日期"},
            "ic_mean": {"type": "DOUBLE", "description": "IC均值"},
            "ic_std": {"type": "DOUBLE", "description": "IC标准差"},
            "ic_ir": {"type": "DOUBLE", "description": "IC_IR = IC均值/IC标准差"},
            "rank_ic_mean": {"type": "DOUBLE", "description": "RankIC均值"},
            "rank_ic_std": {"type": "DOUBLE", "description": "RankIC标准差"},
            "rank_ic_ir": {"type": "DOUBLE", "description": "RankIC_IR"},
            "return_long": {"type": "DOUBLE", "description": "多头收益 (%)"},
            "return_short": {"type": "DOUBLE", "description": "空头收益 (%)"},
            "return_long_short": {"type": "DOUBLE", "description": "多空收益 (%)"},
            "calculated_at": {"type": "TIMESTAMP", "description": "计算时间"},
        },
    },
}


# =============================================================================
# Schema
# =============================================================================


def get_ods_schema(source: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS ods_calendars_{source} (
    date DATE, exchange VARCHAR, is_trading_day BOOLEAN,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, exchange)
);
CREATE TABLE IF NOT EXISTS ods_instruments_{source} (
    symbol VARCHAR, name VARCHAR, list_date DATE, delist_date DATE, market VARCHAR,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol)
);
CREATE TABLE IF NOT EXISTS ods_index_components_{source} (
    index_code VARCHAR, index_name VARCHAR, symbol VARCHAR, in_date DATE, out_date DATE, weight DOUBLE,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, symbol, in_date)
);
CREATE TABLE IF NOT EXISTS ods_daily_ohlcv_{source} (
    date DATE, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, amount DOUBLE, turnover_rate DOUBLE, pct_change DOUBLE, adjust_flag VARCHAR,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);
"""


SCHEMA_DWD = """
CREATE TABLE IF NOT EXISTS dwd_calendars (
    date DATE PRIMARY KEY, is_trading_day BOOLEAN DEFAULT TRUE,
    year INTEGER, quarter INTEGER, month INTEGER, week_of_year INTEGER, day_of_week INTEGER,
    is_month_end BOOLEAN, is_quarter_end BOOLEAN, is_year_end BOOLEAN, is_week_end BOOLEAN,
    exchange VARCHAR DEFAULT 'ALL', updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dwd_instruments (
    symbol VARCHAR PRIMARY KEY, name VARCHAR, list_date DATE, delist_date DATE, market VARCHAR,
    board_type VARCHAR, industry_sw_l1 VARCHAR, industry_sw_l2 VARCHAR, industry_sw_l3 VARCHAR,
    industry_csrc VARCHAR, status VARCHAR DEFAULT 'ACTIVE', updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dwd_index_components (
    id INTEGER PRIMARY KEY, index_code VARCHAR, index_name VARCHAR, symbol VARCHAR,
    in_date DATE, out_date DATE, weight DOUBLE, is_current BOOLEAN DEFAULT FALSE,
    source VARCHAR, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(index_code, symbol, in_date)
);
CREATE TABLE IF NOT EXISTS dwd_daily_ohlcv (
    date DATE, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, amount DOUBLE, turnover_rate DOUBLE, pct_change DOUBLE,
    factor DOUBLE DEFAULT 1.0, raw_close DOUBLE, source VARCHAR, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);
"""


SCHEMA_APP = """
CREATE TABLE IF NOT EXISTS app_monthly_stats (
    symbol VARCHAR, year INTEGER, month INTEGER, start_date DATE, end_date DATE,
    open_first DOUBLE, close_last DOUBLE, high_max DOUBLE, low_min DOUBLE,
    volume_sum DOUBLE, amount_sum DOUBLE, avg_turnover_rate DOUBLE, pct_change_monthly DOUBLE,
    PRIMARY KEY (symbol, year, month)
);
CREATE TABLE IF NOT EXISTS app_yearly_stats (
    symbol VARCHAR, year INTEGER, start_date DATE, end_date DATE,
    open_first DOUBLE, close_last DOUBLE, high_max DOUBLE, low_min DOUBLE,
    volume_sum DOUBLE, amount_sum DOUBLE, avg_turnover_rate DOUBLE, pct_change_yearly DOUBLE,
    PRIMARY KEY (symbol, year)
);
CREATE TABLE IF NOT EXISTS app_index_daily (
    date DATE, index_code VARCHAR, index_name VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, amount DOUBLE, pct_change DOUBLE,
    PRIMARY KEY (date, index_code)
);
CREATE TABLE IF NOT EXISTS app_index_members (
    index_code VARCHAR, index_name VARCHAR, symbol VARCHAR, name VARCHAR, in_date DATE,
    weight DOUBLE, market_cap DOUBLE,
    PRIMARY KEY (index_code, symbol)
);
CREATE TABLE IF NOT EXISTS app_limit_up_down (
    date DATE, symbol VARCHAR, limit_type VARCHAR, prev_close DOUBLE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    amplitude DOUBLE, turnover_rate DOUBLE, reason VARCHAR,
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS app_factors_registry (
    id INTEGER PRIMARY KEY, name VARCHAR UNIQUE, category VARCHAR, description TEXT,
    expression TEXT, parameters JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS app_factors_values (
    date DATE, symbol VARCHAR, factor_name VARCHAR, value DOUBLE, source VARCHAR,
    PRIMARY KEY (date, symbol, factor_name)
);
CREATE TABLE IF NOT EXISTS app_factors_ic (
    id INTEGER PRIMARY KEY, factor_name VARCHAR, index_code VARCHAR, start_date DATE, end_date DATE,
    ic_mean DOUBLE, ic_std DOUBLE, ic_ir DOUBLE, rank_ic_mean DOUBLE, rank_ic_std DOUBLE, rank_ic_ir DOUBLE,
    return_long DOUBLE, return_short DOUBLE, return_long_short DOUBLE, calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(factor_name, index_code, start_date, end_date)
);
"""


SCHEMA_META = """
-- 参数表: 记录每张表最后更新时间
CREATE TABLE IF NOT EXISTS meta_table_params (
    layer VARCHAR, table_name VARCHAR, source VARCHAR,
    last_update_time TIMESTAMP, last_update_records INTEGER,
    status VARCHAR DEFAULT 'OK', error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (layer, table_name, source)
);
-- 更新日志
CREATE TABLE IF NOT EXISTS meta_update_log (
    id INTEGER PRIMARY KEY, layer VARCHAR, table_name VARCHAR, source VARCHAR, update_type VARCHAR,
    start_date DATE, end_date DATE, records_total INTEGER, records_success INTEGER, records_failed INTEGER,
    status VARCHAR, error_message TEXT, started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP
);
-- 校验日志
CREATE TABLE IF NOT EXISTS meta_validation_log (
    id INTEGER PRIMARY KEY, layer VARCHAR, table_name VARCHAR, check_type VARCHAR, check_column VARCHAR,
    expected_value VARCHAR, actual_value VARCHAR, passed BOOLEAN, details JSON, checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- 数据源配置
CREATE TABLE IF NOT EXISTS meta_data_sources (
    id INTEGER PRIMARY KEY,
    source_name VARCHAR UNIQUE, source_type VARCHAR, enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 1, config JSON, last_fetch_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ODS表注册
CREATE TABLE IF NOT EXISTS meta_ods_tables (
    id INTEGER PRIMARY KEY,
    source VARCHAR, table_type VARCHAR, table_name VARCHAR UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# =============================================================================
# Validation Rules
# =============================================================================


@dataclass
class ValidationRule:
    name: str
    layer: str
    table: str
    column: str
    check_type: str
    sql: str
    expected: str
    severity: str = "ERROR"


VALIDATION_RULES = [
    ValidationRule(
        "dwd_calendar_weekend",
        "DWD",
        "dwd_calendars",
        "is_trading_day",
        "consistency",
        "SELECT COUNT(*) FROM dwd_calendars WHERE is_trading_day = TRUE AND day_of_week >= 5",
        "count = 0",
    ),
    ValidationRule(
        "dwd_instruments_list_date",
        "DWD",
        "dwd_instruments",
        "list_date",
        "accuracy",
        "SELECT COUNT(*) FROM dwd_instruments WHERE list_date > CURRENT_DATE OR list_date < '1990-01-01'",
        "count = 0",
    ),
    ValidationRule(
        "dwd_ohlcv_price_consistency",
        "DWD",
        "dwd_daily_ohlcv",
        "high/low/close",
        "consistency",
        "SELECT COUNT(*) FROM dwd_daily_ohlcv WHERE high < low OR high < close OR low > open",
        "count = 0",
    ),
    ValidationRule(
        "dwd_ohlcv_volume_positive",
        "DWD",
        "dwd_daily_ohlcv",
        "volume",
        "accuracy",
        "SELECT COUNT(*) FROM dwd_daily_ohlcv WHERE volume < 0 OR amount < 0",
        "count = 0",
    ),
    ValidationRule(
        "dwd_ohlcv_timeliness",
        "DWD",
        "dwd_daily_ohlcv",
        "date",
        "timeliness",
        "SELECT MAX(date) FROM dwd_daily_ohlcv",
        "max_date >= TODAY - 1",
        "WARNING",
    ),
]


@dataclass
class ValidationResult:
    rule: ValidationRule
    passed: bool
    expected: str
    actual: str
    details: dict | None = None


@dataclass
class UpdateResult:
    """更新结果"""

    layer: str
    table: str
    source: str
    records: int
    status: str
    start_time: datetime
    end_time: datetime | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def is_success(self) -> bool:
        return self.status == "SUCCESS"


# =============================================================================
# QuantDB Class
# =============================================================================


class QuantDB:
    """量化数据库管理器 (3层架构 + ETL分离)"""

    def __init__(
        self, db_path: str = ":memory:", read_only: bool = False, config: dict | None = None
    ):
        self.db_path = db_path
        self.read_only = read_only
        self.config = config or {}
        self._conn: duckdb.DuckDBPyConnection | None = None

        # 连接并初始化 (包括 :memory: 数据库)
        self.connect()
        if not read_only:
            self.init_schema()

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path, read_only=self.read_only, config=self.config)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> QuantDB:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def init_schema(self) -> None:
        conn = self.connect()
        conn.execute(SCHEMA_DWD)
        conn.execute(SCHEMA_APP)
        conn.execute(SCHEMA_META)
        conn.commit()
        print(f"✅ 数据库Schema初始化完成: {self.db_path}")

    def execute(self, sql: str, params: dict | None = None) -> duckdb.DuckDBPyConnection:
        conn = self.connect()
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)
        conn.commit()
        return conn

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        conn = self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()

    # -------------------------------------------------------------------------
    # Table Params - 参数表
    # -------------------------------------------------------------------------

    def update_table_params(
        self,
        layer: str,
        table_name: str,
        records: int,
        source: str = "",
        status: str = "OK",
        error_message: str = None,
    ) -> None:
        """更新表参数 (最后更新时间)"""
        error_val = "NULL" if not error_message else f"'{error_message}'"
        now = datetime.now()

        # 先删除后插入（避免 ON CONFLICT 语法兼容问题）
        self.execute(
            f"DELETE FROM meta_table_params WHERE layer = '{layer}' AND table_name = '{table_name}' AND source = '{source}'"
        )
        self.execute(f"""
            INSERT INTO meta_table_params
            (layer, table_name, source, last_update_time, last_update_records, status, error_message, created_at, updated_at)
            VALUES ('{layer}', '{table_name}', '{source}', '{now}', {records}, '{status}', {error_val}, '{now}', '{now}')
        """)

    def get_table_params(
        self, layer: str | None = None, table_name: str | None = None
    ) -> pd.DataFrame:
        """获取表参数"""
        conditions = []
        if layer:
            conditions.append(f"layer = '{layer}'")
        if table_name:
            conditions.append(f"table_name = '{table_name}'")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT layer, table_name, source, last_update_time,
                   last_update_records, status, error_message
            FROM meta_table_params
            WHERE {where}
            ORDER BY layer, table_name
        """
        return self.query(sql)

    def get_all_table_status(self) -> pd.DataFrame:
        """获取所有表状态"""
        sql = """
            SELECT layer, table_name, source, last_update_time, last_update_records, status,
                   CASE
                       WHEN last_update_time >= CURRENT_TIMESTAMP - INTERVAL '1 day' THEN 'fresh'
                       WHEN last_update_time >= CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 'stale'
                       ELSE 'outdated'
                   END as freshness
            FROM meta_table_params
            ORDER BY layer, table_name
        """
        return self.query(sql)

    # -------------------------------------------------------------------------
    # Data Source
    # -------------------------------------------------------------------------

    def register_source(
        self, source: str, source_type: str = "api", priority: int = 1, config: dict = None
    ) -> None:
        config_json = json.dumps(config or {})
        # 先删除旧记录
        self.execute(f"DELETE FROM meta_data_sources WHERE source_name = '{source}'")
        # 计算新ID
        max_id = self.query("SELECT COALESCE(MAX(id), 0) as max_id FROM meta_data_sources").iloc[
            0, 0
        ]
        new_id = int(max_id) + 1
        self.execute(f"""
            INSERT INTO meta_data_sources
            (id, source_name, source_type, priority, config, enabled)
            VALUES ({new_id}, '{source}', '{source_type}', {priority}, '{config_json}', TRUE)
        """)

    def get_active_sources(self) -> list[str]:
        df = self.query(
            "SELECT source_name FROM meta_data_sources WHERE enabled = TRUE ORDER BY priority ASC"
        )
        return df["source_name"].tolist()

    def get_primary_source(self) -> str | None:
        df = self.query(
            "SELECT source_name FROM meta_data_sources WHERE enabled = TRUE ORDER BY priority ASC LIMIT 1"
        )
        return df.iloc[0]["source_name"] if not df.empty else None

    # -------------------------------------------------------------------------
    # ODS Layer
    # -------------------------------------------------------------------------

    def create_ods_tables(self, source: str) -> None:
        self.execute(get_ods_schema(source))
        for table_type in ["calendars", "instruments", "index_components", "daily_ohlcv"]:
            table_name = f"ods_{table_type}_{source}"
            # 先删除后插入
            self.execute(
                f"DELETE FROM meta_ods_tables WHERE source = '{source}' AND table_type = '{table_type}'"
            )
            # 计算新ID
            max_id = self.query("SELECT COALESCE(MAX(id), 0) as max_id FROM meta_ods_tables").iloc[
                0, 0
            ]
            new_id = int(max_id) + 1
            self.execute(
                f"INSERT INTO meta_ods_tables (id, source, table_type, table_name) VALUES ({new_id}, '{source}', '{table_type}', '{table_name}')"
            )

    def import_ods(self, source: str, table_type: str, df: pd.DataFrame) -> int:
        table_name = f"ods_{table_type}_{source}"

        if table_type not in self.list_ods_tables(source):
            self.create_ods_tables(source)

        df = df.copy()
        df["source"] = source
        if "fetched_at" not in df.columns:
            df["fetched_at"] = datetime.now()

        conn = self.connect()
        records = []

        for _, row in df.iterrows():
            if table_type == "calendars":
                records.append(
                    [row["date"], row["exchange"], row["is_trading_day"], source, row["fetched_at"]]
                )
            elif table_type == "instruments":
                records.append(
                    [
                        row["symbol"],
                        row["name"],
                        row.get("list_date"),
                        row.get("delist_date"),
                        row["market"],
                        source,
                        row["fetched_at"],
                    ]
                )
            elif table_type == "index_components":
                records.append(
                    [
                        row["index_code"],
                        row.get("index_name"),
                        row["symbol"],
                        row.get("in_date"),
                        row.get("out_date"),
                        row.get("weight", 0),
                        source,
                        row["fetched_at"],
                    ]
                )
            elif table_type == "daily_ohlcv":
                records.append(
                    [
                        row["date"],
                        row["symbol"],
                        row.get("open", 0),
                        row.get("high", 0),
                        row.get("low", 0),
                        row.get("close", 0),
                        row.get("volume", 0),
                        row.get("amount", 0),
                        row.get("turnover_rate", 0),
                        row.get("pct_change", 0),
                        row.get("adjust_flag", "2"),
                        source,
                        row["fetched_at"],
                    ]
                )

        if records:
            placeholders = {
                "calendars": "(?, ?, ?, ?, ?)",
                "instruments": "(?, ?, ?, ?, ?, ?, ?)",
                "index_components": "(?, ?, ?, ?, ?, ?, ?, ?)",
                "daily_ohlcv": "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            }
            sql = f"INSERT OR REPLACE INTO {table_name} VALUES {placeholders[table_type]}"
            conn.executemany(sql, records)
            conn.commit()
            self.update_table_params("ODS", f"ods_{table_type}", len(records), source)

        return len(records) if records else 0

    def list_ods_tables(self, source: str | None = None) -> list:
        if source:
            df = self.query(f"SELECT table_name FROM meta_ods_tables WHERE source = '{source}'")
            return df["table_name"].tolist() if not df.empty else []
        else:
            df = self.query("SELECT DISTINCT source FROM meta_ods_tables")
            return df["source"].tolist()

    def get_ods(self, source: str, table_type: str) -> pd.DataFrame:
        return self.query(f"SELECT * FROM ods_{table_type}_{source} ORDER BY date")

    # -------------------------------------------------------------------------
    # DWD Layer
    # -------------------------------------------------------------------------

    def get_trading_days(self, start: str | None = None, end: str | None = None) -> list[str]:
        conditions = ["is_trading_day = TRUE"]
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        where = " AND ".join(conditions)
        df = self.query(f"SELECT date FROM dwd_calendars WHERE {where} ORDER BY date")
        return [str(d.date()) for d in df["date"].tolist()]

    def get_ohlcv(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        where = " AND ".join(conditions) if conditions else "1=1"
        df = self.query(f"SELECT * FROM dwd_daily_ohlcv WHERE {where} ORDER BY date, symbol")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_instruments(self, market: str | None = None, active_only: bool = True) -> pd.DataFrame:
        conditions = []
        if market:
            conditions.append(f"market = '{market}'")
        if active_only:
            conditions.append("status = 'ACTIVE'")
        where = " AND ".join(conditions) if conditions else "1=1"
        return self.query(f"SELECT * FROM dwd_instruments WHERE {where} ORDER BY symbol")

    def get_current_index_members(self, index_code: str) -> pd.DataFrame:
        return self.query(f"""
            SELECT ic.index_code, ic.index_name, ic.symbol, i.name, ic.in_date, ic.weight
            FROM dwd_index_components ic
            LEFT JOIN dwd_instruments i ON ic.symbol = i.symbol
            WHERE ic.index_code = '{index_code}' AND ic.is_current = TRUE
            ORDER BY ic.weight DESC
        """)

    # -------------------------------------------------------------------------
    # APP Layer
    # -------------------------------------------------------------------------

    def aggregate_monthly_stats(self, start: str | None = None, end: str | None = None) -> int:
        where = ""
        if start:
            where += f" WHERE date >= '{start}'"
        if end:
            where += f" WHERE date <= '{end}'" if where else f" AND date <= '{end}'"

        sql = f"""
            INSERT INTO app_monthly_stats
            (symbol, year, month, start_date, end_date, open_first, close_last, high_max, low_min,
             volume_sum, amount_sum, avg_turnover_rate, pct_change_monthly)
            SELECT symbol, EXTRACT(YEAR FROM date)::INTEGER, EXTRACT(MONTH FROM date)::INTEGER,
                   MIN(date), MAX(date), FIRST(open), LAST(close), MAX(high), MIN(low),
                   SUM(volume), SUM(amount), AVG(turnover_rate),
                   (LAST(close) - FIRST(open)) / FIRST(open) * 100
            FROM dwd_daily_ohlcv
            {where}
            GROUP BY symbol, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
            ON CONFLICT(symbol, year, month) DO UPDATE SET
                start_date = excluded.start_date, end_date = excluded.end_date,
                open_first = excluded.open_first, close_last = excluded.close_last,
                high_max = excluded.high_max, low_min = excluded.low_min,
                volume_sum = excluded.volume_sum, amount_sum = excluded.amount_sum,
                avg_turnover_rate = excluded.avg_turnover_rate, pct_change_monthly = excluded.pct_change_monthly
        """
        conn = self.execute(sql)
        self.update_table_params("APP", "app_monthly_stats", conn.rowcount)
        return conn.rowcount

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self, rule: ValidationRule) -> ValidationResult:
        try:
            df = self.query(rule.sql)
            actual = str(df.iloc[0][0]) if len(df) > 0 else "0"
            if "count" in rule.expected.lower():
                expected_count = int(rule.expected.split("=")[1].strip())
                passed = int(actual) == expected_count
            elif "max_date" in rule.expected.lower():
                passed = actual >= str(date.today() - timedelta(days=2))
            else:
                passed = True
            return ValidationResult(rule=rule, passed=passed, expected=rule.expected, actual=actual)
        except Exception as e:
            return ValidationResult(rule=rule, passed=False, expected=rule.expected, actual=str(e))

    def validate_all(self, layer: str | None = None) -> list[ValidationResult]:
        rules = VALIDATION_RULES
        if layer:
            rules = [r for r in rules if r.layer == layer]
        results = []
        for rule in rules:
            result = self.validate(rule)
            results.append(result)
            details_json = json.dumps(result.details or {})

            self.execute(f"""
                INSERT INTO meta_validation_log
                (id, layer, table_name, check_type, check_column, expected_value, actual_value, passed, details)
                VALUES (
                    (SELECT COALESCE(MAX(id), 0) + 1 FROM meta_validation_log),
                    '{result.rule.layer}', '{result.rule.table}', '{result.rule.check_type}',
                    '{result.rule.column}', '{result.expected}', '{result.actual}',
                    {result.passed}, '{details_json}'
                )
            """)
        return results

    # -------------------------------------------------------------------------
    # Update Log
    # -------------------------------------------------------------------------

    def log_update(
        self,
        layer: str,
        table: str,
        source: str,
        update_type: str,
        records: int,
        status: str,
        start_date: str | None = None,
        end_date: str | None = None,
        error: str | None = None,
    ) -> None:
        start_val = "NULL" if not start_date else f"'{start_date}'"
        end_val = "NULL" if not end_date else f"'{end_date}'"
        error_val = "NULL" if not error else f"'{error}'"
        now = datetime.now()
        max_id = self.query("SELECT COALESCE(MAX(id), 0) as max_id FROM meta_update_log").iloc[0, 0]
        new_id = int(max_id) + 1
        self.execute(f"""
            INSERT INTO meta_update_log
            (id, layer, table_name, source, update_type, start_date, end_date, records_total,
             records_success, records_failed, status, error_message, completed_at)
            VALUES ({new_id}, '{layer}', '{table}', '{source}', '{update_type}',
                    {start_val}, {end_val}, {records}, {records}, 0, '{status}',
                    {error_val}, '{now}')
        """)

    def get_update_history(self, table: str | None = None, limit: int = 10) -> pd.DataFrame:
        where = f"WHERE table_name = '{table}'" if table else ""
        return self.query(
            f"SELECT * FROM meta_update_log {where} ORDER BY started_at DESC LIMIT {limit}"
        )

    # -------------------------------------------------------------------------
    # Data Dictionary
    # -------------------------------------------------------------------------

    def get_data_dictionary(self, table_name: str | None = None) -> dict:
        if table_name:
            return {table_name: DATA_DICTIONARY.get(table_name, {})}
        return DATA_DICTIONARY

    def print_data_dictionary(self, table_name: str | None = None) -> None:
        dd = self.get_data_dictionary(table_name)
        for name, info in dd.items():
            if not info:
                continue
            print(f"\n{'='*60}")
            print(f"📋 {name}")
            print(f"{'='*60}")
            print(f"名称: {info.get('name', '')}")
            print(f"层级: {info.get('layer', '')}")
            print(f"描述: {info.get('description', '')}")
            print("\n字段:")
            print("-" * 60)
            print(f"{'字段名':<20} {'类型':<10} {'说明':<30}")
            print("-" * 60)
            for field, props in info.get("fields", {}).items():
                print(f"{field:<20} {props.get('type', ''):<10} {props.get('description', ''):<30}")

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        info = {"db_path": self.db_path, "sources": self.get_active_sources(), "tables": {}}

        info["tables"]["ODS"] = {}
        for source in info["sources"]:
            info["tables"]["ODS"][source] = self.list_ods_tables(source)

        for layer, tables in [
            (
                "DWD",
                ["dwd_calendars", "dwd_instruments", "dwd_index_components", "dwd_daily_ohlcv"],
            ),
            (
                "APP",
                [
                    "app_monthly_stats",
                    "app_yearly_stats",
                    "app_factors_registry",
                    "app_factors_values",
                ],
            ),
        ]:
            info["tables"][layer] = {}
            for table in tables:
                try:
                    count = self.query(f"SELECT COUNT(*) FROM {table}").iloc[0][0]
                    info["tables"][layer][table] = {"rows": int(count)}
                except Exception:
                    info["tables"][layer][table] = {"rows": -1}

        return info
