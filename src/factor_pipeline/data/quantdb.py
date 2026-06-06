"""Quantitative Database - 量化数据库核心模块 (3层架构 + ETL分离)

架构设计:
- ODS (原始数据层): 按数据源分表命名
- DWD (明细数据层): 清洗、标准化后的数据
- APP (应用数据层): 聚合统计、因子数据
- ETL (数据迁移): 独立的迁移脚本，不影响数据源

Philosophy: Keep it simple, make it work, make it fast.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Optional

import duckdb
import pandas as pd


# =============================================================================
# Constants - 常量定义
# =============================================================================

class Market(Enum):
    """市场标识"""
    SSE = "SSE"
    SZSE = "SZSE"
    BSE = "BSE"


class IndexCode(Enum):
    """常用指数代码"""
    SSE50 = ("000016", "SH", "上证50")
    CSI300 = ("000300", "SH", "沪深300")
    CSI500 = ("000905", "SH", "中证500")
    CSI1000 = ("000852", "SH", "中证1000")
    CSI800 = ("000906", "SH", "中证800")
    CSI100 = ("000903", "SH", "中证100")
    
    def __init__(self, code: str, exchange: str, name: str):
        self.code = code
        self.exchange = exchange
        self.name = name
    
    @property
    def full_code(self) -> str:
        return f"{self.code}.{self.exchange}"


# =============================================================================
# Schema Templates - Schema模板 (动态生成)
# =============================================================================

def get_ods_schema(source: str) -> str:
    """获取ODS层Schema (按数据源命名)
    
    Args:
        source: 数据源名称, 如 baostock, akshare
        
    Returns:
        CREATE TABLE SQL
    """
    return f"""
-- ODS: 原始日历数据 ({source})
CREATE TABLE IF NOT EXISTS ods_calendars_{source} (
    date DATE,
    exchange VARCHAR,
    is_trading_day BOOLEAN,
    source VARCHAR DEFAULT '{source}',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, exchange)
);

-- ODS: 原始股票列表 ({source})
CREATE TABLE IF NOT EXISTS ods_instruments_{source} (
    symbol VARCHAR,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,
    source VARCHAR DEFAULT '{source}',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol)
);

-- ODS: 原始指数成分 ({source})
CREATE TABLE IF NOT EXISTS ods_index_components_{source} (
    index_code VARCHAR,
    index_name VARCHAR,
    symbol VARCHAR,
    in_date DATE,
    out_date DATE,
    weight DOUBLE,
    source VARCHAR DEFAULT '{source}',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, symbol, in_date)
);

-- ODS: 原始K线数据 ({source})
CREATE TABLE IF NOT EXISTS ods_daily_ohlcv_{source} (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    turnover_rate DOUBLE,
    pct_change DOUBLE,
    adjust_flag VARCHAR,
    source VARCHAR DEFAULT '{source}',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);
"""


# =============================================================================
# DWD Layer - 明细数据层 (固定)
# =============================================================================

SCHEMA_DWD = """
-- DWD: 交易日历
CREATE TABLE IF NOT EXISTS dwd_calendars (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN DEFAULT TRUE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    week_of_year INTEGER,
    day_of_week INTEGER,
    is_month_end BOOLEAN,
    is_quarter_end BOOLEAN,
    is_year_end BOOLEAN,
    is_week_end BOOLEAN,
    exchange VARCHAR DEFAULT 'ALL',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DWD: 股票基础信息
CREATE TABLE IF NOT EXISTS dwd_instruments (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,
    board_type VARCHAR,
    industry_sw_l1 VARCHAR,
    industry_sw_l2 VARCHAR,
    industry_sw_l3 VARCHAR,
    industry_csrc VARCHAR,
    status VARCHAR DEFAULT 'ACTIVE',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DWD: 指数成分股快照
CREATE TABLE IF NOT EXISTS dwd_index_components (
    id INTEGER PRIMARY KEY,
    index_code VARCHAR,
    index_name VARCHAR,
    symbol VARCHAR,
    in_date DATE,
    out_date DATE,
    weight DOUBLE,
    is_current BOOLEAN DEFAULT FALSE,
    source VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(index_code, symbol, in_date)
);

-- DWD: 日K线数据 (前复权)
CREATE TABLE IF NOT EXISTS dwd_daily_ohlcv (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    turnover_rate DOUBLE,
    pct_change DOUBLE,
    factor DOUBLE DEFAULT 1.0,
    raw_close DOUBLE,
    source VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);
"""


# =============================================================================
# APP Layer - 应用数据层 (固定)
# =============================================================================

SCHEMA_APP = """
-- APP: 月度汇总统计
CREATE TABLE IF NOT EXISTS app_monthly_stats (
    symbol VARCHAR,
    year INTEGER,
    month INTEGER,
    start_date DATE,
    end_date DATE,
    open_first DOUBLE,
    close_last DOUBLE,
    high_max DOUBLE,
    low_min DOUBLE,
    volume_sum DOUBLE,
    amount_sum DOUBLE,
    avg_turnover_rate DOUBLE,
    pct_change_monthly DOUBLE,
    PRIMARY KEY (symbol, year, month)
);

-- APP: 年度汇总统计
CREATE TABLE IF NOT EXISTS app_yearly_stats (
    symbol VARCHAR,
    year INTEGER,
    start_date DATE,
    end_date DATE,
    open_first DOUBLE,
    close_last DOUBLE,
    high_max DOUBLE,
    low_min DOUBLE,
    volume_sum DOUBLE,
    amount_sum DOUBLE,
    avg_turnover_rate DOUBLE,
    pct_change_yearly DOUBLE,
    PRIMARY KEY (symbol, year)
);

-- APP: 指数日行情
CREATE TABLE IF NOT EXISTS app_index_daily (
    date DATE,
    index_code VARCHAR,
    index_name VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    pct_change DOUBLE,
    PRIMARY KEY (date, index_code)
);

-- APP: 当前股票池成员
CREATE TABLE IF NOT EXISTS app_index_members (
    index_code VARCHAR,
    index_name VARCHAR,
    symbol VARCHAR,
    name VARCHAR,
    in_date DATE,
    weight DOUBLE,
    market_cap DOUBLE,
    PRIMARY KEY (index_code, symbol)
);

-- APP: 涨跌停股票
CREATE TABLE IF NOT EXISTS app_limit_up_down (
    date DATE,
    symbol VARCHAR,
    limit_type VARCHAR,
    prev_close DOUBLE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    amplitude DOUBLE,
    turnover_rate DOUBLE,
    reason VARCHAR,
    PRIMARY KEY (date, symbol)
);

-- APP: 因子定义表
CREATE TABLE IF NOT EXISTS app_factors_registry (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE,
    category VARCHAR,
    description TEXT,
    expression TEXT,
    parameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- APP: 因子值缓存表
CREATE TABLE IF NOT EXISTS app_factors_values (
    date DATE,
    symbol VARCHAR,
    factor_name VARCHAR,
    value DOUBLE,
    source VARCHAR,
    PRIMARY KEY (date, symbol, factor_name)
);

-- APP: 因子IC分析结果
CREATE TABLE IF NOT EXISTS app_factors_ic (
    id INTEGER PRIMARY KEY,
    factor_name VARCHAR,
    index_code VARCHAR,
    start_date DATE,
    end_date DATE,
    ic_mean DOUBLE,
    ic_std DOUBLE,
    ic_ir DOUBLE,
    rank_ic_mean DOUBLE,
    rank_ic_std DOUBLE,
    rank_ic_ir DOUBLE,
    return_long DOUBLE,
    return_short DOUBLE,
    return_long_short DOUBLE,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(factor_name, index_code, start_date, end_date)
);
"""


# =============================================================================
# Metadata Layer - 元数据层
# =============================================================================

SCHEMA_META = """
-- 数据更新日志
CREATE TABLE IF NOT EXISTS meta_update_log (
    id INTEGER PRIMARY KEY,
    layer VARCHAR,
    table_name VARCHAR,
    source VARCHAR,
    update_type VARCHAR,
    start_date DATE,
    end_date DATE,
    records_total INTEGER,
    records_success INTEGER,
    records_failed INTEGER,
    status VARCHAR,
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 数据校验日志
CREATE TABLE IF NOT EXISTS meta_validation_log (
    id INTEGER PRIMARY KEY,
    layer VARCHAR,
    table_name VARCHAR,
    check_type VARCHAR,
    check_column VARCHAR,
    expected_value VARCHAR,
    actual_value VARCHAR,
    passed BOOLEAN,
    details JSON,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 数据源配置表
CREATE TABLE IF NOT EXISTS meta_data_sources (
    id INTEGER PRIMARY KEY,
    source_name VARCHAR UNIQUE,
    source_type VARCHAR,
    enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 1,
    config JSON,
    last_fetch_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ODS表注册表 (记录已创建的ODS表)
CREATE TABLE IF NOT EXISTS meta_ods_tables (
    id INTEGER PRIMARY KEY,
    source VARCHAR,
    table_type VARCHAR,
    table_name VARCHAR UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# =============================================================================
# Validation Rules - 数据校验规则
# =============================================================================

@dataclass
class ValidationRule:
    """数据校验规则"""
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
        name="dwd_calendar_weekend",
        layer="DWD",
        table="dwd_calendars",
        column="is_trading_day",
        check_type="consistency",
        sql="SELECT COUNT(*) FROM dwd_calendars WHERE is_trading_day = TRUE AND day_of_week >= 5",
        expected="count = 0",
    ),
    ValidationRule(
        name="dwd_instruments_list_date",
        layer="DWD",
        table="dwd_instruments",
        column="list_date",
        check_type="accuracy",
        sql="SELECT COUNT(*) FROM dwd_instruments WHERE list_date > CURRENT_DATE OR list_date < '1990-01-01'",
        expected="count = 0",
    ),
    ValidationRule(
        name="dwd_ohlcv_price_consistency",
        layer="DWD",
        table="dwd_daily_ohlcv",
        column="high/low/close",
        check_type="consistency",
        sql="SELECT COUNT(*) FROM dwd_daily_ohlcv WHERE high < low OR high < close OR low > open",
        expected="count = 0",
    ),
    ValidationRule(
        name="dwd_ohlcv_volume_positive",
        layer="DWD",
        table="dwd_daily_ohlcv",
        column="volume",
        check_type="accuracy",
        sql="SELECT COUNT(*) FROM dwd_daily_ohlcv WHERE volume < 0 OR amount < 0",
        expected="count = 0",
    ),
    ValidationRule(
        name="dwd_ohlcv_timeliness",
        layer="DWD",
        table="dwd_daily_ohlcv",
        column="date",
        check_type="timeliness",
        sql="SELECT MAX(date) FROM dwd_daily_ohlcv",
        expected="max_date >= TODAY - 1",
        severity="WARNING",
    ),
]


# =============================================================================
# QuantDB Class - 量化数据库类
# =============================================================================

@dataclass
class ValidationResult:
    """校验结果"""
    rule: ValidationRule
    passed: bool
    expected: str
    actual: str
    details: Optional[dict] = None


class QuantDB:
    """量化数据库管理器 (3层架构 + ETL分离)
    
    架构:
    - ODS: 原始数据层 (按数据源分表: ods_xxx_baostock)
    - DWD: 明细数据层 (标准化数据)
    - APP: 应用数据层 (聚合统计、因子)
    - ETL: 独立的迁移脚本
    
    Example:
        db = QuantDB("data/quant.db")
        
        # 注册数据源
        db.register_source("baostock", priority=1)
        
        # 创建ODS表
        db.create_ods_tables("baostock")
        
        # 导入原始数据
        db.import_ods("baostock", "calendars", df)
        db.import_ods("baostock", "ohlcv", df)
        
        # ETL迁移 (独立的迁移脚本)
        from factor_pipeline.data.etl import ETLPipeline
        etl = ETLPipeline(db)
        etl.run(source="baostock")
        
        # 查询DWD数据
        df = db.get_ohlcv(symbols=["000001.SZ"])
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        config: Optional[dict] = None,
    ):
        self.db_path = db_path
        self.read_only = read_only
        self.config = config or {}
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        if db_path != ":memory:" and not os.path.exists(db_path):
            self.init_schema()
        elif db_path != ":memory:":
            self.connect()
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(
                self.db_path,
                read_only=self.read_only,
                config=self.config,
            )
        return self._conn
    
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "QuantDB":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def init_schema(self) -> None:
        """初始化数据库Schema (DWD + APP + META)"""
        conn = self.connect()
        
        # 初始化固定Schema
        conn.execute(SCHEMA_DWD)
        conn.execute(SCHEMA_APP)
        conn.execute(SCHEMA_META)
        
        conn.commit()
        print(f"✅ 数据库Schema初始化完成 (3层架构 + ETL分离): {self.db_path}")
    
    def execute(self, sql: str, params: Optional[dict] = None) -> duckdb.DuckDBPyConnection:
        conn = self.connect()
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)
        conn.commit()
        return conn
    
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        conn = self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()
    
    # -------------------------------------------------------------------------
    # Data Source Management - 数据源管理
    # -------------------------------------------------------------------------
    
    def register_source(self, source: str, source_type: str = "api", priority: int = 1, config: dict = None) -> None:
        """注册数据源"""
        config_json = json.dumps(config or {})
        self.execute(f"""
            INSERT OR REPLACE INTO meta_data_sources 
            (source_name, source_type, priority, config, enabled)
            VALUES ('{source}', '{source_type}', {priority}, '{config_json}', TRUE)
        """)
        print(f"✅ 数据源已注册: {source}")
    
    def get_active_sources(self) -> list[str]:
        """获取活跃数据源列表"""
        df = self.query("""
            SELECT source_name FROM meta_data_sources 
            WHERE enabled = TRUE 
            ORDER BY priority ASC
        """)
        return df["source_name"].tolist()
    
    def get_primary_source(self) -> Optional[str]:
        """获取主数据源 (优先级最高)"""
        df = self.query("""
            SELECT source_name FROM meta_data_sources 
            WHERE enabled = TRUE 
            ORDER BY priority ASC LIMIT 1
        """)
        return df.iloc[0]["source_name"] if not df.empty else None
    
    # -------------------------------------------------------------------------
    # ODS Layer Operations - ODS层操作
    # -------------------------------------------------------------------------
    
    def create_ods_tables(self, source: str) -> None:
        """创建指定数据源的ODS表"""
        schema_sql = get_ods_schema(source)
        self.execute(schema_sql)
        
        # 注册ODS表
        for table_type in ["calendars", "instruments", "index_components", "daily_ohlcv"]:
            table_name = f"ods_{table_type}_{source}"
            self.execute(f"""
                INSERT OR REPLACE INTO meta_ods_tables (source, table_type, table_name)
                VALUES ('{source}', '{table_type}', '{table_name}')
            """)
        
        print(f"✅ ODS表已创建: ods_*_{source}")
    
    def import_ods(
        self,
        source: str,
        table_type: str,
        df: pd.DataFrame,
    ) -> int:
        """导入数据到ODS层
        
        Args:
            source: 数据源名称
            table_type: 表类型 (calendars/instruments/index_components/daily_ohlcv)
            df: 数据DataFrame
        """
        table_name = f"ods_{table_type}_{source}"
        
        # 确保表存在
        if table_type not in self.list_ods_tables(source):
            self.create_ods_tables(source)
        
        df = df.copy()
        df["source"] = source
        if "fetched_at" not in df.columns:
            df["fetched_at"] = datetime.now()
        
        conn = self.connect()
        
        if table_type == "calendars":
            records = [[row["date"], row["exchange"], row["is_trading_day"], source, row["fetched_at"]] 
                      for _, row in df.iterrows()]
            placeholders = "(?, ?, ?, ?, ?)"
            
        elif table_type == "instruments":
            records = [[row["symbol"], row["name"], row.get("list_date"), row.get("delist_date"),
                       row["market"], source, row["fetched_at"]] for _, row in df.iterrows()]
            placeholders = "(?, ?, ?, ?, ?, ?, ?)"
            
        elif table_type == "index_components":
            records = [[row["index_code"], row.get("index_name"), row["symbol"],
                       row.get("in_date"), row.get("out_date"), row.get("weight", 0), 
                       source, row["fetched_at"]] for _, row in df.iterrows()]
            placeholders = "(?, ?, ?, ?, ?, ?, ?, ?)"
            
        elif table_type == "daily_ohlcv":
            records = [[row["date"], row["symbol"], row.get("open", 0), row.get("high", 0),
                       row.get("low", 0), row.get("close", 0), row.get("volume", 0),
                       row.get("amount", 0), row.get("turnover_rate", 0), row.get("pct_change", 0),
                       row.get("adjust_flag", "2"), source, row["fetched_at"]] for _, row in df.iterrows()]
            placeholders = "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        
        if records:
            sql = f"INSERT OR REPLACE INTO {table_name} VALUES {placeholders}"
            conn.executemany(sql, records)
            conn.commit()
        
        return len(records) if records else 0
    
    def list_ods_tables(self, source: Optional[str] = None) -> list[str]:
        """列出ODS表"""
        if source:
            df = self.query(f"SELECT table_name FROM meta_ods_tables WHERE source = '{source}'")
        else:
            df = self.query("SELECT DISTINCT source FROM meta_ods_tables")
            return df["source"].tolist()
        return df["table_name"].tolist() if not df.empty else []
    
    def get_ods(self, source: str, table_type: str) -> pd.DataFrame:
        """获取ODS数据"""
        table_name = f"ods_{table_type}_{source}"
        return self.query(f"SELECT * FROM {table_name} ORDER BY date")
    
    # -------------------------------------------------------------------------
    # DWD Layer Operations - DWD层操作
    # -------------------------------------------------------------------------
    
    def get_trading_days(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[str]:
        """获取交易日列表"""
        conditions = ["is_trading_day = TRUE"]
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        
        where = " AND ".join(conditions)
        sql = f"SELECT date FROM dwd_calendars WHERE {where} ORDER BY date"
        
        df = self.query(sql)
        return [str(d.date()) for d in df["date"].tolist()]
    
    def get_ohlcv(
        self,
        symbols: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取K线数据"""
        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM dwd_daily_ohlcv WHERE {where} ORDER BY date, symbol"
        
        df = self.query(sql)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def get_instruments(
        self,
        market: Optional[str] = None,
        active_only: bool = True,
    ) -> pd.DataFrame:
        """获取股票列表"""
        conditions = []
        if market:
            conditions.append(f"market = '{market}'")
        if active_only:
            conditions.append("status = 'ACTIVE'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM dwd_instruments WHERE {where} ORDER BY symbol"
        
        return self.query(sql)
    
    def get_current_index_members(self, index_code: str) -> pd.DataFrame:
        """获取当前指数成员"""
        sql = f"""
            SELECT 
                ic.index_code,
                ic.index_name,
                ic.symbol,
                i.name,
                ic.in_date,
                ic.weight
            FROM dwd_index_components ic
            LEFT JOIN dwd_instruments i ON ic.symbol = i.symbol
            WHERE ic.index_code = '{index_code}'
              AND ic.is_current = TRUE
            ORDER BY ic.weight DESC
        """
        return self.query(sql)
    
    # -------------------------------------------------------------------------
    # APP Layer Operations - APP层操作
    # -------------------------------------------------------------------------
    
    def aggregate_monthly_stats(self, start: Optional[str] = None, end: Optional[str] = None) -> int:
        """聚合月度统计"""
        where = ""
        if start:
            where += f" WHERE date >= '{start}'"
        if end:
            where += f" WHERE date <= '{end}'" if where else f" AND date <= '{end}'"
        
        sql = f"""
            INSERT INTO app_monthly_stats 
            (symbol, year, month, start_date, end_date, open_first, close_last, 
             high_max, low_min, volume_sum, amount_sum, avg_turnover_rate, pct_change_monthly)
            SELECT
                symbol,
                EXTRACT(YEAR FROM date)::INTEGER as year,
                EXTRACT(MONTH FROM date)::INTEGER as month,
                MIN(date) as start_date,
                MAX(date) as end_date,
                FIRST(open) as open_first,
                LAST(close) as close_last,
                MAX(high) as high_max,
                MIN(low) as low_min,
                SUM(volume) as volume_sum,
                SUM(amount) as amount_sum,
                AVG(turnover_rate) as avg_turnover_rate,
                (LAST(close) - FIRST(open)) / FIRST(open) * 100 as pct_change_monthly
            FROM dwd_daily_ohlcv
            {where}
            GROUP BY symbol, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
            ON CONFLICT(symbol, year, month) DO UPDATE SET
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                open_first = excluded.open_first,
                close_last = excluded.close_last,
                high_max = excluded.high_max,
                low_min = excluded.low_min,
                volume_sum = excluded.volume_sum,
                amount_sum = excluded.amount_sum,
                avg_turnover_rate = excluded.avg_turnover_rate,
                pct_change_monthly = excluded.pct_change_monthly
        """
        conn = self.execute(sql)
        return conn.rowcount
    
    # -------------------------------------------------------------------------
    # Validation - 数据校验
    # -------------------------------------------------------------------------
    
    def validate(self, rule: ValidationRule) -> ValidationResult:
        """执行校验规则"""
        try:
            df = self.query(rule.sql)
            actual = str(df.iloc[0][0]) if len(df) > 0 else "0"
            
            if "count" in rule.expected.lower():
                expected_count = int(rule.expected.split("=")[1].strip())
                actual_count = int(actual)
                passed = (actual_count == expected_count)
            elif "max_date" in rule.expected.lower():
                passed = actual >= str(date.today() - timedelta(days=2))
            else:
                passed = True
            
            return ValidationResult(rule=rule, passed=passed, expected=rule.expected, actual=actual)
        except Exception as e:
            return ValidationResult(rule=rule, passed=False, expected=rule.expected, actual=str(e))
    
    def validate_all(self, layer: Optional[str] = None) -> list[ValidationResult]:
        """执行所有校验规则"""
        rules = VALIDATION_RULES
        if layer:
            rules = [r for r in rules if r.layer == layer]
        
        results = []
        for rule in rules:
            result = self.validate(rule)
            results.append(result)
            self._log_validation(result)
        return results
    
    def _log_validation(self, result: ValidationResult) -> None:
        """记录校验结果"""
        details_json = json.dumps(result.details or {})
        self.execute(f"""
            INSERT INTO meta_validation_log 
            (layer, table_name, check_type, check_column, expected_value, actual_value, passed, details)
            VALUES ('{result.rule.layer}', '{result.rule.table}', '{result.rule.check_type}',
                    '{result.rule.column}', '{result.expected}', '{result.actual}', 
                    {result.passed}, '{details_json}')
        """)
    
    # -------------------------------------------------------------------------
    # Update Log - 更新日志
    # -------------------------------------------------------------------------
    
    def log_update(
        self,
        layer: str,
        table: str,
        source: str,
        update_type: str,
        records: int,
        status: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """记录更新日志"""
        start_val = 'NULL' if not start_date else f"'{start_date}'"
        end_val = 'NULL' if not end_date else f"'{end_date}'"
        error_val = 'NULL' if not error else f"'{error}'"
        
        sql = f"""
            INSERT INTO meta_update_log 
            (layer, table_name, source, update_type, start_date, end_date, records_total, 
             records_success, records_failed, status, error_message, completed_at)
            VALUES ('{layer}', '{table}', '{source}', '{update_type}', 
                    {start_val}, {end_val}, {records}, {records}, 0, '{status}',
                    {error_val}, CURRENT_TIMESTAMP)
        """
        self.execute(sql)
    
    def get_update_history(self, table: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        """获取更新历史"""
        where = f"WHERE table_name = '{table}'" if table else ""
        sql = f"""
            SELECT * FROM meta_update_log 
            {where}
            ORDER BY started_at DESC 
            LIMIT {limit}
        """
        return self.query(sql)
    
    # -------------------------------------------------------------------------
    # Info - 信息
    # -------------------------------------------------------------------------
    
    def info(self) -> dict[str, Any]:
        """获取数据库信息"""
        info = {"db_path": self.db_path, "sources": self.get_active_sources(), "tables": {}}
        
        # ODS表
        info["tables"]["ODS"] = {}
        for source in info["sources"]:
            tables = self.list_ods_tables(source)
            info["tables"]["ODS"][source] = tables
        
        # DWD表
        dwd_tables = ["dwd_calendars", "dwd_instruments", "dwd_index_components", "dwd_daily_ohlcv"]
        info["tables"]["DWD"] = {}
        for table in dwd_tables:
            try:
                count = self.query(f"SELECT COUNT(*) FROM {table}").iloc[0][0]
                info["tables"]["DWD"][table] = {"rows": int(count)}
            except Exception:
                info["tables"]["DWD"][table] = {"rows": -1}
        
        # APP表
        app_tables = ["app_monthly_stats", "app_yearly_stats", "app_factors_registry", "app_factors_values"]
        info["tables"]["APP"] = {}
        for table in app_tables:
            try:
                count = self.query(f"SELECT COUNT(*) FROM {table}").iloc[0][0]
                info["tables"]["APP"][table] = {"rows": int(count)}
            except Exception:
                info["tables"]["APP"][table] = {"rows": -1}
        
        return info
