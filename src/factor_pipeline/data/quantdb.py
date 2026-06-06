"""Quantitative Database - 量化数据库核心模块 (分层设计)

采用典型数仓分层架构:
- ODS (Operational Data Store): 原始数据层
- DWD (Data Warehouse Detail): 明细数据层
- DWS (Data Warehouse Summary): 汇总数据层
- ADS (Application Data Service): 应用数据层
- Factors: 因子数据层

Philosophy: Keep it simple, make it work, make it fast.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd


# =============================================================================
# Constants - 常量定义
# =============================================================================

class Market(Enum):
    """市场标识"""
    SSE = "SSE"   # 上交所
    SZSE = "SZSE" # 深交所
    BSE = "BSE"   # 北交所


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
# ODS Layer - 原始数据层
# =============================================================================

# ODS层Schema：存储从数据源拉取的原始数据，不做任何清洗
SCHEMA_ODS = """
-- ODS: 原始日历数据 (从baostock/akshare直接获取)
CREATE TABLE IF NOT EXISTS ods_calendars_raw (
    date DATE,
    exchange VARCHAR,      -- 交易所代码
    is_trading_day BOOLEAN,
    source VARCHAR,        -- 数据源: baostock, akshare
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, exchange, source)
);

-- ODS: 原始股票列表
CREATE TABLE IF NOT EXISTS ods_instruments_raw (
    symbol VARCHAR,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,
    source VARCHAR,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, source)
);

-- ODS: 原始指数成分
CREATE TABLE IF NOT EXISTS ods_index_components_raw (
    index_code VARCHAR,
    index_name VARCHAR,
    symbol VARCHAR,
    in_date DATE,
    out_date DATE,
    weight DOUBLE,
    source VARCHAR,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, symbol, in_date, source)
);

-- ODS: 原始K线数据
CREATE TABLE IF NOT EXISTS ods_daily_ohlcv_raw (
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
    adjust_flag VARCHAR,   -- 复权标志: 1=后复权 2=前复权 3=不复权
    source VARCHAR,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol, source)
);
"""


# =============================================================================
# DWD Layer - 明细数据层
# =============================================================================

# DWD层Schema：经过清洗、标准化、统一格式的数据
SCHEMA_DWD = """
-- DWD: 交易日历 (标准化后的日历)
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

-- DWD: 股票基础信息 (标准化)
CREATE TABLE IF NOT EXISTS dwd_instruments (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,              -- SSE/SZSE/BSE
    board_type VARCHAR,          -- 主板/创业板/科创板/北交所
    shenzhen_code VARCHAR,       -- 深交所代码
    shanghai_code VARCHAR,      -- 上交所代码
    industry_sw_l1 VARCHAR,     -- 申万一级行业
    industry_sw_l2 VARCHAR,     -- 申万二级行业
    industry_sw_l3 VARCHAR,     -- 申万三级行业
    industry_csrc VARCHAR,      -- 证监会行业
    status VARCHAR DEFAULT 'ACTIVE',  -- ACTIVE/DELISTED/SUSPENDED
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
    factor DOUBLE DEFAULT 1.0,      -- 复权因子
    raw_close DOUBLE,               -- 原始收盘价
    PRIMARY KEY (date, symbol)
);
"""


# =============================================================================
# DWS Layer - 汇总数据层
# =============================================================================

# DWS层Schema：聚合、统计、预计算的数据
SCHEMA_DWS = """
-- DWS: 月度汇总
CREATE TABLE IF NOT EXISTS dws_monthly_stats (
    symbol VARCHAR,
    year INTEGER,
    month INTEGER,
    start_date DATE,
    end_date DATE,
    open_first DOUBLE,         -- 月初开盘价
    close_last DOUBLE,         -- 月末收盘价
    high_max DOUBLE,           -- 月最高价
    low_min DOUBLE,            -- 月最低价
    volume_sum DOUBLE,         -- 总成交量
    amount_sum DOUBLE,         -- 总成交额
    avg_turnover_rate DOUBLE,  -- 平均换手率
    pct_change_monthly DOUBLE, -- 月涨跌幅
    PRIMARY KEY (symbol, year, month)
);

-- DWS: 年度汇总
CREATE TABLE IF NOT EXISTS dws_yearly_stats (
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

-- DWS: 指数日行情
CREATE TABLE IF NOT EXISTS dws_index_daily (
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
"""


# =============================================================================
# ADS Layer - 应用数据层
# =============================================================================

# ADS层Schema：面向应用的数据视图
SCHEMA_ADS = """
-- ADS: 股票池成员 (当前有效)
CREATE TABLE IF NOT EXISTS ads_index_members (
    index_code VARCHAR,
    index_name VARCHAR,
    symbol VARCHAR,
    name VARCHAR,
    in_date DATE,
    weight DOUBLE,
    market_cap DOUBLE,
    PRIMARY KEY (index_code, symbol)
);

-- ADS: 涨跌停股票
CREATE TABLE IF NOT EXISTS ads_limit_up_down (
    date DATE,
    symbol VARCHAR,
    limit_type VARCHAR,       -- LIMIT_UP/LIMIT_DOWN
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
"""


# =============================================================================
# Factors Layer - 因子数据层
# =============================================================================

# Factors层Schema：存储计算好的因子值
SCHEMA_FACTORS = """
-- 因子定义表
CREATE TABLE IF NOT EXISTS factors_registry (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE,
    category VARCHAR,          -- alpha, technical, fundamental
    description TEXT,
    expression TEXT,           -- 因子表达式
    parameters JSON,           -- 参数字典
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 因子值缓存表
CREATE TABLE IF NOT EXISTS factors_values (
    date DATE,
    symbol VARCHAR,
    factor_name VARCHAR,
    value DOUBLE,
    source VARCHAR,           -- 计算来源: gtja191, technical, custom
    PRIMARY KEY (date, symbol, factor_name)
);

-- 因子IC分析结果
CREATE TABLE IF NOT EXISTS factors_ic (
    id INTEGER PRIMARY KEY,
    factor_name VARCHAR,
    index_code VARCHAR,       -- 股票池代码
    start_date DATE,
    end_date DATE,
    ic_mean DOUBLE,
    ic_std DOUBLE,
    ic_ir DOUBLE,             -- IC_IR = IC_mean / IC_std
    rank_ic_mean DOUBLE,
    rank_ic_std DOUBLE,
    rank_ic_ir DOUBLE,
    return_long DOUBLE,        -- 多头收益
    return_short DOUBLE,       -- 空头收益
    return_long_short DOUBLE,  -- 多空收益
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(factor_name, index_code, start_date, end_date)
);
"""


# =============================================================================
# Metadata Layer - 元数据层
# =============================================================================

SCHEMA_METADATA = """
-- 数据更新日志
CREATE TABLE IF NOT EXISTS meta_update_log (
    id INTEGER PRIMARY KEY,
    layer VARCHAR,             -- ODS/DWD/DWS/ADS/FACTORS
    table_name VARCHAR,
    update_type VARCHAR,       -- FULL/INCREMENTAL
    start_date DATE,
    end_date DATE,
    records_total INTEGER,
    records_success INTEGER,
    records_failed INTEGER,
    status VARCHAR,           -- RUNNING/SUCCESS/FAILED
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(layer, table_name, started_at)
);

-- 数据校验日志
CREATE TABLE IF NOT EXISTS meta_validation_log (
    id INTEGER PRIMARY KEY,
    layer VARCHAR,
    table_name VARCHAR,
    check_type VARCHAR,        -- COMPLETENESS/ACCURACY/CONSISTENCY
    check_column VARCHAR,
    expected_value VARCHAR,
    actual_value VARCHAR,
    passed BOOLEAN,
    details JSON,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 数据源配置
CREATE TABLE IF NOT EXISTS meta_data_sources (
    id INTEGER PRIMARY KEY,
    source_name VARCHAR UNIQUE,
    source_type VARCHAR,       -- baostock/akshare/tushare
    config JSON,               -- 配置信息
    enabled BOOLEAN DEFAULT TRUE,
    last_fetch_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 版本信息
CREATE TABLE IF NOT EXISTS meta_version (
    version VARCHAR PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
"""


# =============================================================================
# Combined Schema - 完整Schema
# =============================================================================

def get_full_schema() -> str:
    """获取完整数据库Schema"""
    return "\n".join([
        SCHEMA_ODS,
        SCHEMA_DWD,
        SCHEMA_DWS,
        SCHEMA_ADS,
        SCHEMA_FACTORS,
        SCHEMA_METADATA,
    ])


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
    check_type: str  # completeness, accuracy, consistency, timeliness
    sql: str
    expected: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO


VALIDATION_RULES = [
    # ODS层校验
    ValidationRule(
        name="ods_ohlcv_completeness",
        layer="ODS",
        table="ods_daily_ohlcv_raw",
        column="*",
        check_type="completeness",
        sql="SELECT COUNT(*) FROM ods_daily_ohlcv_raw WHERE date = ? AND source = ?",
        expected="count > 0",
    ),
    ValidationRule(
        name="ods_ohlcv_price_accuracy",
        layer="ODS",
        table="ods_daily_ohlcv_raw",
        column="close",
        check_type="accuracy",
        sql="SELECT COUNT(*) FROM ods_daily_ohlcv_raw WHERE close <= 0 OR close IS NULL",
        expected="count = 0",
    ),
    ValidationRule(
        name="ods_ohlcv_high_low_consistency",
        layer="ODS",
        table="ods_daily_ohlcv_raw",
        column="high/low",
        check_type="consistency",
        sql="SELECT COUNT(*) FROM ods_daily_ohlcv_raw WHERE high < low OR high < close OR low > close",
        expected="count = 0",
    ),
    
    # DWD层校验
    ValidationRule(
        name="dwd_calendar_trading_days",
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
    
    # 时效性校验
    ValidationRule(
        name="timeliness_recent_data",
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
class UpdateResult:
    """更新结果"""
    layer: str
    table: str
    update_type: str
    success: bool
    records: int
    start_time: datetime
    end_time: datetime
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "table": self.table,
            "update_type": self.update_type,
            "success": self.success,
            "records": self.records,
            "duration_sec": round(self.duration, 2),
            "error": self.error
        }


@dataclass
class ValidationResult:
    """校验结果"""
    rule: ValidationRule
    passed: bool
    expected: str
    actual: str
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "rule": self.rule.name,
            "layer": self.rule.layer,
            "table": self.rule.table,
            "check_type": self.rule.check_type,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "details": self.details,
            "severity": self.rule.severity
        }


class QuantDB:
    """量化数据库管理器 (分层架构)
    
    提供分层数据管理:
    - ODS: 原始数据层
    - DWD: 明细数据层  
    - DWS: 汇总数据层
    - ADS: 应用数据层
    - Factors: 因子数据层
    
    Example:
        db = QuantDB("data/quant.db")
        db.init_schema()
        
        # 拉取原始数据
        db.fetch_ohlcv_to_ods(symbols=["000001.SZ"], start="2024-01-01")
        
        # 清洗转换到DWD
        db.transform_ods_to_dwd()
        
        # 计算因子
        db.calculate_factor("RSI", "000001.SZ", "2024-01-01", "2024-12-31")
        
        # 校验数据
        results = db.validate_all()
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        config: Optional[dict] = None,
    ):
        """初始化量化数据库"""
        self.db_path = db_path
        self.read_only = read_only
        self.config = config or {}
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # 自动初始化
        if db_path != ":memory:" and not os.path.exists(db_path):
            self.init_schema()
        elif db_path != ":memory:":
            self.connect()
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = duckdb.connect(
                self.db_path,
                read_only=self.read_only,
                config=self.config,
            )
        return self._conn
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "QuantDB":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def init_schema(self) -> None:
        """初始化数据库Schema"""
        conn = self.connect()
        conn.execute(get_full_schema())
        conn.commit()
        print(f"✅ 数据库Schema初始化完成 (分层架构): {self.db_path}")
    
    def execute(self, sql: str, params: Optional[dict] = None) -> duckdb.DuckDBPyConnection:
        """执行SQL"""
        conn = self.connect()
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)
        conn.commit()
        return conn
    
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """查询SQL并返回DataFrame"""
        conn = self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()
    
    # -------------------------------------------------------------------------
    # ODS Layer Operations - 原始数据层操作
    # -------------------------------------------------------------------------
    
    def import_ods_calendars(self, df: pd.DataFrame, source: str) -> int:
        """导入原始日历数据到ODS层"""
        df = df.copy()
        df["source"] = source
        conn = self.connect()
        conn.executemany(
            "INSERT OR REPLACE INTO ods_calendars_raw VALUES (?, ?, ?, ?, ?)",
            df[["date", "exchange", "is_trading_day", "source", "fetched_at"]].values.tolist()
        )
        conn.commit()
        return len(df)
    
    def import_ods_instruments(self, df: pd.DataFrame, source: str) -> int:
        """导入原始股票信息到ODS层"""
        df = df.copy()
        df["source"] = source
        conn = self.connect()
        conn.executemany(
            "INSERT OR REPLACE INTO ods_instruments_raw VALUES (?, ?, ?, ?, ?, ?, ?)",
            df[["symbol", "name", "list_date", "delist_date", "market", "source", "fetched_at"]].values.tolist()
        )
        conn.commit()
        return len(df)
    
    def import_ods_ohlcv(self, df: pd.DataFrame, source: str) -> int:
        """导入原始K线数据到ODS层"""
        df = df.copy()
        df["source"] = source
        df["fetched_at"] = datetime.now()
        
        conn = self.connect()
        conn.executemany(
            """INSERT OR REPLACE INTO ods_daily_ohlcv_raw 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            df[["date", "symbol", "open", "high", "low", "close", "volume", "amount",
                "turnover_rate", "pct_change", "adjust_flag", "source", "fetched_at"]].values.tolist()
        )
        conn.commit()
        return len(df)
    
    def import_ods_index_components(self, df: pd.DataFrame, source: str) -> int:
        """导入原始指数成分到ODS层"""
        df = df.copy()
        df["source"] = source
        conn = self.connect()
        conn.executemany(
            "INSERT OR REPLACE INTO ods_index_components_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            df[["index_code", "index_name", "symbol", "in_date", "out_date", "weight", "source", "fetched_at"]].values.tolist()
        )
        conn.commit()
        return len(df)
    
    # -------------------------------------------------------------------------
    # DWD Layer Operations - 明细数据层操作
    # -------------------------------------------------------------------------
    
    def transform_calendars_ods_to_dwd(self) -> int:
        """将ODS日历转换到DWD"""
        sql = """
            INSERT INTO dwd_calendars (date, is_trading_day, year, quarter, month, 
                                        week_of_year, day_of_week, is_month_end, 
                                        is_quarter_end, is_year_end, is_week_end)
            SELECT DISTINCT
                date,
                is_trading_day,
                EXTRACT(YEAR FROM date)::INTEGER as year,
                EXTRACT(QUARTER FROM date)::INTEGER as quarter,
                EXTRACT(MONTH FROM date)::INTEGER as month,
                EXTRACT(WEEK FROM date)::INTEGER as week_of_year,
                EXTRACT(DAYOFWEEK FROM date)::INTEGER as day_of_week,
                LAST_DAY(date) = date as is_month_end,
                DATE_TRUNC('quarter', date) + INTERVAL '3 months' - INTERVAL '1 day' = date as is_quarter_end,
                DATE_TRUNC('year', date) + INTERVAL '1 year' - INTERVAL '1 day' = date as is_year_end,
                EXTRACT(DAYOFWEEK FROM date) = 6 as is_week_end,
                'ALL'
            FROM ods_calendars_raw
            WHERE source = (SELECT source FROM ods_calendars_raw ORDER BY fetched_at DESC LIMIT 1)
            ON CONFLICT(date) DO UPDATE SET
                is_trading_day = excluded.is_trading_day,
                year = excluded.year,
                quarter = excluded.quarter,
                month = excluded.month,
                week_of_year = excluded.week_of_year,
                day_of_week = excluded.day_of_week,
                is_month_end = excluded.is_month_end,
                is_quarter_end = excluded.is_quarter_end,
                is_year_end = excluded.is_year_end,
                is_week_end = excluded.is_week_end,
                updated_at = CURRENT_TIMESTAMP
        """
        conn = self.execute(sql)
        return conn.rowcount
    
    def transform_ohlcv_ods_to_dwd(self, adjust: str = "qfq") -> int:
        """将ODS K线转换到DWD (前复权)
        
        Args:
            adjust: 复权类型 qfq=前复权 hfq=后复权 qfq_q=false前复权
        """
        # 确定数据源
        source = "baostock"  # 默认baostock
        adjust_flag = "2" if adjust == "qfq" else "1"  # baostock复权标志
        
        sql = f"""
            INSERT INTO dwd_daily_ohlcv (date, symbol, open, high, low, close, 
                                          volume, amount, turnover_rate, pct_change, factor, raw_close)
            SELECT 
                date,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                amount,
                turnover_rate,
                pct_change,
                1.0 as factor,
                close as raw_close
            FROM ods_daily_ohlcv_raw
            WHERE source = '{source}'
              AND adjust_flag = '{adjust_flag}'
            ON CONFLICT(date, symbol) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                amount = excluded.amount,
                turnover_rate = excluded.turnover_rate,
                pct_change = excluded.pct_change,
                factor = excluded.factor,
                raw_close = excluded.raw_close
        """
        conn = self.execute(sql)
        return conn.rowcount
    
    def transform_instruments_ods_to_dwd(self) -> int:
        """将ODS股票信息转换到DWD"""
        sql = """
            INSERT INTO dwd_instruments (symbol, name, list_date, delist_date, market, 
                                          board_type, industry_sw_l1)
            SELECT DISTINCT ON (symbol)
                symbol,
                MAX(name) as name,
                MIN(list_date) as list_date,
                MAX(delist_date) as delist_date,
                market,
                CASE 
                    WHEN symbol LIKE '688%' THEN '科创板'
                    WHEN symbol LIKE '002%' OR symbol LIKE '003%' THEN '创业板'
                    WHEN symbol LIKE '430%' OR symbol LIKE '830%' THEN '北交所'
                    ELSE '主板'
                END as board_type,
                NULL as industry_sw_l1
            FROM ods_instruments_raw
            GROUP BY symbol, market
            ON CONFLICT(symbol) DO UPDATE SET
                name = excluded.name,
                delist_date = excluded.delist_date,
                updated_at = CURRENT_TIMESTAMP
        """
        conn = self.execute(sql)
        return conn.rowcount
    
    def transform_index_components_ods_to_dwd(self) -> int:
        """将ODS指数成分转换到DWD"""
        # 先更新当前标记
        self.execute("UPDATE dwd_index_components SET is_current = FALSE WHERE is_current = TRUE")
        
        sql = """
            INSERT INTO dwd_index_components (index_code, index_name, symbol, in_date, out_date, weight, is_current)
            SELECT 
                index_code,
                MAX(index_name) as index_name,
                symbol,
                MIN(in_date) as in_date,
                MAX(out_date) as out_date,
                MAX(weight) as weight,
                TRUE as is_current
            FROM ods_index_components_raw
            GROUP BY index_code, symbol
            ON CONFLICT(index_code, symbol, in_date) DO UPDATE SET
                out_date = excluded.out_date,
                weight = excluded.weight
        """
        conn = self.execute(sql)
        return conn.rowcount
    
    def transform_all_ods_to_dwd(self) -> dict:
        """执行所有ODS到DWD的转换"""
        results = {}
        results["calendars"] = self.transform_calendars_ods_to_dwd()
        results["instruments"] = self.transform_instruments_ods_to_dwd()
        results["index_components"] = self.transform_index_components_ods_to_dwd()
        results["ohlcv"] = self.transform_ohlcv_ods_to_dwd()
        return results
    
    # -------------------------------------------------------------------------
    # DWS Layer Operations - 汇总数据层操作
    # -------------------------------------------------------------------------
    
    def aggregate_monthly_stats(self, start: Optional[str] = None, end: Optional[str] = None) -> int:
        """聚合月度统计"""
        where = ""
        if start:
            where += f" WHERE date >= '{start}'"
        if end:
            where += f" WHERE date <= '{end}'" if where else f" AND date <= '{end}'"
        
        sql = f"""
            INSERT INTO dws_monthly_stats 
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
    # ADS Layer Operations - 应用数据层操作
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # Data Validation - 数据校验
    # -------------------------------------------------------------------------
    
    def validate(self, rule: ValidationRule) -> ValidationResult:
        """执行单条校验规则"""
        try:
            df = self.query(rule.sql)
            actual = str(df.iloc[0][0]) if len(df) > 0 else "0"
            
            # 简单判断是否通过
            if "count" in rule.expected.lower():
                expected_count = int(rule.expected.split("=")[1].strip())
                actual_count = int(actual)
                passed = (actual_count == expected_count)
            elif "max_date" in rule.expected.lower():
                passed = actual >= str(date.today() - timedelta(days=2))
            else:
                passed = True
            
            return ValidationResult(
                rule=rule,
                passed=passed,
                expected=rule.expected,
                actual=actual,
                details={"rows_checked": len(df)}
            )
        except Exception as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected=rule.expected,
                actual=str(e),
                details={"error": str(e)}
            )
    
    def validate_all(self, layer: Optional[str] = None) -> list[ValidationResult]:
        """执行所有校验规则"""
        rules = VALIDATION_RULES
        if layer:
            rules = [r for r in rules if r.layer == layer]
        
        results = []
        for rule in rules:
            result = self.validate(rule)
            results.append(result)
            
            # 记录到日志
            self._log_validation(result)
        
        return results
    
    def _log_validation(self, result: ValidationResult) -> None:
        """记录校验结果"""
        self.execute(f"""
            INSERT INTO meta_validation_log 
            (layer, table_name, check_type, check_column, expected_value, actual_value, passed, details)
            VALUES ('{result.rule.layer}', '{result.rule.table}', '{result.rule.check_type}',
                    '{result.rule.column}', '{result.expected}', '{result.actual}', 
                    {result.passed}, '{json.dumps(result.details or {})}')
        """)
    
    def get_validation_report(self, days: int = 7) -> pd.DataFrame:
        """获取校验报告"""
        sql = f"""
            SELECT * FROM meta_validation_log
            WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
            ORDER BY checked_at DESC
        """
        return self.query(sql)
    
    # -------------------------------------------------------------------------
    # Update Log - 更新日志
    # -------------------------------------------------------------------------
    
    def log_update(
        self,
        layer: str,
        table: str,
        update_type: str,
        records: int,
        status: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """记录更新日志"""
