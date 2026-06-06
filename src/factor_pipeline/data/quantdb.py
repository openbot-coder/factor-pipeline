"""Quantitative Database - 量化数据库核心模块

这个模块负责管理A股量化研究的所有数据:
- 交易日历
- 股票基础信息
- 指数成分股
- 日线OHLCV数据
- 因子缓存

Philosophy: Keep it simple, make it work, make it fast.
"""

from __future__ import annotations

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
    # 宽基指数
    SSE50 = ("000016", "SH", "上证50")
    CSI300 = ("000300", "SH", "沪深300")
    CSI500 = ("000905", "SH", "中证500")
    CSI1000 = ("000852", "SH", "中证1000")
    CSI800 = ("000906", "SH", "中证800")
    CSI100 = ("000903", "SH", "中证100")
    
    # 申万一级行业
    SW_BANK = ("801010", "SH", "银行")
    SW_SECURITY = ("801780", "SH", "证券")
    SW_INSURANCE = ("801790", "SH", "保险")
    SW_REAL_ESTATE = ("801180", "SH", "房地产")
    SW_BUILD_MATERIALS = ("801710", "SH", "建筑材料")
    SW_CONSTRUCTION = ("801720", "SH", "建筑装饰")
    SW_MACHINERY = ("801730", "SH", "通用设备")
    SW_SPECIAL_EQUIPMENT = ("801740", "SH", "专用设备")
    SW_AUTOMOBILE = ("801880", "SH", "汽车")
    SW_ELECTRONICS = ("801080", "SH", "电子")
    SW_COMPUTER = ("801750", "SH", "计算机")
    SW_MEDIA = ("801760", "SH", "传媒")
    SW_COMMUNICATION = ("801770", "SH", "通信")
    SW_POWER = ("801730", "SH", "电力设备")
    SW_DEFENSE = ("801750", "SH", "国防军工")
    SW_ELECTRIC = ("801730", "SH", "电气设备")
    SW_CHEMICAL = ("801030", "SH", "化工")
    SW_STEEL = ("801040", "SH", "钢铁")
    SW_NONFERROUS = ("801050", "SH", "有色金属")
    SW_COAL = ("801020", "SH", "煤炭")
    SW_OIL = ("801150", "SH", "石油石化")
    SW_RETAIL = ("801200", "SH", "商业贸易")
    SW_CATERING = ("801210", "SH", "餐饮旅游")
    SW_HOUSEHOLD = ("801130", "SH", "家用电器")
    SW_TEXTILE = ("801140", "SH", "纺织服装")
    SW_LIGHT_MANU = ("801150", "SH", "轻工制造")
    SW_AGRICULTURE = ("801010", "SH", "农林牧渔")
    SW_FOOD_BEVERAGE = ("801120", "SH", "食品饮料")
    SW_MEDICAL = ("801150", "SH", "医药生物")
    SW_PUBLIC = ("801170", "SH", "公用事业")
    
    def __init__(self, code: str, exchange: str, name: str):
        self.code = code
        self.exchange = exchange
        self.name = name
    
    @property
    def full_code(self) -> str:
        return f"{self.code}.{self.exchange}"


# =============================================================================
# Database Schema - 数据库Schema
# =============================================================================

SCHEMA_QUANTDB = """
-- 交易日历表
CREATE TABLE IF NOT EXISTS calendars (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN DEFAULT TRUE,
    exchange VARCHAR,  -- SSE/SZSE/BSE/ALL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 股票基础信息表
CREATE TABLE IF NOT EXISTS instruments (
    symbol VARCHAR PRIMARY KEY,           -- 股票代码 e.g. 000001.SZ
    name VARCHAR,                         -- 股票名称
    list_date DATE,                       -- 上市日期
    delist_date DATE,                     -- 退市日期 (NULL表示仍在交易)
    market VARCHAR,                        -- 所属市场 SSE/SZSE/BSE
    industry_sw VARCHAR,                  -- 申万一级行业
    industry_sw_2 VARCHAR,               -- 申万二级行业
    industry_sw_3 VARCHAR,               -- 申万三级行业
    sector VARCHAR,                       -- 证监会行业分类
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 指数基础信息表
CREATE TABLE IF NOT EXISTS indices (
    index_code VARCHAR PRIMARY KEY,       -- 指数代码 e.g. 000300.SH
    index_name VARCHAR,                   -- 指数名称
    index_full_name VARCHAR,              -- 指数全称
    base_date DATE,                       -- 基期
    base_point DOUBLE,                    -- 基点
    exchange VARCHAR,                     -- 交易所
    category VARCHAR,                      -- 指数类别
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 指数成分股表
CREATE TABLE IF NOT EXISTS index_components (
    id INTEGER PRIMARY KEY,
    index_code VARCHAR,                    -- 指数代码
    symbol VARCHAR,                        -- 股票代码
    in_date DATE,                         -- 纳入日期
    out_date DATE,                        -- 剔除日期 (NULL表示仍在池中)
    weight DOUBLE,                        -- 权重
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(index_code, symbol, in_date)
);

-- 日线OHLCV数据表
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,                        -- 成交量 (股数)
    amount DOUBLE,                        -- 成交额
    turnover_rate DOUBLE,                 -- 换手率
    pct_change DOUBLE,                    -- 涨跌幅
    is_suspended BOOLEAN DEFAULT FALSE,   -- 是否停牌
    factor DOUBLE DEFAULT 1.0,           -- 复权因子
    PRIMARY KEY (date, symbol)
);

-- 因子缓存表
CREATE TABLE IF NOT EXISTS factor_cache (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE,
    description TEXT,
    expression TEXT,
    result_bytes BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 数据更新日志表
CREATE TABLE IF NOT EXISTS data_update_log (
    id INTEGER PRIMARY KEY,
    table_name VARCHAR,
    update_type VARCHAR,                   -- FULL/INCREMENTAL
    start_date DATE,
    end_date DATE,
    records_updated INTEGER,
    status VARCHAR,                       -- SUCCESS/FAILED/RUNNING
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
"""

# 索引
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_calendars_exchange ON calendars(exchange);
CREATE INDEX IF NOT EXISTS idx_instruments_market ON instruments(market);
CREATE INDEX IF NOT EXISTS idx_instruments_list_date ON instruments(list_date);
CREATE INDEX IF NOT EXISTS idx_index_components_code ON index_components(index_code);
CREATE INDEX IF NOT EXISTS idx_index_components_symbol ON index_components(symbol);
CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_date ON daily_ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_symbol ON daily_ohlcv(symbol);
CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_date_symbol ON daily_ohlcv(date, symbol);
"""


# =============================================================================
# QuantDB Class - 量化数据库类
# =============================================================================

@dataclass
class UpdateResult:
    """更新结果"""
    table: str
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
            "table": self.table,
            "success": self.success,
            "records": self.records,
            "duration_sec": round(self.duration, 2),
            "error": self.error
        }


class QuantDB:
    """量化数据库管理器
    
    统一管理A股量化研究的所有数据，提供:
    - 数据库初始化
    - 数据导入导出
    - 增量更新
    - 数据查询
    - 完整性校验
    
    Example:
        db = QuantDB("data/quant.db")
        db.init_schema()
        
        # 查询交易日
        trading_days = db.get_trading_days("2024-01-01", "2024-12-31")
        
        # 查询股票列表
        stocks = db.get_instruments(market="SSE")
        
        # 获取K线数据
        df = db.get_ohlcv(symbols=["000001.SZ"], start="2024-01-01")
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        config: Optional[dict] = None,
    ):
        """初始化量化数据库
        
        Args:
            db_path: 数据库路径，默认为内存数据库
            read_only: 是否只读模式
            config: DuckDB配置
        """
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
        # 执行建表SQL
        for sql in SCHEMA_QUANTDB.strip().split(";"):
            sql = sql.strip()
            if sql and not sql.startswith("--"):
                try:
                    conn.execute(sql)
                except Exception as e:
                    # 表已存在会报错，忽略
                    pass
        
        # 创建索引
        for sql in INDEXES.strip().split(";"):
            sql = sql.strip()
            if sql:
                try:
                    conn.execute(sql)
                except Exception:
                    pass
        
        conn.commit()
        print(f"✅ 数据库Schema初始化完成: {self.db_path}")
    
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
    # Calendar Operations - 日历操作
    # -------------------------------------------------------------------------
    
    def get_trading_days(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        exchange: str = "ALL",
    ) -> list[str]:
        """获取交易日列表
        
        Args:
            start: 开始日期
            end: 结束日期
            exchange: 交易所 (SSE/SZSE/BSE/ALL)
            
        Returns:
            交易日列表 [YYYY-MM-DD, ...]
        """
        conditions = ["is_trading_day = TRUE"]
        if exchange != "ALL":
            conditions.append(f"exchange = '{exchange}'")
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        
        where = " AND ".join(conditions)
        sql = f"SELECT date FROM calendars WHERE {where} ORDER BY date"
        
        df = self.query(sql)
        return [str(d.date()) for d in df["date"].tolist()]
    
    def get_latest_trading_day(self, before: Optional[str] = None) -> Optional[str]:
        """获取最新交易日"""
        if before:
            sql = f"SELECT date FROM calendars WHERE is_trading_day = TRUE AND date <= '{before}' ORDER BY date DESC LIMIT 1"
        else:
            sql = "SELECT date FROM calendars WHERE is_trading_day = TRUE ORDER BY date DESC LIMIT 1"
        
        result = self.query(sql)
        if len(result) > 0:
            return str(result.iloc[0]["date"])
        return None
    
    def import_calendars(self, df: pd.DataFrame) -> int:
        """导入日历数据"""
        conn = self.execute("""
            INSERT INTO calendars (date, is_trading_day, exchange)
            SELECT DISTINCT date, is_trading_day, COALESCE(exchange, 'ALL')
            FROM df
            ON CONFLICT(date) DO UPDATE SET
                is_trading_day = excluded.is_trading_day,
                exchange = excluded.exchange
        """)
        return conn.rowcount
    
    # -------------------------------------------------------------------------
    # Instrument Operations - 股票信息操作
    # -------------------------------------------------------------------------
    
    def get_instruments(
        self,
        market: Optional[str] = None,
        list_date: Optional[str] = None,
        delist_date: Optional[str] = None,
        include_delisted: bool = True,
    ) -> pd.DataFrame:
        """获取股票列表"""
        conditions = []
        if market:
            conditions.append(f"market = '{market}'")
        if list_date:
            conditions.append(f"list_date <= '{list_date}'")
        if not include_delisted:
            conditions.append("delist_date IS NULL")
        elif delist_date:
            conditions.append(f"delist_date >= '{delist_date}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM instruments WHERE {where} ORDER BY symbol"
        
        return self.query(sql)
    
    def get_symbols(
        self,
        market: Optional[str] = None,
        active_only: bool = True,
    ) -> list[str]:
        """获取股票代码列表"""
        df = self.get_instruments(market=market, include_delisted=not active_only)
        return df["symbol"].tolist()
    
    def import_instruments(self, df: pd.DataFrame) -> int:
        """导入股票信息"""
        conn = self.execute("""
            INSERT INTO instruments (symbol, name, list_date, delist_date, market, 
                                   industry_sw, industry_sw_2, industry_sw_3, sector)
            SELECT symbol, name, list_date, delist_date, market,
                   industry_sw, industry_sw_2, industry_sw_3, sector
            FROM df
            ON CONFLICT(symbol) DO UPDATE SET
                name = excluded.name,
                delist_date = excluded.delist_date,
                industry_sw = excluded.industry_sw,
                industry_sw_2 = excluded.industry_sw_2,
                industry_sw_3 = excluded.industry_sw_3,
                sector = excluded.sector,
                updated_at = CURRENT_TIMESTAMP
        """)
        return conn.rowcount
    
    # -------------------------------------------------------------------------
    # Index Operations - 指数操作
    # -------------------------------------------------------------------------
    
    def get_index_components(
        self,
        index_code: str,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取指数成分股
        
        Args:
            index_code: 指数代码，如 '000300.SH'
            date: 日期，默认返回最新成分
            
        Returns:
            成分股列表
        """
        if date is None:
            sql = f"""
                SELECT * FROM index_components 
                WHERE index_code = '{index_code}' 
                  AND out_date IS NULL
                ORDER BY weight DESC
            """
        else:
            sql = f"""
                SELECT * FROM index_components 
                WHERE index_code = '{index_code}' 
                  AND in_date <= '{date}'
                  AND (out_date IS NULL OR out_date > '{date}')
                ORDER BY weight DESC
            """
        
        return self.query(sql)
    
    def import_index_components(self, df: pd.DataFrame) -> int:
        """导入指数成分股"""
        conn = self.execute("""
            INSERT INTO index_components (index_code, symbol, in_date, out_date, weight)
            SELECT index_code, symbol, in_date, out_date, COALESCE(weight, 0)
            FROM df
            ON CONFLICT(index_code, symbol, in_date) DO UPDATE SET
                out_date = excluded.out_date,
                weight = excluded.weight,
                updated_at = CURRENT_TIMESTAMP
        """)
        return conn.rowcount
    
    # -------------------------------------------------------------------------
    # OHLCV Operations - K线数据操作
    # -------------------------------------------------------------------------
    
    def get_ohlcv(
        self,
        symbols: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fields: Optional[list[str]] = None,
        index: bool = True,
    ) -> pd.DataFrame:
        """获取K线数据
        
        Args:
            symbols: 股票代码列表
            start: 开始日期
            end: 结束日期
            fields: 返回字段
            index: 是否使用MultiIndex
            
        Returns:
            OHLCV数据
        """
        fields = fields or ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        select_fields = ", ".join(fields)
        
        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT {select_fields} FROM daily_ohlcv WHERE {where} ORDER BY date, symbol"
        
        df = self.query(sql)
        
        if index and "date" in df.columns and "symbol" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "symbol"]).sort_index()
        
        return df
    
    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[str]:
        """获取最新K线日期"""
        if symbol:
            sql = f"SELECT MAX(date) FROM daily_ohlcv WHERE symbol = '{symbol}'"
        else:
            sql = "SELECT MAX(date) FROM daily_ohlcv"
        
        result = self.query(sql)
        if result.iloc[0][0]:
            return str(result.iloc[0][0])
        return None
    
    def import_ohlcv(self, df: pd.DataFrame, batch_size: int = 10000) -> int:
        """批量导入K线数据
        
        Args:
            df: K线数据DataFrame
            batch_size: 每批次提交数量
            
        Returns:
            导入记录数
        """
        conn = self.connect()
        total = len(df)
        
        # 确保列顺序
        required_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 转换为列表便于分批
        records = df[required_cols].values.tolist()
        placeholders = ", ".join(["?" * len(required_cols)])
        
        # 批量插入
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            conn.executemany(
                f"INSERT INTO daily_ohlcv VALUES ({placeholders})",
                batch
            )
        
        conn.commit()
        return total
    
    def count_ohlcv(self, symbol: Optional[str] = None) -> int:
        """统计K线记录数"""
        if symbol:
            sql = f"SELECT COUNT(*) FROM daily_ohlcv WHERE symbol = '{symbol}'"
        else:
            sql = "SELECT COUNT(*) FROM daily_ohlcv"
        
        return int(self.query(sql).iloc[0][0])
    
    # -------------------------------------------------------------------------
    # Update Log - 更新日志
    # -------------------------------------------------------------------------
    
    def log_update(
        self,
        table: str,
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
            INSERT INTO data_update_log 
            (table_name, update_type, start_date, end_date, records_updated, status, error_message, completed_at)
            VALUES ('{table}', '{update_type}', {start_val}, {end_val}, {records}, '{status}', {error_val}, CURRENT_TIMESTAMP)
        """
        self.execute(sql)
    
    def get_update_history(self, table: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        """获取更新历史"""
        where = f"WHERE table_name = '{table}'" if table else ""
        sql = f"""
            SELECT * FROM data_update_log 
            {where}
            ORDER BY started_at DESC 
            LIMIT {limit}
        """
        return self.query(sql)
    
    # -------------------------------------------------------------------------
    # Utility Methods - 工具方法
    # -------------------------------------------------------------------------
    
    def info(self) -> dict[str, Any]:
        """获取数据库信息"""
        tables = ["calendars", "instruments", "indices", "index_components", "daily_ohlcv", "factor_cache"]
        info = {"db_path": self.db_path, "tables": {}}
        
        for table in tables:
            try:
                count = self.query(f"SELECT COUNT(*) FROM {table}").iloc[0][0]
                info["tables"][table] = {"rows": int(count)}
            except Exception:
                info["tables"][table] = {"rows": -1}
        
        return info
    
    def vacuum(self) -> None:
        """优化数据库"""
        self.execute("VACUUM")
    
    def checkpoint(self) -> None:
        """检查点"""
        self.execute("CHECKPOINT")
