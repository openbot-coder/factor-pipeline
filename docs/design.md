# 产品设计文档

> 版本：0.1.0
> 创建日期：2026-06-09
> 最后更新：2026-06-09

## 1. 项目概述

**项目名称：** factor-pipeline
**开发语言：** python
**框架：** Click (CLI), DuckDB (数据库), Pandas/NumPy (计算)

**项目定位：** 量化因子研发流水线——从数据采集、因子计算、IC 分析到分层回测的完整工具链。

**目标用户：** 量化研究员、个人投资者、金融科技学生

**核心价值主张：**
- Qlib 风格表达式语法，80+ 算子支持
- DuckDB 列存引擎，亚秒级查询
- 3 层数据架构（ODS → DWD → APP），支持增量更新
- GTJA 191 + 技术指标因子库
- HTML 报告一键生成

## 2. 架构设计

### 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    CLI 入口 (fp)                      │
│  ┌──────────┬──────────┬──────────┬──────────┐      │
│  │ fp data   │ fp factor │ fp backtest│fp report│      │
│  └────┬─────┴────┬─────┴────┬─────┴────┬───┘      │
│       │          │          │          │            │
│  ┌────▼─────┐ ┌──▼───────┐ ┌▼────────┐ ┌▼───────┐ │
│  │ 数据层    │ │ 因子层    │ │ 分析层   │ │ 报告层 │ │
│  │ QuantDB  │ │ expr_eng │ │ IC/Layer │ │ HTML   │ │
│  │ Storage  │ │ ops/160+ │ │ Backtest │ │ Charts │ │
│  │ ETL      │ │ registry │ │          │ │        │ │
│  └────┬─────┘ └──────────┘ └──────────┘ └────────┘ │
│       │                                              │
│  ┌────▼─────┐                                       │
│  │ DuckDB   │                                       │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

### 模块划分

| 模块 | 路径 | 职责 |
|------|------|------|
| **CLI** | `src/factor_pipeline/cli/` | Click 命令组：data/factor/backtest/report |
| **数据层** | `src/factor_pipeline/data/` | QuantDB (新) + DuckDBStorage (旧) + ETL |
| **因子层** | `src/factor_pipeline/factors/` | 表达式引擎 + 160+ 算子 + 因子注册表 |
| **分析层** | `src/factor_pipeline/analysis/` | IC 分析 + 分层回测 + HTML 报告 |
| **配置** | `src/factor_pipeline/config/` | YAML 配置 dataclass |

### 数据采集脚本

| 脚本 | 数据源 | 功能 |
|------|--------|------|
| `scripts/init_data.py` | baostock | 3 层架构全量初始化 |
| `scripts/update_data.py` | baostock | 日增量更新 |
| `scripts/download_baostock.py` | baostock | A 股全市场日线（多进程） |
| `scripts/fetch_csi500.py` | 腾讯 API | 中证 500 成分股 |
| `scripts/fill_hs300_gaps.py` | baostock | 沪深 300 缺失补齐 |
| `scripts/csv2duckdb.py` | CSV 文件 | 通用 CSV 导入器 |

### 3 层数据架构

```
ODS (原始层)          →  按数据源分表，保持原始格式
  ods_daily_ohlcv_baostock    日线K线
  ods_minute_ohlcv_baostock   分钟K线（新增）
  ods_instruments_baostock    股票列表
  ods_calendars_baostock      日历
  ods_index_components_baostock  指数成分

DWD (明细层)          →  清洗、标准化、可查询
  dwd_calendars              交易日历（按交易所分列）
  dwd_instruments_info       证券上市/退市时间
  dwd_instruments_pool_registration  股票池进出登记
  dwd_daily_basic_factors    基础日因子（超宽表）
  dwd_minute_ohlcv           分钟K线（新增）
  dwd_daily_financials       财务数据（建议新增）
  dwd_daily_macro            宏观因子（建议新增）

APP (应用层)          →  交易实际使用的因子信息 + 实时交易数据
  app_factors_registry       因子注册表
  app_factors_values         因子值
  app_factors_ic             因子IC分析
  app_monthly_stats          月度统计
  app_limit_up_down          涨跌停监控
```

### 数据存储策略

| 数据类型 | 存储范围 | 74只预估 | 全市场预估 |
|---------|---------|---------|-----------|
| 日K线 | 2005-至今（全量） | ~8 MB | ~500 MB |
| 分钟K线 | 近1年 | ~70 MB | ~4.5 GB |
| 财务数据 | 全量 | ~5 MB | ~300 MB |
| 宏观数据 | 全量 | ~1 MB | ~1 MB |
| 指数数据 | 全量 | ~3 MB | ~3 MB |
| Tick数据 | 不落盘，实时拉取 | — | — |

### 日历更新规则

- 首次初始化：2005-01-01 → 当前（全量）
- 增量更新：当 `date > 今天的交易日 < 10 条` 时
  - 从 MAX(date) → MAX(date) + 1 年 补充交易日数据
  - 实现于 `update_data.py` 的 `check_calendar_needs_update()`

## 3. 数据模型

### 交易日历：dwd_calendars

**职责：** 各交易所的交易日标记 + 日期维度信息。判断某日是否为某交易所的交易日，用于行情对齐和因子计算的时间窗口判断。

**DDL：**

```sql
CREATE TABLE dwd_calendars (
    date DATE PRIMARY KEY,          -- 日历日期
    sse BOOLEAN DEFAULT FALSE,      -- 上交所 1=交易日 0=非交易日
    szse BOOLEAN DEFAULT FALSE,     -- 深交所
    hkse BOOLEAN DEFAULT FALSE,     -- 港交所
    usse BOOLEAN DEFAULT FALSE,     -- 纽交所
    year INTEGER,                   -- 年份 (2026)
    quarter INTEGER,                -- 季度 (1-4)
    month INTEGER,                  -- 月份 (1-12)
    week_of_year INTEGER,           -- 年内第几周 (1-53)
    day_of_week INTEGER,            -- 周几 (0=周一, 6=周日)
    is_month_end BOOLEAN,           -- 是否为月末
    is_quarter_end BOOLEAN,         -- 是否为季末
    is_year_end BOOLEAN,            -- 是否为年末
    is_week_end BOOLEAN,            -- 是否为周末 (周五)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**数据覆盖：** 2005-01-01 ~ 当前 + 1 年（约 22 年 X 365 天 ≈ 8000 行）

| 字段 | 说明 |
|------|------|
| date | 日历日期，主键，保障每日期只有一行 |
| sse/szse/hkse/usse | 交易所级别标记。同一日期 A 股可能同时开市、港股可能休市 |
| day_of_week | 0-6 对应周一到周日，用于过滤周末 |
| is_month_end | 月末标记，用于计算月度因子、月频换仓 |

**ODS 源数据格式：**

```sql
CREATE TABLE ods_calendars_{source} (
    date DATE, exchange VARCHAR, is_trading_day BOOLEAN,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, exchange)
);
```

ODS 层用 `exchange='ALL'` + `is_trading_day` 标识全局交易日，ETL 层将其展开为按交易所分列。

**ETL 转换逻辑：**

```
ODS: (date, exchange='ALL', is_trading_day=TRUE)
  → 按 date 分组，exchange='ALL' 展开到所有交易所
  → 补充 year/quarter/month/day_of_week 等日期维度
  → INSERT ... ON CONFLICT(date) DO UPDATE → dwd_calendars
```

**数据生成规则：**

| 规则 | 说明 |
|------|------|
| 周末 | 周六、周日自动 non-trading |
| 节假日 | 内置中国法定假日简化规则（春节/清明/劳动/端午/中秋/国庆） |
| 开市时间 | 仅 A 股交易日规则，港股/美股规则待接入交易所官方日历 |

**增量更新策略（在 `update_data.py` 中实现）：**

```
1. 查询: SELECT COUNT(*) FROM dwd_calendars
         WHERE date > CURRENT_DATE AND (sse OR szse)
2. 判断: 若 < 10 条 → 触发更新
3. 范围: MAX(date) → MAX(date) + 1 年
4. 写入: generate_calendar_range() → ODS → ETL → DWD
```

**查询示例：**

```sql
-- 查询当前日期的前一个交易日
SELECT MAX(date) FROM dwd_calendars
WHERE sse AND date < CURRENT_DATE;

-- 查询某月所有交易日
SELECT date FROM dwd_calendars
WHERE sse AND year=2026 AND month=6;

-- 判断某天是否为月末最后一个交易日
SELECT is_month_end FROM dwd_calendars
WHERE date = '2026-06-30' AND sse;

-- 查询港股和 A 股同时开市的日期
SELECT date FROM dwd_calendars
WHERE sse AND hkse AND date >= '2026-01-01';
```

### 基础日因子 (超宽表)：dwd_daily_basic_factors

| 字段 | 类型 | 说明 |
|------|------|------|
| date/symbol | DATE/VARCHAR | 交易日期 + 股票代码 (联合主键) |
| open/high/low/close | DOUBLE | 开盘/最高/最低/收盘价 |
| pre_close | DOUBLE | 昨收价 |
| volume/amount | DOUBLE | 成交量(股) / 成交额(元) |
| vwap | DOUBLE | 成交均价 (amount/volume) |
| turnover_rate | DOUBLE | 换手率 |
| pct_change / amplitude | DOUBLE | 涨跌幅 / 振幅 |
| price_limit_up/down | DOUBLE | 涨停价 / 跌停价 |
| source/updated_at | VARCHAR/TIMESTAMP | 数据来源 / 更新时间 |

### 证券上市退市信息：dwd_instruments_info

**职责：** 存储所有证券的基本信息，用于过滤活跃股票、计算上市天数。

**DDL：**

```sql
CREATE TABLE dwd_instruments_info (
    symbol VARCHAR PRIMARY KEY,     -- 证券代码 (如 600000.SSE)
    name VARCHAR,                    -- 证券名称 (如 浦发银行)
    list_date DATE,                  -- 上市日期
    delist_date DATE,                -- 退市日期 (NULL = 未退市)
    market VARCHAR,                  -- 市场 (SSE / SZSE / BSE)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**ODS 源：**

```sql
CREATE TABLE ods_instruments_{source} (
    symbol VARCHAR, name VARCHAR, list_date DATE, delist_date DATE, market VARCHAR,
    source VARCHAR DEFAULT '{source}', fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol)
);
```

**数据源：**

| 数据源 | API | 说明 |
|--------|-----|------|
| baostock | `query_all_stock()` | 已停服，返回空 |
| akshare | `stock_zh_a_spot_em()` | 完整 A 股列表 |
| 回退列表 | init_data.py 内置 | 74 只沪深 300 头部股票 |

**更新规则：**
- 首次：全量拉取所有上市股票
- 增量：每日检查是否有新上市/退市股票

**查询示例：**

```sql
-- 查活跃股票
SELECT symbol, name FROM dwd_instruments_info
WHERE delist_date IS NULL OR delist_date > CURRENT_DATE;

-- 查某只股票存续期
SELECT * FROM dwd_instruments_info WHERE symbol = '600000.SSE';

-- 查某市场全部活跃股票
SELECT symbol, name, list_date FROM dwd_instruments_info
WHERE market = 'SSE' AND (delist_date IS NULL OR delist_date > CURRENT_DATE);
```

---

### 股票池进出登记：dwd_instruments_pool_registration

**职责：** 记录每只股票进入/退出某个股票池的历史（指数成分/自选股/行业板块等）。

**DDL：**

```sql
CREATE TABLE dwd_instruments_pool_registration (
    pool_name VARCHAR,               -- 股票池名称 (联合主键)
    symbol VARCHAR,                  -- 证券代码 (联合主键)
    in_date DATE,                    -- 纳入日期 (联合主键)
    out_date DATE,                   -- 剔除日期 (NULL = 仍在池中)
    weight DOUBLE,                   -- 权重
    source VARCHAR,                  -- 数据来源
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (pool_name, symbol, in_date)
);
```

**pool_name 类型枚举：**

| pool_name | 含义 | 说明 |
|-----------|------|------|
| `csi300` | 沪深 300 | 沪深两市市值最大的 300 只 |
| `csi500` | 中证 500 | 排除沪深 300 后最大的 500 只 |
| `csi1000` | 中证 1000 | 更小盘股 |
| `hang_seng` | 恒生指数 | 港股大盘 |
| `sp500` | 标普 500 | 美股大盘 |
| `all` | 全市场 | 所有股票 |
| `sector_xxx` | 行业板块 | 按申万/CSRC 行业 |
| `custom_xxx` | 自定义池 | 用户自定义的股票池 |

**数据源：**

| 数据源 | API | 说明 |
|--------|-----|------|
| baostock | `query_index_stock_weight()` | 不存在（已停服） |
| akshare | `index_stock_cons_csindex()` | 沪深 300/中证 500/中证 1000 |

**更新规则：**
- 指数成分：每月/季度检查更新（取决于指数调整周期）
- 自选股：用户手动更新

**查询示例：**

```sql
-- 查沪深 300 当前成分
SELECT symbol, in_date, weight
FROM dwd_instruments_pool_registration
WHERE pool_name = 'csi300' AND out_date IS NULL;

-- 查某池某日的成分
SELECT symbol, weight
FROM dwd_instruments_pool_registration
WHERE pool_name = 'csi300'
AND in_date <= '2026-06-01'
AND (out_date IS NULL OR out_date > '2026-06-01');

-- 查某只股票进/出池历史
SELECT pool_name, in_date, out_date, weight
FROM dwd_instruments_pool_registration
WHERE symbol = '600000.SSE'
ORDER BY in_date DESC;

-- 两表 JOIN：查池内证券的详细信息
SELECT i.symbol, i.name, i.market, r.weight
FROM dwd_instruments_pool_registration r
JOIN dwd_instruments_info i ON r.symbol = i.symbol
WHERE r.pool_name = 'csi300' AND r.out_date IS NULL;
```

### 因子结果：FactorResult

```python
@dataclass
class FactorResult:
    values: pd.DataFrame  # MultiIndex (date, stock)
    name: str = ""
    max_window: int = 1
    description: str = ""
```

### IC 分析结果：ICResult

```python
@dataclass
class ICResult:
    ic_series: pd.Series    # 每日 IC 时间序列
    ic_mean: float = 0.0    # IC 均值
    ic_std: float = 0.0     # IC 标准差
    ir: float = 0.0         # 信息比率 (IC_mean / IC_std)
    t_stat: float = 0.0     # t 检验统计量
    n_days: int = 0         # 样本天数
```

## 4. API 设计

### CLI 命令

```bash
# 数据管理
fp data init --db data/ohlcv.duckdb
fp data import data.csv --table daily_ohlcv
fp data info
fp data query "SELECT * FROM daily_ohlcv LIMIT 10"

# 因子计算
fp factor list
fp factor doc alpha001
fp factor run "Mean($close, 20)"
fp factor batch factors.txt --output results/

# 回测分析
fp backtest run --factors alpha001,alpha014
fp backtest ic --factor alpha001
fp backtest layered --factor alpha001

# 报告生成
fp report generate --factors alpha001 --output report.html
```

### Python API

```python
from factor_pipeline import DuckDBStorage, FactorRegistry
from factor_pipeline.analysis.ic import ICAnalysis

# 加载数据
db = DuckDBStorage("data/ohlcv.duckdb")
data = db.get_ohlcv(start_date="2024-01-01")

# 计算因子
factor_fn = FactorRegistry.get("alpha001")
result = factor_fn(data)

# IC 分析
ic = ICAnalysis(factor_values, forward_returns)
ic_result = ic.run("spearman")
```

## 5. 安全设计

- 无认证需求（本地 CLI 工具）
- 数据库文件使用 DuckDB 默认权限
- API Key 通过环境变量注入（baostock 无需 Key）

## 6. 配置与部署

### 依赖管理

```
pandas>=2.0, numpy>=1.24, scipy>=1.10
duckdb>=1.0, click>=8.0, pyarrow>=14.0
matplotlib>=3.7, seaborn>=0.12, plotly>=5.18
ta>=0.10, jinja2>=3.1, pyyaml>=6.0
```

### 安装方式

```bash
pip install -e .           # 开发模式
pip install -e ".[dev]"    # 开发+测试
```

### 环境配置

```python
# 默认数据库路径
DEFAULT_DB = "data/ohlcv.duckdb"
# 可通过环境变量覆盖
os.environ["FACTOR_PIPELINE_DB"] = "custom/path.duckdb"
```

## 7. 非功能性需求

### 性能

- 单只股票 K 线查询：< 2 秒
- DuckDB 聚合查询（74 只）：< 0.5 秒
- Arrow native insert（130k 行）：< 1 秒
- init_data.py 全流程（74 只 × 1 个月）：~285 秒（瓶颈在 baostock 网络）

### 可靠性

- baostock 连接失败时自动重试
- 股票列表 API 返回空时使用内置回退列表
- ETL 支持增量更新（ON CONFLICT）

### 可扩展性

- 新增因子：`@register_factor` 装饰器
- 新增数据源：实现 `_fetch_xxx` 函数
- 新增算子：继承 `Op` 基类

## 8. 已知问题（重构目标）

| 问题 | 严重度 | 影响 |
|------|--------|------|
| CLI 导入路径残留裸路径 | P0 | `factor_list` 和 `factor_doc` 内部 `importlib.import_module("factors.xxx")` 安装后失败 |
| ETL iterrows 性能瓶颈 | P1 | 88k 行数据迁移卡住（已改 Arrow native，待验证） |
| baostock query_index_stock_weight 不存在 | P1 | 指数成分获取失败 |
| baostock query_all_stock 返回空 | P1 | 股票列表 API 停服，依赖回退列表 |
| register_factor 装饰器歧义 | P1 | `@register_factor("name")` 位置参数会报错 |
| FactorRegistry.clear() 缺失 | P2 | 测试中调用但未实现 |
| 测试/CLI 裸路径 | P2 | pip install 后测试和部分 CLI 命令失败 |
| data/cli.py 重复 CLI | P2 | 与主 CLI 重复，增加维护成本 |
| pytest testpaths 配置矛盾 | P2 | pyproject.toml 与 CONTRIBUTING.md 不一致 |

## 9. 重构优先级

### Phase 1: P0 阻断性修复
1. 统一所有 `importlib.import_module()` 为 `factor_pipeline.xxx`
2. 修复 `factor_list` 和 `factor_doc` 的裸路径
3. 修复 `register_factor` 装饰器位置参数歧义

### Phase 2: P1 性能与可靠性
4. 验证 ETL Arrow native insert 在大数据量下的性能
5. 移除 `data/cli.py` 重复 CLI，统一入口
6. 实现 `FactorRegistry.clear()` 方法

### Phase 3: P2 清理与文档
7. 修复测试裸路径
8. 统一 pytest testpaths 配置
9. 补充缺失依赖声明（akshare/baostock）
