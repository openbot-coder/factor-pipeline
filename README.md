# Factor Pipeline

![CI](https://github.com/openbot-coder/factor-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Quantitative factor detection pipeline: calculate → IC/IR analyze → layered backtest → HTML report.

## Features

- **191 GTJA Alpha Factors** - Full implementation of Guotai Junan 191 alpha factors
- **Technical Indicators** - RSI, MACD, Bollinger Bands, ATR, OBV
- **IC Analysis** - Spearman/Pearson IC, IR, t-test statistics
- **Layered Backtest** - Quantile-based portfolio analysis
- **HTML Reports** - Interactive analysis reports with charts
- **Multiple Data Sources** - DuckDB, Parquet, CSV support
- **Unified CLI** - Full command-line interface (`fp` command)

## Installation

```bash
pip install factor-pipeline
```

Or install from source:

```bash
git clone https://github.com/openbot-coder/factor-pipeline.git
cd factor-pipeline
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage (`fp` command)

After installation, use the `fp` command:

```bash
# Data management
fp data init --db data/ohlcv.duckdb
fp data import data.csv --table daily_ohlcv
fp data info
fp data query "SELECT * FROM daily_ohlcv LIMIT 10"
fp data stats --table daily_ohlcv
fp data instruments --market SSE
fp data export "SELECT * FROM daily_ohlcv" --output data.csv

# Factor analysis
fp factor list
fp factor doc alpha001
fp factor run "Mean($close, 20)" --start 2020-01-01 --end 2024-12-31
fp factor batch factors.txt --output results/

# Backtesting
fp backtest run --factors alpha001,alpha014 --start 2020-01-01
fp backtest ic --factor alpha001
fp backtest layered --factor alpha001

# Reports
fp report generate --factors alpha001,alpha014 --output report.html

# Interactive shell
fp shell
```

### Alternative CLI Usage

Without installation, use Python module syntax:

```bash
# Data management
python -m data.cli import-csv data.csv --table daily_ohlcv
python -m data.cli info

# Standalone scripts
python scripts/csv2duckdb.py data.csv --table daily_ohlcv
```

### Python API

```python
from data.storage import DuckDBStorage
from data.loader import DataLoader
from factors.registry import FactorRegistry
from analysis.ic import ICAnalysis
from analysis.layered import LayeredBacktest
import importlib

# 1. Initialize DuckDB and import data
db = DuckDBStorage("data/ohlcv.duckdb")
db.import_csv("daily.csv", table="daily_ohlcv")

# 2. Load data via DataLoader
loader = DataLoader("duckdb", "data/ohlcv.duckdb")
data = loader.load(start="2020-01-01", end="2025-12-31")

# 3. Register factor modules
importlib.import_module("factors.gtja191")
importlib.import_module("factors.technical")

# 4. Compute factor
alpha_fn = FactorRegistry.get("alpha001")
factor_values = alpha_fn(data)

# 5. IC Analysis
close = data["close"]
fwd_ret = close.groupby(level=1).shift(-5) / close - 1
common = factor_values.dropna().index.intersection(fwd_ret.dropna().index)
ic = ICAnalysis(factor_values.loc[common], fwd_ret.loc[common])
result = ic.run("spearman")
print(f"IC Mean: {result.ic_mean:.4f}, IR: {result.ir:.3f}")

# 6. Layered Backtest
lb = LayeredBacktest(factor_values.loc[common], fwd_ret.loc[common], n_quantiles=5)
lb_result = lb.run()

# 7. Generate HTML Report
from analysis.report import FactorReport
report = FactorReport(factor_values, close)
report.to_html("reports/alpha001_report.html")
```

## CLI Command Reference

### Data Commands

| Command | Description |
|---------|-------------|
| `fp data init` | Initialize database schema |
| `fp data import <csv>` | Import CSV file |
| `fp data info` | Show database information |
| `fp data query <sql>` | Execute SQL query |
| `fp data stats` | Show table statistics |
| `fp data tables` | List all tables |
| `fp data instruments` | List instruments |
| `fp data export` | Export query results |

### Factor Commands

| Command | Description |
|---------|-------------|
| `fp factor list` | List all available factors |
| `fp factor doc <name>` | Show factor documentation |
| `fp factor run <expr>` | Run factor expression |
| `fp factor batch <file>` | Run batch factors from file |

### Backtest Commands

| Command | Description |
|---------|-------------|
| `fp backtest run` | Run backtest for factors |
| `fp backtest ic` | Run IC analysis |
| `fp backtest layered` | Run layered backtest |

### Report Commands

| Command | Description |
|---------|-------------|
| `fp report generate` | Generate HTML report |

## Core Metrics

| Metric | Description | Good Threshold |
|--------|-------------|----------------|
| IC Mean | Cross-sectional Spearman correlation mean | > 0.02 |
| IC IR | Information Ratio (IC Mean / IC Std) | > 0.5 |
| IC > 0 Ratio | Percentage of positive IC days | > 50% |
| Long-Short IR | Long-short portfolio IR | > 0.5 |

## Project Structure

```
factor-pipeline/
├── factors/              # Factor implementations
│   ├── base.py          # FactorABC base class + time-series operators
│   ├── registry.py      # FactorRegistry (@register_factor)
│   ├── ops.py           # Expression operators (80+)
│   ├── gtja191.py       # 191 GTJA alpha factors
│   └── technical.py     # Technical indicators
├── data/                # Data loading & preprocessing
│   ├── storage.py       # DuckDB storage layer
│   ├── loader.py        # DataLoader (DuckDB/Parquet/CSV)
│   └── preprocessor.py  # DataPreprocessor
├── analysis/            # Analysis modules
│   ├── ic.py           # ICAnalysis
│   ├── layered.py      # LayeredBacktest
│   └── report.py       # FactorReport
├── cli/                 # CLI interface (fp command)
├── config/             # Configuration
└── .github/workflows/   # CI/CD workflows
```

## Available Factors

### GTJA 191 Factors

`alpha001` - `alpha191` - Full Guotai Junan 191 alpha factor suite

### Technical Indicators

| Factor | Description |
|--------|-------------|
| `rsi14` | 14-day Relative Strength Index |
| `macd_diff` | MACD histogram (12, 26, 9) |
| `bb_pct` | Bollinger Bands %B |
| `atr14` | 14-day Average True Range |
| `obv` | On-Balance Volume |

### Expression Operators (80+)

| Category | Operators |
|----------|-----------|
| Time Series | `Ref`, `Delta`, `Sum`, `Mean`, `Std`, `Max`, `Min`, `Median`, `Corr`, `Cov` |
| Cross-sectional | `Rank`, `Quantile`, `Decile` |
| Decay | `DecayLinear`, `DecayExp`, `WMA`, `EMA`, `SMA` |
| Math | `Log`, `Abs`, `Sign`, `Sqrt`, `Power`, `Exp`, `Tanh` |
| Conditional | `Iif`, `Where`, `FillNa`, `IsNa` |

## Data Format

### DuckDB Schema

```sql
-- OHLCV data
CREATE TABLE daily_ohlcv (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    factor DOUBLE DEFAULT 1.0,
    PRIMARY KEY (date, symbol)
);

-- Instruments
CREATE TABLE instruments (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,
    industry VARCHAR
);
```

### CSV Import

```bash
# Auto-detect columns
fp data import data.csv --table daily_ohlcv

# Manual column mapping
fp data import data.csv --table daily_ohlcv --date-col trade_date --symbol-col ts_code
```

## CI/CD

This project uses GitHub Actions for:

- **CI** - Lint, type check, unit tests on every push
- **Daily Tests** - Automated factor analysis runs
- **Dependency Updates** - Weekly dependency checks
- **Release** - Automated PyPI publishing on tags

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Past performance does not guarantee future results.
