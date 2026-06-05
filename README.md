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

### CLI Usage

```bash
# List all available factors
python run.py factors

# View factor documentation
python run.py doc alpha001

# Run factor analysis
python run.py run \
  --factors alpha001 alpha006 alpha014 \
  --data data/ohlcv.duckdb \
  --start 2020-01-01 \
  --end 2025-12-31
```

### Python API

```python
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from factors.registry import FactorRegistry
from analysis.ic import ICAnalysis
from analysis.layered import LayeredBacktest
from analysis.report import FactorReport
import importlib

# 1. Load data
loader = DataLoader("duckdb", "data/ohlcv.duckdb")
data = loader.load(start="2020-01-01", end="2025-12-31")

# 2. Register factor modules
importlib.import_module("factors.gtja191")
importlib.import_module("factors.technical")

# 3. Compute factor
alpha_fn = FactorRegistry.get("alpha001")
factor_values = alpha_fn(data)

# 4. IC Analysis
close = data["close"]
fwd_ret = close.groupby(level=1).shift(-5) / close - 1
common = factor_values.dropna().index.intersection(fwd_ret.dropna().index)
ic = ICAnalysis(factor_values.loc[common], fwd_ret.loc[common])
result = ic.run("spearman")
print(f"IC Mean: {result.ic_mean:.4f}, IR: {result.ir:.3f}")

# 5. Layered Backtest
lb = LayeredBacktest(factor_values.loc[common], fwd_ret.loc[common], n_quantiles=5)
lb_result = lb.run()

# 6. Generate HTML Report
report = FactorReport(factor_values, close)
report.to_html("reports/alpha001_report.html")
```

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
│   ├── gtja191.py       # 191 GTJA alpha factors
│   └── technical.py     # Technical indicators
├── data/                # Data loading & preprocessing
│   ├── loader.py        # DataLoader (DuckDB/Parquet/CSV)
│   └── preprocessor.py  # DataPreprocessor
├── analysis/            # Analysis modules
│   ├── ic.py           # ICAnalysis
│   ├── layered.py      # LayeredBacktest
│   └── report.py       # FactorReport
├── config/             # Configuration
├── cli/                # CLI interface
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

## Data Format

Expected input data (MultiIndex DataFrame):

```
                 close    volume
date       stock
2020-01-02 600000.SH  10.5     1000000
           000001.SZ   15.2      800000
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
