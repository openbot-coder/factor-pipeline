"""factor-pipeline: Quantitative factor research framework with DuckDB backend.

A Python framework for factor research, featuring:
- 80+ expression operators (Qlib-style)
- GTJA 191 alpha factors
- Technical indicators
- IC analysis and layered backtesting
- DuckDB-powered data management

Example:
    from factor_pipeline import DuckDBStorage, FactorRegistry
    from factor_pipeline.factors.ops import Mean, Ref

    # Load data
    db = DuckDBStorage("data/ohlcv.duckdb")
    data = db.load(symbols=["000001"], start="2024-01-01")

    # Calculate factor
    ma5 = Mean(data["close"], 5)

    # Register and run analysis
    registry = FactorRegistry()
    ...
"""

__version__ = "0.1.0"
__author__ = "Factor Pipeline Team"
