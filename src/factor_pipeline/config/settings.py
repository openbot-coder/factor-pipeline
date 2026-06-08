"""Pipeline configuration — YAML load/dump + dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    source: str = "duckdb"
    path: str = "data/ohlcv.duckdb"
    symbols: list[str] = field(default_factory=list)  # empty = all
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"


@dataclass
class FactorConfig:
    names: list[str] = field(default_factory=lambda: ["alpha001", "alpha002", "alpha014"])
    max_window: int = 250  # trailing days needed for factor calc
    winsorize_pct: float = 0.01  # winsorise tails
    neutralize: bool = True


@dataclass
class BacktestConfig:
    n_quantiles: int = 5
    periods: list[int] = field(default_factory=lambda: [1, 5, 10])
    long_short: bool = True


@dataclass
class OutputConfig:
    report_dir: str = "reports"
    report_format: str = "html"
    save_csv: bool = True
    log_level: str = "INFO"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            data=DataConfig(**d.get("data", {})),
            factor=FactorConfig(**d.get("factor", {})),
            backtest=BacktestConfig(**d.get("backtest", {})),
            output=OutputConfig(**d.get("output", {})),
        )

    def to_yaml(self, path: str | Path):
        import dataclasses

        d = {
            "data": dataclasses.asdict(self.data),
            "factor": dataclasses.asdict(self.factor),
            "backtest": dataclasses.asdict(self.backtest),
            "output": dataclasses.asdict(self.output),
        }
        Path(path).write_text(yaml.dump(d, allow_unicode=True))


def load_config(path: str | None = None) -> PipelineConfig:
    path = path or os.environ.get("FACTOR_PIPELINE_CONFIG", "config/pipeline.yaml")
    if Path(path).exists():
        return PipelineConfig.from_yaml(path)
    return PipelineConfig()
