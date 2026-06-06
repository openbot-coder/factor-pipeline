"""Factor report — HTML + charts output."""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

from analysis.ic import ICAnalysis
from analysis.layered import LayeredBacktest

sns.set_theme(style="whitegrid", font_scale=0.8)


class FactorReport:
    """Generate a comprehensive HTML factor analysis report."""

    def __init__(self, factor: pd.Series, prices: pd.DataFrame):
        """
        Args:
            factor: MultiIndex (date, stock) Series named "factor"
            prices: DataFrame with MultiIndex (date, stock), column "close"
        """
        self.factor = factor.rename("factor") if factor.name is None else factor
        self.prices = prices

    # ------------------------------------------------------------------
    # Alphalens-compatible input
    # ------------------------------------------------------------------

    def alphalens_input(self, quantiles=5, periods=(1, 5, 10)):
        """Build Alphalens-style merged DataFrame (factor + forward returns)."""
        close = self.prices["close"].copy()
        fwd = {}
        for p in periods:
            shifted = close.groupby(level=1).shift(-p)
            fwd[p] = (shifted / close - 1).rename(p)
        ret_df = pd.concat(fwd.values(), axis=1)
        # align index
        common = self.factor.index.intersection(ret_df.index)
        factor_aligned = self.factor.loc[common]
        ret_aligned = ret_df.loc[common]
        merged = pd.concat([factor_aligned, ret_aligned], axis=1)
        merged.columns = ["factor"] + [str(c) for c in ret_aligned.columns]
        return merged

    # ------------------------------------------------------------------
    # Run full analysis
    # ------------------------------------------------------------------

    def run(self, n_quantiles: int = 5, periods: tuple = (1, 5, 10)):
        """Run IC analysis + layered backtest for all forward periods."""
        close = self.prices["close"].copy()
        period = periods[0]   # primary period
        shifted = close.groupby(level=1).shift(-period)
        fwd_ret = (shifted / close - 1).dropna()

        common = self.factor.dropna().index.intersection(fwd_ret.index)
        f = self.factor.loc[common]
        r = fwd_ret.loc[common]

        ic = ICAnalysis(f, r)
        ic_res = ic.run("spearman")

        layered = LayeredBacktest(f, r, n_quantiles)
        lb_res = layered.run()

        return {
            "ic": ic_res,
            "layered": lb_res,
            "periods": periods,
        }

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _plot_ic(self, ic_series: pd.Series, factor_name: str) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        ax = axes[0]
        ic_series.plot(ax=ax, color="steelblue", alpha=0.6, label="Daily IC")
        ic_series.rolling(20).mean().plot(ax=ax, color="red", label="20d MA")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"IC Time Series — {factor_name}")
        ax.legend()
        ax.set_ylabel("IC")

        ax2 = axes[1]
        ic_series.hist(ax=ax2, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="red", lw=1)
        ax2.set_title("IC Distribution")
        ax2.set_xlabel("IC")
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def _plot_cumulative_returns(self, cum_ret: pd.DataFrame, spread: pd.Series) -> str:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = sns.color_palette("RdYlGn_r", n_colors=cum_ret.shape[1])
        for i, col in enumerate(cum_ret.columns):
            (cum_ret[col] * 100).plot(ax=ax, color=colors[i], label=col)
        if not spread.empty:
            (spread * 100).plot(ax=ax, color="black", lw=1.5, label="Long-Short", linestyle="--")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("Cumulative Returns by Quantile")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def _plot_quantile_returns(self, q_ret: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 3))
        q_ret.plot(kind="bar", ax=ax, alpha=0.8, width=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("Mean Daily Return by Quantile")
        ax.set_ylabel("Return")
        ax.set_xticklabels([])
        plt.tight_layout()
        return self._fig_to_b64(fig)

    @staticmethod
    def _fig_to_b64(fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def to_html(self, output_path: str = "reports/factor_report.html", n_quantiles: int = 5):
        """Generate the full HTML report."""
        result = self.run(n_quantiles)
        ic = result["ic"]
        lb = result["layered"]

        # plots
        ic_plot = self._plot_ic(ic.ic_series, self.factor.name or "Factor")
        cum_plot = self._plot_cumulative_returns(lb["cumulative_returns"], lb["long_short_cum"])
        qret_plot = self._plot_quantile_returns(lb["quantile_returns"])

        # summary table
        ic_stats = {
            "IC Mean": f"{ic.ic_mean:.4f}",
            "IC Std": f"{ic.ic_std:.4f}",
            "IR (IC Mean/IC Std)": f"{ic.ir:.4f}",
            "IC > 0 Ratio": f"{ic.ic_positive_ratio:.1%}",
            "t-stat": f"{ic.t_stat:.3f}",
            "p-value": f"{ic.p_value:.4f}",
            "N days": ic.n_days,
        }
        lb_stats = {
            "Top Quantile Mean Ret": f"{lb['top_mean']:.4%}",
            "Bottom Quantile Mean Ret": f"{lb['bottom_mean']:.4%}",
            "Long-Short Mean": f"{lb['spread_mean']:.4%}",
            "Long-Short Std": f"{lb['spread_std']:.4%}",
            "Long-Short IR": f"{lb['spread_ir']:.4f}",
        }

        template = Path(__file__).parent / "templates" / "report.html"
        if template.exists():
            env = Environment(loader=FileSystemLoader(str(template.parent)))
            tpl = env.get_template("report.html")
            html = tpl.render(
                factor_name=self.factor.name or "Factor",
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ic_stats=ic_stats,
                lb_stats=lb_stats,
                ic_plot=ic_plot,
                cum_plot=cum_plot,
                qret_plot=qret_plot,
                ic_series=ic.ic_series.tail(30).to_dict(),
            )
        else:
            html = self._default_html(ic_stats, lb_stats, ic_plot, cum_plot, qret_plot)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html)
        return output_path

    def _default_html(self, ic_stats, lb_stats, ic_plot, cum_plot, qret_plot) -> str:
        def tbl(d):
            rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())
            return f"<table class='stats'>{rows}</table>"

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Factor Report</title>
<style>
body{{font-family:sans-serif;margin:2rem;background:#fafafa}}
h1{{color:#333}}h2{{color:#555;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
.chart{{margin:1.5rem 0;text-align:center}}
.chart img{{max-width:100%;border:1px solid #ddd;border-radius:4px}}
.stats{{border-collapse:collapse;margin:.5rem 0}}
.stats th{{text-align:left;padding:4px 12px 4px 0;color:#555}}
.stats td{{padding:4px 12px;font-weight:bold}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:2rem}}
</style></head><body>
<h1>📊 Factor Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="grid">
<div><h2>IC Analysis</h2>{tbl(ic_stats)}</div>
<div><h2>Layered Backtest</h2>{tbl(lb_stats)}</div>
</div>

<div class="chart"><h2>IC Time Series</h2><img src="data:image/png;base64,{ic_plot}"/></div>
<div class="chart"><h2>Cumulative Returns</h2><img src="data:image/png;base64,{cum_plot}"/></div>
<div class="chart"><h2>Quantile Returns</h2><img src="data:image/png;base64,{qret_plot}"/></div>
</body></html>"""
