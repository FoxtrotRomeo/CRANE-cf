"""
Fusion-strategy comparison plots for all datasets.

For each dataset, produces a 2×2 metric figure and a flip-rate figure with
grouped bars — one bar group per generator, one bar per fusion strategy —
plus a CSV with the aggregated numbers.

Datasets: sepsis, long_covid_tweets, real_or_fake_jobs, memes (hateful memes)

Outputs land in each dataset's data/metric_plots_fusion/ directory.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

METRICS = ["outcome", "proximity", "sparsity", "plausibility"]
METRIC_LABELS = {
    "outcome":      "Outcome ↓",
    "proximity":    "Proximity ↓",
    "sparsity":     "Sparsity ↓",
    "plausibility": "Plausibility ↓",
}

# Fusion-strategy display colours (early / intermediate / late)
STRATEGY_COLORS = ["#2c7bb6", "#d7191c", "#1a9641"]
BAR_COLOR_MISSING = "#aaaaaa"

# ---------------------------------------------------------------------------
# Per-dataset specs
# ---------------------------------------------------------------------------

DATASETS: Dict[str, dict] = {

    "sepsis": {
        "label": "Sepsis",
        "data_dir": ROOT / "sepsis" / "data",
        "folds": [0, 1, 2, 3, 4],
        "fusion_strategies": {
            "Early":        "ablation_runs/fold_{fold}/fusion_early_k200/summary.json",
            "Intermediate": "ablation_runs/fold_{fold}/fusion_intermediate_k200/summary.json",
            "Late":         "ablation_runs/fold_{fold}/fusion_late_k200/summary.json",
        },
        "generators": [
            "Tabular",
            "TS[lab_exams_vitals]",
            "Frankenstein",
            "Combined",
            "EarlyFusion",
            "IntermediateFusion",
        ],
        "gen_short": {
            "Tabular":            "Tab",
            "TS[lab_exams_vitals]": "TS",
            "Frankenstein":       "Frank",
            "Combined":           "Comb",
            "EarlyFusion":        "EFus",
            "IntermediateFusion": "IntFus",
        },
    },

    "long_covid_tweets": {
        "label": "Long COVID Tweets",
        "data_dir": ROOT / "long_covid_tweets" / "data",
        "folds": None,
        "fusion_strategies": {
            "Early":        "ablation_runs/fusion_early_k200/summary.json",
            "Intermediate": "ablation_runs/fusion_intermediate_k200/summary.json",
            "Late":         "ablation_runs/fusion_late_k200/summary.json",
        },
        "generators": [
            "Tabular",
            "Text",
            "Frankenstein",
            "Combined",
            "EarlyFusion",
            "IntermediateFusion",
        ],
        "gen_short": {
            "Tabular":            "Tab",
            "Text":               "Text",
            "Frankenstein":       "Frank",
            "Combined":           "Comb",
            "EarlyFusion":        "EFus",
            "IntermediateFusion": "IntFus",
        },
    },

    "real_or_fake_jobs": {
        "label": "Real or Fake Jobs",
        "data_dir": ROOT / "real_or_fake_jobs" / "data",
        "folds": None,
        "fusion_strategies": {
            "Early":        "ablation_runs/fusion_early_k200/summary.json",
            "Intermediate": "ablation_runs/fusion_intermediate_k200/summary.json",
            "Late":         "ablation_runs/fusion_late_k200/summary.json",
        },
        "generators": [
            "Tabular",
            "Text",
            "Text_company_profile",
            "Text_requirements",
            "Frankenstein",
            "Combined",
            "EarlyFusion",
            "IntermediateFusion",
        ],
        "gen_short": {
            "Tabular":               "Tab",
            "Text":                  "Text",
            "Text_company_profile":  "Text[co]",
            "Text_requirements":     "Text[req]",
            "Frankenstein":          "Frank",
            "Combined":              "Comb",
            "EarlyFusion":           "EFus",
            "IntermediateFusion":    "IntFus",
        },
    },

    "memes": {
        "label": "Hateful Memes",
        "data_dir": ROOT / "memes" / "data",
        "folds": None,
        "fusion_strategies": {
            "Early":        "ablation_runs/fusion_early_k200/summary.json",
            "Intermediate": "ablation_runs/fusion_intermediate_k200/summary.json",
            "Late":         "ablation_runs/fusion_late_k200/summary.json",
        },
        "generators": [
            "Text",
            "Image",
            "Frankenstein",
            "Combined",
            "EarlyFusion",
            "IntermediateFusion",
        ],
        "gen_short": {
            "Text":               "Text",
            "Image":              "Img",
            "Frankenstein":       "Frank",
            "Combined":           "Comb",
            "EarlyFusion":        "EFus",
            "IntermediateFusion": "IntFus",
        },
    },
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        warnings.warn(f"Missing: {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def load_strategy_data(cfg: dict) -> pd.DataFrame:
    """
    Returns one row per (strategy, combo_id, generator) with 4 metric columns.
    For sepsis (multi-fold) the fold is an additional column.
    """
    data_dir = cfg["data_dir"]
    folds    = cfg["folds"]
    generators = cfg["generators"]
    records: List[dict] = []

    for strategy, path_tpl in cfg["fusion_strategies"].items():
        if folds is not None:
            paths = [(f, data_dir / path_tpl.format(fold=f)) for f in folds]
        else:
            paths = [(None, data_dir / path_tpl)]

        for fold, path in paths:
            summary = _load_summary(path)
            if summary is None:
                continue
            for row in summary.get("rows", []):
                if row.get("status") != "ok":
                    continue
                objectives = row.get("objectives", {})
                for gen in generators:
                    obj = objectives.get(gen)
                    if not obj:
                        continue
                    rec = {
                        "strategy":  strategy,
                        "generator": gen,
                        "combo_id":  row.get("combo_id"),
                        "fold":      fold,
                    }
                    for m in METRICS:
                        rec[m] = obj.get(m, np.nan)
                    records.append(rec)

    return pd.DataFrame(records)


def load_strategy_flip_rate(cfg: dict) -> pd.DataFrame:
    """
    Returns one row per (strategy, combo_id, generator) with flip_rate ∈ {0, 1}.
    """
    data_dir   = cfg["data_dir"]
    folds      = cfg["folds"]
    generators = cfg["generators"]
    records: List[dict] = []

    for strategy, path_tpl in cfg["fusion_strategies"].items():
        if folds is not None:
            paths = [(f, data_dir / path_tpl.format(fold=f)) for f in folds]
        else:
            paths = [(None, data_dir / path_tpl)]

        for fold, path in paths:
            summary = _load_summary(path)
            if summary is None:
                continue
            for row in summary.get("rows", []):
                if row.get("status") != "ok":
                    continue
                counts = row.get("candidate_counts", {})
                for gen in generators:
                    records.append({
                        "strategy":  strategy,
                        "generator": gen,
                        "combo_id":  row.get("combo_id"),
                        "fold":      fold,
                        "flip_rate": 1.0 if counts.get(gen, 0) > 0 else 0.0,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _grouped_bar_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    generators: List[str],
    strategies: List[str],
    gen_short: dict,
    colors: List[str],
    title: str,
    ylabel: str = "",
    ylim: Optional[tuple] = None,
    pct_fmt: bool = False,
):
    n_gen  = len(generators)
    n_strat = len(strategies)
    width  = 0.8 / n_strat
    x      = np.arange(n_gen)

    for si, (strategy, color) in enumerate(zip(strategies, colors)):
        sub = df[df["strategy"] == strategy]
        means, yerr_lo, yerr_hi = [], [], []
        for gen in generators:
            vals = sub.loc[sub["generator"] == gen, metric].dropna()
            if vals.empty:
                means.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
            else:
                m = vals.mean()
                s = vals.std(ddof=0)
                means.append(m)
                yerr_lo.append(min(s, m) if not pct_fmt else min(s, m))
                yerr_hi.append(s)

        bar_x = x + (si - (n_strat - 1) / 2) * width
        bars = ax.bar(
            bar_x, means,
            width=width * 0.9,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=strategy,
            alpha=0.88,
        )
        # shade missing bars grey
        for bar, m in zip(bars, means):
            if np.isnan(m):
                bar.set_color(BAR_COLOR_MISSING)
                bar.set_alpha(0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([gen_short.get(g, g) for g in generators],
                       rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if ylim:
        ax.set_ylim(*ylim)
    if pct_fmt:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


# ---------------------------------------------------------------------------
# Per-dataset plotting entry points
# ---------------------------------------------------------------------------

def plot_dataset(name: str, cfg: dict):
    data_dir   = cfg["data_dir"]
    generators = cfg["generators"]
    gen_short  = cfg["gen_short"]
    strategies = list(cfg["fusion_strategies"].keys())
    colors     = STRATEGY_COLORS[: len(strategies)]
    label      = cfg["label"]

    out_dir = data_dir / "metric_plots_fusion"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"{'='*60}")

    df      = load_strategy_data(cfg)
    flip_df = load_strategy_flip_rate(cfg)

    if df.empty:
        print(f"  [skip] No data found for {name}.")
        return

    n_records = len(df)
    print(f"  {n_records} metric records loaded.")

    # ---------------------------------------------------------------- 2×2 metrics
    fig, axes = plt.subplots(
        2, 2,
        figsize=(max(10, 0.85 * len(generators) * len(strategies)), 8),
        squeeze=False,
    )
    fig.suptitle(
        f"Counterfactual metrics by fusion strategy — {label}",
        fontsize=13, fontweight="bold",
    )
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, metric in zip(axes_flat, METRICS):
        _grouped_bar_panel(
            ax, df, metric, generators, strategies, gen_short, colors,
            title=METRIC_LABELS[metric],
        )

    # shared legend
    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    # add missing patch
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=BAR_COLOR_MISSING, alpha=0.5, label="Missing"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(strategies) + 1, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = out_dir / "cf_metrics_by_fusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---------------------------------------------------------------- flip rate
    fig2, ax2 = plt.subplots(
        1, 1,
        figsize=(max(8, 0.85 * len(generators) * len(strategies)), 4),
    )
    fig2.suptitle(
        f"Flip rate ↑ by fusion strategy — {label}",
        fontsize=13, fontweight="bold",
    )
    _grouped_bar_panel(
        ax2, flip_df, "flip_rate", generators, strategies, gen_short, colors,
        title="Flip rate ↑",
        ylabel="Flip rate",
        ylim=(0, 1.05),
        pct_fmt=True,
    )
    handles2, _ = ax2.get_legend_handles_labels()
    handles2.append(Patch(facecolor=BAR_COLOR_MISSING, alpha=0.5, label="Missing"))
    fig2.legend(handles=handles2, loc="lower center",
                ncol=len(strategies) + 1, fontsize=9, frameon=True,
                bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out2 = out_dir / "cf_flip_rate_by_fusion.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ---------------------------------------------------------------- CSV
    rows = []
    for strategy in strategies:
        sub = df[df["strategy"] == strategy]
        fr_sub = flip_df[flip_df["strategy"] == strategy]
        for gen in generators:
            gen_sub    = sub[sub["generator"] == gen]
            gen_fr_sub = fr_sub[fr_sub["generator"] == gen]
            rec = {"dataset": name, "strategy": strategy, "generator": gen}
            for m in METRICS:
                vals = gen_sub[m].dropna()
                rec[f"{m}_mean"] = round(vals.mean(), 4) if not vals.empty else np.nan
                rec[f"{m}_std"]  = round(vals.std(ddof=0), 4) if not vals.empty else np.nan
            fr_vals = gen_fr_sub["flip_rate"].dropna()
            rec["flip_rate_mean"] = round(fr_vals.mean(), 4) if not fr_vals.empty else np.nan
            rec["flip_rate_std"]  = round(fr_vals.std(ddof=0), 4) if not fr_vals.empty else np.nan
            rows.append(rec)

    csv_df = pd.DataFrame(rows)
    out_csv = out_dir / "cf_metrics_by_fusion.csv"
    csv_df.to_csv(out_csv, index=False)
    print(f"\n=== {label} — metrics by fusion strategy ===")
    print(csv_df.to_string(index=False))
    print(f"\n  Saved: {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        plot_dataset(name, cfg)

    print("\nDone.")
