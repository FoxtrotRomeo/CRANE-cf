"""
Plot counterfactual metrics for all generators on the long_covid_tweets dataset.

Generators:
  NN-based (6):   Tabular, Text, Frankenstein, Combined, EarlyFusion, IntermediateFusion
  Genetic (8):    Genetic, Genetic top30, Genetic topText, Genetic top30+text
                  + seeded variants of each

Metrics (4): outcome, proximity, sparsity, plausibility

Outputs (in metric_plots/):
  - cf_metrics_aggregated.png  : 2×2 grid (4 metrics), mean±std bars per generator
  - cf_flip_rate.png           : flip rate per generator (genetic only)
  - cf_metrics.csv             : mean±std for 4 metrics per generator
  - cf_flip_rate.csv           : mean±std flip rate per generator
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE         = Path(__file__).parent
DATA_DIR     = BASE / "data"
FUSION_DIR   = DATA_DIR / "ablation_runs" / "fusion_intermediate_deep"
GENETIC_DIR  = DATA_DIR / "ablation_runs"
SEEDED_DIR   = DATA_DIR / "ablation_runs_seeded"
OUT_DIR      = DATA_DIR / "metric_plots"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = ["outcome", "proximity", "sparsity", "plausibility"]
METRIC_LABELS = {
    "outcome":     "Outcome ↓",
    "proximity":   "Proximity ↓",
    "sparsity":    "Sparsity ↓",
    "plausibility":"Plausibility ↓",
}

NN_GENERATORS = [
    "Tabular", "Text", "MPS", "MC-R", "EarlyFusion", "IntermediateFusion",
]

# summary.json files saved before the MPS/MC-R rename used the old generator
# names ("Frankenstein"/"Combined") as dict keys — fall back to those so
# pre-existing ablation runs still plot.
LEGACY_GEN_KEY_ALIASES = {"MPS": "Frankenstein", "MC-R": "Combined"}

# (display_name, subfolder_name)
GENETIC_VARIANTS = [
    ("Genetic",            "genetic_no_sel"),
    ("Genetic top30",      "genetic_top_30"),
    ("Genetic topText",    "genetic_top_text"),
    ("Genetic top30+text", "genetic_top_30_top_text"),
]
GENETIC_SEEDED_VARIANTS = [
    ("GeneticSeeded",            "genetic_no_sel"),
    ("GeneticSeeded top30",      "genetic_top_30"),
    ("GeneticSeeded topText",    "genetic_top_text"),
    ("GeneticSeeded top30+text", "genetic_top_30_top_text"),
]

GENETIC_GENERATORS = [n for n, _ in GENETIC_VARIANTS + GENETIC_SEEDED_VARIANTS]
ALL_GENERATORS     = NN_GENERATORS + GENETIC_GENERATORS

GEN_SHORT = {
    "Tabular":               "Tab",
    "Text":                  "Text",
    "MPS":                   "MPS",
    "MC-R":                  "MC-R",
    "EarlyFusion":           "EFus",
    "IntermediateFusion":    "IntFus",
    "Genetic":               "Gen",
    "Genetic top30":         "Gen30",
    "Genetic topText":       "GenTxt",
    "Genetic top30+text":    "Gen30T",
    "GeneticSeeded":         "GenS",
    "GeneticSeeded top30":   "GenS30",
    "GeneticSeeded topText": "GenSTxt",
    "GeneticSeeded top30+text": "GenS30T",
}

# Genetic F-column ordering: outcome=0, proximity=1, sparsity=2, plausibility=3
GENETIC_F_COLS = {"outcome": 0, "proximity": 1, "sparsity": 2, "plausibility": 3}

GEN_ORDER = ALL_GENERATORS
X_TICKS   = [GEN_SHORT[g] for g in GEN_ORDER]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nn_data() -> pd.DataFrame:
    """
    Load NN/fusion generator metrics from fusion_intermediate_deep/summary.json.
    Returns one row per (combo, generator) with the 4 objective values.
    The objectives in summary.json are already averaged across patients within
    each combo, so std will be across combos (distance-metric configurations).
    """
    summary_path = FUSION_DIR / "summary.json"
    if not summary_path.exists():
        warnings.warn(f"Missing: {summary_path}")
        return pd.DataFrame()

    with open(summary_path) as fh:
        summary = json.load(fh)

    records = []
    for row in summary.get("rows", []):
        if row.get("status") != "ok":
            continue
        objectives = row.get("objectives", {})
        for gen in NN_GENERATORS:
            obj = objectives.get(gen) or objectives.get(LEGACY_GEN_KEY_ALIASES.get(gen))
            if not obj:
                continue
            rec = {"generator": gen, "combo_id": row.get("combo_id")}
            for m in METRICS:
                rec[m] = obj.get(m, np.nan)
            records.append(rec)
    return pd.DataFrame(records)


def load_nn_flip_rate() -> pd.DataFrame:
    """
    Derive per-combo flip rate for NN generators from candidate_counts.
    flip_rate = 1 if at least one candidate was generated for that generator, else 0.
    (The summary does not expose per-patient feasibility counts for NN runs.)
    """
    summary_path = FUSION_DIR / "summary.json"
    if not summary_path.exists():
        return pd.DataFrame()

    with open(summary_path) as fh:
        summary = json.load(fh)

    records = []
    for row in summary.get("rows", []):
        if row.get("status") != "ok":
            continue
        counts = row.get("candidate_counts", {})
        for gen in NN_GENERATORS:
            count = counts.get(gen, counts.get(LEGACY_GEN_KEY_ALIASES.get(gen), 0))
            records.append({
                "generator": gen,
                "combo_id":  row.get("combo_id"),
                "flip_rate": 1.0 if count > 0 else 0.0,
            })
    return pd.DataFrame(records)


def load_genetic_data(base_dir: Path, variants: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Load genetic counterfactual metrics from pkl files.
    Returns one row per patient per variant with the 4 objectives
    (averaged over Pareto candidates for that patient).
    """
    records = []
    for name, subfolder in variants:
        pkl_path = base_dir / subfolder / "genetic_counterfactuals_split0.pkl"
        if not pkl_path.exists():
            warnings.warn(f"Missing: {pkl_path}")
            continue
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        for patient, result in data.items():
            if not isinstance(result, dict):
                continue
            F = result.get("F")
            if F is None:
                continue
            F = np.asarray(F)
            if F.ndim != 2 or F.shape[1] < 4 or F.shape[0] == 0:
                continue
            rec = {"generator": name, "patient": patient}
            for m, col in GENETIC_F_COLS.items():
                rec[m] = float(np.mean(F[:, col]))
            records.append(rec)
    return pd.DataFrame(records)


def load_genetic_flip_rate(base_dir: Path, variants: list[tuple[str, str]]) -> pd.DataFrame:
    """
    flip_rate per variant = fraction of patients with n_candidates_feasible > 0.
    Returns one row per variant.
    """
    records = []
    for name, subfolder in variants:
        pkl_path = base_dir / subfolder / "genetic_counterfactuals_split0.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        n_total   = len(data)
        if n_total == 0:
            continue
        n_flipped = sum(
            1 for v in data.values()
            if isinstance(v, dict) and v.get("n_candidates_feasible", 0) > 0
        )
        records.append({
            "generator": name,
            "flip_rate": n_flipped / n_total,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

BAR_COLOR_NN      = "#2c7bb6"
BAR_COLOR_GENETIC = "#d7191c"
BAR_COLOR_SEEDED  = "#1a9641"
BAR_COLOR_MISSING = "#aaaaaa"

def _bar_color(gen: str) -> str:
    if gen in NN_GENERATORS:
        return BAR_COLOR_NN
    if "Seeded" in gen:
        return BAR_COLOR_SEEDED
    return BAR_COLOR_GENETIC


def plot_aggregated(summary_df: pd.DataFrame):
    """
    2×2 grid (4 metrics). Each panel: mean±std bars per generator.
    Saves to metric_plots/cf_metrics_aggregated.png.
    """
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.7 * len(ALL_GENERATORS)), 8), squeeze=False)
    fig.suptitle("Counterfactual metrics — long_covid_tweets", fontsize=14, fontweight="bold")

    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, metric in zip(axes_flat, METRICS):
        means, yerr_low, yerr_high, missing = [], [], [], []
        for gen in GEN_ORDER:
            vals = summary_df.loc[summary_df["generator"] == gen, metric].dropna()
            if vals.empty:
                means.append(np.nan)
                yerr_low.append(0.0)
                yerr_high.append(0.0)
                missing.append(True)
            else:
                m = vals.mean()
                s = vals.std(ddof=0)
                means.append(m)
                yerr_low.append(min(s, m))
                yerr_high.append(s)
                missing.append(False)

        x      = np.arange(len(GEN_ORDER))
        colors = [BAR_COLOR_MISSING if m else _bar_color(g) for m, g in zip(missing, GEN_ORDER)]
        ax.bar(
            x, means,
            yerr=[yerr_low, yerr_high],
            capsize=3, color=colors, edgecolor="white", linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(X_TICKS, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BAR_COLOR_NN,      label="NN / Fusion"),
        Patch(facecolor=BAR_COLOR_GENETIC, label="Genetic"),
        Patch(facecolor=BAR_COLOR_SEEDED,  label="Genetic (seeded)"),
        Patch(facecolor=BAR_COLOR_MISSING, label="Missing"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = OUT_DIR / "cf_metrics_aggregated.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_flip_rate(flip_df: pd.DataFrame):
    """
    Single panel: mean flip rate per generator.
    For NN generators, flip_rate is derived from whether any candidates were produced.
    For genetic generators, flip_rate is the fraction of patients with ≥1 feasible CF.
    Saves to metric_plots/cf_flip_rate.png.
    """
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.7 * len(ALL_GENERATORS)), 4))
    fig.suptitle("Flip rate ↑ (% CFs achieving target class) — long_covid_tweets",
                 fontsize=13, fontweight="bold")

    means, yerr_low, yerr_high, missing = [], [], [], []
    for gen in GEN_ORDER:
        vals = flip_df.loc[flip_df["generator"] == gen, "flip_rate"].dropna()
        if vals.empty:
            means.append(np.nan)
            yerr_low.append(0.0)
            yerr_high.append(0.0)
            missing.append(True)
        else:
            m = vals.mean()
            s = vals.std(ddof=0)
            means.append(m)
            yerr_low.append(min(s, m))
            yerr_high.append(s)
            missing.append(False)

    x      = np.arange(len(GEN_ORDER))
    colors = [BAR_COLOR_MISSING if m else _bar_color(g) for m, g in zip(missing, GEN_ORDER)]
    ax.bar(
        x, means,
        yerr=[yerr_low, yerr_high],
        capsize=3, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(X_TICKS, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Flip rate", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "cf_flip_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(nn_df: pd.DataFrame, genetic_df: pd.DataFrame, flip_df: pd.DataFrame):
    """
    Write:
      - cf_metrics.csv   : mean±std for 4 metrics, one row per generator
      - cf_flip_rate.csv : mean±std flip_rate, one row per generator
    """
    raw = pd.concat(
        [df[["generator"] + METRICS] for df in [nn_df, genetic_df] if not df.empty],
        ignore_index=True,
    )
    metrics_agg = (
        raw.groupby("generator")[METRICS]
        .agg(["mean", "std"])
        .round(4)
    )
    metrics_agg.columns = [f"{m}_{s}" for m, s in metrics_agg.columns]
    # Restore canonical generator order
    metrics_agg = metrics_agg.reindex(
        [g for g in GEN_ORDER if g in metrics_agg.index]
    ).reset_index()
    out_metrics = OUT_DIR / "cf_metrics.csv"
    metrics_agg.to_csv(out_metrics, index=False)
    print(f"\n=== Aggregated metrics ===\n{metrics_agg.to_string(index=False)}")
    print(f"Saved: {out_metrics}")

    flip_agg = (
        flip_df.groupby("generator")["flip_rate"]
        .agg(["mean", "std"])
        .round(4)
        .rename(columns={"mean": "flip_rate_mean", "std": "flip_rate_std"})
    )
    flip_agg = flip_agg.reindex(
        [g for g in GEN_ORDER if g in flip_agg.index]
    ).reset_index()
    out_flip = OUT_DIR / "cf_flip_rate.csv"
    flip_agg.to_csv(out_flip, index=False)
    print(f"\n=== Flip rate ===\n{flip_agg.to_string(index=False)}")
    print(f"Saved: {out_flip}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading NN/fusion metrics …")
    nn_df = load_nn_data()
    print(f"  {len(nn_df)} NN records (combo × generator)")

    print("Loading NN flip rates …")
    nn_flip_df = load_nn_flip_rate()

    print("Loading genetic metrics (non-seeded) …")
    genetic_df = load_genetic_data(GENETIC_DIR, GENETIC_VARIANTS)
    print(f"  {len(genetic_df)} genetic patient records")

    print("Loading genetic metrics (seeded) …")
    seeded_df = load_genetic_data(SEEDED_DIR, GENETIC_SEEDED_VARIANTS)
    print(f"  {len(seeded_df)} seeded genetic patient records")

    print("Loading genetic flip rates …")
    genetic_flip_df = load_genetic_flip_rate(GENETIC_DIR, GENETIC_VARIANTS)
    seeded_flip_df  = load_genetic_flip_rate(SEEDED_DIR, GENETIC_SEEDED_VARIANTS)

    all_genetic_df = pd.concat([genetic_df, seeded_df], ignore_index=True)
    flip_df = pd.concat([nn_flip_df, genetic_flip_df, seeded_flip_df], ignore_index=True)

    # Summary df: NN aggregated per (combo, generator), genetic per patient
    nn_agg = (
        nn_df.groupby("generator")[METRICS].mean().reset_index()
        if not nn_df.empty else pd.DataFrame()
    )
    summary_df = pd.concat([nn_df, all_genetic_df], ignore_index=True)

    print("Exporting CSVs …")
    export_csv(nn_df, all_genetic_df, flip_df)

    print("Plotting aggregated figure …")
    plot_aggregated(summary_df)

    print("Plotting flip rate figure …")
    plot_flip_rate(flip_df)

    print(f"\nDone. All outputs in: {OUT_DIR}")
