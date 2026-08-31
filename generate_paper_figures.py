"""
generate_paper_figures.py
=========================
Produces all paper figures and tables from CRANE-cf ablation results.

Outputs land in  paper_figures/
  fig1/   varA_boxplot_{strategy}_{dataset}.pdf   (distributions vs single point)
          varA_barplot_{strategy}_{dataset}.pdf
          varB_barplot_{strategy}_{dataset}.pdf    (everything averaged)
  fig2/   ablation_{strategy}_{dataset}.pdf        (encoder/distance ablation)
  fig3_classifier_table.tex
  fig4/   stability_{dataset}.pdf                 (stability across fusion strategies)
  fig5_runtime_table.tex
  fig6/   pareto_{dataset}.pdf                    (scatter + Pareto frontier)
  fig7_metrics.csv

Usage:
    python generate_paper_figures.py [--outdir paper_figures]
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150,
})

ROOT = Path(__file__).resolve().parent
METRICS     = ["outcome", "proximity", "sparsity", "plausibility"]
METRIC_LABELS = {
    "outcome":      "Outcome ↓",
    "proximity":    "Proximity ↓",
    "sparsity":     "Sparsity ↓",
    "plausibility": "Plausibility ↓",
}
STRATEGIES = ["early", "intermediate", "late"]
STRATEGY_LABELS = {
    "early":        "Early Fusion",
    "intermediate": "Intermediate Fusion",
    "late":         "Late Fusion",
}

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASETS: Dict[str, dict] = {
    "sepsis": {
        "label": "Sepsis",
        "data_dir": ROOT / "sepsis" / "data",
        "fold_structure": True,
        "strategy_run": {
            "early":        "fusion_early_deep",
            "intermediate": "fusion_intermediate_rf",
            "late":         "fusion_late_deep",
        },
        "strategy_folds": {
            "early":        [0, 1, 2, 3, 4],
            "intermediate": [0, 1, 2, 3, 4],
            "late":         [0, 1, 2, 3, 4],
        },
    },
    "tweets": {
        "label": "Long COVID Tweets",
        "data_dir": ROOT / "long_covid_tweets" / "data",
        "fold_structure": False,
        "strategy_run": {
            "early":        "fusion_early_mlp",
            "intermediate": "fusion_intermediate_deep",
            "late":         "fusion_late_nondp",
        },
    },
    "memes": {
        "label": "Hateful Memes",
        "data_dir": ROOT / "memes" / "data",
        "fold_structure": False,
        "strategy_run": {
            "early":        "fusion_early_gbt",
            "intermediate": "fusion_intermediate_deep",
            "late":         "fusion_late_nondp",
        },
    },
    "jobs": {
        "label": "Real or Fake Jobs",
        "data_dir": ROOT / "real_or_fake_jobs" / "data",
        "fold_structure": False,
        "strategy_run": {
            "early":        "fusion_early_mlp",
            "intermediate": "fusion_intermediate_deep",
            "late":         "fusion_late_nondp",
        },
    },
}

REGISTRY_FILES = {
    "sepsis":  ROOT / "sepsis"           / "data" / "fusion_model_registry.json",
    "tweets":  ROOT / "long_covid_tweets"/ "data" / "fusion_model_registry.json",
    "memes":   ROOT / "memes"            / "data" / "fusion_model_registry.json",
    "jobs":    ROOT / "real_or_fake_jobs"/ "data" / "fusion_model_registry.json",
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

CRANE_GENERATORS = [
    "Tabular", "TS[lab_exams_vitals]", "Text", "Image",
    "EarlyFusion", "IntermediateFusion", "Frankenstein", "Combined",
]

_PALETTE = [
    "#4e79a7","#f28e2b","#59a14f","#e15759",
    "#76b7b2","#edc948","#b07aa1",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#dbdb8d","#9edae5","#ad494a","#8c564b","#17becf",
]

_FIXED_COLORS = {
    "Combined": "#000000",  # black highlight
    "Image":    "#f4a7b9",  # light pink
}

GENERATOR_DISPLAY: Dict[str, str] = {
    "Tabular":              "TabularNN",
    "TS[lab_exams_vitals]": "TimeSeriesNN",
    "Text":                 "TextNN",
    "Image":                "ImageNN",
    "Frankenstein":         "MPS",
    "Combined":             "MC-R",
    "EarlyFusion":          "EF-NN",
    "IntermediateFusion":   "IF-NN",
}

def _glabel(name: str) -> str:
    """Return the display label for a generator name."""
    return GENERATOR_DISPLAY.get(name, name)

def _build_color_map(methods: List[str]) -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    palette_idx = 0
    for m in methods:
        if m in _FIXED_COLORS:
            cmap[m] = _FIXED_COLORS[m]
        else:
            cmap[m] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
    return cmap

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_K_ABLATION_FNAME = "k_ablation_summary_k1_5_10_20_50.json"


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_crane(dataset: str, strategy: str) -> pd.DataFrame:
    """Load ablation data for a (dataset, strategy) pair.
    Returns one row per (generator, combo), averaged across folds for sepsis.
    Columns: generator, combo_id, outcome, proximity, sparsity, plausibility,
             n_samples, coverage, text_encoder, text_metric, img_encoder, img_metric,
             tab_metric, ts_metric, runtime_sec.
    """
    cfg = DATASETS[dataset]
    run_name = cfg["strategy_run"].get(strategy)
    if not run_name:
        return pd.DataFrame()

    def _row_to_records(row: dict, fold=None) -> List[dict]:
        if row.get("status") != "ok":
            return []
        n_s = row.get("n_samples", 1) or 1
        tc  = row.get("text_config")  or {}
        ic  = row.get("image_config") or {}
        tab = row.get("tab_metrics")  or {}
        ts  = row.get("ts_metrics")   or {}
        tab_m = list(tab.values())[0] if tab else None
        ts_m  = list(list(ts.values())[0].values())[0] if ts else None
        records = []
        for gen, obj in row.get("objectives", {}).items():
            cov = row.get("candidate_counts", {}).get(gen, 0) / n_s
            records.append({
                "generator":    gen,
                "combo_id":     row.get("combo_id", ""),
                "fold":         fold,
                "outcome":      obj.get("outcome"),
                "proximity":    obj.get("proximity"),
                "sparsity":     obj.get("sparsity"),
                "plausibility": obj.get("plausibility"),
                "n_samples":    n_s,
                "coverage":     cov,
                "text_encoder": tc.get("encoder"),
                "text_metric":  tc.get("metric"),
                "img_encoder":  ic.get("encoder"),
                "img_metric":   ic.get("metric"),
                "tab_metric":   tab_m,
                "ts_metric":    ts_m,
                "runtime_sec":  row.get("runtime_sec"),
            })
        return records

    all_records: List[dict] = []
    if cfg["fold_structure"]:
        folds = cfg.get("strategy_folds", {}).get(strategy, [])
        for fold in folds:
            fold_run_name = f"{run_name}_fold_{fold}"
            p = cfg["data_dir"] / "ablation_runs" / f"fold_{fold}" / fold_run_name / "summary.jsonl"
            for row in _load_jsonl(p):
                all_records.extend(_row_to_records(row, fold=fold))
    else:
        p = cfg["data_dir"] / "ablation_runs" / run_name / "summary.jsonl"
        if not p.exists():
            p = cfg["data_dir"] / "ablation_runs" / f"{run_name}_k50" / "summary.jsonl"
        for row in _load_jsonl(p):
            all_records.extend(_row_to_records(row, fold=None))

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Override all metrics from the k-ablation JSON (k=20) when available.
    # That file reflects the latest evaluation code (proximity normalizer, etc.)
    # and supersedes the potentially stale values in summary.jsonl.
    if not cfg["fold_structure"]:
        k50_run = f"{run_name}_k50"
        kab_path = cfg["data_dir"] / "ablation_runs" / k50_run / _K_ABLATION_FNAME
        if kab_path.exists():
            try:
                kab = json.loads(kab_path.read_text())
                k20_obj = kab.get("results", {}).get("20", {}).get("objectives", {})
                if k20_obj:
                    for metric in METRICS:
                        metric_map = {gen: obj[metric] for gen, obj in k20_obj.items()
                                      if metric in obj}
                        df[metric] = df.apply(
                            lambda r, m=metric, mm=metric_map: mm.get(r["generator"], r[m]),
                            axis=1,
                        )
            except Exception:
                pass

    # For sepsis: average across folds per (generator, combo_id)
    if cfg["fold_structure"]:
        grp = ["generator", "combo_id"]
        num = METRICS + ["coverage", "n_samples", "runtime_sec"]
        df = df.groupby(grp)[num].mean().reset_index()

    return df


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


# ---------------------------------------------------------------------------
# Figure 1 — performance comparison
# ---------------------------------------------------------------------------

def _fig1_data(dataset: str, strategy: str):
    """Returns (crane_df, crane_gens, color_map)."""
    crane = load_crane(dataset, strategy)

    crane_means: Dict[str, Dict[str, float]] = {}
    if not crane.empty:
        for gen, grp in crane.groupby("generator"):
            crane_means[gen] = {m: grp[m].mean() for m in METRICS}

    crane_gens = [g for g in CRANE_GENERATORS if g in crane_means]
    cmap = _build_color_map(crane_gens)
    return crane, crane_gens, cmap


def plot_fig1_varA_boxplot(dataset: str, strategy: str, outdir: Path) -> None:
    """Variant A boxplot: CRANE generators as boxplots (distribution over combos)."""
    crane, crane_gens, cmap = _fig1_data(dataset, strategy)
    if crane.empty or not crane_gens:
        return

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    for ax, metric in zip(axes, METRICS):
        box_data = []
        positions = []
        labels = []
        for pos, gen in enumerate(crane_gens):
            vals = crane[crane["generator"] == gen][metric].dropna().values
            if len(vals):
                box_data.append((pos, vals, cmap[gen]))
                positions.append(pos)
                labels.append(_glabel(gen))

        if box_data:
            bplot = ax.boxplot(
                [d[1] for d in box_data],
                positions=[d[0] for d in box_data],
                widths=0.6,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
            )
            for patch, (_, _, col) in zip(bplot["boxes"], box_data):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)

        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = outdir / f"varA_boxplot_{strategy}_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_fig1_varA_barplot(dataset: str, strategy: str, outdir: Path) -> None:
    """Variant A barplot: CRANE generators as mean±std bars."""
    crane, crane_gens, cmap = _fig1_data(dataset, strategy)
    if crane.empty or not crane_gens:
        return

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    for ax, metric in zip(axes, METRICS):
        xs, heights, errs, colors, labels = [], [], [], [], []
        for pos, gen in enumerate(crane_gens):
            vals = crane[crane["generator"] == gen][metric].dropna().values
            if len(vals):
                xs.append(pos); heights.append(np.nanmean(vals))
                errs.append(np.nanstd(vals)); colors.append(cmap[gen])
                labels.append(_glabel(gen))

        ax.bar(xs, heights, color=colors, alpha=0.8, width=0.7, edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, heights, yerr=errs, fmt="none", color="black", capsize=3, linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = outdir / f"varA_barplot_{strategy}_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_fig1_varB_barplot(dataset: str, strategy: str, outdir: Path) -> None:
    """Variant B barplot: each CRANE generator averaged to a single value per metric."""
    crane, crane_gens, cmap = _fig1_data(dataset, strategy)
    if crane.empty or not crane_gens:
        return

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    for ax, metric in zip(axes, METRICS):
        xs, heights, colors, labels = [], [], [], []
        for pos, gen in enumerate(crane_gens):
            vals = crane[crane["generator"] == gen][metric].dropna().values
            if len(vals):
                xs.append(pos); heights.append(np.nanmean(vals))
                colors.append(cmap[gen]); labels.append(gen)

        ax.bar(xs, heights, color=colors, alpha=0.8, width=0.7, edgecolor="black", linewidth=0.4)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = outdir / f"varB_barplot_{strategy}_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1 global — averaged across all datasets
# ---------------------------------------------------------------------------

def _load_all_crane(datasets: List[str], strategies: List[str]) -> pd.DataFrame:
    """Pool all crane rows across the given datasets and strategies."""
    frames = []
    for ds in datasets:
        for strat in strategies:
            df = load_crane(ds, strat)
            if not df.empty:
                df = df.copy()
                df["dataset"]  = ds
                df["strategy"] = strat
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_fig1_global_boxplot(datasets: List[str], strategies: List[str], outdir: Path) -> None:
    """Boxplot of each CRANE generator's metric distribution across all datasets × strategies."""
    all_crane = _load_all_crane(datasets, strategies)
    if all_crane.empty:
        return

    crane_gens = [g for g in CRANE_GENERATORS if g in all_crane["generator"].unique()]
    if not crane_gens:
        return
    cmap = _build_color_map(crane_gens)

    fig, axes_grid = plt.subplots(2, 2, figsize=(6, 4.3))
    axes = axes_grid.flatten()

    for ax, metric in zip(axes, METRICS):
        box_data  = []
        positions = []
        for pos, gen in enumerate(crane_gens):
            vals = all_crane[all_crane["generator"] == gen][metric].dropna().values
            if len(vals):
                box_data.append((pos, vals, cmap[gen]))
                positions.append(pos)

        if box_data:
            bplot = ax.boxplot(
                [d[1] for d in box_data],
                positions=[d[0] for d in box_data],
                widths=0.6,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            for patch, (_, _, col) in zip(bplot["boxes"], box_data):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels([""] * len(positions))
        ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(facecolor=cmap[gen], alpha=0.7, edgecolor="black",
                               linewidth=0.4, label=_glabel(gen))
               for gen in crane_gens]
    fig.legend(handles=handles, fontsize=11, loc="center left",
               bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=True)
    plt.tight_layout()
    path = outdir / "global_boxplot.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_fig1_global_barplot(datasets: List[str], strategies: List[str], outdir: Path) -> None:
    """Bar chart (mean ± std) per CRANE generator, averaged across all datasets × strategies."""
    all_crane = _load_all_crane(datasets, strategies)
    if all_crane.empty:
        return

    crane_gens = [g for g in CRANE_GENERATORS if g in all_crane["generator"].unique()]
    if not crane_gens:
        return
    cmap = _build_color_map(crane_gens)

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    for ax, metric in zip(axes, METRICS):
        xs, heights, errs, colors, labels = [], [], [], [], []
        for pos, gen in enumerate(crane_gens):
            vals = all_crane[all_crane["generator"] == gen][metric].dropna().values
            if len(vals):
                xs.append(pos); heights.append(np.nanmean(vals))
                errs.append(np.nanstd(vals)); colors.append(cmap[gen])
                labels.append(_glabel(gen))

        ax.bar(xs, heights, color=colors, alpha=0.8, width=0.7, edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, heights, yerr=errs, fmt="none", color="black", capsize=3, linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = outdir / "global_barplot.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 6 global — Pareto across all datasets and fusion strategies
# ---------------------------------------------------------------------------

def plot_fig6_global_pareto(datasets: List[str], strategies: List[str], outdir: Path) -> None:
    """Pareto scatter with each generator's mean across all datasets × strategies."""
    all_crane = _load_all_crane(datasets, strategies)
    if all_crane.empty:
        return

    # Per-generator grand mean across all combos, datasets, strategies
    crane_agg: Dict[str, Dict[str, float]] = {}
    for gen, grp in all_crane.groupby("generator"):
        crane_agg[gen] = {m: grp[m].mean() for m in METRICS}

    if not crane_agg:
        return

    cmap  = _build_color_map(list(crane_agg.keys()))
    pairs = [
        ("outcome",   "plausibility", "Outcome ↓",   "Plausibility (↓)"),
        ("proximity", "plausibility", "Proximity (↓)", "Plausibility (↓)"),
        ("sparsity",  "plausibility", "Sparsity (↓)",  "Plausibility (↓)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8, 2))

    for ax, (mx, my, lx, ly) in zip(axes, pairs):
        pts_list, names_list = [], []
        for name, d in crane_agg.items():
            vx = d.get(mx); vy = d.get(my)
            if vx is None or vy is None:
                continue
            if np.isnan(float(vx)) or np.isnan(float(vy)):
                continue
            pts_list.append([float(vx), float(vy)])
            names_list.append(name)

        if not pts_list:
            continue

        pts = np.array(pts_list)
        pareto_idx = _pareto_front(pts)

        for i, (name, pt) in enumerate(zip(names_list, pts_list)):
            col = cmap.get(name, "#888888")
            on_pareto = i in pareto_idx
            ax.scatter(pt[0], pt[1], color=col, s=80 if on_pareto else 40,
                       marker="o", edgecolors="black",
                       linewidths=1.2 if on_pareto else 0.3,
                       zorder=5 if on_pareto else 3, label=_glabel(name))
            ax.annotate(_glabel(name), (pt[0], pt[1]), fontsize=5,
                        textcoords="offset points", xytext=(4, 2))

        if len(pareto_idx) > 1:
            pf = pts[pareto_idx]
            pf = pf[pf[:, 0].argsort()]
            ax.plot(pf[:, 0], pf[:, 1], "k--", linewidth=0.8, alpha=0.6, label="_Pareto")

        ax.set_xlabel(lx, fontsize=8); ax.set_ylabel(ly, fontsize=8)
        ax.grid(alpha=0.3)

    handles, _labs = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith("_"):
                handles.append(h); _labs.append(l); seen.add(l)
    fig.legend(handles, _labs, loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=min(8, len(handles)), fontsize=6, frameon=True)
    plt.tight_layout(rect=[0, 0.20, 1, 1])
    path = outdir / "global_pareto.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 2 — encoder / distance ablation
# ---------------------------------------------------------------------------

def plot_fig2_ablation(dataset: str, strategy: str, outdir: Path) -> None:
    """Heatmap of metric value per (generator × combo), ordered by combo index."""
    crane = load_crane(dataset, strategy)
    if crane.empty:
        return

    generators = [g for g in CRANE_GENERATORS if g in crane["generator"].unique()]
    if not generators:
        return

    # Build combo label
    def _combo_label(row) -> str:
        parts = []
        if pd.notna(row.get("text_encoder")): parts.append(str(row["text_encoder"]))
        if pd.notna(row.get("text_metric")):  parts.append(str(row["text_metric"]))
        if pd.notna(row.get("img_encoder")):  parts.append(str(row["img_encoder"]))
        if pd.notna(row.get("img_metric")):   parts.append(str(row["img_metric"]))
        if pd.notna(row.get("tab_metric")):   parts.append(f"tab:{row['tab_metric']}")
        if pd.notna(row.get("ts_metric")):    parts.append(f"ts:{row['ts_metric']}")
        return "/".join(parts) or str(row.get("combo_id", ""))

    if "text_encoder" not in crane.columns:
        for col in ["text_encoder","text_metric","img_encoder","img_metric","tab_metric","ts_metric"]:
            crane[col] = None

    crane["combo_label"] = crane.apply(_combo_label, axis=1)
    combos = list(crane["combo_label"].unique())

    n_combos = len(combos)
    n_gen    = len(generators)

    fig, axes = plt.subplots(1, 4, figsize=(8, max(2, n_gen * 0.35 + 0.8)))
    if n_gen == 1 or n_combos == 1:
        axes = np.atleast_1d(axes)

    combo_idx = {c: i for i, c in enumerate(combos)}
    gen_idx   = {g: i for i, g in enumerate(generators)}

    for ax, metric in zip(axes, METRICS):
        mat = np.full((n_gen, n_combos), np.nan)
        for _, row in crane.iterrows():
            g = row["generator"]
            c = row["combo_label"]
            if g in gen_idx and c in combo_idx:
                mat[gen_idx[g], combo_idx[c]] = row[metric]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_yticks(range(n_gen)); ax.set_yticklabels([_glabel(g) for g in generators], fontsize=6)
        ax.set_xticks(range(n_combos))
        ax.set_xticklabels(combos, rotation=90, fontsize=4)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = outdir / f"ablation_{strategy}_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 3 — classifier performance LaTeX table
# ---------------------------------------------------------------------------

def _compute_auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))


# Maps (ds_key, strategy) -> (y_true_path, y_proba_path)
# Sepsis uses per-fold files; other datasets use single files.
_PROBA_FILES = {
    "sepsis": {
        "early":        ("y_true.npy",              "y_proba.npy"),
        "intermediate": ("y_true.npy",              "y_proba_intermediate_rf.npy"),
        "late":         ("y_true.npy",              "y_proba_late_deep.npy"),
    },
    "tweets": {
        "early":        ("y_true_early_fusion_mlp.npy",                  "y_proba_early_fusion_mlp.npy"),
        "intermediate": ("y_true.npy",                                   "y_proba.npy"),
        "late":         ("y_true_late_fusion_tfidf_logreg_nondp.npy",    "y_proba_late_fusion_tfidf_logreg_nondp.npy"),
    },
    "memes": {
        "early":        ("y_true_early_fusion_gbt.npy",  "y_proba_early_fusion_gbt.npy"),
        "intermediate": ("y_true.npy",                   "y_proba.npy"),
        "late":         ("y_true_late_fusion_nondp.npy", "y_proba_late_fusion_nondp.npy"),
    },
    "jobs": {
        "early":        ("y_true_early_fusion_rf.npy",                "y_proba_early_fusion_rf.npy"),
        "intermediate": ("y_true.npy",                                "y_proba.npy"),
        "late":         ("y_true_late_fusion_tfidf_logreg_nondp.npy", "y_proba_late_fusion_tfidf_logreg_nondp.npy"),
    },
}


def _load_auroc(ds_key: str, strategy: str, data_dir: Path) -> Optional[float]:
    files = _PROBA_FILES.get(ds_key, {}).get(strategy)
    if not files:
        return None
    yt_name, yp_name = files
    if ds_key == "sepsis":
        aurocs = []
        for fold in [0, 1, 2, 3, 4]:
            fold_dir = data_dir / f"fold_{fold}"
            yt_p = fold_dir / yt_name
            yp_p = fold_dir / yp_name
            if yt_p.exists() and yp_p.exists():
                aurocs.append(_compute_auroc(np.load(yt_p), np.load(yp_p)))
        return float(np.mean(aurocs)) if aurocs else None
    yt_p = data_dir / yt_name
    yp_p = data_dir / yp_name
    if yt_p.exists() and yp_p.exists():
        return _compute_auroc(np.load(yt_p), np.load(yp_p))
    return None


def generate_classifier_table(outdir: Path) -> None:
    rows_tex = []
    ds_labels = {
        "sepsis": "Sepsis",
        "tweets": "Long COVID Tweets",
        "memes":  "Hateful Memes",
        "jobs":   "Real or Fake Jobs",
    }
    for ds_key in ["sepsis", "tweets", "memes", "jobs"]:
        reg_path = REGISTRY_FILES[ds_key]
        if not reg_path.exists():
            continue
        reg = json.loads(reg_path.read_text())
        best   = reg.get("best_per_strategy", {})
        models = reg.get("models", {})
        label  = ds_labels[ds_key]
        data_dir = DATASETS[ds_key]["data_dir"]

        first = True
        for strat in STRATEGIES:
            model_key = best.get(strat)
            if not model_key or model_key not in models:
                continue
            m = models[model_key]
            f1    = m.get("test_macro_f1")
            acc   = m.get("test_accuracy")
            mtype = m.get("model_type", "—").replace("_", r"\_")
            auroc = _load_auroc(ds_key, strat, data_dir)
            strat_label = STRATEGY_LABELS[strat]

            f1_str    = f"{f1:.4f}"    if f1    is not None else "—"
            acc_str   = f"{acc:.4f}"   if acc   is not None else "—"
            auroc_str = f"{auroc:.4f}" if auroc is not None else "—"

            ds_cell = (r"\multirow{3}{*}{" + label + r"}" if first else "")
            rows_tex.append(
                f"  {ds_cell} & {strat_label} & {mtype} & {f1_str} & {acc_str} & {auroc_str} \\\\"
            )
            first = False
        rows_tex.append(r"  \midrule")

    tex = r"""\begin{table}[ht]
\centering
\caption{Best classifier per fusion strategy and dataset.
         F1 = macro-averaged F1; Acc = test accuracy; AUROC = macro-averaged
         one-vs-rest AUC (binary datasets use standard AUC).
         Sepsis AUROC is averaged across 5 folds.}
\label{tab:classifiers}
\small
\begin{tabular}{lllrrr}
\toprule
Dataset & Fusion Strategy & Model Type & Macro-F1 & Accuracy & AUROC \\
\midrule
""" + "\n".join(rows_tex[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path = outdir / "fig3_classifier_table.tex"
    path.write_text(tex)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 4 — stability across fusion strategies
# ---------------------------------------------------------------------------

def plot_fig4_stability(dataset: str, outdir: Path) -> None:
    """Grouped bar chart: x = CRANE generator, colours = fusion strategy."""
    cfg = DATASETS[dataset]
    strat_colors  = {"early": "#2c7bb6", "intermediate": "#d7191c", "late": "#1a9641"}
    strat_hatches = {"early": "////", "intermediate": "", "late": "\\\\\\\\"}

    strat_crane: Dict[str, pd.DataFrame] = {}
    for strat in STRATEGIES:
        strat_crane[strat] = load_crane(dataset, strat)

    # union of generators present in any strategy
    all_crane_gens: List[str] = []
    for strat, crane in strat_crane.items():
        if not crane.empty:
            for g in CRANE_GENERATORS:
                if g in crane["generator"].unique() and g not in all_crane_gens:
                    all_crane_gens.append(g)

    if not all_crane_gens:
        return

    n_methods = len(all_crane_gens)
    width     = 0.25
    x         = np.arange(n_methods)

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    for ax, metric in zip(axes, METRICS):
        for si, strat in enumerate(STRATEGIES):
            crane = strat_crane[strat]
            heights = []
            for gen in all_crane_gens:
                if not crane.empty:
                    vals = crane[crane["generator"] == gen][metric].dropna().values
                    heights.append(np.nanmean(vals) if len(vals) else np.nan)
                else:
                    heights.append(np.nan)
            offset = (si - 1) * width
            ax.bar(x + offset, heights, width=width,
                   label=STRATEGY_LABELS[strat], color=strat_colors[strat],
                   hatch=strat_hatches[strat],
                   alpha=0.8, edgecolor="black", linewidth=0.3)

        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([_glabel(g) for g in all_crane_gens], rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=3, fontsize=7, frameon=True)
    plt.tight_layout(rect=[0, 0.20, 1, 1])
    path = outdir / f"stability_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 5 — runtime table (LaTeX)
# ---------------------------------------------------------------------------

def generate_runtime_table(outdir: Path) -> None:
    """
    Runtime normalised by n_samples (sec/sample), averaged across all datasets
    and fusion strategies.
    """
    crane_rt: Dict[str, List[float]] = {}
    for ds in DATASETS:
        for strat in STRATEGIES:
            crane = load_crane(ds, strat)
            if crane.empty or "runtime_sec" not in crane.columns:
                continue
            for _, grp in crane.groupby("generator"):
                gen = grp["generator"].iloc[0]
                ns  = grp["n_samples"].mean()
                rt  = grp["runtime_sec"].dropna()
                if len(rt) and ns > 0:
                    crane_rt.setdefault(gen, []).extend((rt / ns).tolist())

    rows_tex = []
    for g in CRANE_GENERATORS:
        if g in crane_rt:
            val = float(np.mean(crane_rt[g]))
            highlight = r"\textbf{" + g + "}" if g == "Combined" else g.replace("_", r"\_")
            rows_tex.append(f"  {highlight} & {val:.3f} \\\\")

    tex = r"""\begin{table}[ht]
\centering
\caption{Average runtime per sample (seconds/sample), averaged across all datasets
         and fusion strategies. Runtime is the mean per-combo wall-clock time
         divided by number of samples.}
\label{tab:runtime}
\small
\begin{tabular}{lr}
\toprule
Generator & Sec / sample \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path = outdir / "fig5_runtime_table.tex"
    path.write_text(tex)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 6 — scatter + Pareto frontier plots
# ---------------------------------------------------------------------------

def _pareto_front(pts: np.ndarray) -> np.ndarray:
    """Indices of Pareto-optimal rows (minimise all columns)."""
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        dominated[i] = False
        is_pareto[dominated] = False
    return np.where(is_pareto)[0]


def plot_fig6_dataset_pareto(dataset: str, strategies: List[str], outdir: Path) -> None:
    """Pareto scatter for one dataset, with each generator averaged across all fusion strategies."""
    cfg   = DATASETS[dataset]
    pairs = [
        ("outcome",   "plausibility", "Outcome ↓",   "Plausibility (↓)"),
        ("proximity", "plausibility", "Proximity (↓)", "Plausibility (↓)"),
        ("sparsity",  "plausibility", "Sparsity (↓)",  "Plausibility (↓)"),
    ]

    frames = []
    for strat in strategies:
        df = load_crane(dataset, strat)
        if not df.empty:
            frames.append(df)
    if not frames:
        return

    all_crane = pd.concat(frames, ignore_index=True)

    crane_agg: Dict[str, Dict[str, float]] = {}
    for gen, grp in all_crane.groupby("generator"):
        crane_agg[gen] = {m: grp[m].mean() for m in METRICS}

    if not crane_agg:
        return

    cmap = _build_color_map(list(crane_agg.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(8, 2))

    for ax, (mx, my, lx, ly) in zip(axes, pairs):
        pts_list, names_list = [], []
        for name, d in crane_agg.items():
            vx = d.get(mx); vy = d.get(my)
            if vx is None or vy is None:
                continue
            if np.isnan(float(vx)) or np.isnan(float(vy)):
                continue
            pts_list.append([float(vx), float(vy)])
            names_list.append(name)

        if not pts_list:
            continue

        pts = np.array(pts_list)
        pareto_idx = _pareto_front(pts)

        for i, (name, pt) in enumerate(zip(names_list, pts_list)):
            col = cmap.get(name, "#888888")
            on_pareto = i in pareto_idx
            ax.scatter(pt[0], pt[1], color=col, s=80 if on_pareto else 40,
                       marker="o", edgecolors="black",
                       linewidths=1.2 if on_pareto else 0.3,
                       zorder=5 if on_pareto else 3, label=_glabel(name))
            ax.annotate(_glabel(name), (pt[0], pt[1]), fontsize=5,
                        textcoords="offset points", xytext=(4, 2))

        if len(pareto_idx) > 1:
            pf = pts[pareto_idx]
            pf = pf[pf[:, 0].argsort()]
            ax.plot(pf[:, 0], pf[:, 1], "k--", linewidth=0.8, alpha=0.6, label="_Pareto")

        ax.set_xlabel(lx, fontsize=8); ax.set_ylabel(ly, fontsize=8)
        ax.grid(alpha=0.3)

    handles, _labs = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith("_"):
                handles.append(h); _labs.append(l); seen.add(l)
    fig.legend(handles, _labs, loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=min(8, len(handles)), fontsize=6, frameon=True)
    plt.tight_layout(rect=[0, 0.20, 1, 1])
    path = outdir / f"pareto_avg_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_fig6_pareto(dataset: str, outdir: Path) -> None:
    """Scatter + Pareto frontier for each fusion strategy, CRANE generators only."""
    cfg   = DATASETS[dataset]
    pairs = [
        ("outcome",   "plausibility", "Outcome ↓",   "Plausibility (↓)"),
        ("proximity", "plausibility", "Proximity (↓)", "Plausibility (↓)"),
        ("sparsity",  "plausibility", "Sparsity (↓)",  "Plausibility (↓)"),
    ]

    for strat in STRATEGIES:
        crane = load_crane(dataset, strat)
        if crane.empty:
            continue

        crane_agg: Dict[str, Dict[str, float]] = {}
        for gen, grp in crane.groupby("generator"):
            crane_agg[gen] = {m: grp[m].mean() for m in METRICS}

        if not crane_agg:
            continue

        cmap = _build_color_map(list(crane_agg.keys()))

        fig, axes = plt.subplots(1, 3, figsize=(8, 2))

        for ax, (mx, my, lx, ly) in zip(axes, pairs):
            pts_list, names_list = [], []
            for name, d in crane_agg.items():
                vx = d.get(mx); vy = d.get(my)
                if vx is None or vy is None:
                    continue
                if np.isnan(float(vx)) or np.isnan(float(vy)):
                    continue
                pts_list.append([float(vx), float(vy)])
                names_list.append(name)

            if not pts_list:
                continue

            pts = np.array(pts_list)
            pareto_idx = _pareto_front(pts)

            for i, (name, pt) in enumerate(zip(names_list, pts_list)):
                col = cmap.get(name, "#888888")
                on_pareto = i in pareto_idx
                ax.scatter(pt[0], pt[1], color=col, s=80 if on_pareto else 40,
                           marker="o", edgecolors="black",
                           linewidths=1.2 if on_pareto else 0.3,
                           zorder=5 if on_pareto else 3, label=_glabel(name))
                ax.annotate(_glabel(name), (pt[0], pt[1]), fontsize=5,
                            textcoords="offset points", xytext=(4, 2))

            if len(pareto_idx) > 1:
                pf = pts[pareto_idx]
                pf = pf[pf[:, 0].argsort()]
                ax.plot(pf[:, 0], pf[:, 1], "k--", linewidth=0.8, alpha=0.6, label="_Pareto")

            ax.set_xlabel(lx, fontsize=8); ax.set_ylabel(ly, fontsize=8)
            ax.grid(alpha=0.3)

        handles, _labs = [], []
        seen = set()
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen and not l.startswith("_"):
                    handles.append(h); _labs.append(l); seen.add(l)
        fig.legend(handles, _labs, loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=min(8, len(handles)), fontsize=6, frameon=True)
        plt.tight_layout(rect=[0, 0.20, 1, 1])
        path = outdir / f"pareto_{strat}_{dataset}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 7 — CSV summary
# ---------------------------------------------------------------------------

def export_fig7_csv(outdir: Path) -> None:
    """
    One row per (dataset, strategy, generator).
    Columns: dataset, strategy, generator, outcome, proximity, sparsity, plausibility,
             coverage_candidates_per_sample, flip_rate_pct (= outcome * 100).
    Note: 'outcome' = mean validity rate (fraction of generated CFs that flip
          the model prediction). 'coverage' = mean candidates per sample.
    """
    records = []
    for ds in DATASETS:
        for strat in STRATEGIES:
            crane = load_crane(ds, strat)
            if crane.empty:
                continue
            for gen, grp in crane.groupby("generator"):
                records.append({
                    "dataset":       ds,
                    "strategy":      strat,
                    "generator":     gen,
                    "outcome":       round(grp["outcome"].mean(), 4),
                    "proximity":     round(grp["proximity"].mean(), 4),
                    "sparsity":      round(grp["sparsity"].mean(), 4),
                    "plausibility":  round(grp["plausibility"].mean(), 4),
                    "coverage_candidates_per_sample": round(grp["coverage"].mean(), 4),
                    "flip_rate_pct": round(grp["outcome"].mean() * 100, 2),
                })

    df = pd.DataFrame(records)
    path = outdir / "fig7_metrics.csv"
    df.to_csv(path, index=False)
    print(f"  saved {path.name}")
    print(f"  NOTE: 'outcome' = mean validity rate (fraction of CFs that flip prediction).")


# ---------------------------------------------------------------------------
# Figure K — k-ablation line plots
# ---------------------------------------------------------------------------

def load_k_ablation(dataset: str, strategy: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Load k-ablation results for one (dataset, strategy).

    Returns {k: {generator: {metric: value}}}, averaged across folds for sepsis.
    Empty dict if no files found.
    """
    cfg      = DATASETS[dataset]
    run_name = cfg["strategy_run"].get(strategy)
    if not run_name:
        return {}

    k50_run = f"{run_name}_k50"

    # accumulator: k -> gen -> metric -> [values across folds]
    acc: Dict[int, Dict[str, Dict[str, list]]] = {}

    def _ingest(path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        for k_str, entry in data.get("results", {}).items():
            k = int(k_str)
            for gen, obj in entry.get("objectives", {}).items():
                for m in METRICS:
                    v = obj.get(m)
                    if v is not None and v == v:  # skip nan
                        acc.setdefault(k, {}).setdefault(gen, {}).setdefault(m, []).append(float(v))

    if cfg["fold_structure"]:
        folds = cfg.get("strategy_folds", {}).get(strategy, [])
        for fold in folds:
            fold_run = f"{k50_run}_fold_{fold}"
            p = cfg["data_dir"] / "ablation_runs" / f"fold_{fold}" / fold_run / _K_ABLATION_FNAME
            _ingest(p)
    else:
        p = cfg["data_dir"] / "ablation_runs" / k50_run / _K_ABLATION_FNAME
        _ingest(p)

    # average across folds
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k, gens in acc.items():
        result[k] = {}
        for gen, metrics in gens.items():
            result[k][gen] = {m: float(np.nanmean(vals)) for m, vals in metrics.items()}
    return result


_UNIMODAL_GENERATORS = {"Tabular", "TS[lab_exams_vitals]", "Text", "Image"}


def _plot_k_lines(k_data: Dict[int, Dict[str, Dict[str, float]]],
                  title: str, path: Path,
                  figsize: Tuple[float, float] = (8, 2),
                  legend_fontsize: int = 6) -> None:
    """Shared renderer: k on x-axis (log scale), one line per generator, 4 metric subplots.
    Unimodal generators: dashed lines. Multimodal generators: solid lines.
    """
    if not k_data:
        return

    k_vals = sorted(k_data.keys())
    all_gens = [g for g in CRANE_GENERATORS
                if any(g in k_data[k] for k in k_vals)]
    if not all_gens:
        return

    cmap = _build_color_map(all_gens)
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    for ax, metric in zip(axes, METRICS):
        for gen in all_gens:
            ys = [k_data[k].get(gen, {}).get(metric, np.nan) for k in k_vals]
            if all(np.isnan(y) for y in ys):
                continue
            ls = "--" if gen in _UNIMODAL_GENERATORS else "-"
            ax.plot(k_vals, ys, marker="o", markersize=4, linewidth=1.4,
                    linestyle=ls, color=cmap[gen], label=_glabel(gen))
        ax.set_xscale("log")
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals], fontsize=7)
        ax.set_xlabel("k", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(alpha=0.3)

    import matplotlib.lines as mlines
    handles = []
    uni  = [g for g in all_gens if g in _UNIMODAL_GENERATORS]
    multi = [g for g in all_gens if g not in _UNIMODAL_GENERATORS]
    if uni:
        for g in uni:
            handles.append(mlines.Line2D([], [], color=cmap[g], linewidth=1.4,
                                         linestyle="--", marker="o", markersize=4, label=_glabel(g)))
    if multi:
        for g in multi:
            handles.append(mlines.Line2D([], [], color=cmap[g], linewidth=1.4,
                                         linestyle="-", marker="o", markersize=4, label=_glabel(g)))
    fig.legend(handles=handles, fontsize=legend_fontsize, loc="lower center",
               bbox_to_anchor=(0.5, 0), ncol=8, frameon=True)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_figK_per_strategy(dataset: str, strategy: str, outdir: Path) -> None:
    """Line plot for one (dataset, strategy)."""
    k_data = load_k_ablation(dataset, strategy)
    cfg    = DATASETS[dataset]
    title  = f"{cfg['label']} — {STRATEGY_LABELS[strategy]}: effect of k"
    path   = outdir / f"k_ablation_{strategy}_{dataset}.pdf"
    _plot_k_lines(k_data, title, path)


def plot_figK_dataset(dataset: str, strategies: List[str], outdir: Path) -> None:
    """Line plot averaged across all fusion strategies for one dataset."""
    cfg = DATASETS[dataset]
    # merge across strategies: k -> gen -> metric -> [values]
    acc: Dict[int, Dict[str, Dict[str, list]]] = {}
    for strat in strategies:
        for k, gens in load_k_ablation(dataset, strat).items():
            for gen, metrics in gens.items():
                for m, v in metrics.items():
                    acc.setdefault(k, {}).setdefault(gen, {}).setdefault(m, []).append(v)
    k_data = {k: {gen: {m: float(np.nanmean(vs)) for m, vs in metrics.items()}
                  for gen, metrics in gens.items()}
              for k, gens in acc.items()}
    title = f"{cfg['label']} — averaged across fusion strategies: effect of k"
    path  = outdir / f"k_ablation_avg_{dataset}.pdf"
    _plot_k_lines(k_data, title, path)


def plot_figK_global(datasets: List[str], strategies: List[str], outdir: Path) -> None:
    """Line plot averaged across all datasets and fusion strategies."""
    acc: Dict[int, Dict[str, Dict[str, list]]] = {}
    for ds in datasets:
        for strat in strategies:
            for k, gens in load_k_ablation(ds, strat).items():
                for gen, metrics in gens.items():
                    for m, v in metrics.items():
                        acc.setdefault(k, {}).setdefault(gen, {}).setdefault(m, []).append(v)
    k_data = {k: {gen: {m: float(np.nanmean(vs)) for m, vs in metrics.items()}
                  for gen, metrics in gens.items()}
              for k, gens in acc.items()}
    title = "All datasets & strategies — effect of k"
    path  = outdir / "k_ablation_global.pdf"
    _plot_k_lines(k_data, title, path, figsize=(6, 1.5), legend_fontsize=8)


def plot_figK_global_2x2(datasets: List[str], strategies: List[str], outdir: Path) -> None:
    """2×2 line plot averaged across all datasets and fusion strategies."""
    acc: Dict[int, Dict[str, Dict[str, list]]] = {}
    for ds in datasets:
        for strat in strategies:
            for k, gens in load_k_ablation(ds, strat).items():
                for gen, metrics in gens.items():
                    for m, v in metrics.items():
                        acc.setdefault(k, {}).setdefault(gen, {}).setdefault(m, []).append(v)
    if not acc:
        return
    k_data = {k: {gen: {m: float(np.nanmean(vs)) for m, vs in metrics.items()}
                  for gen, metrics in gens.items()}
              for k, gens in acc.items()}

    k_vals  = sorted(k_data.keys())
    all_gens = [g for g in CRANE_GENERATORS if any(g in k_data[k] for k in k_vals)]
    if not all_gens:
        return

    cmap = _build_color_map(all_gens)
    fig, axes_grid = plt.subplots(2, 2, figsize=(5, 3.4))
    axes = axes_grid.flatten()

    for ax, metric in zip(axes, METRICS):
        for gen in all_gens:
            ys = [k_data[k].get(gen, {}).get(metric, np.nan) for k in k_vals]
            if all(np.isnan(y) for y in ys):
                continue
            ls = "--" if gen in _UNIMODAL_GENERATORS else "-"
            ax.plot(k_vals, ys, marker="o", markersize=4, linewidth=1.4,
                    linestyle=ls, color=cmap[gen], label=_glabel(gen))
        ax.set_xscale("log")
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals], fontsize=7)
        ax.set_xlabel("k", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(alpha=0.3)

    import matplotlib.lines as mlines
    handles = []
    for g in [g for g in all_gens if g in _UNIMODAL_GENERATORS]:
        handles.append(mlines.Line2D([], [], color=cmap[g], linewidth=1.4,
                                     linestyle="--", marker="o", markersize=4, label=_glabel(g)))
    for g in [g for g in all_gens if g not in _UNIMODAL_GENERATORS]:
        handles.append(mlines.Line2D([], [], color=cmap[g], linewidth=1.4,
                                     linestyle="-", marker="o", markersize=4, label=_glabel(g)))
    fig.legend(handles=handles, fontsize=8, loc="center left",
               bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=True)
    plt.tight_layout()
    path = outdir / "k_ablation_global_2x2.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--skip", nargs="+", default=[],
                    help="skip figures: fig1 fig2 fig3 fig4 fig5 fig6 fig7 figK")
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "fig1").mkdir(parents=True, exist_ok=True)
    (out / "fig2").mkdir(parents=True, exist_ok=True)
    (out / "fig4").mkdir(parents=True, exist_ok=True)
    (out / "fig6").mkdir(parents=True, exist_ok=True)
    (out / "figK").mkdir(parents=True, exist_ok=True)

    skip = set(args.skip)

    # Figure 1
    if "fig1" not in skip:
        print("\n=== Figure 1: performance comparison ===")
        for strat in args.strategies:
            for ds in args.datasets:
                plot_fig1_varA_boxplot(ds, strat, out / "fig1")
                plot_fig1_varA_barplot(ds, strat, out / "fig1")
                plot_fig1_varB_barplot(ds, strat, out / "fig1")
        print("  --- global (all datasets) ---")
        plot_fig1_global_boxplot(args.datasets, args.strategies, out / "fig1")
        plot_fig1_global_barplot(args.datasets, args.strategies, out / "fig1")

    # Figure 2
    if "fig2" not in skip:
        print("\n=== Figure 2: encoder/distance ablation ===")
        for strat in args.strategies:
            for ds in args.datasets:
                plot_fig2_ablation(ds, strat, out / "fig2")

    # Figure 3
    if "fig3" not in skip:
        print("\n=== Figure 3: classifier table (LaTeX) ===")
        generate_classifier_table(out)

    # Figure 4
    if "fig4" not in skip:
        print("\n=== Figure 4: stability across fusion strategies ===")
        for ds in args.datasets:
            plot_fig4_stability(ds, out / "fig4")

    # Figure 5
    if "fig5" not in skip:
        print("\n=== Figure 5: runtime table (LaTeX) ===")
        generate_runtime_table(out)

    # Figure 6
    if "fig6" not in skip:
        print("\n=== Figure 6: scatter + Pareto ===")
        for ds in args.datasets:
            plot_fig6_pareto(ds, out / "fig6")
        print("  --- dataset average (all strategies) ---")
        for ds in args.datasets:
            plot_fig6_dataset_pareto(ds, args.strategies, out / "fig6")
        print("  --- global (all datasets & strategies) ---")
        plot_fig6_global_pareto(args.datasets, args.strategies, out / "fig6")

    # Figure 7
    if "fig7" not in skip:
        print("\n=== Figure 7: metrics CSV ===")
        export_fig7_csv(out)

    # Figure K — k ablation
    if "figK" not in skip:
        print("\n=== Figure K: effect of k ===")
        for strat in args.strategies:
            for ds in args.datasets:
                plot_figK_per_strategy(ds, strat, out / "figK")
        print("  --- dataset average (all strategies) ---")
        for ds in args.datasets:
            plot_figK_dataset(ds, args.strategies, out / "figK")
        print("  --- global (all datasets & strategies) ---")
        plot_figK_global(args.datasets, args.strategies, out / "figK")
        plot_figK_global_2x2(args.datasets, args.strategies, out / "figK")

    print(f"\nAll outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
