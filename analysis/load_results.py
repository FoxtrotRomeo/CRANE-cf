"""
analysis/load_results.py
========================
Load and aggregate ablation results from all project experiments.

Discovery
---------
The loader scans subdirectories of a root path (default: repo root) for the
pattern ``<project>/data/ablation_runs/<run>/summary.json``.  Any directory
that matches is treated as an experiment, so adding a new project requires no
changes here — only a matching directory layout.

Optional per-project metadata
------------------------------
Place a file ``<project>/paper_config.json`` to override the auto-inferred
display name and add task/modality labels::

    {
        "name":       "Long COVID Tweets",
        "task":       "sadness → joy",
        "modalities": ["tabular", "text"]
    }

All fields are optional.  Without the file the project directory name is used.

Typical usage
-------------
    from analysis.load_results import load_all, pivot_objectives, best_combos

    df = load_all()                        # all projects, latest run per project
    df = load_all(latest_only=False)       # every run ever recorded

    # Mean objectives per (dataset, generator) across all combos
    print(df.groupby(["dataset", "generator"])[OBJECTIVES].mean())

    # Pivot: datasets as columns, generators as rows, for one objective
    print(pivot_objectives(df, "proximity"))

    # Best combo per (dataset, generator) ranked by proximity (lower = better)
    print(best_combos(df, rank_by="proximity", ascending=True))
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECTIVES = ["outcome", "proximity", "sparsity", "plausibility"]

# Repo root = parent of this file's directory
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _load_project_config(project_dir: Path) -> dict:
    """Load optional paper_config.json from a project directory."""
    cfg_path = project_dir / "paper_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def discover_experiments(root: Optional[Path | str] = None) -> list[dict]:
    """Return one entry per project that has at least one complete ablation run.

    Each entry is the full ``paper_config.json`` dict (if present) merged with
    auto-detected fields::

        {
            "project_dir": Path,
            "name":        str,          # display name
            "task":        str,          # e.g. "sadness → joy"
            "domain":      str,
            "language":    str,
            "modalities":  list[str],
            "dataset":     dict,         # source, n_train, n_test, classes …
            "model":       dict,         # architecture, performance …
            "counterfactual_experiment": dict,
            "runs":        list[Path],   # sorted by timestamp, newest last
        }

    Parameters
    ----------
    root:
        Directory to search in.  Defaults to the repository root.
    """
    root = Path(root) if root else _REPO_ROOT
    experiments = []

    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        ablation_root = candidate / "data" / "ablation_runs"
        if not ablation_root.is_dir():
            continue

        run_dirs = sorted(
            p for p in ablation_root.iterdir()
            if p.is_dir() and (p / "summary.json").exists()
        )
        if not run_dirs:
            continue

        cfg = _load_project_config(candidate)
        experiments.append({
            "project_dir": candidate,
            "name":        cfg.get("name",       candidate.name),
            "task":        cfg.get("task",        ""),
            "domain":      cfg.get("domain",      ""),
            "language":    cfg.get("language",    ""),
            "modalities":  cfg.get("modalities",  []),
            "dataset":     cfg.get("dataset",     {}),
            "tabular_features": cfg.get("tabular_features", {}),
            "text_features":    cfg.get("text_features",    []),
            "image_features":   cfg.get("image_features",   {}),
            "model":       cfg.get("model",       {}),
            "counterfactual_experiment": cfg.get("counterfactual_experiment", {}),
            "runs":        run_dirs,
        })

    return experiments


# ---------------------------------------------------------------------------
# Single-run loader
# ---------------------------------------------------------------------------

def _flatten_tab_metrics(tab_metrics: dict | None) -> str:
    """Return a single string for the tabular metric config."""
    if not tab_metrics:
        return ""
    # Most runs have one primary metric; join multiples with "+"
    return "+".join(
        f"{k}:{v}" if k != "__primary__" else v
        for k, v in tab_metrics.items()
    )


def _flatten_ts_metrics(ts_metrics: dict | None) -> str:
    """Return a single string for the TS metric config."""
    if not ts_metrics:
        return ""
    return "+".join(f"{k}:{v['metric']}" for k, v in ts_metrics.items())


def load_run(run_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load one ablation run directory into a flat DataFrame.

    Each row corresponds to one (combo, generator) pair.  Only combos with
    ``status == "ok"`` are included.

    Parameters
    ----------
    run_dir:
        Path to the run directory containing ``summary.json``.
    dataset_name:
        Display name for the dataset (used as the ``dataset`` column value).
    """
    summary_path = run_dir / "summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    run_id = run_dir.name
    rows = []

    for combo in summary.get("rows", []):
        if combo.get("status") != "ok":
            continue

        text_cfg   = combo.get("text_config")  or {}
        image_cfg  = combo.get("image_config") or {}

        combo_meta = {
            "dataset":       dataset_name,
            "run_id":        run_id,
            "combo_id":      combo["combo_id"],
            "tab_metric":    _flatten_tab_metrics(combo.get("tab_metrics")),
            "ts_metric":     _flatten_ts_metrics(combo.get("ts_metrics")),
            "text_encoder":  text_cfg.get("encoder", ""),
            "text_metric":   text_cfg.get("metric", ""),
            "image_encoder": image_cfg.get("encoder", ""),
            "image_metric":  image_cfg.get("metric", ""),
            "n_samples":     combo.get("n_samples", np.nan),
            "runtime_sec":   combo.get("runtime_sec", np.nan),
        }

        objectives  = combo.get("objectives", {})
        cand_counts = combo.get("candidate_counts", {})

        for generator, obj in objectives.items():
            rows.append({
                **combo_meta,
                "generator":       generator,
                "candidate_count": cand_counts.get(generator, np.nan),
                "outcome":         obj.get("outcome",     np.nan),
                "proximity":       obj.get("proximity",   np.nan),
                "sparsity":        obj.get("sparsity",    np.nan),
                "plausibility":    obj.get("plausibility",np.nan),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Enforce numeric types
    df[OBJECTIVES] = df[OBJECTIVES].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Multi-project loader
# ---------------------------------------------------------------------------

def load_all(
    root: Optional[Path | str] = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Load ablation results from all discovered projects.

    Parameters
    ----------
    root:
        Repository root to search.  Defaults to the repo root.
    latest_only:
        When True (default), load only the most recent complete run per
        project.  Set to False to load every run ever recorded.

    Returns
    -------
    pandas.DataFrame with columns:
        dataset, run_id, combo_id,
        tab_metric, ts_metric, text_encoder, text_metric,
        image_encoder, image_metric,
        generator, candidate_count, n_samples, runtime_sec,
        outcome, proximity, sparsity, plausibility
    """
    experiments = discover_experiments(root)
    if not experiments:
        warnings.warn("No experiments found. Check that summary.json files exist.")
        return pd.DataFrame()

    frames = []
    for exp in experiments:
        runs = [exp["runs"][-1]] if latest_only else exp["runs"]
        for run_dir in runs:
            df = load_run(run_dir, dataset_name=exp["name"])
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def pivot_objectives(
    df: pd.DataFrame,
    objective: str,
    index: str = "generator",
    columns: str = "dataset",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Pivot mean objective scores with generators as rows and datasets as columns.

    Parameters
    ----------
    df:        Output of ``load_all()``.
    objective: One of ``"outcome"``, ``"proximity"``, ``"sparsity"``,
               ``"plausibility"``.
    index:     Row grouping (default: ``"generator"``).
    columns:   Column grouping (default: ``"dataset"``).
    aggfunc:   Aggregation function name understood by ``pivot_table``.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    return df.pivot_table(
        values=objective,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
    ).round(4)


def best_combos(
    df: pd.DataFrame,
    rank_by: str = "proximity",
    ascending: bool = True,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Return the best combo per (dataset, generator) for a given objective.

    Parameters
    ----------
    df:        Output of ``load_all()``.
    rank_by:   Objective column to rank by.
    ascending: True for objectives where lower is better (proximity, sparsity);
               False for objectives where higher is better (plausibility,
               outcome).
    group_by:  Columns to group by.  Defaults to ``["dataset", "generator"]``.

    Returns
    -------
    DataFrame with one row per group containing the combo that achieved the
    best value of ``rank_by``, plus all other objective columns.
    """
    if rank_by not in OBJECTIVES:
        raise ValueError(f"rank_by must be one of {OBJECTIVES}, got {rank_by!r}")

    group_by = group_by or ["dataset", "generator"]
    config_cols = ["tab_metric", "ts_metric", "text_encoder", "text_metric",
                   "image_encoder", "image_metric"]

    # Average objectives per (group + config) first, then pick best config
    agg = (
        df.groupby(group_by + config_cols, dropna=False)[OBJECTIVES]
        .mean()
        .reset_index()
    )
    idx = (
        agg.groupby(group_by, dropna=False)[rank_by]
        .transform(lambda s: s == (s.min() if ascending else s.max()))
        .astype(bool)
    )
    return agg[idx].sort_values(group_by).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std for all four objectives, grouped by (dataset, generator).

    Useful as a first-pass overview table for a paper.
    """
    records = []
    for (dataset, generator), grp in df.groupby(["dataset", "generator"]):
        row = {"dataset": dataset, "generator": generator}
        for obj in OBJECTIVES:
            vals = grp[obj].dropna()
            row[f"{obj}_mean"] = round(vals.mean(), 4)
            row[f"{obj}_std"]  = round(vals.std(),  4)
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print ablation result summaries for all discovered experiments."
    )
    parser.add_argument(
        "--root", default=None,
        help="Repository root to search (default: auto-detected from file location).",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Load all runs, not just the latest per project.",
    )
    parser.add_argument(
        "--objective", default="proximity",
        choices=OBJECTIVES,
        help="Objective to use for the best-combo table (default: proximity).",
    )
    parser.add_argument(
        "--pivot", action="store_true",
        help="Print a pivot table (generators × datasets) for --objective.",
    )
    args = parser.parse_args()

    df = load_all(root=args.root, latest_only=not args.all_runs)

    if df.empty:
        print("No results found.")
    else:
        print(f"\nLoaded {len(df):,} rows from "
              f"{df['dataset'].nunique()} dataset(s), "
              f"{df['combo_id'].nunique()} combo(s), "
              f"{df['generator'].nunique()} generator(s).\n")

        print("=== Mean objectives (all combos) ===")
        print(summary_table(df).to_string(index=False))

        print(f"\n=== Best combo per (dataset, generator) by {args.objective} ===")
        print(best_combos(df, rank_by=args.objective,
                          ascending=args.objective != "plausibility"
                                    and args.objective != "outcome")
              .to_string(index=False))

        if args.pivot:
            print(f"\n=== Pivot: mean {args.objective} ===")
            print(pivot_objectives(df, args.objective).to_string())
