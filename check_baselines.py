"""
check_baselines.py — Report which baselines have been run for each dataset and
fusion strategy (early / intermediate / late).

Prints a per-dataset table: rows = baseline names, cols = fusion strategies
(intermediate = legacy "baselines/" dir, early/late = "baselines_fusion_early/",
"baselines_fusion_late/").  For Sepsis, folds are shown separately.

Usage:
    python check_baselines.py [--root /path/to/CRANE-cf]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config: expected baselines per dataset
# ---------------------------------------------------------------------------

# fmt: off
EXPECTED: Dict[str, Dict[str, List[str]]] = {
    "long_covid_tweets": {
        "intermediate": [
            "DiCE_tabular", "NICE_tabular",
            "DiCE_text_tweet", "NICE_text_tweet",
            "GRADIENT_tabular", "GRADIENT_text_tweet",
        ],
        "early": [
            "DiCE_tabular", "NICE_tabular",
            "DiCE_text_tweet", "NICE_text_tweet",
        ],
        "late": [
            "DiCE_tabular", "NICE_tabular",
        ],
    },
    "real_or_fake_jobs": {
        "intermediate": [
            "DiCE_tabular", "NICE_tabular",
            "DiCE_text_description",    "NICE_text_description",
            "DiCE_text_company_profile","NICE_text_company_profile",
            "DiCE_text_requirements",   "NICE_text_requirements",
            "GRADIENT_tabular",
            "GRADIENT_text_description",
            "GRADIENT_text_company_profile",
            "GRADIENT_text_requirements",
        ],
        "early": [
            "DiCE_tabular", "NICE_tabular",
            "DiCE_text_description",    "NICE_text_description",
            "DiCE_text_company_profile","NICE_text_company_profile",
            "DiCE_text_requirements",   "NICE_text_requirements",
        ],
        "late": [
            "DiCE_tabular", "NICE_tabular",
        ],
    },
    "memes": {
        "intermediate": [
            "DiCE_text_meme_text", "NICE_text_meme_text",
            "DiCE_image_meme_img", "NICE_image_meme_img",
            "GRADIENT_text_meme_text", "GRADIENT_image_meme_img",
        ],
        "early": [
            "DiCE_text_meme_text", "NICE_text_meme_text",
            "DiCE_image_meme_img", "NICE_image_meme_img",
        ],
        "late": [
            "DiCE_text_meme_text", "NICE_text_meme_text",
            "DiCE_image_meme_img", "NICE_image_meme_img",
        ],
    },
    "sepsis": {
        "intermediate": [
            "DiCE_tabular", "NICE_tabular",
            "GRADIENT_tabular", "GRADIENT_TS_lab_exams_vitals",
            "NATIVEGUIDE_TS_lab_exams_vitals", "COMTE_TS_lab_exams_vitals",
        ],
        "early": [
            "DiCE_tabular", "NICE_tabular",
            "GRADIENT_tabular", "GRADIENT_TS_lab_exams_vitals",
            "NATIVEGUIDE_TS_lab_exams_vitals", "COMTE_TS_lab_exams_vitals",
        ],
        "late": [
            "DiCE_tabular", "NICE_tabular",
            "GRADIENT_tabular", "GRADIENT_TS_lab_exams_vitals",
            "NATIVEGUIDE_TS_lab_exams_vitals", "COMTE_TS_lab_exams_vitals",
        ],
    },
}
# fmt: on

STRATEGIES = ["intermediate", "early", "late"]
SEPSIS_FOLDS = [0, 1, 2, 3, 4]  # fold 0 excluded for early (0 deaths), included for intermediate/late

STRATEGY_DIRS = {
    "intermediate": "baselines",
    "early":        "baselines_fusion_early",
    "late":         "baselines_fusion_late",
}

# hateful_memes dataset folder may be named "memes" or "hateful_memes"
DATASET_FOLDER = {
    "long_covid_tweets": "long_covid_tweets",
    "real_or_fake_jobs": "real_or_fake_jobs",
    "memes":             "memes",
    "sepsis":            "sepsis",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _check_baseline(summary_path: Path) -> Tuple[str, int]:
    """Returns (status_symbol, n_samples). status: ✓ / ✗ / ?"""
    s = _load_summary(summary_path)
    if s is None:
        return "✗", 0
    n = s.get("n_total", len(s.get("sample_indices", [])))
    return "✓", int(n)


# ---------------------------------------------------------------------------
# Per-dataset reporters
# ---------------------------------------------------------------------------

def _check_non_sepsis(dataset: str, data_dir: Path) -> None:
    expected = EXPECTED[dataset]

    # Collect all baseline names across strategies
    all_names: List[str] = []
    for strategy in STRATEGIES:
        for name in expected.get(strategy, []):
            if name not in all_names:
                all_names.append(name)

    # Header
    col_w = max(len(n) for n in all_names) + 2
    strat_w = 22  # wide enough for "intermediate (n=XXXX)"
    header = f"  {'Baseline':<{col_w}}" + "".join(
        f"  {s:<{strat_w}}" for s in STRATEGIES
    )
    print(header)
    print("  " + "-" * (col_w + (strat_w + 2) * len(STRATEGIES)))

    for name in all_names:
        row = f"  {name:<{col_w}}"
        for strategy in STRATEGIES:
            bdir = STRATEGY_DIRS[strategy]
            path = data_dir / bdir / name / "summary.json"
            if name not in expected.get(strategy, []):
                cell = f"  {'N/A':<{strat_w}}"
            else:
                sym, n = _check_baseline(path)
                cell = f"  {sym} (n={n})" if sym == "✓" else f"  {sym}"
                cell = f"{cell:<{strat_w + 2}}"
            row += cell
        print(row)


def _check_sepsis(data_dir: Path) -> None:
    expected = EXPECTED["sepsis"]

    all_names: List[str] = []
    for strategy in STRATEGIES:
        for name in expected.get(strategy, []):
            if name not in all_names:
                all_names.append(name)

    name_w = max(len(n) for n in all_names) + 2

    for strategy in STRATEGIES:
        bdir = STRATEGY_DIRS[strategy]
        print(f"\n  Strategy: {strategy}  →  {bdir}/fold_N/")
        # fold header
        fold_hdr = f"    {'Baseline':<{name_w}}" + "".join(
            f"  fold_{f:<6}" for f in SEPSIS_FOLDS
        )
        print(fold_hdr)
        print("    " + "-" * (name_w + 10 * len(SEPSIS_FOLDS)))
        for name in expected.get(strategy, []):
            row = f"    {name:<{name_w}}"
            for fold in SEPSIS_FOLDS:
                path = data_dir / bdir / f"fold_{fold}" / name / "summary.json"
                sym, n = _check_baseline(path)
                cell = f"✓{n:>4}" if sym == "✓" else "  ✗  "
                row += f"  {cell:<8}"
            print(row)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _count_done(dataset: str, data_dir: Path) -> Tuple[int, int]:
    """Returns (done, total) across all strategies."""
    expected = EXPECTED[dataset]
    done, total = 0, 0
    if dataset == "sepsis":
        for strategy in STRATEGIES:
            bdir = STRATEGY_DIRS[strategy]
            for name in expected.get(strategy, []):
                for fold in SEPSIS_FOLDS:
                    path = data_dir / bdir / f"fold_{fold}" / name / "summary.json"
                    total += 1
                    if path.exists():
                        done += 1
    else:
        for strategy in STRATEGIES:
            bdir = STRATEGY_DIRS[strategy]
            for name in expected.get(strategy, []):
                path = data_dir / bdir / name / "summary.json"
                total += 1
                if path.exists():
                    done += 1
    return done, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Baseline completion report.")
    p.add_argument("--root", default=None,
                   help="Path to CRANE-cf root. Defaults to this script's parent.")
    args = p.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent

    print("=" * 72)
    print("  Baseline completion report")
    print("  Strategies: intermediate=baselines/  early=baselines_fusion_early/"
          "  late=baselines_fusion_late/")
    print("=" * 72)

    for dataset, folder in DATASET_FOLDER.items():
        data_dir = root / folder / "data"
        done, total = _count_done(dataset, data_dir)
        pct = 100 * done // total if total else 0
        print(f"\n{'─'*72}")
        print(f"  {dataset.upper()}  [{done}/{total} done, {pct}%]")
        print(f"{'─'*72}")

        if dataset == "sepsis":
            _check_sepsis(data_dir)
        else:
            _check_non_sepsis(dataset, data_dir)

    print(f"\n{'=' * 72}")
    print("  Legend:  ✓ = summary.json present   ✗ = missing   N/A = not expected")
    print("=" * 72)


if __name__ == "__main__":
    main()
