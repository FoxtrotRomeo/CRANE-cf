"""evaluate_k_ablation.py — Re-evaluate saved-full ablation pickles at multiple k values.

Each combo_{N}_results.pkl produced by run_cf_ablation.py with --save-full stores
candidates sorted in ascending distance order.  Slicing [:k] therefore gives the
exact same k nearest neighbours as if generation had been run with k=k_target,
without repeating any search.

For efficiency, each combo pkl is scored only ONCE (at full k_max depth) using
compute_objectives.  The per-candidate scores are stored in memory and then
aggregated at each k cutoff by slicing — so expensive calls (predict_fn, LOF
scoring, embedding lookups) are never repeated across k values.

This module exposes ``evaluate_k_ablation()``, a function you can call from
within a dataset-specific run_cf_ablation.py so it can reuse the already-loaded
dataset, model, and objectives_kwargs_factory.

Typical usage (from a dataset script):

    if args.eval_pkls:
        from evaluate_k_ablation import evaluate_k_ablation
        k_vals = [int(x) for x in args.k_values.split(",")]
        evaluate_k_ablation(
            run_dirs=[Path(p) for p in args.eval_pkls.split(",")],
            dataset=dataset,
            objectives_kwargs_factory=_objectives_kwargs_factory,
            k_values=k_vals,
            output_path=...,
        )

Command-line usage (standalone, requires --factory):

    python evaluate_k_ablation.py \\
        --run-dir path/to/fusion_xxx_k50 \\
        --factory my_module:build_for_k_eval \\
        --k-values 1,5,10,20,50

    The factory must return a dict with keys:
        "dataset"                   — MultimodalDataset
        "objectives_kwargs_factory" — callable(text_cfg, image_cfg) -> dict | None
        "objectives_kwargs"         — dict (used when factory is absent or returns None)
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    for _p in [str(repo_root), str(Path(__file__).resolve().parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

from counterfactual_evaluation_helpers import compute_objectives
from run_distance_ablation import _json_default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_factory(factory_spec: str):
    if ":" not in factory_spec:
        raise ValueError("Factory must be in format 'module:function_name'.")
    mod_name, fn_name = factory_spec.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"Function '{fn_name}' not found in module '{mod_name}'.")
    return fn


def _load_combo_metadata(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return {combo_id: row_dict} from summary.jsonl (or summary.json)."""
    jsonl = run_dir / "summary.jsonl"
    if jsonl.exists():
        meta: Dict[str, Dict[str, Any]] = {}
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        meta[str(row["combo_id"])] = row
                except json.JSONDecodeError:
                    pass
        return meta

    summary_json = run_dir / "summary.json"
    if summary_json.exists():
        with open(summary_json, encoding="utf-8") as f:
            data = json.load(f)
        return {str(r["combo_id"]): r for r in data.get("rows", []) if r.get("status") == "ok"}

    return {}


def _score_candidates(
    results_by_sample: Dict[int, Dict[str, List[Any]]],
    dataset,
    objectives_kwargs: Dict[str, Any],
) -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """Score every candidate once, returning per-generator per-sample score lists.

    Returns
    -------
    {generator_name: {sample_idx: [{"outcome": v, "proximity": v, ...}, ...]}}

    Candidates within each sample retain their original order (ascending distance),
    so slicing [:k] at aggregation time gives the k-nearest-neighbour subset.
    The embedding_cache and text_metrics_cache are shared across all generators
    and samples within this call, avoiding repeated embedding computations.
    """
    X_train_static = (
        np.asarray(dataset.X_train_static) if dataset.X_train_static is not None else None
    )
    X_test_static = (
        np.asarray(dataset.X_test_static) if dataset.X_test_static is not None else None
    )
    X_train_tabular = dict(dataset.X_train_tabular or {})
    X_test_tabular  = dict(dataset.X_test_tabular or {})
    X_train_ts      = dict(dataset.X_train_ts or {})
    X_test_ts       = dict(dataset.X_test_ts or {})
    X_test_text = (
        dataset.get_text_branch(dataset.primary_text_name, split="test")
        if dataset.primary_text_name is not None else None
    )
    X_test_texts  = dataset.text_modalities(split="test")
    X_test_images = dataset.image_modalities(split="test")

    embedding_cache: Dict = {}
    text_metrics_cache: Dict = {}

    helper_kwargs_keys = {
        "tabular_objective_context", "tabular_objective_contexts",
        "ts_objective_context", "ts_objective_contexts",
        "text_objective_contexts", "image_objective_contexts",
    }
    clean_kw = {
        k: v for k, v in objectives_kwargs.items()
        if not k.startswith("_") and k not in helper_kwargs_keys
    }

    scores: Dict[str, Dict[int, List[Dict[str, float]]]] = {}

    for sample_idx, gen_results in results_by_sample.items():
        x_tab_ref = X_test_static[sample_idx] if X_test_static is not None else None
        text_factual = (
            X_test_text[sample_idx]
            if X_test_text is not None and len(X_test_text) > sample_idx
            else None
        )

        for gen_name, candidates in gen_results.items():
            gen_scores = scores.setdefault(gen_name, {})
            sample_scores: List[Dict[str, float]] = []

            for cand in (candidates or []):
                x_tab_cand = cand.get("static")
                text_cand  = cand.get("text") or cand.get("text_input")

                # --- tabular kwargs ---
                _tabmf = objectives_kwargs.get("_tabular_modalities_fn")
                if _tabmf is not None:
                    extra_tab_kw = _tabmf(sample_idx, cand, x_tab_ref)
                elif isinstance(cand.get("tab"), dict) or x_tab_cand is not None:
                    tab_ctx_default  = objectives_kwargs.get("tabular_objective_context")
                    tab_ctx_by_name  = objectives_kwargs.get("tabular_objective_contexts", {}) or {}
                    tabular_modalities: Dict[str, Any] = {}
                    primary_tab_name = dataset.primary_tabular_name or "__primary__"
                    if x_tab_cand is not None and x_tab_ref is not None and X_train_static is not None:
                        spec = {"x": x_tab_cand, "x_ref": x_tab_ref, "X_obs": X_train_static}
                        spec.update(tab_ctx_by_name.get(primary_tab_name, tab_ctx_default) or {})
                        tabular_modalities[primary_tab_name] = spec
                    for name, cand_tab in (cand.get("tab") or {}).items():
                        X_obs_tab = X_train_tabular.get(name)
                        X_ref_tabs = X_test_tabular.get(name)
                        if cand_tab is None or X_obs_tab is None or X_ref_tabs is None:
                            continue
                        if len(X_ref_tabs) <= sample_idx:
                            continue
                        spec = {"x": cand_tab, "x_ref": X_ref_tabs[sample_idx], "X_obs": X_obs_tab}
                        spec.update(tab_ctx_by_name.get(name, tab_ctx_default) or {})
                        tabular_modalities[name] = spec
                    extra_tab_kw = (
                        {"tabular_modalities": tabular_modalities}
                        if tabular_modalities
                        else {"x_tab": x_tab_cand, "x_tab_ref": x_tab_ref, "X_tab_obs": X_train_static}
                    )
                else:
                    extra_tab_kw = {
                        "x_tab": x_tab_cand, "x_tab_ref": x_tab_ref, "X_tab_obs": X_train_static,
                    }

                # --- time-series kwargs ---
                _tsmf = objectives_kwargs.get("_ts_modalities_fn")
                if _tsmf is not None:
                    extra_ts_kw = _tsmf(sample_idx, cand)
                elif isinstance(cand.get("ts"), dict):
                    ts_ctx_default = objectives_kwargs.get("ts_objective_context")
                    ts_ctx_by_name = objectives_kwargs.get("ts_objective_contexts", {}) or {}
                    ts_modalities: Dict[str, Any] = {}
                    for name, cand_ts in cand["ts"].items():
                        ctx = ts_ctx_by_name.get(name, ts_ctx_default)
                        X_obs_ts = X_train_ts.get(name)
                        X_ref_ts = X_test_ts.get(name)
                        if ctx is None or cand_ts is None or X_obs_ts is None or X_ref_ts is None:
                            continue
                        if len(X_ref_ts) <= sample_idx:
                            continue
                        spec = {"x": cand_ts, "x_ref": X_ref_ts[sample_idx], "X_obs": X_obs_ts}
                        spec.update(ctx or {})
                        ts_modalities[name] = spec
                    extra_ts_kw = {"ts_modalities": ts_modalities} if ts_modalities else {}
                else:
                    extra_ts_kw = {}

                # --- text kwargs ---
                _tmf = objectives_kwargs.get("_text_modalities_fn")
                if _tmf is not None:
                    extra_text_kw = _tmf(sample_idx, cand, text_factual)
                elif isinstance(cand.get("texts"), dict):
                    text_ctx_default = clean_kw.get("text_objective_context")
                    text_ctx_by_name = clean_kw.get("text_objective_contexts", {}) or {}
                    extra_text_kw = {
                        "text_modalities": {
                            name: {
                                "candidate": cand["texts"].get(name),
                                "factual": (
                                    X_test_texts.get(name)[sample_idx]
                                    if name in X_test_texts and len(X_test_texts.get(name)) > sample_idx
                                    else None
                                ),
                                "context": text_ctx_by_name.get(name, text_ctx_default),
                            }
                            for name in cand["texts"].keys()
                        }
                    }
                else:
                    extra_text_kw = {
                        "text_candidate": text_cand if text_cand is not None else text_factual,
                        "text_factual": text_factual,
                    }

                # --- image kwargs ---
                _imf = objectives_kwargs.get("_image_modalities_fn")
                if _imf is not None:
                    extra_image_kw = _imf(sample_idx, cand)
                elif isinstance(cand.get("images"), dict):
                    image_ctx_default = clean_kw.get("image_objective_context")
                    image_ctx_by_name = clean_kw.get("image_objective_contexts", {}) or {}
                    extra_image_kw = {
                        "image_modalities": {
                            name: {
                                "candidate": cand["images"].get(name),
                                "factual": (
                                    X_test_images.get(name)[sample_idx]
                                    if name in X_test_images and len(X_test_images.get(name)) > sample_idx
                                    else None
                                ),
                                "context": image_ctx_by_name.get(name, image_ctx_default),
                            }
                            for name in cand["images"].keys()
                        }
                    }
                else:
                    extra_image_kw = {}

                try:
                    objs = compute_objectives(
                        embedding_cache=embedding_cache,
                        text_metrics_cache=text_metrics_cache,
                        **extra_tab_kw,
                        **extra_ts_kw,
                        **extra_text_kw,
                        **extra_image_kw,
                        **clean_kw,
                    )
                    sample_scores.append({
                        "outcome":     float(objs[0]),
                        "proximity":   float(objs[1]),
                        "sparsity":    float(objs[2]),
                        "plausibility": float(objs[3]),
                    })
                except Exception:
                    pass

            gen_scores[sample_idx] = sample_scores

    return scores


def _score_candidates_verbose(
    results_by_sample: Dict[int, Dict[str, List[Any]]],
    dataset,
    objectives_kwargs: Dict[str, Any],
) -> None:
    """Like _score_candidates but raises on the first exception instead of swallowing.

    Used to surface the root cause when _score_candidates returns 0 scored candidates.
    Only processes the first sample × generator × candidate.
    """
    X_train_static = (
        np.asarray(dataset.X_train_static) if dataset.X_train_static is not None else None
    )
    X_test_static = (
        np.asarray(dataset.X_test_static) if dataset.X_test_static is not None else None
    )
    X_test_texts  = dataset.text_modalities(split="test")
    X_test_images = dataset.image_modalities(split="test")

    helper_kwargs_keys = {
        "tabular_objective_context", "tabular_objective_contexts",
        "ts_objective_context", "ts_objective_contexts",
        "text_objective_contexts", "image_objective_contexts",
    }
    clean_kw = {
        k: v for k, v in objectives_kwargs.items()
        if not k.startswith("_") and k not in helper_kwargs_keys
    }

    sample_idx, gen_results = next(iter(results_by_sample.items()))
    gen_name, candidates = next(iter(gen_results.items()))
    cand = candidates[0]

    x_tab_ref = X_test_static[sample_idx] if X_test_static is not None else None
    x_tab_cand = cand.get("static")

    tab_ctx_default = objectives_kwargs.get("tabular_objective_context")
    tab_ctx_by_name = objectives_kwargs.get("tabular_objective_contexts", {}) or {}
    tabular_modalities: Dict[str, Any] = {}
    primary_tab_name = dataset.primary_tabular_name or "__primary__"
    if x_tab_cand is not None and x_tab_ref is not None and X_train_static is not None:
        spec = {"x": x_tab_cand, "x_ref": x_tab_ref, "X_obs": X_train_static}
        spec.update(tab_ctx_by_name.get(primary_tab_name, tab_ctx_default) or {})
        tabular_modalities[primary_tab_name] = spec
    extra_tab_kw = {"tabular_modalities": tabular_modalities} if tabular_modalities else {
        "x_tab": x_tab_cand, "x_tab_ref": x_tab_ref, "X_tab_obs": X_train_static,
    }

    text_factual = None
    if dataset.primary_text_name is not None:
        xt = dataset.get_text_branch(dataset.primary_text_name, split="test")
        if xt is not None and len(xt) > sample_idx:
            text_factual = xt[sample_idx]

    _tmf = objectives_kwargs.get("_text_modalities_fn")
    if _tmf is not None:
        extra_text_kw = _tmf(sample_idx, cand, text_factual)
    else:
        extra_text_kw = {"text_candidate": cand.get("text"), "text_factual": text_factual}

    embedding_cache: Dict = {}
    text_metrics_cache: Dict = {}
    print(f"  sample={sample_idx}, gen={gen_name}, cand keys={list(cand.keys())}")
    objs = compute_objectives(
        embedding_cache=embedding_cache,
        text_metrics_cache=text_metrics_cache,
        **extra_tab_kw,
        **extra_text_kw,
        **clean_kw,
    )
    print(f"  Result: outcome={objs[0]:.4f} proximity={objs[1]:.4f} "
          f"sparsity={objs[2]:.4f} plausibility={objs[3]:.4f}")


def _aggregate_at_k(
    per_candidate_scores: Dict[str, Dict[int, List[Dict[str, float]]]],
    k_target: int,
) -> Dict[str, Dict[str, float]]:
    """Average the first k_target per-candidate scores across all samples and generators."""
    result: Dict[str, Dict[str, float]] = {}
    for gen_name, sample_scores in per_candidate_scores.items():
        vals: Dict[str, List[float]] = {"outcome": [], "proximity": [], "sparsity": [], "plausibility": []}
        for scores_list in sample_scores.values():
            for score in scores_list[:k_target]:
                for metric, v in score.items():
                    if not np.isnan(v):
                        vals[metric].append(v)
        result[gen_name] = {
            m: float(np.mean(vs)) if vs else float("nan") for m, vs in vals.items()
        }
    return result


def _aggregate_combo_scores(
    all_combo_scores: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Average per-generator objective dicts across combos."""
    accum: Dict[str, Dict[str, List[float]]] = {}
    for scores in all_combo_scores:
        for gen, metrics in scores.items():
            bucket = accum.setdefault(gen, {"outcome": [], "proximity": [], "sparsity": [], "plausibility": []})
            for metric, val in metrics.items():
                if val is not None and not np.isnan(val):
                    bucket[metric].append(float(val))
    return {
        gen: {m: float(np.mean(vs)) if vs else float("nan") for m, vs in buckets.items()}
        for gen, buckets in accum.items()
    }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_k_ablation(
    run_dirs: Sequence[Union[str, Path]],
    dataset,
    objectives_kwargs_factory: Optional[Callable] = None,
    objectives_kwargs: Optional[Dict[str, Any]] = None,
    k_values: Sequence[int] = (1, 5, 10, 20, 50),
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Re-evaluate saved-full ablation pickles at multiple k values.

    Each combo pkl is scored exactly once (at its full depth).  Per-candidate
    scores are stored in memory and aggregated at each k cutoff by slicing —
    so predict_fn, LOF scoring, and embedding lookups are never repeated.

    Parameters
    ----------
    run_dirs
        One or more directories produced by run_cf_ablation.py with --save-full.
        Each must contain ``summary.jsonl`` (or ``summary.json``) and
        ``combo_*_results.pkl`` files.
    dataset
        The same MultimodalDataset used during counterfactual generation.
    objectives_kwargs_factory
        ``callable(text_cfg: dict | None, image_cfg: dict | None) -> dict | None``.
        Called once per combo to build the objectives kwargs for that encoder/metric
        combination.  When it returns None, ``objectives_kwargs`` is used as fallback.
    objectives_kwargs
        Combo-independent objectives kwargs.  Used when ``objectives_kwargs_factory``
        is None or returns None.
    k_values
        k cutoffs to evaluate.  All must be ≤ the k used during generation (50).
    output_path
        If given, the result dict is written as JSON to this path.

    Returns
    -------
    dict
        ``results`` is keyed by str(k), each entry containing ``candidate_counts``
        and (if objectives were evaluated) ``objectives`` per generator.
    """
    run_dirs = [Path(d) for d in run_dirs]
    k_values = sorted(set(int(k) for k in k_values))

    all_meta: Dict[str, Dict[str, Any]] = {}
    all_pkls: List[Path] = []
    for run_dir in run_dirs:
        all_meta.update(_load_combo_metadata(run_dir))
        all_pkls.extend(sorted(run_dir.glob("combo_*_results.pkl")))

    if not all_pkls:
        raise FileNotFoundError(
            f"No combo_*_results.pkl files found in: {[str(d) for d in run_dirs]}"
        )

    print(f"[k-eval] {len(all_pkls)} pkl file(s) across {len(run_dirs)} run dir(s).")
    print(f"[k-eval] Evaluating at k = {k_values}  (each combo scored once, results sliced per k)")
    has_objectives = objectives_kwargs_factory is not None or objectives_kwargs is not None

    counts_by_k: Dict[int, Dict[str, int]] = {k: {} for k in k_values}
    # per k: list of per-combo {gen: {metric: mean}} dicts for final aggregation
    scores_by_k: Dict[int, List[Dict[str, Dict[str, float]]]] = {k: [] for k in k_values}
    n_combos_done = 0

    def _build_and_save(final: bool = False) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for k_target in k_values:
            entry: Dict[str, Any] = {
                "k": k_target,
                "n_combos_evaluated": len(scores_by_k[k_target]),
                "candidate_counts": counts_by_k[k_target],
            }
            if scores_by_k[k_target]:
                entry["objectives"] = _aggregate_combo_scores(scores_by_k[k_target])
            results[str(k_target)] = entry
        payload = {
            "k_values": k_values,
            "n_combos_total": len(all_pkls),
            "n_combos_done": n_combos_done,
            "complete": final,
            "run_dirs": [str(d) for d in run_dirs],
            "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "results": results,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=_json_default)
        return payload

    if output_path is not None:
        output_path = Path(output_path)

    for pkl_path in all_pkls:
        combo_id = pkl_path.stem.replace("_results", "")
        meta     = all_meta.get(combo_id, {})
        text_cfg = meta.get("text_config")
        image_cfg = meta.get("image_config")

        obj_kw = objectives_kwargs
        if objectives_kwargs_factory is not None:
            try:
                result = objectives_kwargs_factory(text_cfg, image_cfg)
                if result is not None:
                    obj_kw = result
            except Exception as exc:
                print(f"[k-eval] Warning: objectives_kwargs_factory failed for {combo_id}: {exc}")

        print(f"[k-eval] {pkl_path.name} ...", end=" ", flush=True)
        try:
            with open(pkl_path, "rb") as f:
                full_results: Dict[int, Dict[str, List[Any]]] = pickle.load(f)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        max_k = max(
            (len(c) for gr in full_results.values() for c in gr.values()),
            default=0,
        )
        n_gens = len(next(iter(full_results.values()), {}).keys()) if full_results else 0
        print(f"generators={n_gens}, max_k={max_k}")

        # Score all candidates once
        per_candidate: Optional[Dict[str, Dict[int, List[Dict[str, float]]]]] = None
        if has_objectives and obj_kw is not None and dataset is not None:
            try:
                per_candidate = _score_candidates(full_results, dataset, dict(obj_kw))
            except Exception as exc:
                print(f"[k-eval] Warning: scoring failed for {combo_id}: {exc}")

        # Accumulate counts and objectives at each k — no re-scoring
        for k_target in k_values:
            for gen_res in full_results.values():
                for gen_name, cands in gen_res.items():
                    n = min(len(cands), k_target)
                    counts_by_k[k_target][gen_name] = counts_by_k[k_target].get(gen_name, 0) + n

            if per_candidate is not None:
                scores_by_k[k_target].append(_aggregate_at_k(per_candidate, k_target))

        n_combos_done += 1
        _build_and_save(final=False)
        print(f"[k-eval] Progress: {n_combos_done}/{len(all_pkls)} combos saved.")

    summary = _build_and_save(final=True)
    if output_path is not None:
        print(f"[k-eval] Final results saved to: {output_path}")

    print("\n=== K-Ablation Summary ===")
    for k_str, row in summary["results"].items():
        if "objectives" in row:
            print(f"\n  k={k_str}  (aggregated over {row['n_combos_evaluated']} combos):")
            for gen_name, metrics in sorted(row["objectives"].items()):
                parts = [f"{m}={v:.4f}" for m, v in metrics.items()]
                print(f"    {gen_name:<22s}  {', '.join(parts)}")
        else:
            counts = row["candidate_counts"]
            print(f"\n  k={k_str}  candidate counts: {counts}")

    return summary


# ---------------------------------------------------------------------------
# Re-evaluation: rewrite summary.jsonl with corrected objectives
# ---------------------------------------------------------------------------

def reeval_summaries(
    run_dir: Union[str, Path],
    dataset,
    objectives_kwargs_factory: Optional[Callable] = None,
    objectives_kwargs: Optional[Dict[str, Any]] = None,
    k: int = 50,
    backup: bool = True,
) -> None:
    """Re-score saved pkl candidates and overwrite summary.jsonl with corrected objectives.

    This is a targeted fix for runs where summary.jsonl was produced without a
    proximity normalizer (or with an incorrect one).  The pkl files are the
    source of truth — they contain the raw candidate feature vectors; only the
    aggregated objective values in summary.jsonl are overwritten.

    Parameters
    ----------
    run_dir
        Directory produced by run_cf_ablation.py --save-full.
        Must contain ``summary.jsonl`` and ``combo_*_results.pkl`` files.
    dataset
        The same MultimodalDataset used during counterfactual generation.
    objectives_kwargs_factory
        ``callable(text_cfg, image_cfg) -> dict | None`` — same factory as used
        in run_cf_ablation.py (includes proximity_normalizer, embed_fn, etc.).
    objectives_kwargs
        Fallback kwargs when the factory returns None.
    k
        Aggregate the first *k* candidates per sample per generator.
        Should match the k used during generation (default 50).
    backup
        If True, save a ``summary.jsonl.bak`` copy before overwriting.
    """
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "summary.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"summary.jsonl not found in {run_dir}")

    # Load all existing rows (including non-ok ones) so we can rewrite verbatim
    all_rows: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    rows_by_id: Dict[str, Dict[str, Any]] = {
        str(r.get("combo_id", "")): r for r in all_rows
    }

    if backup:
        backup_path = jsonl_path.with_suffix(".jsonl.bak")
        shutil.copy2(jsonl_path, backup_path)
        print(f"[reeval] Backed up original to {backup_path}")

    pkls = sorted(run_dir.glob("combo_*_results.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No combo_*_results.pkl files found in {run_dir}")

    print(f"[reeval] Re-evaluating {len(pkls)} pkl(s) in {run_dir}  (k={k})")

    n_done = 0
    for pkl_path in pkls:
        combo_id = pkl_path.stem.replace("_results", "")
        row = rows_by_id.get(combo_id, {})
        text_cfg  = row.get("text_config")
        image_cfg = row.get("image_config")

        obj_kw = objectives_kwargs
        if objectives_kwargs_factory is not None:
            try:
                result = objectives_kwargs_factory(text_cfg, image_cfg)
                if result is not None:
                    obj_kw = result
            except Exception as exc:
                print(f"[reeval] Warning: factory failed for {combo_id}: {exc}")

        if obj_kw is None:
            print(f"[reeval] Skipping {combo_id} — no objectives_kwargs available.")
            continue

        print(f"[reeval] {pkl_path.name} ...", end=" ", flush=True)
        try:
            with open(pkl_path, "rb") as f:
                full_results: Dict[int, Dict[str, List[Any]]] = pickle.load(f)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        try:
            per_candidate = _score_candidates(full_results, dataset, dict(obj_kw))
        except Exception as exc:
            print(f"SCORING FAILED ({exc})")
            import traceback; traceback.print_exc()
            continue

        n_scored = sum(len(s) for gs in per_candidate.values() for s in gs.values())
        n_total  = sum(
            len(cands)
            for gs in full_results.values()
            for cands in gs.values()
        )
        if n_scored == 0 and n_total > 0:
            # _score_candidates swallows per-candidate exceptions; run one candidate
            # manually to surface the actual error.
            _first_sample = next(iter(full_results.values()))
            _first_gen    = next(iter(_first_sample.values()))
            if _first_gen:
                print(f"\n[reeval] WARNING: 0/{n_total} candidates scored. "
                      "Re-running first candidate without exception suppression:")
                _score_candidates_verbose(full_results, dataset, dict(obj_kw))
            continue

        updated_objectives = _aggregate_at_k(per_candidate, k)

        if combo_id in rows_by_id:
            rows_by_id[combo_id]["objectives"] = updated_objectives
        else:
            print(f"[reeval] Warning: {combo_id} not found in summary.jsonl — skipping write.")

        n_done += 1
        gen_summary = ", ".join(
            f"{g}:prox={v.get('proximity', float('nan')):.3f}"
            for g, v in updated_objectives.items()
        )
        print(f"done  ({n_scored}/{n_total} candidates, {gen_summary})")

    # Overwrite summary.jsonl with corrected objectives
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")

    print(f"[reeval] Updated {n_done}/{len(pkls)} combos — wrote {jsonl_path}")


# ---------------------------------------------------------------------------
# Standalone CLI (requires --factory)
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Re-evaluate saved ablation pickles at multiple k values."
    )
    p.add_argument(
        "--run-dir", nargs="+", required=True, metavar="DIR",
        help="Directory (or directories) with combo_*_results.pkl + summary.jsonl.",
    )
    p.add_argument(
        "--factory", default=None, metavar="MODULE:FUNC",
        help=(
            "Factory returning dataset + objectives setup: 'module:function_name'. "
            "Must return a dict with 'dataset' and 'objectives_kwargs_factory' keys."
        ),
    )
    p.add_argument(
        "--factory-kwargs-json", default=None, metavar="JSON",
        help="Optional JSON object passed as **kwargs to the factory.",
    )
    p.add_argument(
        "--k-values", default="1,5,10,20,50",
        help="Comma-separated k values to evaluate (default: 1,5,10,20,50).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON file. Defaults to <first-run-dir>/k_ablation_summary_k<vals>.json.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    run_dirs = [Path(d) for d in args.run_dir]
    for d in run_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")

    k_values = _parse_int_csv(args.k_values)
    _ksuffix = "_k" + "_".join(str(k) for k in sorted(k_values))
    output_path = (
        Path(args.output) if args.output
        else run_dirs[0] / f"k_ablation_summary{_ksuffix}.json"
    )

    dataset = None
    objectives_kwargs_factory = None
    objectives_kwargs = None

    if args.factory:
        factory_fn = _load_factory(args.factory)
        factory_kw: Dict[str, Any] = {}
        if args.factory_kwargs_json:
            factory_kw = json.loads(args.factory_kwargs_json)
        produced = factory_fn(**factory_kw)
        if not isinstance(produced, dict):
            raise TypeError("Factory must return a dict.")
        dataset = produced.get("dataset")
        objectives_kwargs_factory = produced.get("objectives_kwargs_factory")
        objectives_kwargs = produced.get("objectives_kwargs")
    else:
        print("[k-eval] No --factory given; candidate counts only (no objectives recomputed).")

    evaluate_k_ablation(
        run_dirs=run_dirs,
        dataset=dataset,
        objectives_kwargs_factory=objectives_kwargs_factory,
        objectives_kwargs=objectives_kwargs,
        k_values=k_values,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
