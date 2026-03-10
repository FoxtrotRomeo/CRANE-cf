"""Distance-ablation runner for cf_lib generators.

This script enumerates combinations of distance choices across available
modalities in a ``MultimodalDataset`` and runs counterfactual generation for
each combination.

Usage (CLI)
-----------
Provide a factory that returns either:
  - ``(dataset, model)``
  - ``(dataset, model, text_backend_kwargs)``
  - ``{"dataset": ..., "model": ..., "text_backend_kwargs": {...}}``

Example:
    python run_distance_ablation.py \
      --factory my_project.my_loader:build_dataset_and_model \
      --output-dir ablation_runs \
      --max-samples 25 \
      --max-combinations 200 \
      --save-full

Template factory in this repo:
    python run_distance_ablation.py \
      --factory ablation_factory_template:build_synthetic_dataset_and_model \
      --max-combinations 5 \
      --max-samples 3
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import json
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cf_lib import CounterfactualLibrary
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN


def _parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_float_csv(value: Optional[str]) -> List[float]:
    vals = _parse_csv(value)
    return [float(v) for v in vals]


def _parse_int_csv(value: Optional[str]) -> List[int]:
    vals = _parse_csv(value)
    return [int(v) for v in vals]


def _safe_len(obj: Any) -> int:
    try:
        return int(len(obj))
    except Exception:
        return 0


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _load_factory(factory_spec: str):
    if ":" not in factory_spec:
        raise ValueError("Factory must be in format 'module.submodule:function_name'.")
    mod_name, fn_name = factory_spec.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"Factory function '{fn_name}' not found in module '{mod_name}'.")
    if not callable(fn):
        raise TypeError(f"Factory '{factory_spec}' is not callable.")
    return fn


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def _normalize_ts_metric(metric: str) -> str:
    m = str(metric).strip().lower()
    aliases = {
        "dynamic_time_warping": "dtw",
        "dtw_distance": "dtw",
        "euclidean_distance": "euclidean",
        "ed": "euclidean",
        "lcss_distance": "lcss",
    }
    return aliases.get(m, m)


def _build_ts_metric_options(ts_metrics: Sequence[str], dtw_windows: Sequence[float]) -> List[Dict[str, Any]]:
    opts: List[Dict[str, Any]] = []
    for m in ts_metrics:
        mn = _normalize_ts_metric(m)
        if mn == "dtw":
            windows = list(dtw_windows) if len(dtw_windows) > 0 else [0.10]
            for w in windows:
                opts.append({"metric": "dtw", "dtw_window": float(w)})
        else:
            opts.append({"metric": mn, "dtw_window": None})
    return opts


def _text_encoder_available(encoder: str, text_backend_kwargs: Dict[str, Any]) -> bool:
    enc = str(encoder).strip().lower()
    if enc in {"tfidf", "raw"}:
        return True
    if enc in {"e5"}:
        return (
            text_backend_kwargs.get("e5_embed_fn") is not None
            or (
                text_backend_kwargs.get("e5_tokenizer") is not None
                and text_backend_kwargs.get("e5_model") is not None
                and text_backend_kwargs.get("e5_device") is not None
            )
        )
    if enc in {"bert", "sbert", "sentence-bert"}:
        return (
            text_backend_kwargs.get("bert_embed_fn") is not None
            or (
                text_backend_kwargs.get("bert_tokenizer") is not None
                and text_backend_kwargs.get("bert_model") is not None
            )
        )
    if enc in {"word2vec", "w2v"}:
        return (
            text_backend_kwargs.get("word2vec_embed_fn") is not None
            or text_backend_kwargs.get("word2vec_model") is not None
        )
    if enc == "custom":
        return text_backend_kwargs.get("text_embed_fn") is not None
    return False


def _build_text_configs(
    *,
    has_text: bool,
    text_encoders: Sequence[str],
    text_vector_metrics: Sequence[str],
    text_direct_metrics: Sequence[str],
    text_backend_kwargs: Dict[str, Any],
) -> List[Optional[Dict[str, Any]]]:
    if not has_text:
        return [None]

    configs: List[Optional[Dict[str, Any]]] = []
    for enc in text_encoders:
        enc_norm = str(enc).strip().lower()
        if not _text_encoder_available(enc_norm, text_backend_kwargs):
            print(f"[ablation] Skipping unavailable text encoder '{enc_norm}'.")
            continue

        if enc_norm == "raw":
            for m in text_direct_metrics:
                configs.append({"encoder": enc_norm, "metric": str(m).strip().lower()})
        else:
            for m in text_vector_metrics:
                configs.append({"encoder": enc_norm, "metric": str(m).strip().lower()})

    if len(configs) == 0:
        print("[ablation] No text configs available; text modality will be skipped.")
        return [None]
    return configs


def _build_generators_for_combo(
    *,
    k: int,
    tab_metrics_by_name: Dict[str, str],
    ts_metrics_by_name: Dict[str, Dict[str, Any]],
    text_cfg: Optional[Dict[str, Any]],
    text_backend_kwargs: Dict[str, Any],
):
    generators = {}

    # Primary static/tabular + named tabular modalities
    for tab_name, metric in tab_metrics_by_name.items():
        if tab_name == "__primary__":
            generators["Tabular"] = TabularNN(k=k, distance_metric=metric)
        else:
            generators[f"Tabular[{tab_name}]"] = TabularNN(
                tab_name=tab_name,
                k=k,
                distance_metric=metric,
            )

    # Time-series modalities
    for ts_name, ts_cfg in ts_metrics_by_name.items():
        generators[f"TS[{ts_name}]"] = TimeSeriesNN(
            ts_name=ts_name,
            k=k,
            distance_metric=ts_cfg["metric"],
            dtw_window=ts_cfg.get("dtw_window"),
        )

    # Text modality (optional)
    if text_cfg is not None:
        text_kwargs = dict(
            k=k,
            text_encoder=text_cfg["encoder"],
            text_distance_metric=text_cfg["metric"],
        )
        text_kwargs.update(_filter_kwargs_for_callable(TextNN.__init__, text_backend_kwargs))
        generators["Text"] = TextNN(**text_kwargs)

    return generators


def _summarize_results(results_by_sample: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for method_results in results_by_sample.values():
        for method_name, cands in method_results.items():
            counts[method_name] = counts.get(method_name, 0) + _safe_len(cands)
    return counts


def run_distance_ablation(
    dataset,
    model,
    *,
    sample_indices: Optional[Sequence[int]] = None,
    max_samples: Optional[int] = 25,
    target_value: int = 0,
    k: int = 20,
    tab_metrics: Optional[Sequence[str]] = None,
    ts_metrics: Optional[Sequence[str]] = None,
    dtw_windows: Optional[Sequence[float]] = None,
    text_encoders: Optional[Sequence[str]] = None,
    text_vector_metrics: Optional[Sequence[str]] = None,
    text_direct_metrics: Optional[Sequence[str]] = None,
    text_backend_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    save_full: bool = False,
    max_combinations: Optional[int] = None,
) -> Dict[str, Any]:
    """Run distance-combination ablation and return summary metadata.

    Parameters
    ----------
    dataset, model
        Input objects used by cf_lib generators.
    sample_indices
        Test indices to evaluate. If None, uses the first ``max_samples`` test rows.
    max_samples
        Used only when ``sample_indices`` is None.
    """
    tab_metrics = list(tab_metrics or ["euclidean", "manhattan", "hamming"])
    ts_metrics = list(ts_metrics or ["dtw", "euclidean", "lcss"])
    dtw_windows = list(dtw_windows or [0.10])
    text_encoders = list(text_encoders or ["e5", "bert", "tfidf", "word2vec", "raw"])
    text_vector_metrics = list(text_vector_metrics or ["cosine", "euclidean", "manhattan"])
    text_direct_metrics = list(text_direct_metrics or ["rouge_l", "lcs", "bleu"])
    text_backend_kwargs = dict(text_backend_kwargs or {})

    n_test = int(np.asarray(dataset.X_test_static).shape[0])
    if sample_indices is None:
        if max_samples is None:
            sample_indices = list(range(n_test))
        else:
            sample_indices = list(range(min(int(max_samples), n_test)))
    else:
        sample_indices = [int(i) for i in sample_indices]

    if len(sample_indices) == 0:
        raise ValueError("No sample indices provided for ablation.")

    tab_names = ["__primary__"] + sorted(list((dataset.X_train_tab or {}).keys()))
    ts_names = sorted(list((dataset.X_train_ts or {}).keys()))
    has_text = dataset.X_train_text is not None and dataset.X_test_text is not None

    tab_assignments: List[Dict[str, str]] = []
    for combo in itertools.product(tab_metrics, repeat=len(tab_names)):
        tab_assignments.append({name: metric for name, metric in zip(tab_names, combo)})

    ts_metric_opts = _build_ts_metric_options(ts_metrics, dtw_windows)
    ts_assignments: List[Dict[str, Dict[str, Any]]] = []
    if len(ts_names) == 0:
        ts_assignments = [{}]
    else:
        for combo in itertools.product(ts_metric_opts, repeat=len(ts_names)):
            ts_assignments.append({name: cfg for name, cfg in zip(ts_names, combo)})

    text_configs = _build_text_configs(
        has_text=has_text,
        text_encoders=text_encoders,
        text_vector_metrics=text_vector_metrics,
        text_direct_metrics=text_direct_metrics,
        text_backend_kwargs=text_backend_kwargs,
    )

    combo_iter = itertools.product(tab_assignments, ts_assignments, text_configs)
    if max_combinations is not None and int(max_combinations) > 0:
        combo_iter = itertools.islice(combo_iter, int(max_combinations))

    rows = []
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    run_root = None
    jsonl_path = None
    if output_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = run_name or f"distance_ablation_{stamp}"
        run_root = Path(output_dir) / run_label
        run_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_root / "summary.jsonl"

    for combo_idx, (tab_cfg, ts_cfg, text_cfg) in enumerate(combo_iter, start=1):
        combo_id = f"combo_{combo_idx:05d}"
        print(f"[ablation] Running {combo_id} ...")

        row = {
            "combo_id": combo_id,
            "tab_metrics": tab_cfg,
            "ts_metrics": {
                name: {
                    "metric": cfg["metric"],
                    "dtw_window": cfg.get("dtw_window"),
                }
                for name, cfg in ts_cfg.items()
            },
            "text_config": text_cfg,
            "status": "ok",
            "error": None,
            "runtime_sec": None,
            "candidate_counts": {},
            "total_candidates": 0,
            "n_samples": len(sample_indices),
        }

        t0 = time.time()
        results = {}
        try:
            generators = _build_generators_for_combo(
                k=k,
                tab_metrics_by_name=tab_cfg,
                ts_metrics_by_name=ts_cfg,
                text_cfg=text_cfg,
                text_backend_kwargs=text_backend_kwargs,
            )
            lib = CounterfactualLibrary(generators=generators)
            results = lib.generate_batch(
                dataset=dataset,
                sample_indices=list(sample_indices),
                model=model,
                target_value=target_value,
                k=k,
            )
            counts = _summarize_results(results)
            row["candidate_counts"] = counts
            row["total_candidates"] = int(sum(counts.values()))
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        finally:
            row["runtime_sec"] = float(time.time() - t0)

        rows.append(row)

        if run_root is not None:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")

            if save_full and row["status"] == "ok":
                with open(run_root / f"{combo_id}_results.pkl", "wb") as f:
                    pickle.dump(results, f)
            elif row["status"] == "error":
                with open(run_root / f"{combo_id}_error.txt", "w", encoding="utf-8") as f:
                    f.write(str(row.get("traceback", row.get("error"))))

    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_runs": len(rows),
        "sample_indices": list(sample_indices),
        "target_value": int(target_value),
        "k": int(k),
        "tab_metrics_space": tab_metrics,
        "ts_metrics_space": ts_metrics,
        "dtw_windows_space": dtw_windows,
        "text_encoders_space": text_encoders,
        "text_vector_metrics_space": text_vector_metrics,
        "text_direct_metrics_space": text_direct_metrics,
        "rows": rows,
    }

    if run_root is not None:
        with open(run_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"[ablation] Results saved to: {run_root}")

    return payload


def parse_args():
    p = argparse.ArgumentParser(description="Run distance/encoder ablation across cf_lib modalities.")
    p.add_argument(
        "--factory",
        type=str,
        required=True,
        help="Factory in format module.submodule:function_name. Must return dataset/model (+optional kwargs).",
    )
    p.add_argument(
        "--factory-kwargs-json",
        type=str,
        default=None,
        help="Optional JSON string with kwargs passed to the factory.",
    )
    p.add_argument("--output-dir", type=str, default="ablation_runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-full", action="store_true", help="Persist per-combo full result pickles.")

    p.add_argument("--sample-indices", type=str, default=None, help="Comma-separated test indices.")
    p.add_argument("--max-samples", type=int, default=25, help="Used if --sample-indices is omitted.")

    p.add_argument("--target-value", type=int, default=0)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-combinations", type=int, default=None)

    p.add_argument("--tab-metrics", type=str, default="euclidean,manhattan,hamming")
    p.add_argument("--ts-metrics", type=str, default="dtw,euclidean,lcss")
    p.add_argument("--dtw-windows", type=str, default="0.10")

    p.add_argument("--text-encoders", type=str, default="e5,bert,tfidf,word2vec,raw")
    p.add_argument("--text-vector-metrics", type=str, default="cosine,euclidean,manhattan")
    p.add_argument("--text-direct-metrics", type=str, default="rouge_l,lcs,bleu")
    return p.parse_args()


def main():
    args = parse_args()
    factory = _load_factory(args.factory)

    factory_kwargs = {}
    if args.factory_kwargs_json:
        factory_kwargs = json.loads(args.factory_kwargs_json)
        if not isinstance(factory_kwargs, dict):
            raise TypeError("--factory-kwargs-json must decode to a JSON object.")

    produced = factory(**factory_kwargs)
    text_backend_kwargs = {}
    if isinstance(produced, dict):
        dataset = produced.get("dataset")
        model = produced.get("model")
        text_backend_kwargs = dict(produced.get("text_backend_kwargs", {}))
    elif isinstance(produced, (list, tuple)):
        if len(produced) == 0:
            raise ValueError("Factory returned an empty tuple/list.")
        dataset = produced[0]
        model = produced[1] if len(produced) > 1 else None
        text_backend_kwargs = dict(produced[2]) if len(produced) > 2 else {}
    else:
        raise TypeError(
            "Factory must return (dataset, model), (dataset, model, text_backend_kwargs), "
            "or a dict with keys {'dataset', 'model', 'text_backend_kwargs'}."
        )

    if dataset is None:
        raise ValueError("Factory did not provide a dataset.")

    sample_indices = (
        _parse_int_csv(args.sample_indices) if args.sample_indices is not None else None
    )

    run_distance_ablation(
        dataset=dataset,
        model=model,
        sample_indices=sample_indices,
        max_samples=args.max_samples,
        target_value=args.target_value,
        k=args.k,
        tab_metrics=_parse_csv(args.tab_metrics),
        ts_metrics=_parse_csv(args.ts_metrics),
        dtw_windows=_parse_float_csv(args.dtw_windows),
        text_encoders=_parse_csv(args.text_encoders),
        text_vector_metrics=_parse_csv(args.text_vector_metrics),
        text_direct_metrics=_parse_csv(args.text_direct_metrics),
        text_backend_kwargs=text_backend_kwargs,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_full=bool(args.save_full),
        max_combinations=args.max_combinations,
    )


if __name__ == "__main__":
    main()
