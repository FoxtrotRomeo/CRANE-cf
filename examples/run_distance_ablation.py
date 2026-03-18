"""Distance-ablation runner for cf_lib generators.

This script enumerates combinations of distance choices across available
modalities in a ``MultimodalDataset`` and runs counterfactual generation for
each combination.

Usage (CLI)
-----------
Provide a factory that returns either:
  - ``(dataset, model)``
  - ``(dataset, model, text_backend_kwargs)``
  - ``{"dataset": ..., "model": ..., "text_backend_kwargs": {...}, "image_backend_kwargs": {...}}``

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
import concurrent.futures
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
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN, ImageNN
from counterfactual_evaluation_helpers import compute_objectives


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


def _call_factory_compat(fn, *args):
    """Call a user-provided factory while tolerating legacy shorter signatures."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn(*args)

    positional_params = [
        p for p in params
        if p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return fn(*args[: len(positional_params)])


def _infer_n_test(dataset) -> int:
    """Return the number of test samples, regardless of which modalities are present."""
    if dataset.X_test_static is not None:
        return int(np.asarray(dataset.X_test_static).shape[0])
    if dataset.y_test is not None:
        return int(len(dataset.y_test))
    if dataset.X_test_text is not None:
        return int(len(dataset.X_test_text))
    if dataset.X_test_img is not None:
        return int(len(dataset.X_test_img))
    if dataset.X_test_ts:
        return int(next(iter(dataset.X_test_ts.values())).shape[0])
    raise ValueError(
        "Cannot determine n_test from dataset — provide --sample-indices explicitly."
    )


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


def _image_encoder_available(encoder: str, image_backend_kwargs: Dict[str, Any]) -> bool:
    enc = str(encoder).strip().lower()
    if enc == "precomputed":
        return True
    if enc == "custom":
        return image_backend_kwargs.get("embed_fn") is not None
    if enc in {"resnet50", "efficientnet_b0"}:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            return True
        except ImportError:
            return False
    if enc == "clip_vit_b32":
        try:
            import clip  # noqa: F401
            return True
        except ImportError:
            return False
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


def _build_image_configs(
    *,
    has_image: bool,
    image_encoders: Sequence[str],
    image_distance_metrics: Sequence[str],
    image_backend_kwargs: Dict[str, Any],
) -> List[Optional[Dict[str, Any]]]:
    """Build list of image (encoder, metric) config dicts, or [None] when absent.

    Each config: ``{"encoder": "resnet50", "metric": "cosine"}``.
    Returns ``[None]`` when the dataset has no image modality or no encoders
    are available, meaning image is simply skipped in that combo.
    """
    if not has_image:
        return [None]

    configs: List[Optional[Dict[str, Any]]] = []
    for enc in image_encoders:
        enc_norm = str(enc).strip().lower()
        if not _image_encoder_available(enc_norm, image_backend_kwargs):
            print(f"[ablation] Skipping unavailable image encoder '{enc_norm}'.")
            continue
        for m in image_distance_metrics:
            configs.append({"encoder": enc_norm, "metric": str(m).strip().lower()})

    if len(configs) == 0:
        print("[ablation] No image configs available; image modality will be skipped.")
        return [None]
    return configs


def _build_generators_for_combo(
    *,
    k: int,
    tab_metrics_by_name: Dict[str, str],
    ts_metrics_by_name: Dict[str, Dict[str, Any]],
    text_cfg: Optional[Dict[str, Any]],
    text_backend_kwargs: Dict[str, Any],
    image_cfg: Optional[Dict[str, Any]],
    image_backend_kwargs: Dict[str, Any],
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

    # Image modality (optional)
    if image_cfg is not None:
        image_kwargs = dict(
            k=k,
            image_encoder=image_cfg["encoder"],
            image_distance_metric=image_cfg["metric"],
        )
        image_kwargs.update(_filter_kwargs_for_callable(ImageNN.__init__, image_backend_kwargs))
        generators["Image"] = ImageNN(**image_kwargs)

    return generators


def _evaluate_combo_objectives(
    results_by_sample: Dict[int, Dict[str, Any]],
    dataset,
    objectives_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Compute mean compute_objectives scores per generator for one ablation combo.

    For each sample and each generator, calls ``compute_objectives`` on every
    candidate using data auto-filled from ``dataset``. Returns a dict mapping
    generator name → mean objectives dict with keys
    ``outcome``, ``proximity``, ``sparsity``, ``plausibility``.

    Parameters
    ----------
    results_by_sample
        ``{sample_idx: {generator_name: [candidate_dicts]}}``
    dataset
        ``MultimodalDataset`` — provides factual data and training observations.
    objectives_kwargs
        Extra keyword arguments forwarded to ``compute_objectives`` (e.g.
        ``text_objective_context``, ``plausibility_normalizer``,
        ``predict_fn``, ``y_target``, ``modality_weights``).
        Data-dependent args (``x_tab``, ``x_tab_ref``, ``X_tab_obs``,
        ``text_candidate``, ``text_factual``) are filled automatically.
    """
    X_train_static = (
        np.asarray(dataset.X_train_static)
        if dataset.X_train_static is not None else None
    )
    X_test_static = (
        np.asarray(dataset.X_test_static)
        if dataset.X_test_static is not None else None
    )
    X_test_text = getattr(dataset, "X_test_text", None)

    # accumulators: {generator_name: {objective: [values]}}
    accum: Dict[str, Dict[str, list]] = {}
    embedding_cache: Dict = {}
    text_metrics_cache: Dict = {}

    for sample_idx, gen_results in results_by_sample.items():
        x_tab_ref = X_test_static[sample_idx] if X_test_static is not None else None
        text_factual = (
            X_test_text[sample_idx]
            if X_test_text is not None and len(X_test_text) > sample_idx
            else None
        )

        for gen_name, candidates in gen_results.items():
            if gen_name not in accum:
                accum[gen_name] = {"outcome": [], "proximity": [], "sparsity": [], "plausibility": []}
            for cand in (candidates or []):
                x_tab_cand = cand.get("static")
                text_cand = cand.get("text") or cand.get("text_input")
                try:
                    objs = compute_objectives(
                        x_tab=x_tab_cand,
                        x_tab_ref=x_tab_ref,
                        X_tab_obs=X_train_static,
                        text_candidate=text_cand if text_cand is not None else text_factual,
                        text_factual=text_factual,
                        embedding_cache=embedding_cache,
                        text_metrics_cache=text_metrics_cache,
                        **objectives_kwargs,
                    )
                    accum[gen_name]["outcome"].append(float(objs[0]))
                    accum[gen_name]["proximity"].append(float(objs[1]))
                    accum[gen_name]["sparsity"].append(float(objs[2]))
                    accum[gen_name]["plausibility"].append(float(objs[3]))
                except Exception:
                    pass

    def _mean(vals):
        finite = [v for v in vals if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        gen_name: {k: _mean(v) for k, v in scores.items()}
        for gen_name, scores in accum.items()
    }


def _summarize_results(results_by_sample: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for method_results in results_by_sample.values():
        for method_name, cands in method_results.items():
            counts[method_name] = counts.get(method_name, 0) + _safe_len(cands)
    return counts


def _run_ablation_combo(
    *,
    combo_id: str,
    dataset,
    model,
    sample_indices: Sequence[int],
    target_value: int,
    k: int,
    tab_cfg: Dict[str, str],
    ts_cfg: Dict[str, Dict[str, Any]],
    text_cfg: Optional[Dict[str, Any]],
    text_backend_kwargs: Dict[str, Any],
    image_cfg: Optional[Dict[str, Any]],
    image_backend_kwargs: Dict[str, Any],
    objectives_kwargs: Optional[Dict[str, Any]],
    objectives_kwargs_factory: Optional[Any],
    extra_generators_factory: Optional[Any],
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
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
        "image_config": image_cfg,
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
            image_cfg=image_cfg,
            image_backend_kwargs=image_backend_kwargs,
        )
        if extra_generators_factory is not None:
            extras = _call_factory_compat(
                extra_generators_factory,
                tab_cfg,
                ts_cfg,
                text_cfg,
                text_backend_kwargs,
                image_cfg,
                image_backend_kwargs,
            )
            if extras:
                generators.update(extras)
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

        combo_objectives_kwargs = None
        if objectives_kwargs_factory is not None:
            combo_objectives_kwargs = _call_factory_compat(
                objectives_kwargs_factory,
                text_cfg,
                image_cfg,
            )
        if combo_objectives_kwargs is None:
            combo_objectives_kwargs = objectives_kwargs

        if combo_objectives_kwargs is not None and results:
            try:
                row["objectives"] = _evaluate_combo_objectives(
                    results, dataset, dict(combo_objectives_kwargs)
                )
            except Exception as exc_obj:
                row["objectives_error"] = f"{type(exc_obj).__name__}: {exc_obj}"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    finally:
        row["runtime_sec"] = float(time.time() - t0)

    return row, results


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
    image_encoders: Optional[Sequence[str]] = None,
    image_distance_metrics: Optional[Sequence[str]] = None,
    image_backend_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    save_full: bool = False,
    max_combinations: Optional[int] = None,
    n_jobs: int = 1,
    objectives_kwargs: Optional[Dict[str, Any]] = None,
    objectives_kwargs_factory: Optional[Any] = None,
    extra_generators_factory: Optional[Any] = None,
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
    image_encoders
        List of image encoder names to sweep over: ``"precomputed"``,
        ``"resnet50"``, ``"efficientnet_b0"``, ``"clip_vit_b32"``, or
        ``"custom"``. Encoders whose dependencies are missing are skipped.
        Defaults to ``["precomputed"]``.
    image_distance_metrics
        Distance metrics for the image NN search (e.g. ``["cosine", "euclidean"]``).
        Defaults to ``["cosine"]``.
    image_backend_kwargs
        Backend objects forwarded to ``ImageNN``: ``device``, ``batch_size``,
        ``embed_fn``. Typically produced by the factory function.
    objectives_kwargs
        Optional dict of extra keyword arguments forwarded to
        ``compute_objectives`` for each candidate after generation.
        When provided, mean objectives (outcome, proximity, sparsity,
        plausibility) per generator are added to each summary row under
        ``"objectives"``. Typical keys:

        - ``text_objective_context`` — E5/BERT backend dict
        - ``plausibility_normalizer`` — LOF normaliser dict
        - ``predict_fn`` — callable ``(x_tab, x_ts, text) -> float``
        - ``y_target`` — integer target class
        - ``modality_weights`` — ``{modality_name: float}``

        Data args (``x_tab``, ``x_tab_ref``, ``X_tab_obs``,
        ``text_candidate``, ``text_factual``) are filled automatically
        from ``dataset`` and each candidate dict.
    objectives_kwargs_factory
        Optional callable ``(text_cfg: dict | None, image_cfg: dict | None) -> dict | None``.
        Called once per combo with the combo's text and image config dicts.
        The returned dict is used as ``objectives_kwargs`` for that combo,
        overriding the static ``objectives_kwargs`` when both are supplied.
        Return ``None`` to skip objective evaluation for that combo.
    extra_generators_factory
        Optional callable
        ``(tab_cfg, ts_cfg, text_cfg, text_backend_kwargs, image_cfg, image_backend_kwargs) -> dict | None``.
        Called once per combo. The returned dict of ``{name: generator}``
        is merged into the combo's generator set alongside the standard
        unimodal generators. Return ``None`` or ``{}`` to add nothing.
    n_jobs
        Number of ablation combos to run concurrently. Values above 1 use a
        thread pool that shares the same dataset, model, and backend objects.
    """
    tab_metrics = list(tab_metrics or ["euclidean", "manhattan", "hamming"])
    ts_metrics = list(ts_metrics or ["dtw", "euclidean", "lcss"])
    dtw_windows = list(dtw_windows or [0.10])
    text_encoders = list(text_encoders or ["e5", "bert", "tfidf", "word2vec", "raw"])
    text_vector_metrics = list(text_vector_metrics or ["cosine", "euclidean", "manhattan"])
    text_direct_metrics = list(text_direct_metrics or ["rouge_l", "lcs", "bleu"])
    text_backend_kwargs = dict(text_backend_kwargs or {})
    image_encoders = list(image_encoders or ["precomputed"])
    image_distance_metrics = list(image_distance_metrics or ["cosine"])
    image_backend_kwargs = dict(image_backend_kwargs or {})

    n_test = _infer_n_test(dataset)
    if sample_indices is None:
        if max_samples is None:
            sample_indices = list(range(n_test))
        else:
            sample_indices = list(range(min(int(max_samples), n_test)))
    else:
        sample_indices = [int(i) for i in sample_indices]

    if len(sample_indices) == 0:
        raise ValueError("No sample indices provided for ablation.")

    has_static = dataset.X_train_static is not None
    tab_names = (["__primary__"] if has_static else []) + sorted(list((dataset.X_train_tab or {}).keys()))
    ts_names = sorted(list((dataset.X_train_ts or {}).keys()))
    has_text = dataset.X_train_text is not None and dataset.X_test_text is not None
    has_image = dataset.X_train_img is not None and dataset.X_test_img is not None

    tab_assignments: List[Dict[str, str]] = []
    if len(tab_names) == 0:
        tab_assignments = [{}]
    else:
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

    image_configs = _build_image_configs(
        has_image=has_image,
        image_encoders=image_encoders,
        image_distance_metrics=image_distance_metrics,
        image_backend_kwargs=image_backend_kwargs,
    )

    combo_iter = itertools.product(tab_assignments, ts_assignments, text_configs, image_configs)
    if max_combinations is not None and int(max_combinations) > 0:
        combo_iter = itertools.islice(combo_iter, int(max_combinations))

    rows = []
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    n_jobs = max(1, int(n_jobs))

    run_root = None
    jsonl_path = None
    if output_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = run_name or f"distance_ablation_{stamp}"
        run_root = Path(output_dir) / run_label
        run_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_root / "summary.jsonl"

    def _persist_combo_output(row: Dict[str, Any], results: Dict[int, Dict[str, Any]]) -> None:
        rows.append(row)

        if run_root is None:
            return

        combo_id = str(row["combo_id"])
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

        if save_full and row["status"] == "ok":
            with open(run_root / f"{combo_id}_results.pkl", "wb") as f:
                pickle.dump(results, f)
        elif row["status"] == "error":
            with open(run_root / f"{combo_id}_error.txt", "w", encoding="utf-8") as f:
                f.write(str(row.get("traceback", row.get("error"))))

    def _submit_combo(
        executor: concurrent.futures.Executor,
        combo_idx: int,
        combo_args: Tuple[
            Dict[str, str],
            Dict[str, Dict[str, Any]],
            Optional[Dict[str, Any]],
            Optional[Dict[str, Any]],
        ],
    ) -> concurrent.futures.Future:
        combo_id = f"combo_{combo_idx:05d}"
        print(f"[ablation] Running {combo_id} ...")
        tab_cfg, ts_cfg, text_cfg, image_cfg = combo_args
        return executor.submit(
            _run_ablation_combo,
            combo_id=combo_id,
            dataset=dataset,
            model=model,
            sample_indices=sample_indices,
            target_value=target_value,
            k=k,
            tab_cfg=tab_cfg,
            ts_cfg=ts_cfg,
            text_cfg=text_cfg,
            text_backend_kwargs=text_backend_kwargs,
            image_cfg=image_cfg,
            image_backend_kwargs=image_backend_kwargs,
            objectives_kwargs=objectives_kwargs,
            objectives_kwargs_factory=objectives_kwargs_factory,
            extra_generators_factory=extra_generators_factory,
        )

    combo_enumerator = enumerate(combo_iter, start=1)
    if n_jobs == 1:
        for combo_idx, combo_args in combo_enumerator:
            combo_id = f"combo_{combo_idx:05d}"
            print(f"[ablation] Running {combo_id} ...")
            tab_cfg, ts_cfg, text_cfg, image_cfg = combo_args
            row, results = _run_ablation_combo(
                combo_id=combo_id,
                dataset=dataset,
                model=model,
                sample_indices=sample_indices,
                target_value=target_value,
                k=k,
                tab_cfg=tab_cfg,
                ts_cfg=ts_cfg,
                text_cfg=text_cfg,
                text_backend_kwargs=text_backend_kwargs,
                image_cfg=image_cfg,
                image_backend_kwargs=image_backend_kwargs,
                objectives_kwargs=objectives_kwargs,
                objectives_kwargs_factory=objectives_kwargs_factory,
                extra_generators_factory=extra_generators_factory,
            )
            _persist_combo_output(row, results)
    else:
        print(f"[ablation] Running up to {n_jobs} combos in parallel.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            in_flight: Dict[concurrent.futures.Future, str] = {}

            for _ in range(n_jobs):
                try:
                    combo_idx, combo_args = next(combo_enumerator)
                except StopIteration:
                    break
                future = _submit_combo(executor, combo_idx, combo_args)
                in_flight[future] = f"combo_{combo_idx:05d}"

            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    combo_id = in_flight.pop(future)
                    row, results = future.result()
                    print(
                        f"[ablation] Finished {combo_id} "
                        f"({row['status']}, {row['runtime_sec']:.2f}s)."
                    )
                    _persist_combo_output(row, results)

                    try:
                        combo_idx, combo_args = next(combo_enumerator)
                    except StopIteration:
                        continue
                    new_future = _submit_combo(executor, combo_idx, combo_args)
                    in_flight[new_future] = f"combo_{combo_idx:05d}"

    rows.sort(key=lambda row: row["combo_id"])

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
        "image_encoders_space": image_encoders,
        "image_distance_metrics_space": image_distance_metrics,
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
    p.add_argument("--n-jobs", type=int, default=1, help="Number of ablation combos to run in parallel.")

    p.add_argument("--tab-metrics", type=str, default="euclidean,manhattan,hamming")
    p.add_argument("--ts-metrics", type=str, default="dtw,euclidean,lcss")
    p.add_argument("--dtw-windows", type=str, default="0.10")

    p.add_argument("--text-encoders", type=str, default="e5,bert,tfidf,word2vec,raw")
    p.add_argument("--text-vector-metrics", type=str, default="cosine,euclidean,manhattan")
    p.add_argument("--text-direct-metrics", type=str, default="rouge_l,lcs,bleu")

    p.add_argument(
        "--image-encoders",
        type=str,
        default="precomputed",
        help="Comma-separated image encoders to sweep: precomputed,resnet50,efficientnet_b0,clip_vit_b32.",
    )
    p.add_argument(
        "--image-distance-metrics",
        type=str,
        default="cosine",
        help="Comma-separated distance metrics for image NN search (e.g. cosine,euclidean).",
    )
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
    image_backend_kwargs = {}
    if isinstance(produced, dict):
        dataset = produced.get("dataset")
        model = produced.get("model")
        text_backend_kwargs = dict(produced.get("text_backend_kwargs", {}))
        image_backend_kwargs = dict(produced.get("image_backend_kwargs", {}))
    elif isinstance(produced, (list, tuple)):
        if len(produced) == 0:
            raise ValueError("Factory returned an empty tuple/list.")
        dataset = produced[0]
        model = produced[1] if len(produced) > 1 else None
        text_backend_kwargs = dict(produced[2]) if len(produced) > 2 else {}
        image_backend_kwargs = dict(produced[3]) if len(produced) > 3 else {}
    else:
        raise TypeError(
            "Factory must return (dataset, model), (dataset, model, text_backend_kwargs), "
            "or a dict with keys {'dataset', 'model', 'text_backend_kwargs', 'image_backend_kwargs'}."
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
        image_encoders=_parse_csv(args.image_encoders),
        image_distance_metrics=_parse_csv(args.image_distance_metrics),
        image_backend_kwargs=image_backend_kwargs,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_full=bool(args.save_full),
        max_combinations=args.max_combinations,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
