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
import contextlib
import importlib
import inspect
import itertools
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from cf_lib import CounterfactualLibrary
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN, ImageNN
from cf_lib.counterfactual_evaluation_helpers import compute_objectives


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


def _limit_worker_threads(max_threads: Optional[int]):
    if max_threads is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return contextlib.nullcontext()
    return threadpool_limits(limits=max(1, int(max_threads)))


def _infer_n_test(dataset) -> int:
    """Return the number of test samples, regardless of which modalities are present."""
    if dataset.X_test_static is not None:
        return int(np.asarray(dataset.X_test_static).shape[0])
    test_tabulars = dataset.X_test_tabular or {}
    if test_tabulars:
        return int(np.asarray(next(iter(test_tabulars.values()))).shape[0])
    if dataset.y_test is not None:
        return int(len(dataset.y_test))
    test_texts = dataset.text_modalities(split="test")
    if test_texts:
        return int(len(next(iter(test_texts.values()))))
    test_images = dataset.image_modalities(split="test")
    if test_images:
        return int(len(next(iter(test_images.values()))))
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
    if enc == "minilm":
        return (
            text_backend_kwargs.get("minilm_embed_fn") is not None
            or text_backend_kwargs.get("text_embed_fn") is not None
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
    if enc in {"resnet50", "efficientnet_b0", "vit_b_16"}:
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


def _text_branch_names_for_ablation(dataset, text_backend_kwargs: Dict[str, Any]) -> List[str]:
    auto_text_branches = bool(text_backend_kwargs.get("auto_text_branch_generators", True))
    precompute_all = bool(text_backend_kwargs.get("precompute_all_text_branches", False))
    if auto_text_branches or precompute_all:
        return list(dataset.text_names())
    return [dataset.primary_text_name] if dataset.primary_text_name is not None else []


def _image_branch_names_for_ablation(dataset, image_backend_kwargs: Dict[str, Any]) -> List[str]:
    auto_image_branches = bool(image_backend_kwargs.get("auto_image_branch_generators", True))
    if auto_image_branches:
        return list(dataset.image_names())
    return [dataset.primary_image_name] if dataset.primary_image_name is not None else []


def _build_text_generator_for_branch(
    *,
    dataset,
    k: int,
    text_name: str,
    text_cfg: Dict[str, Any],
    text_backend_kwargs: Dict[str, Any],
):
    text_kwargs = dict(
        text_name=text_name,
        k=k,
        text_encoder=text_cfg["encoder"],
        text_distance_metric=text_cfg["metric"],
    )
    text_kwargs.update(_filter_kwargs_for_callable(TextNN.__init__, text_backend_kwargs))
    if (
        str(text_cfg["encoder"]).strip().lower() == "minilm"
        and text_backend_kwargs.get("minilm_embed_fn") is not None
    ):
        text_kwargs["text_embed_fn"] = text_backend_kwargs["minilm_embed_fn"]

    precomputed_by_encoder = text_backend_kwargs.get("precomputed_text_embeddings_by_encoder")
    if isinstance(precomputed_by_encoder, dict):
        precomputed = precomputed_by_encoder.get(text_name)
        if precomputed is None and text_name == dataset.primary_text_name:
            precomputed = precomputed_by_encoder.get(text_cfg["encoder"])
        elif isinstance(precomputed, dict):
            precomputed = precomputed.get(text_cfg["encoder"])
        if isinstance(precomputed, dict):
            text_kwargs["precomputed_train_text_embeddings"] = precomputed.get("train")
            text_kwargs["precomputed_test_text_embeddings"] = precomputed.get("test")

    gen_name = "Text" if text_name == dataset.primary_text_name else f"Text[{text_name}]"
    return gen_name, TextNN(**text_kwargs)


def _build_image_generator_for_branch(
    *,
    dataset,
    k: int,
    image_name: str,
    image_cfg: Dict[str, Any],
    image_backend_kwargs: Dict[str, Any],
):
    image_kwargs = dict(
        image_name=image_name,
        k=k,
        image_encoder=image_cfg["encoder"],
        image_distance_metric=image_cfg["metric"],
    )
    image_kwargs.update(_filter_kwargs_for_callable(ImageNN.__init__, image_backend_kwargs))
    gen_name = "Image" if image_name == dataset.primary_image_name else f"Image[{image_name}]"
    return gen_name, ImageNN(**image_kwargs)


def _build_generators_for_combo(
    *,
    dataset,
    k: int,
    tab_metrics_by_name: Dict[str, str],
    ts_metrics_by_name: Dict[str, Dict[str, Any]],
    text_cfg: Optional[Dict[str, Any]],
    text_backend_kwargs: Dict[str, Any],
    image_cfg: Optional[Dict[str, Any]],
    image_backend_kwargs: Dict[str, Any],
):
    generators = {}
    primary_tab_name = dataset.primary_tabular_name

    # Primary static/tabular + named tabular modalities
    for tab_name, metric in tab_metrics_by_name.items():
        if tab_name == primary_tab_name:
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
        text_branch_names = _text_branch_names_for_ablation(dataset, text_backend_kwargs)
        for text_name in text_branch_names:
            gen_name, gen = _build_text_generator_for_branch(
                dataset=dataset,
                k=k,
                text_name=text_name,
                text_cfg=text_cfg,
                text_backend_kwargs=text_backend_kwargs,
            )
            generators[gen_name] = gen

    # Image modality (optional)
    if image_cfg is not None:
        image_branch_names = _image_branch_names_for_ablation(dataset, image_backend_kwargs)
        for image_name in image_branch_names:
            gen_name, gen = _build_image_generator_for_branch(
                dataset=dataset,
                k=k,
                image_name=image_name,
                image_cfg=image_cfg,
                image_backend_kwargs=image_backend_kwargs,
            )
            generators[gen_name] = gen

    return generators


def _build_text_embedding_lookup(
    precomputed_by_encoder: Dict[str, Any],
    dataset,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build ``{encoder: {text_string: vector}}`` lookup tables from precomputed embeddings.

    Covers the primary text branch.  Handles both the flat
    ``{encoder: {"train": arr, "test": arr}}`` format and the named-branch
    ``{branch_name: {encoder: {"train": arr, "test": arr}}}`` format.
    Returns an empty dict when ``precomputed_by_encoder`` is empty.
    """
    if not precomputed_by_encoder:
        return {}

    # Detect format by inspecting the first value
    first_val = next(iter(precomputed_by_encoder.values()), None)
    if isinstance(first_val, dict) and "train" in first_val:
        enc_splits = precomputed_by_encoder          # flat {enc: splits}
    else:
        primary = getattr(dataset, "primary_text_name", "__primary__")
        enc_splits = precomputed_by_encoder.get(primary, {})  # named-branch: pick primary

    _tt = dataset.get_text_branch(dataset.primary_text_name, split="train")
    train_texts = _tt if _tt is not None else []
    _ts = dataset.get_text_branch(dataset.primary_text_name, split="test")
    test_texts  = _ts if _ts is not None else []

    lookup: Dict[str, Dict[str, np.ndarray]] = {}
    for enc, splits in enc_splits.items():
        if not isinstance(splits, dict):
            continue
        lk: Dict[str, np.ndarray] = {}
        train_arr = splits.get("train")
        test_arr  = splits.get("test")
        if train_arr is not None:
            for t, v in zip(train_texts, train_arr):
                lk[str(t) if t is not None else ""] = v
        if test_arr is not None:
            for t, v in zip(test_texts, test_arr):
                lk.setdefault(str(t) if t is not None else "", v)
        lookup[enc] = lk
    return lookup


def _wrap_embed_fn_with_lookup(
    objectives_kwargs: Dict[str, Any],
    text_cfg: Optional[Dict[str, Any]],
    text_emb_lookup: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Return a copy of *objectives_kwargs* whose ``text_objective_context.embed_fn``
    is backed by ``text_emb_lookup``, with the original live fn as fallback.

    No-ops when the lookup is empty, the encoder is not in the lookup, or
    ``text_objective_context`` / ``embed_fn`` are absent.
    """
    if not text_emb_lookup:
        return objectives_kwargs
    ctx = (objectives_kwargs or {}).get("text_objective_context")
    if not ctx or "embed_fn" not in ctx:
        return objectives_kwargs

    encoder = (text_cfg or {}).get("encoder", "raw")
    lookup_enc = "tfidf" if encoder == "raw" else encoder
    lookup = text_emb_lookup.get(lookup_enc)
    if not lookup:
        return objectives_kwargs

    live_fn = ctx["embed_fn"]

    def _lookup_fn(texts, _lk=lookup, _live=live_fn):
        vecs = [None] * len(texts)
        miss_pos: List[int] = []
        miss_txt: List[str] = []
        for i, t in enumerate(texts):
            v = _lk.get(str(t) if t is not None else "")
            if v is not None:
                vecs[i] = v
            else:
                miss_pos.append(i)
                miss_txt.append(str(t) if t is not None else "")
        if miss_txt:
            live_embs = np.asarray(_live(miss_txt), dtype=np.float32)
            for j, pos in enumerate(miss_pos):
                vecs[pos] = live_embs[j]
        return np.stack(vecs)

    result = dict(objectives_kwargs)
    result["text_objective_context"] = {**ctx, "embed_fn": _lookup_fn}
    return result


def _prepare_unimodal_search_cache(
    *,
    dataset,
    sample_indices: Sequence[int],
    target_value: int,
    k: int,
    combo_args_list: Sequence[Tuple[Dict[str, str], Dict[str, Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
    text_backend_kwargs: Dict[str, Any],
    image_backend_kwargs: Dict[str, Any],
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """Precompute unimodal NN searches once per unique modality config.

    The returned cache stores, for each unique unimodal config, the neighbor
    indices/distances and the already-materialized unimodal candidate dicts for
    the selected sample batch. These cached results can then be reused across
    ablation combos that differ only in the *other* modalities.
    """
    selected = [int(idx) for idx in sample_indices]
    if not combo_args_list:
        return {"tabular": {}, "ts": {}, "text": {}, "image": {}}

    cache: Dict[str, Dict[Any, Dict[str, Any]]] = {
        "tabular": {},
        "ts": {},
        "text": {},
        "image": {},
    }

    needed_tab = set()
    needed_ts = set()
    needed_text = set()
    needed_image = set()

    text_branch_names = _text_branch_names_for_ablation(dataset, text_backend_kwargs)
    image_branch_names = _image_branch_names_for_ablation(dataset, image_backend_kwargs)

    for tab_cfg, ts_cfg, text_cfg, image_cfg in combo_args_list:
        for tab_name, metric in (tab_cfg or {}).items():
            needed_tab.add((tab_name, str(metric)))
        for ts_name, cfg in (ts_cfg or {}).items():
            needed_ts.add((ts_name, str(cfg.get("metric")), cfg.get("dtw_window")))
        if text_cfg is not None:
            text_sig = (
                str(text_cfg.get("encoder")),
                str(text_cfg.get("metric")),
            )
            for text_name in text_branch_names:
                needed_text.add((text_name, *text_sig))
        if image_cfg is not None:
            image_sig = (
                str(image_cfg.get("encoder")),
                str(image_cfg.get("metric")),
            )
            for image_name in image_branch_names:
                needed_image.add((image_name, *image_sig))

    for tab_name, metric in sorted(needed_tab):
        gen = (
            TabularNN(k=k, distance_metric=metric)
            if tab_name == dataset.primary_tabular_name
            else TabularNN(tab_name=tab_name, k=k, distance_metric=metric)
        )
        print(f"[ablation] Precomputing tabular search: {tab_name} / {metric}")
        cands, idx, dist = gen.generate_raw_batch(
            dataset,
            selected,
            target_value=target_value,
            k=k,
        )
        cache["tabular"][(tab_name, metric)] = {
            "pc_key": "tabular" if tab_name == dataset.primary_tabular_name else tab_name,
            "indices": idx,
            "distances": dist,
            "candidates": cands,
        }

    for ts_name, metric, dtw_window in sorted(needed_ts):
        gen = TimeSeriesNN(
            ts_name=ts_name,
            k=k,
            distance_metric=metric,
            dtw_window=dtw_window,
        )
        label = f"{metric}" if dtw_window is None else f"{metric}(window={dtw_window})"
        print(f"[ablation] Precomputing TS search: {ts_name} / {label}")
        cands, idx, dist = gen.generate_raw_batch(
            dataset,
            selected,
            target_value=target_value,
            k=k,
        )
        cache["ts"][(ts_name, metric, dtw_window)] = {
            "pc_key": ts_name,
            "indices": idx,
            "distances": dist,
            "candidates": cands,
        }

    for text_name, encoder, metric in sorted(needed_text):
        _, gen = _build_text_generator_for_branch(
            dataset=dataset,
            k=k,
            text_name=text_name,
            text_cfg={"encoder": encoder, "metric": metric},
            text_backend_kwargs=text_backend_kwargs,
        )
        print(f"[ablation] Precomputing text search: {text_name} / {encoder} / {metric}")
        cands, idx, dist, train_text_str = gen.generate_raw_batch(
            dataset,
            selected,
            target_value=target_value,
            k=k,
        )
        cache["text"][(text_name, encoder, metric)] = {
            "pc_key": ("text", text_name),
            "indices": idx,
            "distances": dist,
            "candidates": cands,
            "train_text_str": train_text_str,
        }

    for image_name, encoder, metric in sorted(needed_image):
        _, gen = _build_image_generator_for_branch(
            dataset=dataset,
            k=k,
            image_name=image_name,
            image_cfg={"encoder": encoder, "metric": metric},
            image_backend_kwargs=image_backend_kwargs,
        )
        print(f"[ablation] Precomputing image search: {image_name} / {encoder} / {metric}")
        cands, idx, dist = gen.generate_raw_batch(
            dataset,
            selected,
            target_value=target_value,
            k=k,
        )
        cache["image"][(image_name, encoder, metric)] = {
            "pc_key": ("image", image_name),
            "indices": idx,
            "distances": dist,
            "candidates": cands,
        }

    return cache


def _assemble_combo_precomputed(
    *,
    dataset,
    tab_cfg: Dict[str, str],
    ts_cfg: Dict[str, Dict[str, Any]],
    text_cfg: Optional[Dict[str, Any]],
    image_cfg: Optional[Dict[str, Any]],
    unimodal_search_cache: Dict[str, Dict[Any, Dict[str, Any]]],
    text_backend_kwargs: Dict[str, Any],
    image_backend_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    pc: Dict[str, Any] = {}
    candidate_cache_by_key: Dict[Any, Dict[int, List[Dict[str, Any]]]] = {}

    for tab_name, metric in (tab_cfg or {}).items():
        entry = unimodal_search_cache.get("tabular", {}).get((tab_name, str(metric)))
        if entry is None:
            continue
        pc[entry["pc_key"]] = (entry["indices"], entry["distances"])
        candidate_cache_by_key[entry["pc_key"]] = entry["candidates"]

    for ts_name, cfg in (ts_cfg or {}).items():
        sig = (ts_name, str(cfg.get("metric")), cfg.get("dtw_window"))
        entry = unimodal_search_cache.get("ts", {}).get(sig)
        if entry is None:
            continue
        pc[entry["pc_key"]] = (entry["indices"], entry["distances"])
        candidate_cache_by_key[entry["pc_key"]] = entry["candidates"]

    if text_cfg is not None:
        encoder = str(text_cfg.get("encoder"))
        metric = str(text_cfg.get("metric"))
        for text_name in _text_branch_names_for_ablation(dataset, text_backend_kwargs):
            entry = unimodal_search_cache.get("text", {}).get((text_name, encoder, metric))
            if entry is None:
                continue
            pc[entry["pc_key"]] = (entry["indices"], entry["distances"])
            candidate_cache_by_key[entry["pc_key"]] = entry["candidates"]
            if entry.get("train_text_str") is not None:
                pc.setdefault("train_text_str", {})[text_name] = entry["train_text_str"]

    if image_cfg is not None:
        encoder = str(image_cfg.get("encoder"))
        metric = str(image_cfg.get("metric"))
        for image_name in _image_branch_names_for_ablation(dataset, image_backend_kwargs):
            entry = unimodal_search_cache.get("image", {}).get((image_name, encoder, metric))
            if entry is None:
                continue
            pc[entry["pc_key"]] = (entry["indices"], entry["distances"])
            candidate_cache_by_key[entry["pc_key"]] = entry["candidates"]

    if candidate_cache_by_key:
        pc["_candidate_cache_by_key"] = candidate_cache_by_key
    return pc


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
    X_train_tabular = dict(dataset.X_train_tabular or {})
    X_test_tabular = dict(dataset.X_test_tabular or {})
    X_train_ts = dict(dataset.X_train_ts or {})
    X_test_ts = dict(dataset.X_test_ts or {})
    X_test_text = (
        dataset.get_text_branch(dataset.primary_text_name, split="test")
        if dataset.primary_text_name is not None
        else None
    )
    X_test_texts = dataset.text_modalities(split="test")
    X_test_images = dataset.image_modalities(split="test")

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
                # Strip private/internal keys (those starting with "_") before forwarding
                # to compute_objectives, which does not accept arbitrary kwargs.
                helper_kwargs = {
                    "tabular_objective_context",
                    "tabular_objective_contexts",
                    "ts_objective_context",
                    "ts_objective_contexts",
                    "text_objective_contexts",
                    "image_objective_contexts",
                }
                clean_kw = {k: v for k, v in objectives_kwargs.items()
                            if not k.startswith("_") and k not in helper_kwargs}
                _tabmf = objectives_kwargs.get("_tabular_modalities_fn")
                if _tabmf is not None:
                    extra_tab_kw = _tabmf(sample_idx, cand, x_tab_ref)
                elif isinstance(cand.get("tab"), dict) or x_tab_cand is not None:
                    tab_ctx_default = objectives_kwargs.get("tabular_objective_context")
                    tab_ctx_by_name = objectives_kwargs.get("tabular_objective_contexts", {}) or {}
                    tabular_modalities = {}

                    primary_tab_name = dataset.primary_tabular_name or "__primary__"
                    if x_tab_cand is not None and x_tab_ref is not None and X_train_static is not None:
                        primary_spec = {
                            "x": x_tab_cand,
                            "x_ref": x_tab_ref,
                            "X_obs": X_train_static,
                        }
                        primary_spec.update(tab_ctx_by_name.get(primary_tab_name, tab_ctx_default) or {})
                        tabular_modalities[primary_tab_name] = primary_spec

                    for name, cand_tab in (cand.get("tab") or {}).items():
                        X_obs_tab = X_train_tabular.get(name)
                        X_ref_tabs = X_test_tabular.get(name)
                        if cand_tab is None or X_obs_tab is None or X_ref_tabs is None:
                            continue
                        if len(X_ref_tabs) <= sample_idx:
                            continue
                        spec = {
                            "x": cand_tab,
                            "x_ref": X_ref_tabs[sample_idx],
                            "X_obs": X_obs_tab,
                        }
                        spec.update(tab_ctx_by_name.get(name, tab_ctx_default) or {})
                        tabular_modalities[name] = spec

                    extra_tab_kw = (
                        {"tabular_modalities": tabular_modalities}
                        if tabular_modalities
                        else {
                            "x_tab": x_tab_cand,
                            "x_tab_ref": x_tab_ref,
                            "X_tab_obs": X_train_static,
                        }
                    )
                else:
                    extra_tab_kw = {
                        "x_tab": x_tab_cand,
                        "x_tab_ref": x_tab_ref,
                        "X_tab_obs": X_train_static,
                    }
                _tsmf = objectives_kwargs.get("_ts_modalities_fn")
                if _tsmf is not None:
                    extra_ts_kw = _tsmf(sample_idx, cand)
                elif isinstance(cand.get("ts"), dict):
                    ts_ctx_default = objectives_kwargs.get("ts_objective_context")
                    ts_ctx_by_name = objectives_kwargs.get("ts_objective_contexts", {}) or {}
                    ts_modalities = {}
                    for name, cand_ts in cand["ts"].items():
                        ctx = ts_ctx_by_name.get(name, ts_ctx_default)
                        X_obs_ts = X_train_ts.get(name)
                        X_ref_ts = X_test_ts.get(name)
                        if ctx is None or cand_ts is None or X_obs_ts is None or X_ref_ts is None:
                            continue
                        if len(X_ref_ts) <= sample_idx:
                            continue
                        spec = {
                            "x": cand_ts,
                            "x_ref": X_ref_ts[sample_idx],
                            "X_obs": X_obs_ts,
                        }
                        spec.update(ctx or {})
                        ts_modalities[name] = spec
                    extra_ts_kw = {"ts_modalities": ts_modalities} if ts_modalities else {}
                else:
                    extra_ts_kw = {}
                # Support per-candidate multi-field text objectives via _text_modalities_fn.
                # When present, it returns {"text_modalities": {...}} keyed by field name,
                # replacing the single flat text_candidate/text_factual args.
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
    worker_threads_limit: Optional[int] = None,
    text_embedding_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    precomputed_seed: Optional[Dict[str, Any]] = None,
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
        with _limit_worker_threads(worker_threads_limit):
            generators = _build_generators_for_combo(
                dataset=dataset,
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
                precomputed_seed=precomputed_seed,
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

            if combo_objectives_kwargs is not None and text_embedding_lookup:
                combo_objectives_kwargs = _wrap_embed_fn_with_lookup(
                    combo_objectives_kwargs, text_cfg, text_embedding_lookup
                )

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
    precompute_unimodal_searches: bool = True,
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
        ``"resnet50"``, ``"efficientnet_b0"``, ``"vit_b_16"``,
        ``"clip_vit_b32"``, or ``"custom"``. Encoders whose dependencies
        are missing are skipped.
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
        - ``tabular_objective_context(s)`` — per-tabular-branch plausibility specs
        - ``ts_objective_context(s)`` — per-time-series specs such as
          ``segments``, ``tau_c``, and TS plausibility normalizers

        Data args for tabular, time-series, text, and image modalities are
        filled automatically from ``dataset`` and each candidate dict.
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
    precompute_unimodal_searches
        When True (default), precompute unique unimodal NN searches once per
        modality config across the selected combo set, and reuse them across
        combos that differ only in other modalities.
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

    tab_names = list(dataset.tabular_names(include_primary=True))
    ts_names = sorted(dataset.ts_names())
    has_text = bool(dataset.text_names()) and bool(dataset.text_modalities(split="test"))
    has_image = bool(dataset.image_names()) and bool(dataset.image_modalities(split="test"))

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
    combo_args_list = list(combo_iter)

    # Build lookup tables once; each combo worker reuses them to avoid GPU
    # embed_fn calls during objective evaluation.
    text_embedding_lookup = _build_text_embedding_lookup(
        text_backend_kwargs.get("precomputed_text_embeddings_by_encoder") or {},
        dataset,
    )

    unimodal_search_cache = {"tabular": {}, "ts": {}, "text": {}, "image": {}}
    if precompute_unimodal_searches and combo_args_list:
        print("[ablation] Precomputing reusable unimodal NN searches across combos.")
        unimodal_search_cache = _prepare_unimodal_search_cache(
            dataset=dataset,
            sample_indices=sample_indices,
            target_value=target_value,
            k=k,
            combo_args_list=combo_args_list,
            text_backend_kwargs=text_backend_kwargs,
            image_backend_kwargs=image_backend_kwargs,
        )

    rows = []
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    n_jobs = max(1, int(n_jobs))
    worker_threads_limit = 1 if n_jobs > 1 else None

    run_root = None
    jsonl_path = None
    _completed_combo_ids: set = set()
    if output_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = run_name or f"distance_ablation_{stamp}"
        run_root = Path(output_dir) / run_label
        run_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_root / "summary.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _entry = json.loads(_line)
                        _cid = str(_entry.get("combo_id", ""))
                        if _cid and _entry.get("status") == "ok":
                            _completed_combo_ids.add(_cid)
                            rows.append(_entry)
                    except json.JSONDecodeError:
                        pass
            if _completed_combo_ids:
                print(f"[ablation] Resuming: {len(_completed_combo_ids)} combo(s) already done, skipping.")

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
            worker_threads_limit=None,  # limit applied once in main thread for n_jobs>1
            text_embedding_lookup=text_embedding_lookup,
            precomputed_seed=_assemble_combo_precomputed(
                dataset=dataset,
                tab_cfg=tab_cfg,
                ts_cfg=ts_cfg,
                text_cfg=text_cfg,
                image_cfg=image_cfg,
                unimodal_search_cache=unimodal_search_cache,
                text_backend_kwargs=text_backend_kwargs,
                image_backend_kwargs=image_backend_kwargs,
            ),
        )

    combo_enumerator = enumerate(combo_args_list, start=1)

    def _next_pending(enumerator):
        """Advance enumerator past already-completed combos; raises StopIteration when exhausted."""
        while True:
            idx, args = next(enumerator)
            cid = f"combo_{idx:05d}"
            if cid not in _completed_combo_ids:
                return idx, args
            print(f"[ablation] Skipping {cid} (already done).")

    if n_jobs == 1:
        for combo_idx, combo_args in combo_enumerator:
            combo_id = f"combo_{combo_idx:05d}"
            if combo_id in _completed_combo_ids:
                print(f"[ablation] Skipping {combo_id} (already done).")
                continue
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
                worker_threads_limit=worker_threads_limit,
                text_embedding_lookup=text_embedding_lookup,
                precomputed_seed=_assemble_combo_precomputed(
                    dataset=dataset,
                    tab_cfg=tab_cfg,
                    ts_cfg=ts_cfg,
                    text_cfg=text_cfg,
                    image_cfg=image_cfg,
                    unimodal_search_cache=unimodal_search_cache,
                    text_backend_kwargs=text_backend_kwargs,
                    image_backend_kwargs=image_backend_kwargs,
                ),
            )
            _persist_combo_output(row, results)
    else:
        print(f"[ablation] Running up to {n_jobs} combos in parallel.")
        with _limit_worker_threads(worker_threads_limit), \
             concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            in_flight: Dict[concurrent.futures.Future, str] = {}

            for _ in range(n_jobs):
                try:
                    combo_idx, combo_args = _next_pending(combo_enumerator)
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
                        combo_idx, combo_args = _next_pending(combo_enumerator)
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
        "precompute_unimodal_searches": bool(precompute_unimodal_searches),
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
        help="Comma-separated image encoders to sweep: precomputed,resnet50,efficientnet_b0,vit_b_16,clip_vit_b32.",
    )
    p.add_argument(
        "--image-distance-metrics",
        type=str,
        default="cosine",
        help="Comma-separated distance metrics for image NN search (e.g. cosine,euclidean).",
    )
    p.add_argument(
        "--no-precompute-unimodal-searches",
        action="store_true",
        help="Disable cross-combo caching of unimodal NN searches.",
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
        precompute_unimodal_searches=not args.no_precompute_unimodal_searches,
    )


if __name__ == "__main__":
    main()
