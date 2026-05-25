"""cf_lib — NN-based counterfactual generator library.

Quickstart
----------
::

    from cf_lib import MultimodalDataset, CounterfactualLibrary
    from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN
    from cf_lib.multimodal import FrankensteinNN, CombinedNN

    dataset = MultimodalDataset(
        X_train_static=...,
        X_train_ts={"ts1": arr_train_1, "ts2": arr_train_2},
        X_train_tab={"tab1": tab_train_1, "tab2": tab_train_2},
        X_train_text=...,
        y_train=...,
        X_test_static=...,
        X_test_ts={"ts1": arr_test_1, "ts2": arr_test_2},
        X_test_tab={"tab1": tab_test_1, "tab2": tab_test_2},
        X_test_text=...,
    )

    # Run a single generator for one test sample
    gen = TabularNN(k=50)                      # primary X_train_static
    gen2 = TabularNN(tab_name="tab1", k=50)    # named tabular modality
    candidates = gen.generate(dataset, sample_idx=3, target_value=0)

    # Or use the library to run many generators at once
    lib = CounterfactualLibrary(
        generators={
            "Tabular":       TabularNN(),
            "Tab1":          TabularNN(tab_name="tab1"),
            "Time Series 1": TimeSeriesNN("ts1"),
            "Time Series 2": TimeSeriesNN("ts2"),
            "Frankenstein":  FrankensteinNN(),
            "Combined":      CombinedNN(),
        }
    )
    results = lib.generate(dataset, sample_idx=3, model=trained_model, target_value=0)
    # results -> {"Tabular": [...], "Time Series 1": [...], ...}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from cf_lib.dataset import MultimodalDataset
from cf_lib.base import CounterfactualGenerator
from cf_lib.unimodal import TabularNN, TimeSeriesNN, LabsNN, MedsNN, TextNN, ImageNN
from cf_lib.multimodal import EarlyFusionNN, IntermediateFusionNN, FrankensteinNN, CombinedNN

__all__ = [
    "MultimodalDataset",
    "CounterfactualGenerator",
    "CounterfactualLibrary",
    "TabularNN",
    "TimeSeriesNN",
    "LabsNN",
    "MedsNN",
    "TextNN",
    "ImageNN",
    "EarlyFusionNN",
    "IntermediateFusionNN",
    "FrankensteinNN",
    "CombinedNN",
]


class CounterfactualLibrary:
    """Orchestrates multiple counterfactual generators over a shared dataset.

    Parameters
    ----------
    generators : dict[str, CounterfactualGenerator]
        Mapping from a human-readable method name to a generator instance.
        Names are returned as-is in the output dict.

    Example
    -------
    ::

        lib = CounterfactualLibrary({
            "Tabular":       TabularNN(k=50),
            "Time Series 1": TimeSeriesNN("ts1", k=50),
            "Time Series 2": TimeSeriesNN("ts2", k=50),
            "Frankenstein":  FrankensteinNN(),
            "Combined":      CombinedNN(),
        })
        results = lib.generate(dataset, sample_idx=7, target_value=0)
    """

    def __init__(self, generators: Dict[str, CounterfactualGenerator]):
        self.generators = generators

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_precomputed_key(gen, dataset: MultimodalDataset):
        if isinstance(gen, TimeSeriesNN):
            return gen.ts_name
        if isinstance(gen, TabularNN):
            tab_name = dataset.resolve_tabular_branch_name(gen.tab_name)
            return tab_name if tab_name != dataset.primary_tabular_name else "tabular"
        if isinstance(gen, TextNN):
            return ("text", dataset.resolve_text_branch_name(gen.text_name))
        if isinstance(gen, ImageNN):
            return ("image", dataset.resolve_image_branch_name(gen.image_name))
        return None

    @staticmethod
    def _precomputed_key_present(key, dataset: MultimodalDataset, avail) -> bool:
        if key is None:
            return False
        if isinstance(key, tuple) and key[0] == "text":
            return key[1] in dataset.text_names()
        if isinstance(key, tuple) and key[0] == "image":
            return key[1] in dataset.image_names()
        return key in avail

    @staticmethod
    def _materialize_from_precomputed_single(
        gen,
        key,
        dataset: MultimodalDataset,
        sample_idx: int,
        precomputed: Dict[str, Any],
    ):
        idx_dist = precomputed.get(key)
        if idx_dist is None:
            return None
        indices_dict, _ = idx_dist
        if isinstance(gen, TabularNN):
            return TabularNN._materialize(
                indices_dict,
                sample_idx,
                dataset,
                distance_metric_label=gen._resolve_distance_metric_label(),
            )
        if isinstance(gen, TimeSeriesNN):
            return TimeSeriesNN._materialize(
                indices_dict,
                sample_idx,
                dataset,
                gen.ts_name,
                distance_metric_label=gen._resolve_distance_metric_label(),
            )
        if isinstance(gen, TextNN):
            text_name = dataset.resolve_text_branch_name(gen.text_name)
            train_text_raw = dataset.get_text_branch(text_name, split="train")
            if train_text_raw is None:
                return []
            return TextNN._materialize(
                indices_dict,
                sample_idx,
                dataset,
                train_text_raw,
                text_name,
                distance_metric_label=gen._resolve_distance_metric_label(),
                text_encoder_label=gen._resolve_text_encoder_label(),
            )
        if isinstance(gen, ImageNN):
            image_name = dataset.resolve_image_branch_name(gen.image_name)
            train_img = dataset.get_image_branch(image_name, split="train")
            if train_img is None:
                return []
            return ImageNN._materialize(
                indices_dict,
                sample_idx,
                dataset,
                train_img,
                image_name,
                distance_metric_label=gen._resolve_distance_metric_label(),
                image_encoder_label=gen._resolve_encoder_label(),
            )
        return None

    def _materialize_from_precomputed_batch(
        self,
        gen,
        key,
        dataset: MultimodalDataset,
        sample_indices: List[int],
        precomputed: Dict[str, Any],
    ):
        cached_by_key = precomputed.get("_candidate_cache_by_key", {}) or {}
        if key in cached_by_key:
            cached = cached_by_key[key]
            return {int(idx): cached.get(int(idx), []) for idx in sample_indices}
        return {
            int(idx): self._materialize_from_precomputed_single(
                gen, key, dataset, int(idx), precomputed
            ) or []
            for idx in sample_indices
        }

    # ------------------------------------------------------------------
    def _collect_precomputed(
        self,
        dataset: MultimodalDataset,
        sample_idx: int,
        target_value: int,
        k: Optional[int],
        precomputed_seed: Optional[Dict[str, Any]] = None,
    ):
        """Run generate_raw() on any registered unimodal generators once.

        Returns
        -------
        pc : dict
            Forwarded to FrankensteinNN / CombinedNN so they skip repeated
            searches.  Keys include ``"tabular"``, one key per
            ``TimeSeriesNN.ts_name``, and tuple keys of the form
            ``("text", text_name)`` / ``("image", image_name)``.
        candidates_cache : dict
            ``{id(gen): candidates}`` — already-computed candidate lists for
            each unimodal generator so the main loop can return them directly.
        """
        pc: Dict[str, Any] = dict(precomputed_seed or {})
        candidates_cache: Dict[int, List] = {}
        avail = dataset.available_modalities

        for gen in self.generators.values():
            try:
                key = self._resolve_precomputed_key(gen, dataset)
                if key is None:
                    continue

                key_present = self._precomputed_key_present(key, dataset, avail)
                if not key_present:
                    continue

                if key in pc:
                    cached = self._materialize_from_precomputed_single(
                        gen, key, dataset, int(sample_idx), pc
                    )
                    if cached is not None:
                        candidates_cache[id(gen)] = cached
                    continue

                raw = gen.generate_raw(dataset, sample_idx, target_value=target_value, k=k)
                if isinstance(gen, TextNN):
                    cands, idx, dist, train_text_str = raw
                    pc[key] = (idx, dist)
                    pc.setdefault("train_text_str", {})[key[1]] = train_text_str
                else:
                    cands, idx, dist = raw   # TabularNN, TimeSeriesNN, ImageNN
                    pc[key] = (idx, dist)
                candidates_cache[id(gen)] = cands
            except Exception as exc:  # noqa: BLE001
                print(f"[CounterfactualLibrary] precomputed collection for '{key}' failed: {exc}")
        return pc, candidates_cache

    def _collect_precomputed_batch(
        self,
        dataset: MultimodalDataset,
        sample_indices: List[int],
        target_value: int,
        k: Optional[int],
        precomputed_seed: Optional[Dict[str, Any]] = None,
    ):
        """Run generate_raw_batch() on unimodal generators once for many samples."""
        pc: Dict[str, Any] = dict(precomputed_seed or {})
        candidates_cache: Dict[int, Dict[int, List]] = {}
        avail = dataset.available_modalities
        selected = [int(idx) for idx in sample_indices]

        for gen in self.generators.values():
            try:
                key = self._resolve_precomputed_key(gen, dataset)
                if key is None:
                    continue

                key_present = self._precomputed_key_present(key, dataset, avail)
                if not key_present:
                    continue

                if key in pc:
                    candidates_cache[id(gen)] = self._materialize_from_precomputed_batch(
                        gen, key, dataset, selected, pc
                    )
                    continue

                if hasattr(gen, "generate_raw_batch"):
                    raw = gen.generate_raw_batch(
                        dataset, selected, target_value=target_value, k=k
                    )
                else:
                    per_sample = {}
                    idx_merged: Dict[int, Any] = {}
                    dist_merged: Dict[int, Any] = {}
                    train_text_str = None
                    for sample_idx in selected:
                        raw_single = gen.generate_raw(
                            dataset, sample_idx, target_value=target_value, k=k
                        )
                        if isinstance(gen, TextNN):
                            cands, idx, dist, train_text_str = raw_single
                        else:
                            cands, idx, dist = raw_single
                        per_sample[int(sample_idx)] = cands
                        idx_merged.update(idx)
                        dist_merged.update(dist)
                    raw = (
                        per_sample,
                        idx_merged,
                        dist_merged,
                        train_text_str,
                    ) if isinstance(gen, TextNN) else (
                        per_sample,
                        idx_merged,
                        dist_merged,
                    )

                if isinstance(gen, TextNN):
                    cands_by_sample, idx, dist, train_text_str = raw
                    pc[key] = (idx, dist)
                    pc.setdefault("train_text_str", {})[key[1]] = train_text_str
                else:
                    cands_by_sample, idx, dist = raw
                    pc[key] = (idx, dist)
                candidates_cache[id(gen)] = cands_by_sample
            except Exception as exc:  # noqa: BLE001
                print(f"[CounterfactualLibrary] batch precomputed collection for '{key}' failed: {exc}")
        return pc, candidates_cache

    # ------------------------------------------------------------------
    def generate(
        self,
        dataset: MultimodalDataset,
        sample_idx: int,
        model: Optional[Any] = None,
        target_value: int = 0,
        k: Optional[int] = None,
        precomputed_seed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run all registered generators for a single test sample.

        If any unimodal generators (TabularNN, TimeSeriesNN, TextNN) are
        registered alongside FrankensteinNN or CombinedNN, their search results
        are automatically reused — the compound generators will not repeat the
        same NN searches.

        Parameters
        ----------
        dataset      : MultimodalDataset
        sample_idx   : index into dataset.X_test_* arrays
        model        : trained classifier (required by IntermediateFusionNN)
        target_value : desired counterfactual class label (default 0)
        k            : optional k override forwarded to every generator

        Returns
        -------
        dict mapping each generator name to its list of candidate dicts.
        Generators that raise an exception are skipped (entry is empty list).
        """
        _compound_types = (FrankensteinNN, CombinedNN)
        _unimodal_types = (TabularNN, TimeSeriesNN, TextNN, ImageNN)
        has_compound = any(isinstance(g, _compound_types) for g in self.generators.values())
        has_unimodal = any(isinstance(g, _unimodal_types) for g in self.generators.values())

        if (has_compound and has_unimodal) or precomputed_seed:
            precomputed, candidates_cache = self._collect_precomputed(
                dataset, sample_idx, target_value, k, precomputed_seed=precomputed_seed
            )
        else:
            precomputed, candidates_cache = {}, {}

        results: Dict[str, List[Dict[str, Any]]] = {}
        for name, gen in self.generators.items():
            try:
                # Unimodal generators: return the already-computed candidates.
                if id(gen) in candidates_cache:
                    results[name] = candidates_cache[id(gen)]
                # Compound generators: forward the precomputed cache.
                elif isinstance(gen, _compound_types) and precomputed:
                    results[name] = gen.generate(
                        dataset=dataset,
                        sample_idx=sample_idx,
                        model=model,
                        target_value=target_value,
                        k=k,
                        precomputed=precomputed,
                    )
                # Everything else (EF NN, IF, or standalone generators).
                else:
                    results[name] = gen.generate(
                        dataset=dataset,
                        sample_idx=sample_idx,
                        model=model,
                        target_value=target_value,
                        k=k,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[CounterfactualLibrary] '{name}' failed for sample {sample_idx}: {exc}")
                results[name] = []
        return results

    # ------------------------------------------------------------------
    def generate_batch(
        self,
        dataset: MultimodalDataset,
        sample_indices: List[int],
        model: Optional[Any] = None,
        target_value: int = 0,
        k: Optional[int] = None,
        precomputed_seed: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """Run all generators for multiple test samples.

        Returns
        -------
        dict mapping each sample_idx to the per-generator result dict.
        """
        _compound_types = (FrankensteinNN, CombinedNN)
        _unimodal_types = (TabularNN, TimeSeriesNN, TextNN, ImageNN)
        has_compound = any(isinstance(g, _compound_types) for g in self.generators.values())
        has_unimodal = any(isinstance(g, _unimodal_types) for g in self.generators.values())
        selected = [int(idx) for idx in sample_indices]

        if (has_compound and has_unimodal) or precomputed_seed:
            precomputed, candidates_cache = self._collect_precomputed_batch(
                dataset, selected, target_value, k, precomputed_seed=precomputed_seed
            )
        else:
            precomputed, candidates_cache = {}, {}

        results: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
            int(idx): {} for idx in selected
        }
        for name, gen in self.generators.items():
            try:
                if id(gen) in candidates_cache:
                    batch_results = candidates_cache[id(gen)]
                elif isinstance(gen, _compound_types) and precomputed:
                    gen_batch = getattr(gen, "generate_batch", None)
                    if callable(gen_batch):
                        batch_results = gen_batch(
                            dataset=dataset,
                            sample_indices=selected,
                            model=model,
                            target_value=target_value,
                            k=k,
                            precomputed=precomputed,
                        )
                    else:
                        batch_results = {
                            int(idx): gen.generate(
                                dataset=dataset,
                                sample_idx=int(idx),
                                model=model,
                                target_value=target_value,
                                k=k,
                                precomputed=precomputed,
                            )
                            for idx in selected
                        }
                else:
                    gen_batch = getattr(gen, "generate_batch", None)
                    if callable(gen_batch):
                        if precomputed and getattr(gen, "_accepts_precomputed_batch", False):
                            batch_results = gen_batch(
                                dataset=dataset,
                                sample_indices=selected,
                                model=model,
                                target_value=target_value,
                                k=k,
                                precomputed=precomputed,
                            )
                        else:
                            batch_results = gen_batch(
                                dataset=dataset,
                                sample_indices=selected,
                                model=model,
                                target_value=target_value,
                                k=k,
                            )
                    else:
                        batch_results = {
                            int(idx): gen.generate(
                                dataset, int(idx), model=model, target_value=target_value, k=k
                            )
                            for idx in selected
                        }
            except Exception as exc:  # noqa: BLE001
                print(f"[CounterfactualLibrary] '{name}' failed for batch: {exc}")
                batch_results = {int(idx): [] for idx in selected}

            for idx in selected:
                results[int(idx)][name] = batch_results.get(int(idx), [])

        return results
