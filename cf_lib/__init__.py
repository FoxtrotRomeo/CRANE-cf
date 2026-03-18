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
    def _collect_precomputed(
        self,
        dataset: MultimodalDataset,
        sample_idx: int,
        target_value: int,
        k: Optional[int],
    ):
        """Run generate_raw() on any registered unimodal generators once.

        Returns
        -------
        pc : dict
            Forwarded to FrankensteinNN / CombinedNN so they skip repeated
            searches.  Keys: ``"tabular"``, ``"text"``, ``"train_text_str"``,
            and one key per ``TimeSeriesNN.ts_name`` encountered.
        candidates_cache : dict
            ``{id(gen): candidates}`` — already-computed candidate lists for
            each unimodal generator so the main loop can return them directly.
        """
        pc: Dict[str, Any] = {}
        candidates_cache: Dict[int, List] = {}
        avail = dataset.available_modalities

        for gen in self.generators.values():
            # Determine the precomputed key for this generator.
            if isinstance(gen, TimeSeriesNN):
                key = gen.ts_name
            elif isinstance(gen, TabularNN):
                key = gen.tab_name if gen.tab_name is not None else "tabular"
            elif isinstance(gen, TextNN):
                key = "text"
            elif isinstance(gen, ImageNN):
                key = "image"
            else:
                continue  # compound or fusion generator — not a unimodal source

            # Skip if absent from dataset or already cached.
            if key in pc or key not in avail:
                continue

            try:
                raw = gen.generate_raw(dataset, sample_idx, target_value=target_value, k=k)
                if isinstance(gen, TextNN):
                    cands, idx, dist, train_text_str = raw
                    pc[key] = (idx, dist)
                    if "train_text_str" not in pc:
                        pc["train_text_str"] = train_text_str
                else:
                    cands, idx, dist = raw   # TabularNN, TimeSeriesNN, ImageNN
                    pc[key] = (idx, dist)
                candidates_cache[id(gen)] = cands
            except Exception as exc:  # noqa: BLE001
                print(f"[CounterfactualLibrary] precomputed collection for '{key}' failed: {exc}")
        return pc, candidates_cache

    # ------------------------------------------------------------------
    def generate(
        self,
        dataset: MultimodalDataset,
        sample_idx: int,
        model: Optional[Any] = None,
        target_value: int = 0,
        k: Optional[int] = None,
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

        if has_compound and has_unimodal:
            precomputed, candidates_cache = self._collect_precomputed(
                dataset, sample_idx, target_value, k
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
    ) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """Run all generators for multiple test samples.

        Returns
        -------
        dict mapping each sample_idx to the per-generator result dict.
        """
        return {
            idx: self.generate(dataset, idx, model=model, target_value=target_value, k=k)
            for idx in sample_indices
        }
