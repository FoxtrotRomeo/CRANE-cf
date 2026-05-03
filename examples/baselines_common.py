"""Shared utilities for the optimisation-based baseline runner.

Used by run_baselines.py.  Does NOT import any dataset-specific code.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float) and (obj != obj):   # NaN check for plain float
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_summary(payload: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


# ---------------------------------------------------------------------------
# Summary payload builder (mirrors the ablation-runner schema)
# ---------------------------------------------------------------------------

def make_summary_payload(
    *,
    baseline_name: str,
    dataset: str,
    method: str,
    modality: str,
    held_fixed: List[str],
    embedding_space: bool,
    sample_indices: Sequence[int],
    target_value: int,
    k: int,
    objectives_by_generator: Dict[str, Dict[str, float]],
    candidate_counts: Dict[str, int],
    n_converged: Optional[int],
    n_total: int,
    runtime_sec: float,
    started_at: datetime.datetime,
    finished_at: datetime.datetime,
    notes: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
    row_status: str = "ok",
    row_error: Optional[str] = None,
) -> dict:
    """Return a dict that mirrors the ablation-runner summary.json schema.

    Extra fields added at the top level (not inside rows) carry baseline-
    specific metadata so the file can be decoded unambiguously when compared
    to ablation results.
    """
    row: Dict[str, Any] = {
        "combo_id": baseline_name,
        # ablation-runner fields kept for schema compatibility
        "tab_metrics": None,
        "ts_metrics": None,
        "text_config": None,
        "image_config": None,
        "status": row_status,
        "error": row_error,
        "runtime_sec": round(runtime_sec, 3),
        "candidate_counts": {k_: int(v) for k_, v in candidate_counts.items()},
        "total_candidates": int(sum(candidate_counts.values())),
        "n_samples": int(len(sample_indices)),
        "objectives": objectives_by_generator if row_status == "ok" else None,
        "traceback": None,
        "objectives_error": None,
    }

    payload: Dict[str, Any] = {
        # --- ablation-runner schema ---
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "n_runs": 1,
        "sample_indices": [int(i) for i in sample_indices],
        "target_value": int(target_value),
        "k": int(k),
        # --- baseline-specific metadata (top-level, outside rows) ---
        "baseline_name": baseline_name,
        "dataset": dataset,
        "method": method,
        "modality": modality,
        "held_fixed": held_fixed,
        "embedding_space": embedding_space,
        "n_converged": int(n_converged) if n_converged is not None else None,
        "n_total": int(n_total),
        "notes": notes,
        **(extra_meta or {}),
        "rows": [row],
    }
    return payload


# ---------------------------------------------------------------------------
# Feature type inference
# ---------------------------------------------------------------------------

def infer_feature_types(
    X_train: np.ndarray,
    col_names: List[str],
) -> Tuple[List[str], List[str]]:
    """Return *(continuous_cols, categorical_cols)*.

    A column is classified as categorical if every non-NaN value in the
    training set is in {0.0, 1.0} and there are at most 2 distinct values.
    Everything else is continuous.

    Ambiguous features (integers with range > 2 but < 10 distinct values)
    are treated as continuous and a warning is printed.
    """
    continuous, categorical = [], []
    ambiguous = []
    for i, col in enumerate(col_names):
        vals = X_train[:, i]
        finite = vals[np.isfinite(vals)]
        unique = np.unique(finite)
        if set(unique).issubset({0.0, 1.0}):
            categorical.append(col)
        else:
            continuous.append(col)
            # Flag values that look like small integers (possible ordinal)
            if len(unique) < 10 and np.all(finite == finite.astype(int)):
                ambiguous.append(col)
    if ambiguous:
        print(
            f"[baselines_common] Ambiguous features treated as continuous "
            f"(small integer range): {ambiguous}"
        )
    return continuous, categorical


# ---------------------------------------------------------------------------
# DiCE sklearn-compatible model wrapper
# ---------------------------------------------------------------------------

class DiceModelWrapper:
    """Wrap a callable ``predict_proba_fn(arr) -> (n, n_classes)`` for DiCE.

    DiCE's sklearn backend expects an object with ``.predict_proba(df)`` and
    ``.predict(df)`` methods.  This wrapper translates between the two APIs.
    """

    def __init__(self, predict_proba_fn, feature_cols: List[str]):
        self._fn = predict_proba_fn
        self._cols = feature_cols

    def predict_proba(self, df_input) -> np.ndarray:
        import pandas as pd  # noqa: F401
        if hasattr(df_input, "values"):
            arr = df_input[self._cols].values.astype(np.float32)
        else:
            arr = np.asarray(df_input, dtype=np.float32)
        return self._fn(arr)

    def predict(self, df_input) -> np.ndarray:
        return self.predict_proba(df_input).argmax(axis=1)


# ---------------------------------------------------------------------------
# Tabular plausibility: pre-fit LOF once and reuse
# ---------------------------------------------------------------------------

def fit_tabular_lof(X_train: np.ndarray, n_neighbors: int = 20) -> LocalOutlierFactor:
    """Fit a novelty-mode LOF on ``X_train`` for reuse across candidates."""
    k = max(2, min(n_neighbors, len(X_train) - 1))
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
    lof.fit(X_train)
    return lof


def tabular_plausibility_lof(x_tab: np.ndarray, lof: LocalOutlierFactor) -> float:
    """LOF outlierness score (higher = less plausible, consistent with helpers)."""
    return -float(lof.score_samples(x_tab[np.newaxis, :])[0])


# ---------------------------------------------------------------------------
# Objective computation
# ---------------------------------------------------------------------------

def compute_tabular_objectives(
    x_cf: np.ndarray,
    x_factual: np.ndarray,
    lof: LocalOutlierFactor,
    predict_proba_fn,
    target_value: int,
) -> np.ndarray:
    """Return ``[outcome, proximity, sparsity, plausibility]`` for one tabular candidate.

    Definitions mirror the evaluation helpers:
      - outcome:     |argmax(predict(x_cf)) - target_value|
      - proximity:   mean |x_cf - x_factual|  (Gower-style)
      - sparsity:    count of changed features (L0)
      - plausibility: LOF outlierness (lower = more plausible)
    """
    proba = predict_proba_fn(x_cf[np.newaxis, :])[0]
    pred_label = float(np.argmax(proba))
    outcome = abs(pred_label - float(target_value))

    proximity = float(np.mean(np.abs(x_cf - x_factual)))
    sparsity = float(np.sum(x_cf != x_factual))
    plausibility = tabular_plausibility_lof(x_cf, lof)

    return np.array([outcome, proximity, sparsity, plausibility], dtype=np.float64)


def compute_embedding_objectives(
    emb_cf: np.ndarray,
    emb_factual: np.ndarray,
    predict_proba_fn,
    target_value: int,
) -> np.ndarray:
    """Return ``[outcome, proximity, sparsity, plausibility]`` for an embedding candidate.

    Plausibility is set to NaN because no LOF is fitted for embedding space.
    proximity:  1 - cosine_similarity(cf, factual)  ∈ [0, 2]
    sparsity:   mean |cf - factual| (normalised L1 change per dimension)
    """
    proba = predict_proba_fn(emb_cf[np.newaxis, :])[0]
    pred_label = float(np.argmax(proba))
    outcome = abs(pred_label - float(target_value))

    n_cf = np.linalg.norm(emb_cf)
    n_fac = np.linalg.norm(emb_factual)
    if n_cf < 1e-12 or n_fac < 1e-12:
        proximity = 1.0
    else:
        proximity = float(1.0 - np.dot(emb_cf, emb_factual) / (n_cf * n_fac))

    sparsity = float(np.mean(np.abs(emb_cf - emb_factual)))

    return np.array([outcome, proximity, sparsity, float("nan")], dtype=np.float64)


def run_native_guide(
    *,
    X_train_ts: np.ndarray,
    y_train: np.ndarray,
    x_ts_query: np.ndarray,
    x_static_factual: np.ndarray,
    target_value: int,
    predict_proba_ts_fn,
    k: int,
    n_segments: int = 10,
    distance_metric: str = "dtw",
) -> Tuple[List[np.ndarray], bool]:
    """Native Guide counterfactual generation for multivariate time series.

    Methodology note
    ----------------
    This is a from-scratch implementation following Delaney et al. (2021)
    "Instance-Based Counterfactual Explanations for Time Series Classification"
    (ECAI 2021).  The original repository (e-delaney/Instance-Based_CFE_TSC) is
    notebook-only and provides no importable package.

    Algorithm (NativeGuide-ISW variant):
      1. Retrieve the k nearest unlike neighbors (NUNs) from the training set
         using DTW distance on the multivariate TS (channels concatenated for
         distance purposes).
      2. For each NUN, greedily swap time-window segments from the NUN onto the
         factual TS in order of largest absolute channel-level difference between
         the two, one window at a time, testing the model prediction after each
         swap.  Stop when the prediction flips to the target class.
      3. Return the resulting counterfactual TS for each NUN that converged.

    ``n_segments``: number of equal-width windows into which the time axis is
    divided.  Each swap copies one window for all channels simultaneously.

    ``predict_proba_ts_fn(arr)``: callable receiving a (B, T, C) batch of TS
    arrays and returning (B, n_classes) probability arrays.  The static
    (tabular) branch is held fixed at factual values inside this closure.

    Returns *(cf_list, converged)* where cf_list contains up to k CFs (shape
    (T, C) each) and converged=True if at least one was found.
    """
    from aeon.distances import dtw_distance, euclidean_distance

    T, C = x_ts_query.shape
    target_mask = y_train == target_value
    X_opp = X_train_ts[target_mask]  # (N_opp, T, C)
    if len(X_opp) == 0:
        return [], False

    # --- 1. Retrieve k nearest unlike neighbors ---
    dist_fn = dtw_distance if distance_metric == "dtw" else euclidean_distance

    dists = np.array([
        dist_fn(x_ts_query.T, X_opp[i].T)   # aeon expects (C, T)
        for i in range(len(X_opp))
    ])
    k_actual = min(k, len(X_opp))
    nun_indices = np.argsort(dists)[:k_actual]

    # Segment boundaries along the time axis
    seg_edges = np.linspace(0, T, n_segments + 1, dtype=int)

    found: List[np.ndarray] = []

    for nun_idx in nun_indices:
        nun = X_opp[nun_idx]  # (T, C)
        candidate = x_ts_query.copy()

        # Rank segments by mean absolute difference across all channels
        seg_diffs = []
        for s in range(n_segments):
            t0, t1 = seg_edges[s], seg_edges[s + 1]
            diff = float(np.mean(np.abs(nun[t0:t1] - x_ts_query[t0:t1])))
            seg_diffs.append((diff, s))
        seg_diffs.sort(reverse=True)

        converged = False
        for _, s in seg_diffs:
            t0, t1 = seg_edges[s], seg_edges[s + 1]
            candidate[t0:t1] = nun[t0:t1]

            proba = predict_proba_ts_fn(candidate[np.newaxis, :, :])  # (1, n_classes)
            if proba[0].argmax() == target_value:
                converged = True
                break

        if converged:
            found.append(candidate.copy())
            if len(found) >= k:
                break

    return found, len(found) > 0


def run_comte(
    *,
    X_train_ts: np.ndarray,
    y_train: np.ndarray,
    x_ts_query: np.ndarray,
    x_static_factual: np.ndarray,
    target_value: int,
    predict_proba_ts_fn,
    k: int,
    n_segments: int = 10,
    distance_metric: str = "dtw",
) -> Tuple[List[np.ndarray], bool]:
    """CoMTE-style counterfactual generation for multivariate time series.

    Methodology note
    ----------------
    CoMTE (Ates et al., 2021, "Counterfactual Explanations for Multivariate
    Time Series") requires a Rust-compiled statistical feature extractor
    (``fast_features``) and wraps a scikit-learn Pipeline.  The original repo
    (peaclab/CoMTE) cannot be used directly with a PyTorch model or raw NumPy
    arrays without substantial re-engineering.

    This implementation follows CoMTE's greedy segment-swap logic — equivalent
    to CoMTE's ``BruteForceSearch`` fallback — without the mlrose hill-climbing
    optimisation layer and without the statistical feature extraction step:

    Algorithm:
      1. Find the k nearest unlike neighbors (NUNs) from opposite-class
         training samples using DTW distance.
      2. For each NUN, rank (channel, segment) pairs by absolute mean difference
         between the NUN and the factual TS.
      3. Greedily copy (channel, segment) pairs from the NUN onto the factual TS
         in descending order of difference magnitude, testing the model after
         each copy.  Stop as soon as the prediction flips.
      4. Return the resulting minimal-change TS.

    The key difference from Native Guide is that CoMTE swaps **per-channel**
    segments independently (finer-grained), whereas NativeGuide-ISW swaps all
    channels in a window simultaneously.

    ``predict_proba_ts_fn``, ``n_segments``, return format: same as
    ``run_native_guide``.
    """
    from aeon.distances import dtw_distance, euclidean_distance

    T, C = x_ts_query.shape
    target_mask = y_train == target_value
    X_opp = X_train_ts[target_mask]  # (N_opp, T, C)
    if len(X_opp) == 0:
        return [], False

    dist_fn = dtw_distance if distance_metric == "dtw" else euclidean_distance

    dists = np.array([
        dist_fn(x_ts_query.T, X_opp[i].T)
        for i in range(len(X_opp))
    ])
    k_actual = min(k, len(X_opp))
    nun_indices = np.argsort(dists)[:k_actual]

    seg_edges = np.linspace(0, T, n_segments + 1, dtype=int)

    found: List[np.ndarray] = []

    for nun_idx in nun_indices:
        nun = X_opp[nun_idx]  # (T, C)
        candidate = x_ts_query.copy()

        # Rank (channel, segment) pairs by absolute mean difference
        cs_diffs = []
        for c in range(C):
            for s in range(n_segments):
                t0, t1 = seg_edges[s], seg_edges[s + 1]
                diff = float(np.mean(np.abs(nun[t0:t1, c] - x_ts_query[t0:t1, c])))
                cs_diffs.append((diff, c, s))
        cs_diffs.sort(reverse=True)

        converged = False
        for _, c, s in cs_diffs:
            t0, t1 = seg_edges[s], seg_edges[s + 1]
            candidate[t0:t1, c] = nun[t0:t1, c]

            proba = predict_proba_ts_fn(candidate[np.newaxis, :, :])  # (1, n_classes)
            if proba[0].argmax() == target_value:
                converged = True
                break

        if converged:
            found.append(candidate.copy())
            if len(found) >= k:
                break

    return found, len(found) > 0


def run_gradient_vector(
    *,
    x_factual: np.ndarray,
    target_value: int,
    n_classes: int,
    forward_fn,
    predict_proba_fn,
    k: int,
    n_steps: int = 300,
    lr: float = 0.05,
    lambda_prox: float = 0.1,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], bool]:
    """Gradient-based (Wachter-style) counterfactual for a flat vector input.

    Applies Adam optimisation to a continuous input vector (tabular features or
    a pre-computed encoder embedding), holding all other model inputs fixed
    inside ``forward_fn``.

    Parameters
    ----------
    x_factual     : (D,) numpy array — the vector to perturb.
    n_classes     : 1 for models that output a binary sigmoid scalar;
                    >1 for models that output class logits.
    forward_fn    : differentiable callable ``tensor (D,) → tensor``.
                    Must return a scalar sigmoid when n_classes==1, or a
                    (n_classes,) logits tensor when n_classes>1.  Other-modality
                    inputs should be captured as constants inside the closure.
    predict_proba_fn : numpy callable ``(B, D) → (B, n_classes)`` used only
                    for convergence checking (no_grad).
    k             : number of restarts (each produces one candidate).
    n_steps, lr, lambda_prox : Adam optimisation hyper-parameters.

    Loss
    ----
    For binary sigmoid (n_classes==1):
        BCE(f(x_cf), y*) + lambda * MSE(x_cf, x_factual)
    For logits (n_classes>1):
        CrossEntropy(f(x_cf), y*) + lambda * MSE(x_cf, x_factual)

    Returns *(cf_list, converged)* where cf_list has up to k entries and
    converged=True if at least one flip was achieved.
    """
    import torch
    import torch.nn.functional as F

    x_fac_t = torch.tensor(x_factual, dtype=torch.float32).to(device)
    target_t = torch.tensor([float(target_value)], dtype=torch.float32).to(device)
    target_cls_t = torch.tensor([target_value], dtype=torch.long).to(device)

    found: List[np.ndarray] = []
    rng = np.random.RandomState(seed=42)

    for restart in range(k):
        noise_scale = 0.01 * (restart + 1)
        x_var = (x_fac_t + noise_scale * torch.tensor(
            rng.randn(*x_fac_t.shape), dtype=torch.float32, device=device
        )).detach()
        x_var.requires_grad_(True)

        optimizer = torch.optim.Adam([x_var], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            out = forward_fn(x_var)
            if n_classes == 1:
                loss_outcome = F.binary_cross_entropy(
                    out.squeeze().clamp(1e-7, 1 - 1e-7), target_t.squeeze()
                )
            else:
                loss_outcome = F.cross_entropy(out.unsqueeze(0), target_cls_t)
            loss_prox = F.mse_loss(x_var, x_fac_t)
            (loss_outcome + lambda_prox * loss_prox).backward()
            optimizer.step()

        x_cf = x_var.detach().cpu().numpy()
        proba = predict_proba_fn(x_cf[np.newaxis, :])
        if proba[0].argmax() == target_value:
            found.append(x_cf.copy())
            if len(found) >= k:
                break

    return found, len(found) > 0


def compute_ts_gradient_objectives(
    x_ts_cf: np.ndarray,
    x_ts_factual: np.ndarray,
    predict_proba_fn,
    target_value: int,
) -> np.ndarray:
    """Return ``[outcome, proximity, sparsity, plausibility]`` for a TS gradient candidate.

    proximity:   mean |x_ts_cf - x_ts_factual|  (element-wise, across T×C)
    sparsity:    fraction of channels where mean change > 0 (channel-level L0)
    plausibility: NaN (no LOF for TS)
    """
    proba = predict_proba_fn(x_ts_cf[np.newaxis, :])[0]
    pred_label = float(np.argmax(proba))
    outcome = abs(pred_label - float(target_value))

    proximity = float(np.mean(np.abs(x_ts_cf - x_ts_factual)))
    # Channel-level sparsity: how many channels changed non-trivially
    channel_changes = np.mean(np.abs(x_ts_cf - x_ts_factual), axis=0)  # (C,)
    eps = 1e-6
    sparsity = float(np.mean(channel_changes > eps))

    return np.array([outcome, proximity, sparsity, float("nan")], dtype=np.float64)


def run_dice_embedding_fast(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    target_value: int,
    predict_proba_fn,
    k: int,
    sample_size: int = 500,
    random_seed: int = 42,
) -> Tuple[List[np.ndarray], bool]:
    """Fast DiCE-random-equivalent for high-dimensional embedding spaces.

    Avoids DiCE's pandas-per-row overhead (which is catastrophic for 768-dim+
    feature spaces) by using pure numpy operations.  Implements the same
    core algorithm as DiCE random:

    1. Sample *sample_size* random opposite-class training embeddings.
    2. Iterate from varying 1 feature up to all features.
    3. In each step, randomly copy one more dimension per candidate from its
       paired training embedding, then batch-evaluate predictions.
    4. Stop early once *k* valid flip-counterfactuals are found.

    Returns *(cf_list, converged)* identical to ``run_dice_on_features``.
    """
    rng = np.random.RandomState(random_seed)
    target_mask = y_train == target_value
    X_target = X_train[target_mask]
    if len(X_target) == 0:
        return [], False

    dim = X_train.shape[1]

    # Sample random reference embeddings (from target class)
    ref_idx = rng.randint(0, len(X_target), sample_size)
    refs = X_target[ref_idx].astype(np.float32)   # (sample_size, dim)

    # Start candidates as copies of the factual query
    x_cf_batch = np.tile(x_query.astype(np.float32), (sample_size, 1))  # (sample_size, dim)

    # Pre-generate the feature-permutation order per candidate
    feature_order = np.stack([rng.permutation(dim) for _ in range(sample_size)], axis=0)
    # feature_order[i, j] = j-th feature to change for candidate i

    found: List[np.ndarray] = []

    for step in range(1, dim + 1):
        # Apply the step-th feature change for each candidate
        for i in range(sample_size):
            feat = feature_order[i, step - 1]
            x_cf_batch[i, feat] = refs[i, feat]

        # Batch evaluate
        proba = predict_proba_fn(x_cf_batch)          # (sample_size, n_classes)
        flipped = proba.argmax(axis=1) == target_value

        for i in np.where(flipped)[0]:
            found.append(x_cf_batch[i].copy())
            if len(found) >= k:
                return found[:k], True

        if len(found) >= k:
            break

    return found[:k], len(found) >= k


def run_nice(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    target_value: int,
    predict_proba_fn,
    k: int,  # kept for API symmetry with DiCE runners; NICE always returns ≤ 1 CF
    cat_feat: Optional[List[int]] = None,
    optimization: str = "proximity",
) -> Tuple[List[np.ndarray], bool]:
    """NICE counterfactual generation via the NICEx reference implementation.

    Wraps the NICEx package (Brughmans et al., 2023).  NICEx generates one
    counterfactual per call (starting from the nearest opposite-class training
    instance), which is the published algorithm's design.  The ``k`` parameter
    is accepted for API symmetry with the DiCE runners but the return list
    contains at most **one** CF per call.  This difference from DiCE (which
    targets k CFs) is documented as a methodological distinction in the paper.

    NICEx is re-instantiated once per call so that ``predict_proba_fn``
    (a per-sample closure that holds other-modality embeddings fixed at the
    current query's factual values) is correctly bound.

    ``cat_feat``: list of categorical feature indices (0-based).  If ``None``,
    columns whose training values are all in {0, 1} are inferred as categorical.

    Returns *(cf_list, converged)* where *cf_list* has 0 or 1 elements and
    *converged* is ``True`` iff a CF was found.
    """
    from nice import NICE as _NICE  # NICEx reference package

    X_train = np.asarray(X_train, dtype=np.float32)
    x_q = x_query.astype(np.float32)[np.newaxis, :]  # (1, D)

    # Infer categorical feature indices if not provided
    if cat_feat is None:
        cat_feat = []
        for i in range(X_train.shape[1]):
            finite = X_train[:, i][np.isfinite(X_train[:, i])]
            if len(finite) > 0 and set(np.unique(finite)).issubset({0.0, 1.0}):
                cat_feat.append(i)

    # Instantiate NICEx with the per-sample predict function.
    # __init__ calls predict_proba_fn(X_train) once to identify correctly
    # classified training samples (justified_cf=True).
    try:
        nice = _NICE(
            predict_fn=predict_proba_fn,
            X_train=X_train,
            cat_feat=cat_feat,
            y_train=y_train.astype(int),
            optimization=optimization,
            justified_cf=True,
            distance_metric="HEOM",
            num_normalization="minmax",
        )
    except Exception as exc:
        print(f"    [NICE] init error: {exc}")
        return [], False

    try:
        cf = nice.explain(x_q, target_class=[target_value])  # (1, D)
    except Exception as exc:
        print(f"    [NICE] explain error: {exc}")
        return [], False

    return [cf[0].astype(np.float32).copy()], True


def aggregate_objectives(objectives_list: List[np.ndarray]) -> Dict[str, float]:
    """Average a list of [outcome, proximity, sparsity, plausibility] arrays."""
    if not objectives_list:
        nan = float("nan")
        return {"outcome": nan, "proximity": nan, "sparsity": nan, "plausibility": nan}
    arr = np.array(objectives_list, dtype=np.float64)

    def _col_mean(col: np.ndarray):
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            return None
        return float(np.mean(finite))

    return {
        "outcome":      _col_mean(arr[:, 0]),
        "proximity":    _col_mean(arr[:, 1]),
        "sparsity":     _col_mean(arr[:, 2]),
        "plausibility": _col_mean(arr[:, 3]),
    }


# ---------------------------------------------------------------------------
# DiCE runner
# ---------------------------------------------------------------------------

def run_dice_on_features(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    col_names: List[str],
    continuous_cols: List[str],
    categorical_cols: List[str],
    x_query: np.ndarray,
    target_value: int,
    predict_proba_fn,
    k: int,
    sample_size: int = 1000,
    random_seed: int = 42,
    stopping_threshold: float = 0.5,
    outcome_col: str = "outcome",
) -> Tuple[List[np.ndarray], bool]:
    """Run DiCE random method on a flat feature space.

    Returns *(cf_list, converged)* where *cf_list* is a list of numpy arrays
    (each shape ``(D,)``) and *converged* is True iff DiCE found exactly *k*
    valid CFs.
    """
    import pandas as pd
    import dice_ml

    # Build training DataFrame
    train_df = pd.DataFrame(X_train, columns=col_names)
    train_df[outcome_col] = y_train.astype(int)

    # Query DataFrame
    query_df = pd.DataFrame([x_query], columns=col_names)

    wrapper = DiceModelWrapper(predict_proba_fn, col_names)

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_cols,
        outcome_name=outcome_col,
    )
    m = dice_ml.Model(model=wrapper, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    try:
        result = exp.generate_counterfactuals(
            query_df,
            total_CFs=k,
            desired_class=int(target_value),
            random_seed=random_seed,
            sample_size=sample_size,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=0.0,   # skip post-hoc to keep embedding CFs clean
        )
    except Exception as exc:
        print(f"    [DiCE] generation error: {exc}")
        return [], False

    cf_examples = result.cf_examples_list[0]
    cf_df = cf_examples.final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        return [], False

    cfs = [cf_df.iloc[i][col_names].values.astype(np.float32) for i in range(len(cf_df))]
    converged = len(cfs) >= k
    return cfs, converged
