import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Sequence
import threading


_EMBEDDING_CACHE_LOCK = threading.Lock()


def _to_numpy(x):
    """Convert backend tensors/arrays to numpy, moving GPU tensors to CPU."""
    # Torch tensor path (works for cuda/cpu tensors without importing torch).
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    # TensorFlow tensor / eager tensor path.
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def outcome_objective(pred, y_target):
    """
    pred: scalar model prediction (e.g. probability)
    y_target: tuple (low, high)
    """

    low = y_target
    high = y_target
    if low <= pred <= high:
        return 0.0
    return min(abs(pred - low), abs(pred - high))

################################################################
# Text-specific objectives (E5 + LOF + token edit sparsity)
################################################################

TEXT_MAX_PENALTY_PROX = 2.0
TEXT_MAX_PENALTY_PLAUS = 1.0


def embed_e5(
    texts: List[str],
    *,
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 512,
    prefix: str = "query: ",
) -> np.ndarray:
    """Compute L2-normalized E5 sentence embeddings for a list of raw texts.

    Embedding recipe:
      1) Prefix each text with "query: " (or custom `prefix`)
      2) Tokenize with truncation/padding (max_length)
      3) Mean-pool last hidden state using attention mask
      4) L2-normalize per embedding
    """
    import torch

    if texts is None:
        raise ValueError("texts must not be None")
    if len(texts) == 0:
        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Cannot infer embedding dim for empty texts input.")
        return np.empty((0, int(hidden)), dtype=np.float32)

    prefixed = [f"{prefix}{'' if t is None else str(t)}" for t in texts]
    model.eval()

    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(prefixed), max(1, int(batch_size))):
            batch = prefixed[i : i + max(1, int(batch_size))]
            toks = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )

            # Guard against tokenizer/model vocabulary mismatch that can trigger
            # CUDA gather index-out-of-bounds assertions inside embedding lookup.
            try:
                inp = toks.get("input_ids", None)
                emb_layer = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
                n_emb = int(getattr(emb_layer, "num_embeddings", 0)) if emb_layer is not None else 0
                if inp is not None and n_emb > 0 and inp.numel() > 0:
                    max_id = int(inp.max().item())
                    min_id = int(inp.min().item())
                    if min_id < 0 or max_id >= n_emb:
                        raise ValueError(
                            f"E5 token id out of range for model embeddings: "
                            f"min_id={min_id}, max_id={max_id}, num_embeddings={n_emb}. "
                            "Check tokenizer/model checkpoint compatibility."
                        )
            except Exception:
                # Re-raise to be handled by caller fallbacks (max-penalty objective),
                # rather than crashing in a GPU kernel.
                raise

            toks = {k: v.to(device) for k, v in toks.items()}

            outputs = model(**toks)
            hidden = outputs.last_hidden_state  # (B, L, D)
            mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)

            # Mean pooling with mask.
            summed = (hidden * mask).sum(dim=1)  # (B, D)
            denom = mask.sum(dim=1).clamp_min(1e-8)  # (B, 1)
            emb = summed / denom

            # L2 normalize.
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out_chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out_chunks, axis=0)


def text_proximity_e5(
    text_factual: str,
    text_candidate: str,
    *,
    tokenizer,
    model,
    device,
    max_penalty: float = TEXT_MAX_PENALTY_PROX,
    embedding_cache: dict = None,
) -> float:
    """E5 cosine-distance proximity objective (lower is better)."""
    if text_candidate is None or str(text_candidate).strip() == "":
        return float(max_penalty)

    try:
        with _EMBEDDING_CACHE_LOCK:
            if embedding_cache is None:
                embedding_cache = {}
            emb_f = embedding_cache.get(text_factual or "")
            emb_c = embedding_cache.get(text_candidate)

        texts_to_embed = []
        indices_to_update = []
        if emb_f is None:
            texts_to_embed.append(text_factual or "")
            indices_to_update.append(0)
        if emb_c is None:
            texts_to_embed.append(text_candidate)
            indices_to_update.append(1)

        if texts_to_embed:
            new_emb = embed_e5(
                texts_to_embed,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=len(texts_to_embed),
            )
            with _EMBEDDING_CACHE_LOCK:
                for i, emb in zip(indices_to_update, new_emb):
                    if i == 0:
                        embedding_cache[text_factual or ""] = emb
                        emb_f = emb
                    elif i == 1:
                        embedding_cache[text_candidate] = emb
                        emb_c = emb
        cos = float(np.sum(emb_f * emb_c))
        prox_txt = 1.0 - cos
        return float(np.clip(prox_txt, 0.0, float(max_penalty)))
    except Exception:
        return float(max_penalty)


def percentile_rank(a: Sequence[float], x: float) -> float:
    """Empirical CDF percentile rank: mean(a <= x) in [0,1]."""
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 1.0
    return float(np.mean(arr <= float(x)))


def text_plausibility_lof(
    text_candidate: str,
    *,
    lof_model_txt,
    train_raw_scores_txt: np.ndarray,
    tokenizer,
    model,
    device,
    max_penalty: float = TEXT_MAX_PENALTY_PLAUS,
    embedding_cache: dict = None,
) -> float:
    """LOF plausibility percentile in [0,1] (lower is better)."""
    if text_candidate is None or str(text_candidate).strip() == "":
        return float(max_penalty)

    try:
        with _EMBEDDING_CACHE_LOCK:
            if embedding_cache is None:
                embedding_cache = {}
            emb = embedding_cache.get(text_candidate)

        if emb is None:
            emb = embed_e5(
                [str(text_candidate)],
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=1,
                max_length=512,
                prefix="query: ",
            )[0]
            with _EMBEDDING_CACHE_LOCK:
                embedding_cache[text_candidate] = emb
        raw_score = -float(lof_model_txt.score_samples(emb.reshape(1, -1))[0])
        plaus_txt = percentile_rank(train_raw_scores_txt, raw_score)
        return float(np.clip(plaus_txt, 0.0, 1.0))
    except Exception:
        return float(max_penalty)


def fit_text_lof_reference(
    train_texts: List[str],
    *,
    tokenizer,
    model,
    device,
    n_neighbors: int = 20,
) -> dict:
    """One-time helper: fit LOF on training E5 embeddings and cache raw scores."""
    embs = embed_e5(
        train_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=32,
        max_length=512,
        prefix="query: ",
    )
    if embs.shape[0] < 3:
        raise ValueError("Need at least 3 training texts to fit LOF plausibility model.")

    k = max(2, min(int(n_neighbors), embs.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=k, metric="euclidean", novelty=True)
    lof.fit(embs)
    train_raw = -np.asarray(lof.score_samples(embs), dtype=float)
    return {"lof_model_txt": lof, "train_raw_scores_txt": train_raw, "train_embs_txt": embs}


def _token_levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Classic token-level Levenshtein distance (insert/delete/substitute, cost=1)."""
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # delete
                curr[j - 1] + 1,  # insert
                prev[j - 1] + cost,  # substitute
            )
        prev, curr = curr, prev
    return int(prev[m])


def text_sparsity_token_edit(tokens_factual: List[str], tokens_candidate: List[str]) -> float:
    """Token-edit sparsity = Levenshtein(token)/max(1, len(factual_tokens))."""
    t1 = list(tokens_factual or [])
    t2 = list(tokens_candidate or [])
    edits = _token_levenshtein_distance(t1, t2)
    denom = max(1, len(t1))
    return float(edits) / float(denom)


def _n_model_inputs(model_like) -> int:
    """Best-effort number of inputs for a Keras model/layer."""
    try:
        return len(model_like.inputs)
    except Exception:
        return 1


def _concat_tab_ts_single(x_tab: np.ndarray, x_ts: np.ndarray) -> np.ndarray:
    """Return combined input shaped (1, T, D_tab + D_ts) for single-input early-fusion models."""
    x_tab = np.asarray(x_tab)
    x_ts = np.asarray(x_ts)
    if x_tab.ndim != 1:
        raise ValueError(f"x_tab must be 1D (D_tab,), got shape={x_tab.shape}")
    if x_ts.ndim != 2:
        raise ValueError(f"x_ts must be 2D (T, C), got shape={x_ts.shape}")

    timesteps = x_ts.shape[0]
    repeat_x_tab = np.broadcast_to(x_tab[None, :], (timesteps, x_tab.shape[0]))
    model_input = np.concatenate([repeat_x_tab, x_ts], axis=1)
    return model_input[None, :, :]


def _concat_tab_ts_batch(X_tab: np.ndarray, X_ts: np.ndarray) -> np.ndarray:
    """Return combined input shaped (N, T, D_tab + D_ts) for single-input early-fusion models."""
    X_tab = np.asarray(X_tab)
    X_ts = np.asarray(X_ts)
    if X_tab.ndim != 2:
        raise ValueError(f"X_tab must be 2D (N, D_tab), got shape={X_tab.shape}")
    if X_ts.ndim != 3:
        raise ValueError(f"X_ts must be 3D (N, T, C), got shape={X_ts.shape}")
    if X_tab.shape[0] != X_ts.shape[0]:
        raise ValueError(f"Mismatched N: X_tab has {X_tab.shape[0]}, X_ts has {X_ts.shape[0]}")

    n, timesteps, _ = X_ts.shape
    repeat_x_tab = np.broadcast_to(X_tab[:, None, :], (n, timesteps, X_tab.shape[1]))
    return np.concatenate([repeat_x_tab, X_ts], axis=2)


def _call_keras(model_like, inputs, training: bool = False):
    """Call a Keras model/layer without ever passing `training` twice.

    IMPORTANT: for multi-input models, `inputs` must be a list/tuple of arrays.
    """
    # Torch backend can retain autograd graphs across repeated inference calls
    # if gradients are enabled. Force inference/no-grad for all evaluation-time
    # forward passes to avoid GPU memory growth.
    try:
        import torch

        with torch.inference_mode():
            return model_like(inputs, training=training)
    except Exception:
        return model_like(inputs, training=training)


def _ts_dims_from_4input_model(model_like):
    """Infer (meds_dim, labs_dim, text_dim) from a 4-input model.

    Expected input order: [meds, labs, static, text].
    """
    try:
        meds_dim = model_like.inputs[0].shape[-1]
        labs_dim = model_like.inputs[1].shape[-1]
        text_dim = model_like.inputs[3].shape[-1]
    except Exception as e:
        raise ValueError("Could not infer modality dimensions from 4-input model.") from e

    if meds_dim is None or labs_dim is None:
        raise ValueError(
            "4-input model must expose known last-dimension sizes for meds and labs inputs."
        )

    meds_dim = int(meds_dim)
    labs_dim = int(labs_dim)
    text_dim = 1 if text_dim is None else int(text_dim)
    return meds_dim, labs_dim, text_dim


def _split_ts_for_4input(model_like, X_ts: np.ndarray):
    """Split combined editable TS into (X_labs, X_meds) for 4-input models.

    Editable TS is expected to be laid out as [labs | meds]. Text is handled separately.
    """
    meds_dim, labs_dim, _ = _ts_dims_from_4input_model(model_like)

    n_channels = X_ts.shape[-1]
    needed = labs_dim + meds_dim
    if n_channels < needed:
        raise ValueError(
            f"Combined x_ts has {n_channels} channels, but model needs at least "
            f"{needed} editable channels (=labs:{labs_dim}+meds:{meds_dim})."
        )

    x_labs = X_ts[:, :, :labs_dim]
    x_meds = X_ts[:, :, labs_dim : labs_dim + meds_dim]
    return x_labs, x_meds


def _coerce_text_input_for_model(text_input, text_input_spec, batch_size: int, timesteps: int):
    """Coerce text input to the rank/dtype expected by the model text branch."""
    rank = len(text_input_spec.shape)
    arr = np.asarray(text_input)

    if rank == 3:
        # Expect (B, T, D)
        if arr.ndim == 3:
            out = arr
        elif arr.ndim == 2:
            out = np.repeat(arr[None, :, :], batch_size, axis=0)
        elif arr.ndim == 1:
            out = np.repeat(arr[None, None, :], batch_size, axis=0)
            out = np.repeat(out, timesteps, axis=1)
        elif arr.ndim == 0:
            out = np.full((batch_size, timesteps, 1), float(arr))
        else:
            raise ValueError(f"Unsupported text input shape {arr.shape} for rank-3 text branch")
    elif rank == 2:
        # Expect (B, D)
        if arr.ndim == 2:
            out = arr
        elif arr.ndim == 1:
            out = np.repeat(arr[None, :], batch_size, axis=0)
        elif arr.ndim == 0:
            out = np.full((batch_size, 1), float(arr))
        else:
            raise ValueError(f"Unsupported text input shape {arr.shape} for rank-2 text branch")
    elif rank == 1:
        # Expect (B,)
        if arr.ndim == 1:
            out = arr
        elif arr.ndim == 0:
            out = np.full((batch_size,), float(arr))
        elif arr.ndim == 2 and arr.shape[1] == 1:
            out = arr[:, 0]
        else:
            raise ValueError(f"Unsupported text input shape {arr.shape} for rank-1 text branch")
    else:
        raise ValueError(f"Unsupported text input rank={rank} for 4-input model.")

    in_dtype = getattr(text_input_spec, "dtype", None)
    if in_dtype is not None:
        try:
            out = np.asarray(out).astype(np.dtype(str(in_dtype)))
        except Exception:
            out = np.asarray(out)
    return out


def _is_split_ts_dict(x) -> bool:
    return isinstance(x, dict) and "labs" in x and "meds" in x


def _as_ts_sample_or_batch(x, *, name: str):
    arr = np.asarray(x)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"{name} must be 2D or 3D, got shape={arr.shape}")


def _ensure_split_ts_sample_dict(x_ts, *, name: str):
    if not _is_split_ts_dict(x_ts):
        raise ValueError(f"{name} must be a dict with keys 'labs' and 'meds'.")
    x_labs = np.asarray(x_ts["labs"])
    x_meds = np.asarray(x_ts["meds"])
    if x_labs.ndim != 2 or x_meds.ndim != 2:
        raise ValueError(
            f"{name}['labs'] and {name}['meds'] must be 2D (T, C). "
            f"Got labs={x_labs.shape}, meds={x_meds.shape}"
        )
    return {"labs": x_labs, "meds": x_meds}


def _ensure_split_ts_batch_dict(X_ts, *, name: str):
    if not _is_split_ts_dict(X_ts):
        raise ValueError(f"{name} must be a dict with keys 'labs' and 'meds'.")
    X_labs = _as_ts_sample_or_batch(X_ts["labs"], name=f"{name}['labs']")
    X_meds = _as_ts_sample_or_batch(X_ts["meds"], name=f"{name}['meds']")
    if X_labs.shape[0] != X_meds.shape[0]:
        raise ValueError(
            f"{name} batch mismatch between labs and meds: "
            f"{X_labs.shape[0]} vs {X_meds.shape[0]}"
        )
    return {"labs": X_labs, "meds": X_meds}

###############################################################
# Tabular data evaluation metrics
###############################################################

def tabular_proximity(x_tab, x_tab_ref):
    # Gower distance
    return np.mean(np.abs(x_tab - x_tab_ref))

def tabular_sparsity(x_tab, x_tab_ref):
    # L0 norm
    return np.sum(x_tab != x_tab_ref)

def tabular_plausibility(x_tab, X_tab_obs):
    # Local Outlier Factor (LOF) of x_tab w.r.t. X_tab_obs
    # Lower value => more plausible (less outlier-like)

    n_obs = len(X_tab_obs)
    if n_obs <= 2:
        # Fallback to 1-NN distance in tabular space when LOF is not well-defined
        dists = np.mean(np.abs(X_tab_obs - x_tab), axis=1)
        return float(np.min(dists))

    n_neighbors = max(2, min(20, n_obs - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_tab_obs)

    # score_samples: higher = more inlier-like; convert to outlierness (lower = better)
    outlierness = -float(lof.score_samples(x_tab[np.newaxis, :])[0])
    return outlierness

################################################################
# Time series data evaluation metrics
################################################################

def _ts_proximity_array(x_ts, x_ts_ref):
    x_ts = np.asarray(x_ts)
    x_ts_ref = np.asarray(x_ts_ref)
    if x_ts.ndim != 2:
        raise ValueError(f"x_ts must be 2D (T, C), got shape={x_ts.shape}")
    if x_ts_ref.shape != x_ts.shape:
        raise ValueError(
            f"x_ts_ref must have same shape as x_ts; got {x_ts_ref.shape} vs {x_ts.shape}"
        )
    return float(np.mean(np.abs(x_ts - x_ts_ref)))


def ts_proximity(x_ts, x_ts_ref):
    # Mean absolute error; supports single array modality or split dict {'labs','meds'}.
    if _is_split_ts_dict(x_ts) or _is_split_ts_dict(x_ts_ref):
        x_ts = _ensure_split_ts_sample_dict(x_ts, name="x_ts")
        x_ts_ref = _ensure_split_ts_sample_dict(x_ts_ref, name="x_ts_ref")
        total_abs = 0.0
        total_count = 0
        for key in ("labs", "meds"):
            delta = np.abs(x_ts[key] - x_ts_ref[key])
            total_abs += float(np.sum(delta))
            total_count += int(delta.size)
        return 0.0 if total_count == 0 else float(total_abs / float(total_count))
    return _ts_proximity_array(x_ts, x_ts_ref)

def ts_segment_sparsity(
    x_ts,
    x_ts_ref,
    segments,
    tau_c,
    rho=0.0
):
    """Compute timestep-level sparsity of a counterfactual time series.

    Methods-style description
    ------------------------
    We quantify how much a candidate time series deviates from its reference by
    thresholding absolute per-timestep changes using channel-specific noise
    thresholds. For each channel $c$, we mark a timestep $t$ as modified when
    $|x_{t,c} - x^{ref}_{t,c}| > \tau_c[c]$, then compute the fraction of modified
    timesteps in that channel. The reported sparsity is the *sum* of these
    per-channel fractions, so higher values indicate more widespread changes
    across time and/or channels.

    Example: if $T=24$ and the channels have 3, 8, and 4 modified timesteps, the
    result is $3/24 + 8/24 + 4/24$.

    Notes
    -----
    The parameters `segments` and `rho` are accepted for backward compatibility
    with older segment-based definitions, but are ignored.
    """
    if _is_split_ts_dict(x_ts) or _is_split_ts_dict(x_ts_ref):
        x_ts = _ensure_split_ts_sample_dict(x_ts, name="x_ts")
        x_ts_ref = _ensure_split_ts_sample_dict(x_ts_ref, name="x_ts_ref")
        if not isinstance(tau_c, dict):
            raise ValueError("tau_c must be a dict {'labs','meds'} when x_ts is split.")
        total = 0.0
        for key in ("labs", "meds"):
            seg_k = segments.get(key) if isinstance(segments, dict) else segments
            total += _ts_segment_sparsity_array(
                x_ts[key],
                x_ts_ref[key],
                seg_k,
                tau_c[key],
                rho=rho,
            )
        return float(total)
    return _ts_segment_sparsity_array(x_ts, x_ts_ref, segments, tau_c, rho=rho)


def _ts_segment_sparsity_array(
    x_ts,
    x_ts_ref,
    segments,
    tau_c,
    rho=0.0,
):
    x_ts = np.asarray(x_ts)
    x_ts_ref = np.asarray(x_ts_ref)

    if x_ts.ndim != 2:
        raise ValueError(f"x_ts must be 2D (T, C), got shape={x_ts.shape}")
    if x_ts_ref.shape != x_ts.shape:
        raise ValueError(
            f"x_ts_ref must have same shape as x_ts; got {x_ts_ref.shape} vs {x_ts.shape}"
        )

    timesteps, n_channels = x_ts.shape
    if timesteps == 0:
        return 0.0

    tau_c = np.asarray(tau_c)
    if tau_c.ndim == 0:
        tau_c = np.full((n_channels,), float(tau_c), dtype=float)
    if tau_c.shape != (n_channels,):
        raise ValueError(
            f"tau_c must have shape ({n_channels},) or be a scalar; got shape={tau_c.shape}"
        )

    delta = np.abs(x_ts - x_ts_ref)  # (T, C)
    changed_mask = delta > tau_c[None, :]
    changed_counts = changed_mask.sum(axis=0).astype(float)  # (C,)
    fractions = changed_counts / float(timesteps)
    return float(np.sum(fractions))

def ts_plausibility(x_ts, X_ts_obs):
    """
    Local Outlier Factor (LOF) of x_ts w.r.t. X_ts_obs.

    Notes
    -----
    We use sklearn's LOF on a flattened representation (T*C) per sample.
    This is robust across sktime versions and avoids Panel mtype conversion
    issues that can otherwise crash optimization loops.

    Lower value => more plausible (less outlier-like).

    x_ts: (T, C)
    X_ts_obs: (N, T, C)
    """
    if _is_split_ts_dict(x_ts) or _is_split_ts_dict(X_ts_obs):
        x_ts = _ensure_split_ts_sample_dict(x_ts, name="x_ts")
        X_ts_obs = _ensure_split_ts_batch_dict(X_ts_obs, name="X_ts_obs")
        numer = 0.0
        denom = 0.0
        for key in ("labs", "meds"):
            raw = _ts_plausibility_array(x_ts[key], X_ts_obs[key])
            weight = float(max(1, X_ts_obs[key].shape[1] * X_ts_obs[key].shape[2]))
            numer += weight * float(raw)
            denom += weight
        return float(numer / max(1e-8, denom))

    return _ts_plausibility_array(x_ts, X_ts_obs)


def _ts_plausibility_array(x_ts, X_ts_obs):
    n_obs = len(X_ts_obs)
    if n_obs <= 2:
        # Fallback to 1-NN distance in TS space when LOF is not well-defined
        dists = np.mean(np.abs(X_ts_obs - x_ts), axis=(1, 2))
        return float(np.min(dists))

    X_flat = np.asarray(X_ts_obs).reshape(n_obs, -1)
    x_flat = np.asarray(x_ts).reshape(1, -1)

    n_neighbors = max(2, min(20, n_obs - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_flat)

    # score_samples: higher = more inlier-like; convert to outlierness (lower = better)
    outlierness = -float(lof.score_samples(x_flat)[0])
    return outlierness


def _normalize_unit_interval(value: float, low: float, high: float, eps: float = 1e-8) -> float:
    """Robustly map `value` to [0, 1] given calibration bounds."""
    span = float(high) - float(low)
    if not np.isfinite(span) or span < eps:
        return 0.5
    return float(np.clip((float(value) - float(low)) / span, 0.0, 1.0))


def fit_plausibility_normalizer(
    X_tab_obs,
    X_ts_obs,
    q_low: float = 5.0,
    q_high: float = 95.0,
):
    """Fit reusable calibration for merged tabular+TS plausibility.

    Returns a dict with fitted LOF models and robust score bounds used to
    normalize tabular and time-series plausibility onto a shared [0, 1] scale.
    """
    X_tab_obs = np.asarray(X_tab_obs)

    n_tab_obs = int(len(X_tab_obs))

    n_tab_neighbors = max(2, min(20, n_tab_obs - 1)) if n_tab_obs > 2 else None

    tab_lof = None
    tab_scores_obs = None

    if n_tab_neighbors is not None:
        tab_lof = LocalOutlierFactor(n_neighbors=n_tab_neighbors, novelty=True)
        tab_lof.fit(X_tab_obs)
        tab_scores_obs = -np.asarray(tab_lof.score_samples(X_tab_obs), dtype=float)

    # Fallback bounds if LOF is not well-defined (very small reference sets).
    if tab_scores_obs is None or tab_scores_obs.size == 0:
        tab_low, tab_high = 0.0, 1.0
    else:
        tab_low, tab_high = np.percentile(tab_scores_obs, [q_low, q_high]).astype(float)

    normalizer = {
        "tab_lof": tab_lof,
        "tab_low": float(tab_low),
        "tab_high": float(tab_high),
    }

    if _is_split_ts_dict(X_ts_obs):
        X_ts_obs = _ensure_split_ts_batch_dict(X_ts_obs, name="X_ts_obs")
        modal_cfg = {}
        for key in ("labs", "meds"):
            X_obs_k = np.asarray(X_ts_obs[key])
            n_obs_k = int(len(X_obs_k))
            n_neighbors_k = max(2, min(20, n_obs_k - 1)) if n_obs_k > 2 else None
            lof_k = None
            scores_k = None
            if n_neighbors_k is not None:
                X_flat_k = X_obs_k.reshape(n_obs_k, -1)
                lof_k = LocalOutlierFactor(n_neighbors=n_neighbors_k, novelty=True)
                lof_k.fit(X_flat_k)
                scores_k = -np.asarray(lof_k.score_samples(X_flat_k), dtype=float)
            if scores_k is None or scores_k.size == 0:
                low_k, high_k = 0.0, 1.0
            else:
                low_k, high_k = np.percentile(scores_k, [q_low, q_high]).astype(float)
            modal_cfg[key] = {
                "lof": lof_k,
                "low": float(low_k),
                "high": float(high_k),
                "weight": float(max(1, X_obs_k.shape[1] * X_obs_k.shape[2])),
            }
        normalizer["ts_modal"] = modal_cfg
        return normalizer

    X_ts_obs = np.asarray(X_ts_obs)
    n_ts_obs = int(len(X_ts_obs))
    n_ts_neighbors = max(2, min(20, n_ts_obs - 1)) if n_ts_obs > 2 else None
    ts_lof = None
    ts_scores_obs = None
    X_ts_flat = X_ts_obs.reshape(n_ts_obs, -1) if n_ts_obs > 0 else np.empty((0, 0), dtype=float)
    if n_ts_neighbors is not None:
        ts_lof = LocalOutlierFactor(n_neighbors=n_ts_neighbors, novelty=True)
        ts_lof.fit(X_ts_flat)
        ts_scores_obs = -np.asarray(ts_lof.score_samples(X_ts_flat), dtype=float)

    if ts_scores_obs is None or ts_scores_obs.size == 0:
        ts_low, ts_high = 0.0, 1.0
    else:
        ts_low, ts_high = np.percentile(ts_scores_obs, [q_low, q_high]).astype(float)
    normalizer["ts_lof"] = ts_lof
    normalizer["ts_low"] = float(ts_low)
    normalizer["ts_high"] = float(ts_high)
    return normalizer


def normalized_merged_plausibility(
    x_tab,
    x_ts,
    X_tab_obs,
    X_ts_obs,
    plausibility_normalizer=None,
    tab_weight: float = 0.5,
    *,
    text_candidate: str = None,
    text_objective_context: dict = None,
    embedding_cache: dict = None,
):
    """Merged tabular+TS plausibility after robust per-modality normalization."""
    """Merged multimodal plausibility after robust per-modality normalization.

    This combines:
    - Tabular plausibility (normalized LOF).
    - Time-series plausibility (normalized LOF).
    - Text plausibility (LOF percentile in E5 space), if text context is provided.

    The final score is a weighted average of the per-modality scores.
    """
    if plausibility_normalizer is None:
        plausibility_normalizer = fit_plausibility_normalizer(X_tab_obs, X_ts_obs)

    tab_lof = plausibility_normalizer.get("tab_lof", None)
    if tab_lof is None:
        tab_raw = tabular_plausibility(x_tab, X_tab_obs)
    else:
        tab_raw = -float(tab_lof.score_samples(np.asarray(x_tab)[None, :])[0])

    tab_norm = _normalize_unit_interval(
        tab_raw,
        plausibility_normalizer["tab_low"],
        plausibility_normalizer["tab_high"],
    )

    if _is_split_ts_dict(x_ts) or "ts_modal" in plausibility_normalizer:
        x_ts = _ensure_split_ts_sample_dict(x_ts, name="x_ts")
        modal_cfg = plausibility_normalizer.get("ts_modal", None)
        if modal_cfg is None:
            # Backward-compat fallback: derive modal normalizers from observed data.
            modal_cfg = fit_plausibility_normalizer(X_tab_obs, X_ts_obs).get("ts_modal", {})
        ts_num = 0.0
        ts_den = 0.0
        X_ts_obs_split = _ensure_split_ts_batch_dict(X_ts_obs, name="X_ts_obs")
        for key in ("labs", "meds"):
            cfg = modal_cfg.get(key, {})
            lof_k = cfg.get("lof", None)
            if lof_k is None:
                raw_k = _ts_plausibility_array(x_ts[key], X_ts_obs_split[key])
            else:
                raw_k = -float(lof_k.score_samples(np.asarray(x_ts[key]).reshape(1, -1))[0])
            low_k = float(cfg.get("low", 0.0))
            high_k = float(cfg.get("high", 1.0))
            norm_k = _normalize_unit_interval(raw_k, low_k, high_k)
            w_k = float(cfg.get("weight", max(1, X_ts_obs_split[key].shape[1] * X_ts_obs_split[key].shape[2])))
            ts_num += w_k * norm_k
            ts_den += w_k
        ts_norm = float(ts_num / max(1e-8, ts_den))
    else:
        ts_lof = plausibility_normalizer.get("ts_lof", None)
        if ts_lof is None:
            ts_raw = ts_plausibility(x_ts, X_ts_obs)
        else:
            x_flat = np.asarray(x_ts).reshape(1, -1)
            ts_raw = -float(ts_lof.score_samples(x_flat)[0])
        ts_norm = _normalize_unit_interval(
            ts_raw,
            plausibility_normalizer["ts_low"],
            plausibility_normalizer["ts_high"],
        )

    w_tab = float(np.clip(tab_weight, 0.0, 1.0))
    w_ts = 1.0 - w_tab
    return float(w_tab * tab_norm + w_ts * ts_norm)
    base_plaus = float(w_tab * tab_norm + w_ts * ts_norm)

    if text_objective_context is not None:
        tok = text_objective_context.get("tokenizer", None)
        e5_model = text_objective_context.get("e5_model", None)
        e5_device = text_objective_context.get("device", None)
        lof_model_txt = text_objective_context.get("lof_model_txt", None)
        train_raw_scores_txt = text_objective_context.get("train_raw_scores_txt", None)
        if embedding_cache is None:
            embedding_cache = text_objective_context.get("embedding_cache", {})

        if (
            tok is not None
            and e5_model is not None
            and e5_device is not None
            and lof_model_txt is not None
            and train_raw_scores_txt is not None
        ):
            plaus_txt = text_plausibility_lof(
                text_candidate=text_candidate,
                lof_model_txt=lof_model_txt,
                train_raw_scores_txt=train_raw_scores_txt,
                tokenizer=tok,
                model=e5_model,
                device=e5_device,
                embedding_cache=embedding_cache,
            )

            w_base_plaus = float(text_objective_context.get("w_base_plaus", 1.0))
            w_text_plaus = float(text_objective_context.get("w_text_plaus", 1.0))
            plaus_merge = text_objective_context.get("plaus_merge", "weighted_mean")

            if plaus_merge == "weighted_sum":
                return (w_base_plaus * base_plaus) + (w_text_plaus * plaus_txt)
            else:  # weighted_mean
                denom = max(1e-8, w_base_plaus + w_text_plaus)
                return ((w_base_plaus * base_plaus) + (w_text_plaus * plaus_txt)) / denom

    return base_plaus

###############################################################
# Multimodal evaluation metrics
###############################################################

def multimodal_proximity(
    x_tab,
    x_ts,
    x_tab_ref,
    x_ts_ref,
    *,
    text_factual: str = None,
    text_candidate: str = None,
    text_objective_context: dict = None,
    embedding_cache: dict = None,
):
    """Compute multimodal proximity across tabular, time-series, and optionally text data.

    This combines the tabular proximity (`tabular_proximity`) and the
    time-series proximity (`ts_proximity`) into a single objective.
    This combines:
    - Tabular proximity (`tabular_proximity`).
    - Time-series proximity (`ts_proximity`).
    - Text proximity (`text_proximity_e5`), if text context is provided.

    If text is included, the final score is a weighted sum based on weights
    in `text_objective_context`.
    """
    base_prox = float(tabular_proximity(x_tab, x_tab_ref)) + float(ts_proximity(x_ts, x_ts_ref))

    if text_objective_context is not None:
        tok = text_objective_context.get("tokenizer", None)
        e5_model = text_objective_context.get("e5_model", None)
        e5_device = text_objective_context.get("device", None)
        if embedding_cache is None:
            embedding_cache = text_objective_context.get("embedding_cache", {})

        if tok is not None and e5_model is not None and e5_device is not None:
            prox_txt = text_proximity_e5(
                text_factual=text_factual,
                text_candidate=text_candidate,
                tokenizer=tok,
                model=e5_model,
                device=e5_device,
                embedding_cache=embedding_cache,
            )
            w_base_prox = float(text_objective_context.get("w_base_prox", 1.0))
            w_text_prox = float(text_objective_context.get("w_text_prox", 1.0))
            return (w_base_prox * base_prox) + (w_text_prox * prox_txt)

    return base_prox

def multimodal_sparsity(
    x_tab,
    x_ts,
    x_tab_ref,
    x_ts_ref,
    segments,
    tau_c,
    rho: float = 0.0,
    *,
    text_tokens_factual: List[str] = None,
    text_tokens_candidate: List[str] = None,
    text_objective_context: dict = None,
):

    """Compute multimodal sparsity across tabular, time-series, and optionally text data.

    This combines the tabular sparsity (number of changed tabular features) and
    the time-series sparsity (sum of per-channel fractions of modified timesteps;
    see `ts_segment_sparsity`).
    This combines:
    - Tabular sparsity (L0 norm).
    - Time-series sparsity (sum of per-channel fractions of modified timesteps).
    - Text sparsity (token-level Levenshtein edit fraction), if text is provided.

    Notes
    -----
    `segments` is accepted for compatibility and forwarded to
    `ts_segment_sparsity`, where it is currently ignored.
    If text is included, the final score is a weighted sum based on weights
    in `text_objective_context`.
    """
    base_sparse = float(tabular_sparsity(x_tab, x_tab_ref)) + float(
        ts_segment_sparsity(x_ts, x_ts_ref, segments, tau_c, rho)
    )

    if text_objective_context is not None and text_tokens_factual is not None and text_tokens_candidate is not None:
        sparse_txt = text_sparsity_token_edit(text_tokens_factual, text_tokens_candidate)
        w_base_sparse = float(text_objective_context.get("w_base_sparse", 1.0))
        w_text_sparse = float(text_objective_context.get("w_text_sparse", 1.0))
        return (w_base_sparse * base_sparse) + (w_text_sparse * sparse_txt)

    return base_sparse

def multimodal_plausibility(
    x_tab,
    x_ts,
    X_tab_obs,
    X_ts_obs,
    encoder
):
    """
    Plausibility of (x_tab, x_ts) measured as LOF outlierness in the encoder's latent space.

    encoder: callable mapping (x_tab, x_ts) -> latent vector (or tensor)
    Lower value => more plausible (less outlier-like).
    """
    n_in = _n_model_inputs(encoder)

    # Single-input early-fusion encoder: (N, T, D)
    if n_in == 1:
        z = _call_keras(encoder, _concat_tab_ts_single(x_tab, x_ts), training=False)
        Z_obs = _call_keras(encoder, _concat_tab_ts_batch(X_tab_obs, X_ts_obs), training=False)

    # Two-input encoder: (N, D_tab) and (N, T, C)
    elif n_in == 2:
        x_tab_b = np.asarray(x_tab)[None, :]
        x_ts_b = np.asarray(x_ts)[None, :, :]
        z = _call_keras(encoder, [x_tab_b, x_ts_b], training=False)
        Z_obs = _call_keras(encoder, [np.asarray(X_tab_obs), np.asarray(X_ts_obs)], training=False)

    else:
        raise ValueError(f"Unsupported encoder with {n_in} inputs")

    Z_obs = _to_numpy(Z_obs)
    z_arr = _to_numpy(z)

    # Ensure shapes: Z_obs -> (N, D), z_arr -> (1, D)
    if Z_obs.ndim != 2:
        Z_obs = Z_obs.reshape(Z_obs.shape[0], -1)

    if z_arr.ndim >= 2:
        z_arr = z_arr[0]
    z_arr = z_arr.reshape(1, -1)

    n_obs = Z_obs.shape[0]
    if n_obs <= 2:
        # Fallback when LOF is not well-defined: 1-NN distance in latent space
        dists = np.linalg.norm(Z_obs - z_arr, axis=1)
        return float(np.min(dists))

    n_neighbors = max(2, min(20, n_obs - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(Z_obs)

    # score_samples: higher = more inlier-like; convert to outlierness (lower = better)
    outlierness = -float(lof.score_samples(z_arr)[0])
    return outlierness

def compute_tau_c(X_ts_obs, factor=1.1):
    """
    X_ts_obs: (N, T, C) normalized time series OR {'labs','meds'} dict of such arrays
    Returns: tau_c of shape (C,) or dict with per-modality tau.
    """
    if _is_split_ts_dict(X_ts_obs):
        X_ts_obs = _ensure_split_ts_batch_dict(X_ts_obs, name="X_ts_obs")
        return {
            "labs": compute_tau_c(X_ts_obs["labs"], factor=factor),
            "meds": compute_tau_c(X_ts_obs["meds"], factor=factor),
        }

    # Flatten over samples and time
    _, _, C = X_ts_obs.shape
    tau_c = np.zeros(C)

    for c in range(C):
        values = X_ts_obs[:, :, c].ravel()
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        tau_c[c] = factor * mad

    return tau_c

################################################################
# One function to compute them all
################################################################

def compute_objectives(
    x_tab,
    x_ts,
    x_tab_ref,
    x_ts_ref,
    X_tab_obs,
    X_ts_obs,
    model,
    segments,
    tau_c,
    y_target,
    rho=0.0,
    plausibility_normalizer=None,
    tab_plausibility_weight: float = 0.5,
    text_factual: str = None,
    text_candidate: str = None,
    text_tokens_factual: List[str] = None,
    text_tokens_candidate: List[str] = None,
    text_objective_context: dict = None,
    text_input_ref=None,
    text_input_cf=None,
    embedding_cache: dict = None,
    pred_override: float = None,
    text_metrics_cache: dict = None,
):

    """Compute merged objectives for tabular, time-series, and text modalities.

    If `text_objective_context` is provided, proximity/sparsity/plausibility
    objectives are augmented with text-specific terms:
      - proximity: E5 cosine distance
      - sparsity: token-level Levenshtein edit fraction
      - plausibility: LOF percentile in E5 space
    This function computes four objectives for a counterfactual candidate:
    1. Outcome: Distance to the target prediction.
    2. Proximity: A weighted combination of distances in tabular, time-series,
       and (if provided) text embedding space.
    3. Sparsity: A weighted combination of feature changes in tabular,
       time-series, and (if provided) text token space.
    4. Plausibility: A weighted combination of plausibility scores (LOF-based)
       for tabular, time-series, and (if provided) text modalities.

    The text-related computations are only performed if `text_objective_context`
    is provided.
    """
    if embedding_cache is None:
        embedding_cache = {}
    if pred_override is None:
        n_in = _n_model_inputs(model)
        if n_in in (1, 2) and _is_split_ts_dict(x_ts):
            raise ValueError(
                "Split TS dict inputs require a 4-input model (meds, labs, static, text)."
            )

        if n_in == 1:
            model_input = _concat_tab_ts_single(x_tab, x_ts)
            pred = _call_keras(model, model_input, training=False)
        elif n_in == 2:
            x_tab_b = np.asarray(x_tab)[None, :]
            x_ts_b = np.asarray(x_ts)[None, :, :]
            pred = _call_keras(model, [x_tab_b, x_ts_b], training=False)
        elif n_in == 4:
            x_tab_b = np.asarray(x_tab)[None, :]
            if _is_split_ts_dict(x_ts):
                split = _ensure_split_ts_sample_dict(x_ts, name="x_ts")
                x_labs_b = split["labs"][None, :, :]
                x_meds_b = split["meds"][None, :, :]
            else:
                x_ts_b = np.asarray(x_ts)[None, :, :]
                x_labs_b, x_meds_b = _split_ts_for_4input(model, x_ts_b)
            x_text_src = text_input_cf if text_input_cf is not None else text_input_ref
            if x_text_src is None:
                # Backward-compatible fallback: zero text input if no explicit text value is provided.
                x_text_src = 0.0
            x_text_b = _coerce_text_input_for_model(
                x_text_src,
                model.inputs[3],
                batch_size=1,
                timesteps=x_labs_b.shape[1],
            )
            pred = _call_keras(model, [x_meds_b, x_labs_b, x_tab_b, x_text_b], training=False)
        else:
            raise ValueError(f"Unsupported model with {n_in} inputs")

        pred = float(_to_numpy(pred).squeeze())
    else:
        pred = float(pred_override)
    obj_outcome = outcome_objective(pred, y_target)

    # Pass text-related arguments to the multimodal objective functions
    obj_prox = multimodal_proximity(
        x_tab,
        x_ts,
        x_tab_ref,
        x_ts_ref,
        text_factual=text_factual,
        text_candidate=text_candidate,
        text_objective_context=text_objective_context,
        embedding_cache=embedding_cache,
    )
    obj_sparse = multimodal_sparsity(
        x_tab, x_ts, x_tab_ref, x_ts_ref, segments, tau_c, rho,
        text_tokens_factual=text_tokens_factual,
        text_tokens_candidate=text_tokens_candidate,
        text_objective_context=text_objective_context,
    )
    obj_plaus = normalized_merged_plausibility(
        x_tab, x_ts, X_tab_obs, X_ts_obs,
        plausibility_normalizer=plausibility_normalizer,
        tab_weight=tab_plausibility_weight,
        text_candidate=text_candidate,
        text_objective_context=text_objective_context,
        embedding_cache=embedding_cache,
    )

    # Optional text objective augmentation.
    if text_objective_context is not None:
        tok = text_objective_context.get("tokenizer", None)
        e5_model = text_objective_context.get("e5_model", None)
        e5_device = text_objective_context.get("device", None)

        w_base_prox = float(text_objective_context.get("w_base_prox", 1.0))
        w_text_prox = float(text_objective_context.get("w_text_prox", 1.0))
        w_base_sparse = float(text_objective_context.get("w_base_sparse", 1.0))
        w_text_sparse = float(text_objective_context.get("w_text_sparse", 1.0))
        w_base_plaus = float(text_objective_context.get("w_base_plaus", 1.0))
        w_text_plaus = float(text_objective_context.get("w_text_plaus", 1.0))
        plaus_merge = text_objective_context.get("plaus_merge", "weighted_mean")

        # Proximity (E5 cosine distance)
        txt_key = "" if text_candidate is None else str(text_candidate)
        cache_bucket = None
        if text_metrics_cache is not None:
            cache_bucket = text_metrics_cache.setdefault(txt_key, {})

        if tok is not None and e5_model is not None and e5_device is not None:
            prox_txt = cache_bucket.get("prox_txt") if cache_bucket is not None else None
            if prox_txt is None:
                prox_txt = text_proximity_e5(
                    text_factual=text_factual,
                    text_candidate=text_candidate,
                    tokenizer=tok,
                    model=e5_model,
                    device=e5_device,
                    embedding_cache=embedding_cache,
                )
                if cache_bucket is not None:
                    cache_bucket["prox_txt"] = float(prox_txt)
            obj_prox = (w_base_prox * float(obj_prox)) + (w_text_prox * float(prox_txt))

        # Sparsity (token edit fraction)
        tokenize_fn = text_objective_context.get("tokenize_fn", None)
        if text_tokens_factual is None and text_factual is not None and callable(tokenize_fn):
            try:
                text_tokens_factual = list(tokenize_fn(text_factual))
            except Exception:
                text_tokens_factual = []
        if text_tokens_candidate is None and text_candidate is not None and callable(tokenize_fn):
            try:
                text_tokens_candidate = list(tokenize_fn(text_candidate))
            except Exception:
                text_tokens_candidate = []

        if text_tokens_factual is not None and text_tokens_candidate is not None:
            sparse_txt = cache_bucket.get("sparse_txt") if cache_bucket is not None else None
            if sparse_txt is None:
                sparse_txt = text_sparsity_token_edit(text_tokens_factual, text_tokens_candidate)
                if cache_bucket is not None:
                    cache_bucket["sparse_txt"] = float(sparse_txt)
            obj_sparse = (w_base_sparse * float(obj_sparse)) + (w_text_sparse * float(sparse_txt))

        # Plausibility (LOF percentile in E5 space)
        lof_model_txt = text_objective_context.get("lof_model_txt", None)
        train_raw_scores_txt = text_objective_context.get("train_raw_scores_txt", None)
        embedding_cache = text_objective_context.get("embedding_cache", {})
        if (
            tok is not None
            and e5_model is not None
            and e5_device is not None
            and lof_model_txt is not None
            and train_raw_scores_txt is not None
        ):
            plaus_txt = cache_bucket.get("plaus_txt") if cache_bucket is not None else None
            if plaus_txt is None:
                plaus_txt = text_plausibility_lof(
                    text_candidate=text_candidate,
                    lof_model_txt=lof_model_txt,
                    train_raw_scores_txt=train_raw_scores_txt,
                    tokenizer=tok,
                    model=e5_model,
                    device=e5_device,
                    embedding_cache=embedding_cache,
                )
                if cache_bucket is not None:
                    cache_bucket["plaus_txt"] = float(plaus_txt)
            if plaus_merge == "weighted_sum":
                obj_plaus = (w_base_plaus * float(obj_plaus)) + (w_text_plaus * float(plaus_txt))
            else:
                denom = max(1e-8, w_base_plaus + w_text_plaus)
                obj_plaus = ((w_base_plaus * float(obj_plaus)) + (w_text_plaus * float(plaus_txt))) / denom

    return np.array([obj_outcome, obj_prox, obj_sparse, obj_plaus], dtype=float)


def _as_2d_tab(x_tab: np.ndarray, n_tab: int) -> np.ndarray:
    x_tab = np.asarray(x_tab)
    if x_tab.ndim == 1:
        if x_tab.shape[0] != n_tab:
            raise ValueError(f"x_tab has shape {x_tab.shape}, expected ({n_tab},)")
        return x_tab[None, :]
    if x_tab.ndim == 2:
        if x_tab.shape[1] != n_tab:
            raise ValueError(f"x_tab has shape {x_tab.shape}, expected (N, {n_tab})")
        return x_tab
    raise ValueError(f"x_tab must be 1D or 2D, got shape={x_tab.shape}")


def _as_3d_ts(x_ts: np.ndarray, timesteps: int, n_channels: int) -> np.ndarray:
    x_ts = np.asarray(x_ts)
    if x_ts.ndim == 2:
        if x_ts.shape != (timesteps, n_channels):
            raise ValueError(
                f"x_ts has shape {x_ts.shape}, expected ({timesteps}, {n_channels})"
            )
        return x_ts[None, :, :]
    if x_ts.ndim == 3:
        if x_ts.shape[1:] != (timesteps, n_channels):
            raise ValueError(
                f"x_ts has shape {x_ts.shape}, expected (N, {timesteps}, {n_channels})"
            )
        return x_ts
    raise ValueError(f"x_ts must be 2D or 3D, got shape={x_ts.shape}")


def _split_early_fusion(
    X_ef: np.ndarray,
    n_tab: int,
    n_channels: int,
):
    """Split early-fusion array(s) into (X_tab, X_ts).

    Supports:
      - (T, n_tab + C) for a single sample
      - (N, T, n_tab + C) for a batch
    Assumes the first n_tab features are the repeated tabular part.
    """
    X_ef = np.asarray(X_ef)
    if X_ef.ndim == 2:
        if X_ef.shape[1] != n_tab + n_channels:
            raise ValueError(
                f"X_ef has shape {X_ef.shape}, expected (T, {n_tab + n_channels})"
            )
        x_tab = np.asarray(X_ef[0, :n_tab])
        x_ts = np.asarray(X_ef[:, n_tab:])
        return x_tab[None, :], x_ts[None, :, :]

    if X_ef.ndim == 3:
        if X_ef.shape[2] != n_tab + n_channels:
            raise ValueError(
                f"X_ef has shape {X_ef.shape}, expected (N, T, {n_tab + n_channels})"
            )
        X_tab = np.asarray(X_ef[:, 0, :n_tab])
        X_ts = np.asarray(X_ef[:, :, n_tab:])
        return X_tab, X_ts

    raise ValueError(f"X_ef must be 2D or 3D, got shape={X_ef.shape}")


def _predict_scores(model, X_tab: np.ndarray, X_ts: np.ndarray, X_text=None) -> np.ndarray:
    """Return model scores as a 1D float array."""
    n_in = _n_model_inputs(model)
    if n_in in (1, 2) and _is_split_ts_dict(X_ts):
        raise ValueError(
            "Split TS dict batches require a 4-input model (meds, labs, static, text)."
        )

    if n_in == 1:
        model_input = _concat_tab_ts_batch(X_tab, X_ts)
        pred = _call_keras(model, model_input, training=False)
    elif n_in == 2:
        pred = _call_keras(model, [np.asarray(X_tab), np.asarray(X_ts)], training=False)
    elif n_in == 4:
        X_tab = np.asarray(X_tab)
        if _is_split_ts_dict(X_ts):
            split = _ensure_split_ts_batch_dict(X_ts, name="X_ts")
            X_labs = split["labs"]
            X_meds = split["meds"]
        else:
            X_ts = np.asarray(X_ts)
            X_labs, X_meds = _split_ts_for_4input(model, X_ts)
        if X_tab.shape[0] != X_labs.shape[0] or X_tab.shape[0] != X_meds.shape[0]:
            raise ValueError(
                f"Batch mismatch between X_tab ({X_tab.shape[0]}) and split TS "
                f"(labs={X_labs.shape[0]}, meds={X_meds.shape[0]})."
            )
        x_text_src = X_text if X_text is not None else 0.0
        X_text_b = _coerce_text_input_for_model(
            x_text_src,
            model.inputs[3],
            batch_size=X_tab.shape[0],
            timesteps=X_labs.shape[1],
        )
        pred = _call_keras(model, [X_meds, X_labs, X_tab, X_text_b], training=False)
    else:
        raise ValueError(f"Unsupported model with {n_in} inputs")

    pred = _to_numpy(pred)
    if pred.ndim == 0:
        return np.array([float(pred)], dtype=float)
    if pred.ndim == 1:
        return pred.astype(float)
    # Common Keras shapes: (N, 1) or (N, K). We assume binary => use squeeze.
    return pred.reshape(pred.shape[0], -1)[:, 0].astype(float)


def _predict_scores(model, X_tab: np.ndarray, X_ts: np.ndarray, X_text=None, batch_size: int = 32) -> np.ndarray:
    """Return model scores as a 1D float array, with batching."""
    n_in = _n_model_inputs(model)
    if n_in in (1, 2) and _is_split_ts_dict(X_ts):
        raise ValueError(
            "Split TS dict batches require a 4-input model (meds, labs, static, text)."
        )

    num_samples = X_tab.shape[0]
    if num_samples == 0:
        return np.array([], dtype=float)

    preds = []
    for i in range(0, num_samples, batch_size):
        batch_end = i + batch_size
        X_tab_batch = X_tab[i:batch_end]

        if n_in == 1:
            X_ts_batch = X_ts[i:batch_end]
            model_input = _concat_tab_ts_batch(X_tab_batch, X_ts_batch)
            pred_batch = _call_keras(model, model_input, training=False)
        elif n_in == 2:
            X_ts_batch = X_ts[i:batch_end]
            pred_batch = _call_keras(model, [np.asarray(X_tab_batch), np.asarray(X_ts_batch)], training=False)
        elif n_in == 4:
            X_tab_batch_np = np.asarray(X_tab_batch)
            if _is_split_ts_dict(X_ts):
                X_labs_batch = X_ts["labs"][i:batch_end]
                X_meds_batch = X_ts["meds"][i:batch_end]
            else:
                X_ts_batch = np.asarray(X_ts)[i:batch_end]
                X_labs_batch, X_meds_batch = _split_ts_for_4input(model, X_ts_batch)

            x_text_src = X_text[i:batch_end] if X_text is not None else 0.0

            X_text_b = _coerce_text_input_for_model(
                x_text_src, model.inputs[3], batch_size=X_tab_batch_np.shape[0], timesteps=X_labs_batch.shape[1]
            )
            pred_batch = _call_keras(model, [X_meds_batch, X_labs_batch, X_tab_batch_np, X_text_b], training=False)
        else:
            raise ValueError(f"Unsupported model with {n_in} inputs")

        preds.append(_to_numpy(pred_batch))

    pred = np.concatenate(preds, axis=0)
    return pred.reshape(pred.shape[0], -1)[:, 0].astype(float)

def _evaluate_counterfactuals_split_ts(
    *,
    x_tab_ref,
    x_ts_ref,
    counterfactuals,
    X_tab_obs,
    X_ts_obs,
    model,
    segments,
    tau_c,
    y_target,
    rho,
    plausibility_normalizer,
    tab_plausibility_weight,
    pred_to_label,
    flip_against,
    text_factual,
    text_candidates,
    text_tokens_factual,
    text_tokens_candidates,
    text_objective_context,
    text_input_ref,
    text_input_candidates,
    prediction_batch_size: int = 32,
):
    x_tab_ref = np.asarray(x_tab_ref)
    x_ts_ref = _ensure_split_ts_sample_dict(x_ts_ref, name="x_ts_ref")
    X_ts_obs = _ensure_split_ts_batch_dict(X_ts_obs, name="X_ts_obs")
    n_tab = int(x_tab_ref.shape[0])
    x_labs_ref = x_ts_ref["labs"]
    x_meds_ref = x_ts_ref["meds"]

    if pred_to_label is None:
        pred_to_label = lambda p: int(p >= 0.5)

    if isinstance(counterfactuals, list) and len(counterfactuals) == 0:
        pred_factual = float(
            _predict_scores(
                model,
                x_tab_ref[None, :],
                {"labs": x_labs_ref[None, :, :], "meds": x_meds_ref[None, :, :]},
                X_text=text_input_ref,
            )[0]
        )
        label_factual = int(pred_to_label(pred_factual))
        return {
            "pred_factual": pred_factual,
            "label_factual": label_factual,
            "pred_cf": np.empty((0,), dtype=float),
            "label_cf": np.empty((0,), dtype=int),
            "flipped": np.empty((0,), dtype=bool),
            "objectives": np.empty((0, 4), dtype=float),
            "X_tab_cf": np.empty((0, n_tab), dtype=float),
            "X_ts_cf": {
                "labs": np.empty((0,) + x_labs_ref.shape, dtype=float),
                "meds": np.empty((0,) + x_meds_ref.shape, dtype=float),
            },
            "cf_kind": [],
        }

    cf_kind = None

    if (
        isinstance(counterfactuals, (tuple, list))
        and len(counterfactuals) == 2
        and not isinstance(counterfactuals[0], dict)
    ):
        try:
            X_tab_in, X_ts_in = counterfactuals
            X_tab = _as_2d_tab(X_tab_in, n_tab)
            X_ts_split = _ensure_split_ts_batch_dict(X_ts_in, name="X_ts")
            X_labs = X_ts_split["labs"]
            X_meds = X_ts_split["meds"]
        except Exception:
            counterfactuals = list(counterfactuals)
        else:
            if X_tab.shape[0] == 1 and X_labs.shape[0] > 1:
                X_tab = np.broadcast_to(X_tab, (X_labs.shape[0], n_tab)).copy()
            if X_tab.shape[0] != X_labs.shape[0]:
                raise ValueError(
                    f"Mismatched N: X_tab has {X_tab.shape[0]}, split TS has {X_labs.shape[0]}"
                )
            cf_kind = ["both"] * X_tab.shape[0]

    if cf_kind is None and isinstance(counterfactuals, list) and len(counterfactuals) > 0 and isinstance(counterfactuals[0], dict):
        X_tab_list = []
        X_labs_list = []
        X_meds_list = []
        cf_kind = []
        for item in counterfactuals:
            if not isinstance(item, dict):
                raise ValueError("When counterfactuals is a list, all items must be dicts")

            has_tab = "tab" in item and item["tab"] is not None
            has_ts = "ts" in item and item["ts"] is not None
            if not (has_tab or has_ts):
                raise ValueError("Each counterfactual dict must have at least one of: 'tab', 'ts'")

            tab = np.asarray(item["tab"]).reshape(-1) if has_tab else x_tab_ref
            if has_ts:
                ts_item = _ensure_split_ts_sample_dict(item["ts"], name="counterfactual['ts']")
                labs = ts_item["labs"]
                meds = ts_item["meds"]
            else:
                labs = x_labs_ref
                meds = x_meds_ref

            X_tab_list.append(tab)
            X_labs_list.append(labs)
            X_meds_list.append(meds)
            if has_tab and has_ts:
                cf_kind.append("both")
            elif has_tab:
                cf_kind.append("tab-only")
            else:
                cf_kind.append("ts-only")

        X_tab = _as_2d_tab(np.stack(X_tab_list, axis=0), n_tab)
        X_labs = _as_ts_sample_or_batch(np.stack(X_labs_list, axis=0), name="X_labs")
        X_meds = _as_ts_sample_or_batch(np.stack(X_meds_list, axis=0), name="X_meds")

    if cf_kind is None:
        arr = np.asarray(counterfactuals)
        while arr.ndim >= 3 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]

        if arr.ndim in (1, 2) and (arr.ndim == 1 or arr.shape[1] == n_tab):
            X_tab = _as_2d_tab(arr, n_tab)
            X_labs = np.broadcast_to(x_labs_ref[None, :, :], (X_tab.shape[0],) + x_labs_ref.shape).copy()
            X_meds = np.broadcast_to(x_meds_ref[None, :, :], (X_tab.shape[0],) + x_meds_ref.shape).copy()
            cf_kind = ["tab-only"] * X_tab.shape[0]
        else:
            raise ValueError(
                "Split TS mode supports tab-only arrays, list[{'tab','ts'}], or (X_tab, {'labs','meds'})."
            )

    X_ts_cf = {"labs": X_labs, "meds": X_meds}

    pred_factual = float(
        _predict_scores(
            model,
            x_tab_ref[None, :],
            {"labs": x_labs_ref[None, :, :], "meds": x_meds_ref[None, :, :]},
            batch_size=1,
            X_text=text_input_ref,
        )[0]
    )
    label_factual = int(pred_to_label(pred_factual))
    pred_cf = _predict_scores(model, X_tab, X_ts_cf, X_text=text_input_candidates, batch_size=prediction_batch_size)
    label_cf = np.array([int(pred_to_label(float(p))) for p in pred_cf], dtype=int)

    if flip_against != "factual":
        raise ValueError("Only flip_against='factual' is currently supported")
    flipped = label_cf != label_factual

    if plausibility_normalizer is None:
        plausibility_normalizer = fit_plausibility_normalizer(X_tab_obs, X_ts_obs)

    def _try_get(seq, i):
        if seq is None:
            return None
        try:
            return seq[i]
        except Exception:
            return seq

    N = int(X_tab.shape[0])
    objectives = np.zeros((N, 4), dtype=float)
    embedding_cache = text_objective_context.get("embedding_cache", {}) if text_objective_context else {}
    for i in range(N):
        objectives[i] = compute_objectives(
            x_tab=X_tab[i],
            x_ts={"labs": X_labs[i], "meds": X_meds[i]},
            x_tab_ref=x_tab_ref,
            x_ts_ref={"labs": x_labs_ref, "meds": x_meds_ref},
            X_tab_obs=X_tab_obs,
            X_ts_obs=X_ts_obs,
            model=model,
            segments=segments,
            tau_c=tau_c,
            y_target=y_target,
            rho=rho,
            plausibility_normalizer=plausibility_normalizer,
            tab_plausibility_weight=tab_plausibility_weight,
            text_factual=text_factual,
            text_candidate=_try_get(text_candidates, i),
            text_tokens_factual=text_tokens_factual,
            text_tokens_candidate=_try_get(text_tokens_candidates, i),
            text_objective_context=text_objective_context,
            text_input_ref=text_input_ref,
            text_input_cf=_try_get(text_input_candidates, i),
            embedding_cache=embedding_cache,
        )

    return {
        "pred_factual": pred_factual,
        "label_factual": label_factual,
        "pred_cf": pred_cf,
        "label_cf": label_cf,
        "flipped": flipped,
        "objectives": objectives,
        "X_tab_cf": X_tab,
        "X_ts_cf": X_ts_cf,
        "cf_kind": cf_kind,
    }


def evaluate_counterfactuals(
    *,
    x_tab_ref: np.ndarray,
    x_ts_ref: np.ndarray,
    counterfactuals,
    X_tab_obs: np.ndarray,
    X_ts_obs: np.ndarray,
    model,
    segments,
    tau_c: np.ndarray,
    y_target,
    rho: float = 0.0,
    plausibility_normalizer=None,
    tab_plausibility_weight: float = 0.5,
    pred_to_label=None,
    flip_against: str = "factual",
    text_factual: str = None,
    text_candidates: Sequence[str] = None,
    text_tokens_factual: List[str] = None,
    text_tokens_candidates: Sequence[Sequence[str]] = None,
    text_objective_context: dict = None,
    text_input_ref=None,
    text_input_candidates=None,
    prediction_batch_size: int = 32,
):
    """Evaluate counterfactual candidates for a single factual instance.

    Parameters
    ----------
    x_tab_ref, x_ts_ref:
        The factual instance (tabular and time-series).

    counterfactuals:
        One of the following:
        - Tab-only candidates: (N, D_tab) or (D_tab,)
        - TS-only candidates: (N, T, C) or (T, C)
        - Tuple/list: (X_tab, X_ts) where either side can be a single sample or batch
        - Early-fusion array(s): (T, D_tab + C) or (N, T, D_tab + C)
        - List of dicts, each like {"tab": ..., "ts": ...} (either side optional)

    flip_against:
        "factual" (default) compares predicted labels vs. the factual prediction.

    Returns
    -------
    dict with keys:
      - pred_factual, label_factual
      - pred_cf: (N,)
      - label_cf: (N,)
      - flipped: (N,) bool
            - objectives: (N, 4)
      - X_tab_cf: (N, D_tab)
      - X_ts_cf: (N, T, C)
      - cf_kind: list[str] length N
    """
    if _is_split_ts_dict(x_ts_ref):
        return _evaluate_counterfactuals_split_ts(
            x_tab_ref=x_tab_ref,
            x_ts_ref=x_ts_ref,
            counterfactuals=counterfactuals,
            X_tab_obs=X_tab_obs,
            X_ts_obs=X_ts_obs,
            model=model,
            segments=segments,
            tau_c=tau_c,
            y_target=y_target,
            rho=rho,
            plausibility_normalizer=plausibility_normalizer,
            tab_plausibility_weight=tab_plausibility_weight,
            pred_to_label=pred_to_label,
            flip_against=flip_against,
            text_factual=text_factual,
            text_candidates=text_candidates,
            text_tokens_factual=text_tokens_factual,
            text_tokens_candidates=text_tokens_candidates,
            text_objective_context=text_objective_context,
            text_input_ref=text_input_ref,
            text_input_candidates=text_input_candidates,
            prediction_batch_size=prediction_batch_size,
        )

    x_tab_ref = np.asarray(x_tab_ref)
    x_ts_ref = np.asarray(x_ts_ref)
    n_tab = int(x_tab_ref.shape[0])
    timesteps = int(x_ts_ref.shape[0])
    n_channels = int(x_ts_ref.shape[1])

    if pred_to_label is None:
        pred_to_label = lambda p: int(p >= 0.5)

    # Materialize counterfactuals into (X_tab, X_ts) with missing modalities filled from reference.
    cf_kind = None

    # Empty input: return an empty evaluation with factual baseline.
    if isinstance(counterfactuals, list) and len(counterfactuals) == 0:
        pred_factual = float(
            _predict_scores(model, x_tab_ref[None, :], x_ts_ref[None, :, :], X_text=text_input_ref, batch_size=1)[0]
        )
        label_factual = int(pred_to_label(pred_factual))
        return {
            "pred_factual": pred_factual,
            "label_factual": label_factual,
            "pred_cf": np.empty((0,), dtype=float),
            "label_cf": np.empty((0,), dtype=int),
            "flipped": np.empty((0,), dtype=bool),
            "objectives": np.empty((0, 4), dtype=float),
            "X_tab_cf": np.empty((0, n_tab), dtype=float),
            "X_ts_cf": np.empty((0, timesteps, n_channels), dtype=float),
            "cf_kind": [],
        }

    if isinstance(counterfactuals, (tuple, list)) and len(counterfactuals) == 2 and not isinstance(counterfactuals[0], dict):
        # Ambiguous case: could be (X_tab, X_ts) OR a list of exactly 2 candidates.
        # We only treat it as (X_tab, X_ts) if the shapes validate.
        try:
            X_tab_in, X_ts_in = counterfactuals
            X_tab = _as_2d_tab(X_tab_in, n_tab)
            X_ts = _as_3d_ts(X_ts_in, timesteps, n_channels)
        except Exception:
            counterfactuals = list(counterfactuals)
        else:
            if X_tab.shape[0] != X_ts.shape[0]:
                raise ValueError(
                    f"Mismatched N: X_tab has {X_tab.shape[0]}, X_ts has {X_ts.shape[0]}"
                )
            cf_kind = ["both"] * X_tab.shape[0]

    # If we didn't resolve the (X_tab, X_ts) case above, we may still have a list of candidates.
    if cf_kind is None and isinstance(counterfactuals, list) and len(counterfactuals) > 0 and not isinstance(counterfactuals[0], dict):
        stacked = np.stack([np.asarray(a) for a in counterfactuals], axis=0)
        while stacked.ndim >= 3 and stacked.shape[1] == 1:
            stacked = stacked[:, 0, ...]
        counterfactuals = stacked

    if cf_kind is None and isinstance(counterfactuals, list) and isinstance(counterfactuals[0], dict):
        X_tab_list = []
        X_ts_list = []
        cf_kind = []
        for item in counterfactuals:
            if not isinstance(item, dict):
                raise ValueError("When counterfactuals is a list, all items must be dicts")
            has_tab = "tab" in item and item["tab"] is not None
            has_ts = "ts" in item and item["ts"] is not None
            if not (has_tab or has_ts):
                raise ValueError("Each counterfactual dict must have at least one of: 'tab', 'ts'")

            if has_tab:
                tab = np.asarray(item["tab"]).reshape(-1)
            else:
                tab = x_tab_ref

            if has_ts:
                ts = np.asarray(item["ts"])
            else:
                ts = x_ts_ref

            X_tab_list.append(tab)
            X_ts_list.append(ts)
            if has_tab and has_ts:
                cf_kind.append("both")
            elif has_tab:
                cf_kind.append("tab-only")
            else:
                cf_kind.append("ts-only")

        X_tab = _as_2d_tab(np.stack(X_tab_list, axis=0), n_tab)
        X_ts = _as_3d_ts(np.stack(X_ts_list, axis=0), timesteps, n_channels)

    if cf_kind is None:
        arr = np.asarray(counterfactuals)

        # Be tolerant to extra singleton batch axes that can appear after stacking.
        # Examples:
        # - tab-only list stacked -> (N, 1, D_tab)
        # - ts-only list stacked -> (N, 1, T, C)
        # - early-fusion list stacked -> (N, 1, T, D)
        while arr.ndim >= 3 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]

        if arr.ndim in (2, 3) and arr.shape[-1] == n_tab + n_channels:
            # Early fusion input(s)
            X_tab, X_ts = _split_early_fusion(arr, n_tab=n_tab, n_channels=n_channels)
            cf_kind = ["early-fusion"] * X_tab.shape[0]
        elif arr.ndim in (1, 2) and (arr.ndim == 1 or arr.shape[1] == n_tab):
            # Tab-only
            X_tab = _as_2d_tab(arr, n_tab)
            X_ts = np.broadcast_to(x_ts_ref[None, :, :], (X_tab.shape[0], timesteps, n_channels)).copy()
            cf_kind = ["tab-only"] * X_tab.shape[0]
        elif arr.ndim in (2, 3) and (arr.ndim == 2 or arr.shape[2] == n_channels):
            # TS-only
            X_ts = _as_3d_ts(arr, timesteps, n_channels)
            X_tab = np.broadcast_to(x_tab_ref[None, :], (X_ts.shape[0], n_tab)).copy()
            cf_kind = ["ts-only"] * X_ts.shape[0]
        else:
            raise ValueError(
                "Unsupported counterfactuals format. Provide tab-only (N,D_tab), ts-only (N,T,C), "
                "(X_tab, X_ts), early-fusion (N,T,D_tab+C), or list of dicts."
            )

    # Factual prediction/label
    pred_factual = float(
        _predict_scores(model, x_tab_ref[None, :], x_ts_ref[None, :, :], X_text=text_input_ref, batch_size=1)[0]
    )
    label_factual = int(pred_to_label(pred_factual))

    # CF predictions/labels
    pred_cf = _predict_scores(model, X_tab, X_ts, X_text=text_input_candidates, batch_size=prediction_batch_size)
    label_cf = np.array([int(pred_to_label(float(p))) for p in pred_cf], dtype=int)

    if flip_against != "factual":
        raise ValueError("Only flip_against='factual' is currently supported")
    flipped = label_cf != label_factual

    if plausibility_normalizer is None:
        plausibility_normalizer = fit_plausibility_normalizer(X_tab_obs, X_ts_obs)

    # Objectives (run compute_objectives on each CF)
    N = int(X_tab.shape[0])
    objectives = np.zeros((N, 4), dtype=float)
    embedding_cache = text_objective_context.get("embedding_cache", {}) if text_objective_context else {}
    for i in range(N):
        text_cand_i = None
        text_tokens_cand_i = None
        if text_candidates is not None and i < len(text_candidates):
            text_cand_i = text_candidates[i]
        if text_tokens_candidates is not None and i < len(text_tokens_candidates):
            text_tokens_cand_i = list(text_tokens_candidates[i])
        text_input_cf_i = None
        if text_input_candidates is not None and i < len(text_input_candidates):
            text_input_cf_i = text_input_candidates[i]

        objectives[i] = compute_objectives(
            x_tab=X_tab[i],
            x_ts=X_ts[i],
            x_tab_ref=x_tab_ref,
            x_ts_ref=x_ts_ref,
            X_tab_obs=X_tab_obs,
            X_ts_obs=X_ts_obs,
            model=model,
            segments=segments,
            tau_c=tau_c,
            y_target=y_target,
            rho=rho,
            plausibility_normalizer=plausibility_normalizer,
            tab_plausibility_weight=tab_plausibility_weight,
            text_factual=text_factual,
            text_candidate=text_cand_i,
            text_tokens_factual=text_tokens_factual,
            text_tokens_candidate=text_tokens_cand_i,
            text_objective_context=text_objective_context,
            text_input_ref=text_input_ref,
            text_input_cf=text_input_cf_i,
            embedding_cache=embedding_cache,
        )

    return {
        "pred_factual": pred_factual,
        "label_factual": label_factual,
        "pred_cf": pred_cf,
        "label_cf": label_cf,
        "flipped": flipped,
        "objectives": objectives,
        "X_tab_cf": X_tab,
        "X_ts_cf": X_ts,
        "cf_kind": cf_kind,
    }
