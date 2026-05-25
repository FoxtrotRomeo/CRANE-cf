"""Optimisation-based counterfactual baselines for all four datasets.

Design principle
----------------
Each baseline operates independently on **one modality at a time**, holding
all other modalities fixed at their factual values.  This is an intentional
design choice to serve as a controlled comparison point that exposes the
known weakness of per-modality optimisation: the resulting counterfactual may
be locally plausible within a single modality but globally incoherent across
modalities.  Baseline names are suffixed with the modality they vary (e.g.
``DiCE_tabular``, ``DiCE_text_description``) so this design choice is
transparent in results tables.

Baselines implemented
---------------------
Long COVID Tweets (tabular + text)
  - DiCE_tabular          — DiCE random on tabular features; tweet held fixed
  - NICE_tabular          — NICE on tabular features; tweet held fixed
  - DiCE_text_tweet       — DiCE random on tweet CLS-embedding; tabular held fixed
  - NICE_text_tweet       — NICE on tweet CLS-embedding; tabular held fixed
                            ⚠ text results are perturbed embeddings, NOT actual text

Real or Fake Jobs (tabular + 3×text)
  - DiCE_tabular,          NICE_tabular
  - DiCE_text_description, NICE_text_description
  - DiCE_text_company_profile, NICE_text_company_profile
  - DiCE_text_requirements,    NICE_text_requirements   (each is a perturbed embedding)

Hateful Memes (text + image, no tabular)
  - DiCE_text_meme_text,  NICE_text_meme_text   — embedding perturbation; image fixed
  - DiCE_image_meme_img,  NICE_image_meme_img   — embedding perturbation; text fixed

Sepsis (tabular + time-series)
  - DiCE_tabular,              NICE_tabular         — static features; TS held fixed
  - GRADIENT_tabular                                — Adam gradient on tabular; TS held fixed
  - GRADIENT_TS_lab_exams_vitals                    — Adam gradient on TS; tabular held fixed
  - NATIVEGUIDE_TS_lab_exams_vitals                 — NativeGuide-ISW (Delaney 2021)
  - COMTE_TS_lab_exams_vitals                       — CoMTE greedy segment swap (Ates 2021)
    ⚠ NativeGuide and CoMTE are from-scratch implementations: original repos have no
      importable package / require Rust extension incompatible with PyTorch model.

Gradient baselines (all datasets, tabular + embeddings)
  - GRADIENT_tabular             — Wachter-style Adam on tabular features
  - GRADIENT_text_*              — Wachter-style Adam on text encoder CLS embeddings
  - GRADIENT_image_*             — Wachter-style Adam on image encoder embeddings
    ⚠ Text/image gradient results are perturbed embeddings, NOT actual text or images.

NICE implementation note
------------------------
NICEx (pip install NICEx) exists but cannot be used here directly: its
``__init__`` calls predict_fn on the entire training set, while our predict
functions are per-sample closures that hold other-modality embeddings fixed at
the current query's factual values.  The custom ``run_nice()`` in
baselines_common.py implements the same greedy-feature-copy algorithm:
for each of the k nearest opposite-class training instances, features are
sorted by absolute difference (largest first) and copied one-by-one until the
prediction flips — following Brughmans et al. (2023).

Output
------
Each baseline writes a ``summary.json`` in the ablation-runner format to:
    <dataset>/data/baselines/<baseline_name>/summary.json

Usage
-----
    # All baselines for Long COVID
    venv/bin/python examples/run_baselines.py --dataset long_covid_tweets --gpu 0

    # Smoke test (3 samples, specific dataset)
    venv/bin/python examples/run_baselines.py --dataset sepsis --fold 0 \\
        --max-samples 3 --k 3 --ts-steps 100 --gpu 0

    # All datasets, explicit sample indices
    venv/bin/python examples/run_baselines.py --dataset hateful_memes \\
        --sample-indices 0,1,2,3,4 --target-value 0 --gpu 0
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Make cf_lib and examples importable from any working directory ----
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines_common import (
    aggregate_objectives,
    compute_embedding_objectives,
    compute_tabular_objectives,
    compute_ts_gradient_objectives,
    fit_tabular_lof,
    infer_feature_types,
    make_summary_payload,
    run_comte,
    run_dice_embedding_fast,
    run_dice_on_features,
    run_gradient_vector,
    run_native_guide,
    run_nice,
    write_summary,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# Shared model class definitions
# (Inline — avoids importing evaluate.py files that have module-level argparse)
# ===========================================================================

class _TabHead(nn.Module):
    def __init__(self, d_in: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Long COVID / Real-or-Fake-Jobs multimodal classifier
# (architecture identical to evaluate.py in both datasets)
class _TweetClassifier(nn.Module):
    """XLM-RoBERTa + tabular head → emotion classifier."""

    TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"

    def __init__(self, d_tab: int, n_classes: int,
                 tab_hidden_dims: List[int], text_hidden_dim: int, dropout: float):
        super().__init__()
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_MODEL)
        enc_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(enc_dim, text_hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.tab_head = _TabHead(d_tab, tab_hidden_dims, dropout)
        self.classifier = nn.Linear(text_hidden_dim + self.tab_head.out_dim, n_classes)

    def forward(self, input_ids, attention_mask, X_static):
        cls = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.classifier(
            torch.cat([self.text_proj(cls), self.tab_head(X_static)], dim=1)
        )


class _JobsClassifier(nn.Module):
    """DistilBERT × 3 branches + tabular head → real/fake classifier."""

    TEXT_MODEL = "distilbert-base-uncased"

    def __init__(self, d_tab: int, n_classes: int,
                 tab_hidden_dims: List[int], text_hidden_dim: int, dropout: float):
        super().__init__()
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_MODEL)
        enc_dim = self.text_encoder.config.hidden_size

        def _proj():
            return nn.Sequential(
                nn.Linear(enc_dim, text_hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        self.proj_description = _proj()
        self.proj_company_profile = _proj()
        self.proj_requirements = _proj()
        self.tab_head = _TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim = 3 * text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, desc_ids, desc_mask, profile_ids, profile_mask,
                reqs_ids, reqs_mask, X_static):
        B = desc_ids.size(0)
        all_ids = torch.cat([desc_ids, profile_ids, reqs_ids], dim=0)
        all_mask = torch.cat([desc_mask, profile_mask, reqs_mask], dim=0)
        cls_all = self.text_encoder(
            input_ids=all_ids, attention_mask=all_mask
        ).last_hidden_state[:, 0]
        d, p, r = cls_all.split(B, dim=0)
        d_emb = self.proj_description(d)
        p_emb = self.proj_company_profile(p)
        r_emb = self.proj_requirements(r)
        tab_emb = self.tab_head(X_static)
        return self.classifier(torch.cat([d_emb, p_emb, r_emb, tab_emb], dim=1))


# ===========================================================================
# Additional model classes for early / late fusion strategies
# ===========================================================================

class _EarlyFusionMLP(nn.Module):
    """MLP for early-fusion (tweets, jobs, memes).

    Supports variable depth so it can load checkpoints trained with either 2 or
    3 hidden layers.  Use ``from_checkpoint`` to build the right architecture
    directly from a saved state_dict.
    """
    def __init__(self, d_in: int, n_classes: int, hidden: int = 256,
                 dropout: float = 0.3, depth: int = 2):
        super().__init__()
        sizes = [hidden // (2 ** i) for i in range(depth)]
        layers: list = []
        prev = d_in
        for sz in sizes:
            layers += [nn.Linear(prev, sz), nn.ReLU(), nn.Dropout(dropout)]
            prev = sz
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def depth_from_state_dict(state_dict: dict) -> int:
        max_idx = max(int(k.split(".")[1]) for k in state_dict if k.startswith("net."))
        # Each hidden block occupies 3 indices (Linear/ReLU/Dropout).
        # Output Linear is at index 3*depth, so depth = max_idx // 3.
        return max_idx // 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SepsisIFModel(nn.Module):
    """Sepsis intermediate-fusion MLP encoder (mirrors train_intermediate_fusion.py).

    Architecture matches the saved checkpoint from train_intermediate_fusion.py:
    - static_enc: Linear → ReLU → Dropout  (NOT Tanh, despite spec pseudocode)
    - fusion: Linear → Tanh → BatchNorm1d → Dropout  (matches SepsisIFModel.fusion)
    """
    def __init__(self, n_ts: int = 53, n_static: int = 47,
                 ts_hidden: int = 32, static_hidden: int = 32,
                 fused_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru        = nn.GRU(n_ts, ts_hidden, batch_first=True)
        self.ts_drop    = nn.Dropout(dropout)
        self.static_enc = nn.Sequential(
            nn.Linear(n_static, static_hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(ts_hidden + static_hidden, fused_hidden),
            nn.Tanh(),
            nn.BatchNorm1d(fused_hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(fused_hidden, 1)

    def encode(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        h = self.ts_drop(h.squeeze(0))
        s = self.static_enc(x_static)
        return self.fusion(torch.cat([h, s], dim=1))

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


class _SepsisTSModel(nn.Module):
    """GRU-only branch for Sepsis late fusion (mirrors train_late_fusion.py TSOnlyModel)."""
    def __init__(self, n_ts: int = 53, gru_hidden: int = 32,
                 dense_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru   = nn.GRU(n_ts, gru_hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.dense = nn.Linear(gru_hidden, dense_hidden)
        self.bn    = nn.BatchNorm1d(dense_hidden, momentum=0.01, eps=0.001)
        self.out   = nn.Linear(dense_hidden, 1)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        h = self.drop(h.squeeze(0))
        h = torch.tanh(self.dense(h))
        h = self.bn(h)
        return torch.sigmoid(self.out(h))


class _SepsisStaticModel(nn.Module):
    """MLP-only branch for Sepsis late fusion (mirrors train_late_fusion.py StaticOnlyModel)."""
    def __init__(self, n_static: int = 47, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, hidden), nn.Tanh(),
            nn.BatchNorm1d(hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x_static))


class _SepsisLateWrapper(nn.Module):
    """Averages TSOnly + StaticOnly branch outputs (same interface as SepsisModel)."""
    def __init__(self, ts_model: nn.Module, static_model: nn.Module):
        super().__init__()
        self.ts = ts_model
        self.st = static_model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return (self.ts(x_ts) + self.st(x_static)) / 2.0

    def eval(self):  # type: ignore[override]
        self.ts.eval()
        self.st.eval()
        return super().eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        self.ts.train(mode)
        self.st.train(mode)
        return super().train(mode)


class _SepsisIFRFWrapper(nn.Module):
    """Intermediate-fusion RF wrapper — encoder encodes to latent, RF classifies."""
    def __init__(self, if_encoder: _SepsisIFModel, rf):
        super().__init__()
        self.enc = if_encoder
        self.rf  = rf

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.enc.encode(x_ts, x_static).cpu().numpy()
        p = self.rf.predict_proba(latent)[:, 1].astype(np.float32)
        return torch.tensor(p, device=x_ts.device).unsqueeze(1)

    def eval(self):  # type: ignore[override]
        self.enc.eval()
        return super().eval()


# ===========================================================================
# Sepsis model — imported from the sepsis package
def _load_sepsis_model(model_path: Path, device: str):
    sys.path.insert(0, str(_ROOT / "sepsis"))
    from modeling import SepsisModel  # type: ignore
    m = SepsisModel()
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.to(device).eval()
    return m


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimisation-based counterfactual baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   choices=["long_covid_tweets", "real_or_fake_jobs",
                            "hateful_memes", "sepsis"],
                   help="Which dataset to run baselines for.")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (None → auto-detect / CPU).")
    p.add_argument("--k", type=int, default=5,
                   help="Counterfactuals per sample.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on number of test samples to process.")
    p.add_argument("--sample-indices", type=str, default=None,
                   help="Comma-separated explicit test indices (overrides max-samples).")
    p.add_argument("--target-value", type=int, default=None,
                   help="Target class index.  Inferred from dataset if not given.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Root output directory.  Defaults to <dataset>/data/baselines/.")
    p.add_argument("--dice-sample-size", type=int, default=1000,
                   help="DiCE random sample_size per iteration.")
    p.add_argument("--dice-seed", type=int, default=42,
                   help="Random seed for DiCE.")
    # Sepsis-only
    p.add_argument("--fold", type=int, default=0,
                   help="(Sepsis only) Fold index to process.")
    p.add_argument("--ts-steps", type=int, default=300,
                   help="(Sepsis TS) Adam optimisation steps per restart.")
    p.add_argument("--ts-lr", type=float, default=0.05,
                   help="(Sepsis TS) Adam learning rate.")
    p.add_argument("--ts-lambda-prox", type=float, default=0.1,
                   help="(Sepsis TS) Proximity regularisation weight.")
    p.add_argument("--ts-n-segments", type=int, default=10,
                   help="(Sepsis TS) Number of equal-width time windows for "
                        "NativeGuide-ISW and CoMTE segment swap.")
    p.add_argument("--fusion-strategy", default="intermediate",
                   choices=["early", "intermediate", "late"],
                   help="Which fusion-strategy model to load for baselines. "
                        "Default: intermediate (existing behaviour unchanged).")
    return p.parse_args()


# ===========================================================================
# Utilities
# ===========================================================================

def _resolve_device(gpu: Optional[int]) -> str:
    if gpu is not None and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_csv_indices(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _default_output_dir(dataset: str, root: Path) -> Path:
    return root / "data" / "baselines"


def _select_source_samples(
    y_pred: np.ndarray,
    source_class: int,
    max_samples: Optional[int],
) -> List[int]:
    """Return test indices predicted as *source_class*.

    Only uses y_pred. Raises if no predicted source-class samples exist.
    """
    indices = [i for i, p in enumerate(y_pred) if int(p) == source_class]
    if not indices:
        raise ValueError(
            f"y_pred has no samples predicted as class {source_class}. "
            "Cannot generate counterfactuals without predicted source-class samples."
        )
    if max_samples:
        indices = indices[:max_samples]
    return indices


def _encode_texts_batch(
    texts: List[str],
    tokenizer,
    model: nn.Module,
    device: str,
    batch_size: int = 64,
    max_len: int = 128,
    pooling: str = "cls",
    normalize: bool = False,
) -> np.ndarray:
    """Encode a list of strings → (n, hidden_dim) embedding array."""
    all_embs = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            last = model(input_ids=ids, attention_mask=mask).last_hidden_state
            if pooling == "cls":
                emb = last[:, 0]
            else:  # mean pooling
                m = mask.unsqueeze(-1).float()
                emb = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if normalize:
                emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _encode_images_batch(
    images: np.ndarray,
    encoder: nn.Module,
    transform,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode (n, H, W, C) uint8 images → (n, feat_dim) float32 array."""
    from PIL import Image

    all_feats = []
    encoder.eval()
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start : start + batch_size]
        tensors = torch.stack(
            [transform(Image.fromarray(img)) for img in batch_imgs]
        ).to(device)
        with torch.no_grad():
            feats = encoder(tensors)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def _build_emb_col_names(dim: int) -> List[str]:
    return [f"emb_{i}" for i in range(dim)]


# ===========================================================================
# Long COVID Tweets baselines
# ===========================================================================

def run_long_covid_baselines(args: argparse.Namespace) -> None:
    dataset_dir = _ROOT / "long_covid_tweets"
    data_dir = Path(args.output_dir).parent if args.output_dir else dataset_dir / "data"
    fusion_strategy = getattr(args, "fusion_strategy", "intermediate")
    device = _resolve_device(args.gpu)

    # Determine output root per fusion strategy
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif fusion_strategy == "intermediate":
        out_root = dataset_dir / "data" / "baselines"
    elif fusion_strategy == "early":
        out_root = dataset_dir / "data" / "baselines_fusion_early"
    else:  # late
        out_root = dataset_dir / "data" / "baselines_fusion_late"

    # ---- Load data ----
    npz = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_train_static = npz["X_train_static"].astype(np.float32)
    X_test_static  = npz["X_test_static"].astype(np.float32)
    y_train        = npz["y_train"].astype(int)
    y_test         = npz["y_test"].astype(int)
    X_train_text   = npz["X_train_text"].tolist()
    X_test_text    = npz["X_test_text"].tolist()
    tab_col_names  = npz["tab_cols"].tolist()
    label_classes  = npz["label_classes"].tolist()

    # ---- Load strategy-specific model ----
    # is_deep: whether gradient-based baselines are applicable
    # has_text: whether text-based baselines are applicable
    is_deep   = True
    has_text  = True
    tokenizer = None
    model     = None

    if fusion_strategy == "intermediate":
        # Existing behaviour — load the intermediate (best) model
        ckpt = torch.load(data_dir / "best_model.pt", map_location="cpu")
        cfg  = ckpt["config"]
        model = _TweetClassifier(
            d_tab           = len(ckpt["tab_cols"]),
            n_classes       = len(ckpt["label_classes"]),
            tab_hidden_dims = cfg["tab_hidden_dims"],
            text_hidden_dim = cfg["text_hidden_dim"],
            dropout         = cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_TweetClassifier.TEXT_MODEL)

    elif fusion_strategy == "early":
        # Early fusion MLP: uses pre-computed CLS embeddings
        ef_ckpt = torch.load(data_dir / "best_model_early_fusion_mlp.pt", map_location="cpu")
        ef_state = ef_ckpt["state_dict"]
        ef_d_in     = ef_ckpt["d_in"]
        ef_n_cls    = ef_ckpt["n_classes"]
        ef_hidden   = ef_ckpt.get("hidden", 256)
        ef_dropout  = ef_ckpt.get("dropout", 0.3)
        ef_label_classes = ef_ckpt["label_classes"]
        label_classes = ef_label_classes  # override
        ef_depth = _EarlyFusionMLP.depth_from_state_dict(ef_state)
        model = _EarlyFusionMLP(ef_d_in, ef_n_cls, ef_hidden, ef_dropout, depth=ef_depth).to(device)
        model.load_state_dict(ef_state)
        model.eval()
        is_deep  = True
        has_text = True  # text embeddings are precomputed; no live tokenizer needed
        tokenizer = None  # not used for early fusion

    elif fusion_strategy == "late":
        # Late fusion: tfidf+logreg for text; GBT for tabular (may fail to load)
        # Text-based baselines not applicable (sklearn models, no gradient)
        is_deep  = False
        has_text = False
        print("[long_covid] late fusion: text baselines skipped (non-deep sklearn model)")

    # ---- Determine sample indices ----
    target_value = args.target_value
    if target_value is None:
        # Default: explain "sadness" → "joy"
        source_class = label_classes.index("sadness") if "sadness" in label_classes else 0
        target_value = label_classes.index("joy") if "joy" in label_classes else 1

    _tweet_pred_file = {
        "intermediate": "y_pred.npy",
        "early":        "y_pred_early_fusion_mlp.npy",
        "late":         "y_pred_late_fusion_tfidf_logreg_nondp.npy",
    }.get(fusion_strategy, "y_pred.npy")
    y_pred_path = data_dir / _tweet_pred_file
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else y_test
    sample_indices: List[int]
    if args.sample_indices:
        sample_indices = _parse_csv_indices(args.sample_indices)
    else:
        source_class = int(label_classes.index(
            next((c for c in label_classes if c != label_classes[target_value]),
                 label_classes[0])
        ))
        sample_indices = _select_source_samples(
            y_pred, source_class, args.max_samples
        )

    k = args.k
    print(f"[long_covid] {len(sample_indices)} samples → target={label_classes[target_value]}"
          f"  fusion_strategy={fusion_strategy}")

    # ---- Pre-compute feature types ----
    continuous_tab, categorical_tab = infer_feature_types(X_train_static, tab_col_names)
    print(f"[long_covid] Tabular: {len(continuous_tab)} continuous, {len(categorical_tab)} categorical")

    # ---- Pre-fit tabular LOF ----
    print("[long_covid] Fitting tabular LOF …")
    tab_lof = fit_tabular_lof(X_train_static)

    if fusion_strategy == "intermediate":
        # ---- Existing intermediate-fusion path ----
        # Pre-compute text embeddings (train + test)
        print("[long_covid] Encoding training text …")
        X_train_emb = _encode_texts_batch(
            X_train_text, tokenizer, model.text_encoder, device
        )  # (n_train, 768)
        print("[long_covid] Encoding test text …")
        X_test_emb = _encode_texts_batch(
            X_test_text, tokenizer, model.text_encoder, device
        )  # (n_test, 768)
        emb_dim = X_train_emb.shape[1]
        emb_col_names = _build_emb_col_names(emb_dim)

        _run_long_covid_dice_tabular(
            args, out_root, device, model, tokenizer,
            X_train_static, y_train, X_test_static, X_test_text,
            tab_col_names, continuous_tab, categorical_tab, tab_lof,
            sample_indices, target_value, label_classes, k,
        )
        _run_long_covid_nice_tabular(
            args, out_root, device, model, tokenizer,
            X_train_static, y_train, X_test_static, X_test_text,
            tab_lof,
            sample_indices, target_value, k,
        )
        _run_long_covid_dice_text(
            args, out_root, device, model,
            X_train_static, X_train_emb, y_train,
            X_test_static, X_test_emb,
            emb_col_names,
            sample_indices, target_value, label_classes, k,
        )
        _run_long_covid_nice_text(
            args, out_root, device, model,
            X_train_static, X_train_emb, y_train,
            X_test_static, X_test_emb,
            sample_indices, target_value, k,
        )
        _run_long_covid_gradient_tab(
            args, out_root, device, model, tokenizer,
            X_train_static, y_train, X_test_static, X_test_text,
            tab_lof, sample_indices, target_value, label_classes, k,
        )
        _run_long_covid_gradient_text(
            args, out_root, device, model,
            X_train_static, X_train_emb, y_train,
            X_test_static, X_test_emb,
            sample_indices, target_value, label_classes, k,
        )

    elif fusion_strategy == "early":
        # ---- Early fusion path ----
        # Load precomputed text CLS embeddings
        print("[long_covid] Loading precomputed early-fusion text embeddings …")
        emb_data   = torch.load(data_dir / "early_fusion_text_embeddings.pt",
                                map_location="cpu")
        X_train_emb = emb_data["train"].numpy().astype(np.float32)  # (n_train, 768)
        X_test_emb  = emb_data["test"].numpy().astype(np.float32)   # (n_test,  768)
        emb_dim     = X_train_emb.shape[1]
        emb_col_names = _build_emb_col_names(emb_dim)

        # Build tabular predict_proba factory for early fusion:
        # Concatenate precomputed text emb (fixed) + tabular (varied)
        # model input: concat(text_emb, x_tab) → (d_in,)
        n_classes = model.net[-1].out_features

        def _make_ef_predict_proba_tab(sample_idx, x_tab_factual):
            text_emb_fixed = torch.tensor(
                X_test_emb[sample_idx][np.newaxis, :], dtype=torch.float32
            ).to(device)
            def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
                x_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = x_t.size(0)
                inp = torch.cat([text_emb_fixed.expand(B, -1), x_t], dim=1)
                with torch.no_grad():
                    logits = model(inp)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            return predict_proba_tab

        def _make_ef_predict_proba_text(sample_idx, x_tab_factual):
            tab_fixed = torch.tensor(
                x_tab_factual[np.newaxis, :], dtype=torch.float32
            ).to(device)
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = emb_t.size(0)
                inp = torch.cat([emb_t, tab_fixed.expand(B, -1)], dim=1)
                with torch.no_grad():
                    logits = model(inp)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            return predict_proba_emb

        _run_long_covid_ef_dice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_col_names, continuous_tab, categorical_tab, tab_lof,
            sample_indices, target_value, label_classes, k,
            _make_ef_predict_proba_tab,
        )
        _run_long_covid_ef_nice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_lof, sample_indices, target_value, k,
            _make_ef_predict_proba_tab,
        )
        _run_long_covid_ef_dice_text(
            args, out_root,
            X_train_emb, y_train, X_test_static, X_test_emb,
            emb_col_names, sample_indices, target_value, label_classes, k,
            _make_ef_predict_proba_text,
        )
        _run_long_covid_ef_nice_text(
            args, out_root,
            X_train_emb, y_train, X_test_static, X_test_emb,
            sample_indices, target_value, k,
            _make_ef_predict_proba_text,
        )
        print("[skip] GRADIENT_tabular — gradient not supported for early-fusion MLP "
              "(use intermediate strategy for gradient baselines)")
        print("[skip] GRADIENT_text_tweet — gradient not supported for early-fusion MLP")

    else:  # late
        # ---- Late fusion path ----
        # Load tfidf+logreg for text; GBT for tabular (may fail)
        import pickle

        # Text model: tfidf + LogisticRegression
        text_model_path = data_dir / "best_model_late_fusion_tfidf_logreg_text.pkl"
        text_model_dict = None
        if text_model_path.exists():
            with open(text_model_path, "rb") as f:
                text_model_dict = pickle.load(f)
            print("[long_covid] Loaded late-fusion text model (tfidf+logreg)")

        # GBT tabular model (may fail due to numpy RNG compatibility)
        gbt_model_path = data_dir / "best_model_late_fusion_gbt_tabular.pkl"
        gbt_model = None
        if gbt_model_path.exists():
            try:
                with open(gbt_model_path, "rb") as f:
                    gbt_model = pickle.load(f)
                if isinstance(gbt_model, dict):
                    gbt_model = gbt_model["model"]
                print("[long_covid] Loaded late-fusion GBT tabular model")
            except Exception as e:
                print(f"[long_covid] Warning: failed to load GBT model ({e}); "
                      "tabular baselines will be skipped")

        # Run tabular baselines only if GBT loaded successfully
        if gbt_model is not None:
            n_classes_late = (len(label_classes)
                              if hasattr(gbt_model, "classes_") else 2)

            def _make_late_predict_proba_tab(sample_idx, x_tab_factual):
                def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
                    return gbt_model.predict_proba(arr).astype(np.float32)
                return predict_proba_tab

            _run_long_covid_ef_dice_tabular(
                args, out_root,
                X_train_static, y_train, X_test_static,
                tab_col_names, continuous_tab, categorical_tab, tab_lof,
                sample_indices, target_value, label_classes, k,
                _make_late_predict_proba_tab,
            )
            _run_long_covid_ef_nice_tabular(
                args, out_root,
                X_train_static, y_train, X_test_static,
                tab_lof, sample_indices, target_value, k,
                _make_late_predict_proba_tab,
            )
        else:
            print("[skip] DiCE_tabular / NICE_tabular — GBT model could not be loaded")

        print("[skip] Text baselines — late fusion text model is non-deep "
              "(no gradient, skipping DiCE/NICE embedding perturbation for this strategy)")
        print("[skip] GRADIENT_tabular — gradient not supported for non-deep strategy")
        print("[skip] GRADIENT_text_tweet — gradient not supported for non-deep strategy")


def _run_long_covid_ef_dice_tabular(
    args, out_root,
    X_train_static, y_train, X_test_static,
    tab_col_names, continuous_tab, categorical_tab, tab_lof,
    sample_indices, target_value, label_classes, k,
    make_predict_proba_tab,
):
    """DiCE_tabular for early/late fusion using a factory-produced predict_proba."""
    name = "DiCE_tabular"
    print(f"\n[long_covid] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}
    n_classes = len(label_classes)
    threshold = max(0.15, 1.0 / n_classes)

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        predict_proba_tab = make_predict_proba_tab(sample_idx, x_tab_factual)
        cfs, converged = run_dice_on_features(
            X_train=X_train_static, y_train=y_train,
            col_names=tab_col_names,
            continuous_cols=continuous_tab, categorical_cols=categorical_tab,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
            stopping_threshold=threshold,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="dice_random", modality="tabular",
        held_fixed=["text_tweet"], embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_ef_nice_tabular(
    args, out_root,
    X_train_static, y_train, X_test_static,
    tab_lof, sample_indices, target_value, k,
    make_predict_proba_tab,
):
    """NICE_tabular for early/late fusion using a factory-produced predict_proba."""
    name = "NICE_tabular"
    print(f"\n[long_covid] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        predict_proba_tab = make_predict_proba_tab(sample_idx, x_tab_factual)
        cfs, converged = run_nice(
            X_train=X_train_static, y_train=y_train,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="nice", modality="tabular",
        held_fixed=["text_tweet"], embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_ef_dice_text(
    args, out_root,
    X_train_emb, y_train, X_test_static, X_test_emb,
    emb_col_names, sample_indices, target_value, label_classes, k,
    make_predict_proba_text,
):
    """DiCE_text_tweet for early fusion using precomputed embeddings."""
    name = "DiCE_text_tweet"
    print(f"\n[long_covid] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        emb_factual   = X_test_emb[sample_idx]
        predict_proba_emb = make_predict_proba_text(sample_idx, x_tab_factual)
        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="dice_random", modality="text_tweet",
        held_fixed=["tabular"], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "Result is a perturbed CLS-token embedding, NOT actual text. "
            "Proximity and sparsity are measured in embedding space (cosine / mean-L1). "
            "Plausibility is NaN (no LOF fitted for embedding space)."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_ef_nice_text(
    args, out_root,
    X_train_emb, y_train, X_test_static, X_test_emb,
    sample_indices, target_value, k,
    make_predict_proba_text,
):
    """NICE_text_tweet for early fusion using precomputed embeddings."""
    name = "NICE_text_tweet"
    print(f"\n[long_covid] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        emb_factual   = X_test_emb[sample_idx]
        predict_proba_emb = make_predict_proba_text(sample_idx, x_tab_factual)
        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="nice", modality="text_tweet",
        held_fixed=["tabular"], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "Result is a perturbed CLS-token embedding, NOT actual text. "
            "Proximity and sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_dice_tabular(
    args, out_root, device, model, tokenizer,
    X_train_static, y_train, X_test_static, X_test_text,
    tab_col_names, continuous_tab, categorical_tab, tab_lof,
    sample_indices, target_value, label_classes, k,
):
    name = "DiCE_tabular"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        text_factual  = X_test_text[sample_idx]

        # Pre-encode factual text once — CLS embedding is fixed for this sample
        enc = tokenizer(
            text_factual, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids_fixed  = enc["input_ids"].to(device)
        mask_fixed = enc["attention_mask"].to(device)

        # Predict-proba: vary tabular, hold text at factual CLS embedding
        with torch.no_grad():
            cls_fixed  = model.text_encoder(
                input_ids=ids_fixed, attention_mask=mask_fixed
            ).last_hidden_state[:, 0]                          # (1, 768)
            text_emb_fixed = model.text_proj(cls_fixed)        # (1, text_hidden_dim)

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            # arr: (B, D_tab)
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb  = model.tab_head(x_t)
                t_exp    = text_emb_fixed.expand(B, -1)
                logits   = model.classifier(torch.cat([t_exp, tab_emb], dim=1))
                probs    = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        # Multi-class (6 emotions): lower the stopping threshold from 0.5 so
        # DiCE accepts CFs where the target class has the highest probability
        # even if below 0.5.
        n_classes = len(label_classes)
        threshold = max(0.15, 1.0 / n_classes)

        cfs, converged = run_dice_on_features(
            X_train=X_train_static, y_train=y_train,
            col_names=tab_col_names,
            continuous_cols=continuous_tab, categorical_cols=categorical_tab,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
            stopping_threshold=threshold,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    runtime_sec = (finished_at - started_at).total_seconds()

    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="dice_random", modality="tabular",
        held_fixed=["text_tweet"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=runtime_sec, started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_dice_text(
    args, out_root, device, model,
    X_train_static, X_train_emb, y_train,
    X_test_static, X_test_emb,
    emb_col_names,
    sample_indices, target_value, label_classes, k,
):
    name = "DiCE_text_tweet"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    # All embedding dimensions are continuous
    continuous_emb = emb_col_names
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        x_tab_factual = X_test_static[sample_idx]

        # Pre-compute fixed tabular embedding for this sample
        x_tab_t = torch.tensor(x_tab_factual[np.newaxis, :],
                               dtype=torch.float32).to(device)
        with torch.no_grad():
            tab_emb_fixed = model.tab_head(x_tab_t)  # (1, tab_out_dim)

        def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
            # arr: (B, emb_dim) — each row is a CLS-style text embedding
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                text_emb = model.text_proj(emb_t)
                tab_exp  = tab_emb_fixed.expand(B, -1)
                logits   = model.classifier(torch.cat([text_emb, tab_exp], dim=1))
                probs    = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    runtime_sec = (finished_at - started_at).total_seconds()

    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="dice_random", modality="text_tweet",
        held_fixed=["tabular"],
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=runtime_sec, started_at=started_at, finished_at=finished_at,
        notes=(
            "Result is a perturbed CLS-token embedding, NOT actual text. "
            "Proximity and sparsity are measured in embedding space (cosine / mean-L1). "
            "Plausibility is NaN (no LOF fitted for embedding space)."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_nice_tabular(
    args, out_root, device, model, tokenizer,
    X_train_static, y_train, X_test_static, X_test_text,
    tab_lof,
    sample_indices, target_value, k,
):
    name = "NICE_tabular"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        text_factual  = X_test_text[sample_idx]

        enc = tokenizer(
            text_factual, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids_fixed  = enc["input_ids"].to(device)
        mask_fixed = enc["attention_mask"].to(device)

        with torch.no_grad():
            cls_fixed      = model.text_encoder(
                input_ids=ids_fixed, attention_mask=mask_fixed
            ).last_hidden_state[:, 0]
            text_emb_fixed = model.text_proj(cls_fixed)

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb = model.tab_head(x_t)
                t_exp   = text_emb_fixed.expand(B, -1)
                logits  = model.classifier(torch.cat([t_exp, tab_emb], dim=1))
                probs   = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_nice(
            X_train=X_train_static, y_train=y_train,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="nice", modality="tabular",
        held_fixed=["text_tweet"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_nice_text(
    args, out_root, device, model,
    X_train_static, X_train_emb, y_train,
    X_test_static, X_test_emb,
    sample_indices, target_value, k,
):
    name = "NICE_text_tweet"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        x_tab_factual = X_test_static[sample_idx]

        x_tab_t = torch.tensor(x_tab_factual[np.newaxis, :],
                               dtype=torch.float32).to(device)
        with torch.no_grad():
            tab_emb_fixed = model.tab_head(x_tab_t)

        def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                text_emb = model.text_proj(emb_t)
                tab_exp  = tab_emb_fixed.expand(B, -1)
                logits   = model.classifier(torch.cat([text_emb, tab_exp], dim=1))
                probs    = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="nice", modality="text_tweet",
        held_fixed=["tabular"],
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "Result is a perturbed CLS-token embedding, NOT actual text. "
            "Proximity and sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


# ===========================================================================
# Real or Fake Jobs baselines
# ===========================================================================

def run_jobs_baselines(args: argparse.Namespace) -> None:
    dataset_dir = _ROOT / "real_or_fake_jobs"
    data_dir = dataset_dir / "data"
    fusion_strategy = getattr(args, "fusion_strategy", "intermediate")
    device = _resolve_device(args.gpu)

    # Determine output root per fusion strategy
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif fusion_strategy == "intermediate":
        out_root = dataset_dir / "data" / "baselines"
    elif fusion_strategy == "early":
        out_root = dataset_dir / "data" / "baselines_fusion_early"
    else:  # late
        out_root = dataset_dir / "data" / "baselines_fusion_late"

    # ---- Load data ----
    npz = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_train_static = npz["X_train_static"].astype(np.float32)
    X_test_static  = npz["X_test_static"].astype(np.float32)
    y_train        = npz["y_train"].astype(int)
    y_test         = npz["y_test"].astype(int)
    tab_col_names  = npz["tab_cols"].tolist()
    label_classes  = npz["label_classes"].tolist()

    text_branches = {
        "description":     (npz["X_train_description"].tolist(),
                            npz["X_test_description"].tolist()),
        "company_profile": (npz["X_train_company_profile"].tolist(),
                            npz["X_test_company_profile"].tolist()),
        "requirements":    (npz["X_train_requirements"].tolist(),
                            npz["X_test_requirements"].tolist()),
    }

    # ---- Determine sample indices (fake jobs → real) ----
    target_value = args.target_value
    if target_value is None:
        target_value = label_classes.index("real") if "real" in label_classes else 0

    _jobs_pred_file = {
        "intermediate": "y_pred.npy",
        "early":        "y_pred_early_fusion_mlp.npy",
        "late":         "y_pred_late_fusion_tfidf_logreg_nondp.npy",
    }.get(fusion_strategy, "y_pred.npy")
    y_pred_path = data_dir / _jobs_pred_file
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else y_test
    if args.sample_indices:
        sample_indices = _parse_csv_indices(args.sample_indices)
    else:
        source_class = 1 - target_value
        sample_indices = _select_source_samples(
            y_pred, source_class, args.max_samples
        )

    k = args.k
    print(f"[jobs] {len(sample_indices)} samples → target={label_classes[target_value]}"
          f"  fusion_strategy={fusion_strategy}")

    # ---- Feature types + LOF ----
    continuous_tab, categorical_tab = infer_feature_types(X_train_static, tab_col_names)
    print(f"[jobs] Tabular: {len(continuous_tab)} continuous, {len(categorical_tab)} categorical")
    print("[jobs] Fitting tabular LOF …")
    tab_lof = fit_tabular_lof(X_train_static)

    if fusion_strategy == "intermediate":
        # ---- Existing intermediate-fusion path ----
        ckpt = torch.load(data_dir / "best_model.pt", map_location="cpu")
        cfg  = ckpt["config"]
        model = _JobsClassifier(
            d_tab           = len(ckpt["tab_cols"]),
            n_classes       = len(ckpt["label_classes"]),
            tab_hidden_dims = cfg["tab_hidden_dims"],
            text_hidden_dim = cfg["text_hidden_dim"],
            dropout         = cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_JobsClassifier.TEXT_MODEL)

        # Pre-compute text embeddings for each branch
        branch_train_embs: Dict[str, np.ndarray] = {}
        branch_test_embs:  Dict[str, np.ndarray] = {}
        for branch, (tr_texts, te_texts) in text_branches.items():
            print(f"[jobs] Encoding {branch} (train) …")
            branch_train_embs[branch] = _encode_texts_batch(
                tr_texts, tokenizer, model.text_encoder, device, max_len=512
            )
            print(f"[jobs] Encoding {branch} (test) …")
            branch_test_embs[branch] = _encode_texts_batch(
                te_texts, tokenizer, model.text_encoder, device, max_len=512
            )

        emb_dim       = branch_train_embs["description"].shape[1]
        emb_col_names = _build_emb_col_names(emb_dim)

        # Pre-compute fixed tab embeddings for all test samples (batch once)
        X_test_t = torch.tensor(X_test_static, dtype=torch.float32).to(device)
        with torch.no_grad():
            tab_embs_test_all = model.tab_head(X_test_t).cpu().numpy()

        proj_modules = {
            "description":     model.proj_description,
            "company_profile": model.proj_company_profile,
            "requirements":    model.proj_requirements,
        }

        def _precompute_fixed_embs_intermediate(bte, dev):
            result: Dict[str, np.ndarray] = {}
            for bname, proj in proj_modules.items():
                emb_t = torch.tensor(bte[bname], dtype=torch.float32).to(dev)
                with torch.no_grad():
                    result[bname] = proj(emb_t).cpu().numpy()
            return result

        fixed_proj_embs = _precompute_fixed_embs_intermediate(branch_test_embs, device)

        _run_jobs_dice_tabular(
            args, out_root, device, model, tokenizer,
            X_train_static, y_train, X_test_static, text_branches,
            tab_col_names, continuous_tab, categorical_tab, tab_lof,
            fixed_proj_embs,
            sample_indices, target_value, label_classes, k,
        )
        _run_jobs_nice_tabular(
            args, out_root, device, model, tokenizer,
            X_train_static, y_train, X_test_static, text_branches,
            tab_lof, fixed_proj_embs,
            sample_indices, target_value, k,
        )
        for branch in text_branches:
            _run_jobs_dice_text(
                args, out_root, device, model, branch,
                branch_train_embs[branch], y_train,
                branch_test_embs[branch],
                emb_col_names,
                tab_embs_test_all,
                fixed_proj_embs,
                proj_modules[branch],
                sample_indices, target_value, label_classes, k,
            )
            _run_jobs_nice_text(
                args, out_root, device, model, branch,
                branch_train_embs[branch], y_train,
                branch_test_embs[branch],
                tab_embs_test_all,
                fixed_proj_embs,
                proj_modules[branch],
                sample_indices, target_value, k,
            )
            _run_jobs_gradient_text(
                args, out_root, device, model, branch,
                branch_train_embs[branch], y_train,
                branch_test_embs[branch],
                tab_embs_test_all,
                fixed_proj_embs,
                proj_modules[branch],
                sample_indices, target_value, label_classes, k,
            )
        _run_jobs_gradient_tab(
            args, out_root, device, model,
            X_train_static, y_train, X_test_static,
            tab_lof, fixed_proj_embs,
            sample_indices, target_value, label_classes, k,
        )

    elif fusion_strategy == "early":
        # ---- Early fusion path ----
        # Load early fusion MLP (d_in = 3*768 + D_TAB)
        ef_ckpt = torch.load(data_dir / "best_model_early_fusion_mlp.pt", map_location="cpu")
        ef_d_in    = ef_ckpt["d_in"]
        ef_n_cls   = ef_ckpt["n_classes"]
        ef_hidden  = ef_ckpt.get("hidden", 256)
        ef_dropout = ef_ckpt.get("dropout", 0.3)
        label_classes = ef_ckpt["label_classes"]
        ef_depth = _EarlyFusionMLP.depth_from_state_dict(ef_ckpt["state_dict"])
        ef_model = _EarlyFusionMLP(ef_d_in, ef_n_cls, ef_hidden, ef_dropout, depth=ef_depth).to(device)
        ef_model.load_state_dict(ef_ckpt["state_dict"])
        ef_model.eval()

        # Load precomputed text CLS embeddings per branch
        print("[jobs] Loading precomputed early-fusion text embeddings …")
        emb_data = torch.load(data_dir / "early_fusion_text_embeddings.pt",
                              map_location="cpu")
        # emb_data: {train: {desc/company_profile/requirements: (n,768)},
        #            test:  {desc/company_profile/requirements: (n,768)}}
        branch_train_embs = {b: emb_data["train"][b].numpy().astype(np.float32)
                             for b in ("description", "company_profile", "requirements")}
        branch_test_embs  = {b: emb_data["test"][b].numpy().astype(np.float32)
                             for b in ("description", "company_profile", "requirements")}
        emb_dim       = branch_train_embs["description"].shape[1]
        emb_col_names = _build_emb_col_names(emb_dim)

        D_TAB = X_train_static.shape[1]

        def _make_ef_jobs_predict_proba_tab(sample_idx, x_tab_factual):
            # Fix all three text embs; vary tabular
            desc_fixed   = torch.tensor(branch_test_embs["description"][sample_idx][np.newaxis, :],
                                        dtype=torch.float32).to(device)
            prof_fixed   = torch.tensor(branch_test_embs["company_profile"][sample_idx][np.newaxis, :],
                                        dtype=torch.float32).to(device)
            reqs_fixed   = torch.tensor(branch_test_embs["requirements"][sample_idx][np.newaxis, :],
                                        dtype=torch.float32).to(device)
            def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
                x_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = x_t.size(0)
                inp = torch.cat([
                    desc_fixed.expand(B, -1),
                    prof_fixed.expand(B, -1),
                    reqs_fixed.expand(B, -1),
                    x_t,
                ], dim=1)
                with torch.no_grad():
                    logits = ef_model(inp)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            return predict_proba_tab

        def _make_ef_jobs_predict_proba_text(branch_name, sample_idx, x_tab_factual):
            other_branches = [b for b in ("description", "company_profile", "requirements")
                              if b != branch_name]
            other_fixed = {b: torch.tensor(branch_test_embs[b][sample_idx][np.newaxis, :],
                                           dtype=torch.float32).to(device)
                           for b in other_branches}
            tab_fixed = torch.tensor(x_tab_factual[np.newaxis, :],
                                     dtype=torch.float32).to(device)
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = emb_t.size(0)
                parts = []
                for b in ("description", "company_profile", "requirements"):
                    if b == branch_name:
                        parts.append(emb_t)
                    else:
                        parts.append(other_fixed[b].expand(B, -1))
                parts.append(tab_fixed.expand(B, -1))
                inp = torch.cat(parts, dim=1)
                with torch.no_grad():
                    logits = ef_model(inp)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            return predict_proba_emb

        _run_jobs_ef_dice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_col_names, continuous_tab, categorical_tab, tab_lof,
            sample_indices, target_value, label_classes, k,
            _make_ef_jobs_predict_proba_tab,
        )
        _run_jobs_ef_nice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_lof, sample_indices, target_value, k,
            _make_ef_jobs_predict_proba_tab,
        )
        for branch in ("description", "company_profile", "requirements"):
            make_pptext = lambda si, xtf, b=branch: _make_ef_jobs_predict_proba_text(b, si, xtf)
            _run_jobs_ef_dice_text(
                args, out_root, branch,
                branch_train_embs[branch], y_train, X_test_static,
                branch_test_embs[branch],
                emb_col_names, sample_indices, target_value, label_classes, k,
                make_pptext,
            )
            _run_jobs_ef_nice_text(
                args, out_root, branch,
                branch_train_embs[branch], y_train, X_test_static,
                branch_test_embs[branch],
                sample_indices, target_value, k,
                make_pptext,
            )
            print(f"[skip] GRADIENT_text_{branch} — gradient not supported for "
                  "early-fusion strategy")
        print("[skip] GRADIENT_tabular — gradient not supported for early-fusion strategy")

    else:  # late
        # ---- Late fusion path ----
        print("[skip] DiCE_text_* / NICE_text_* — late fusion non-deep text model")
        print("[skip] GRADIENT_tabular / GRADIENT_text_* — non-deep strategy")

        # Load late-fusion tabular MLP: Linear(108→128)→ReLU→Dropout→Linear(128→64)→ReLU→Dropout→Linear(64→2)
        lf_tab_ckpt = torch.load(data_dir / "best_model_late_fusion_mlp_tabular.pt", map_location="cpu")
        lf_sd = {k.removeprefix("net."): v for k, v in lf_tab_ckpt["state_dict"].items()}
        d_in  = lf_sd["0.weight"].shape[1]
        h1    = lf_sd["0.weight"].shape[0]
        h2    = lf_sd["3.weight"].shape[0]
        n_cls = lf_sd["6.weight"].shape[0]
        import torch.nn as _nn
        lf_tab_model = _nn.Sequential(
            _nn.Linear(d_in, h1), _nn.ReLU(), _nn.Dropout(),
            _nn.Linear(h1, h2),  _nn.ReLU(), _nn.Dropout(),
            _nn.Linear(h2, n_cls),
        ).to(device)
        lf_tab_model.load_state_dict(lf_sd)
        lf_tab_model.eval()
        lf_label_classes = lf_tab_ckpt["label_classes"]

        def _make_lf_jobs_predict_proba_tab(sample_idx, x_tab_factual):
            def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
                x_t = torch.tensor(arr, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = lf_tab_model(x_t)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            return predict_proba_tab

        _run_jobs_ef_dice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_col_names, continuous_tab, categorical_tab, tab_lof,
            sample_indices, target_value, lf_label_classes, k,
            _make_lf_jobs_predict_proba_tab,
        )
        _run_jobs_ef_nice_tabular(
            args, out_root,
            X_train_static, y_train, X_test_static,
            tab_lof, sample_indices, target_value, k,
            _make_lf_jobs_predict_proba_tab,
        )


def _run_jobs_ef_dice_tabular(
    args, out_root,
    X_train_static, y_train, X_test_static,
    tab_col_names, continuous_tab, categorical_tab, tab_lof,
    sample_indices, target_value, label_classes, k,
    make_predict_proba_tab,
):
    """DiCE_tabular for jobs early fusion using a factory predict_proba."""
    name = "DiCE_tabular"
    print(f"\n[jobs] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        predict_proba_tab = make_predict_proba_tab(sample_idx, x_tab_factual)
        cfs, converged = run_dice_on_features(
            X_train=X_train_static, y_train=y_train,
            col_names=tab_col_names,
            continuous_cols=continuous_tab, categorical_cols=categorical_tab,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="dice_random", modality="tabular",
        held_fixed=["text_description", "text_company_profile", "text_requirements"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_ef_nice_tabular(
    args, out_root,
    X_train_static, y_train, X_test_static,
    tab_lof, sample_indices, target_value, k,
    make_predict_proba_tab,
):
    """NICE_tabular for jobs early fusion using a factory predict_proba."""
    name = "NICE_tabular"
    print(f"\n[jobs] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        predict_proba_tab = make_predict_proba_tab(sample_idx, x_tab_factual)
        cfs, converged = run_nice(
            X_train=X_train_static, y_train=y_train,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="nice", modality="tabular",
        held_fixed=["text_description", "text_company_profile", "text_requirements"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_ef_dice_text(
    args, out_root, branch_name,
    X_train_emb, y_train, X_test_static, X_test_emb,
    emb_col_names, sample_indices, target_value, label_classes, k,
    make_predict_proba_text,
):
    """DiCE_text_<branch> for jobs early fusion using a factory predict_proba."""
    name = f"DiCE_text_{branch_name}"
    other_branches = [b for b in ("description", "company_profile", "requirements")
                      if b != branch_name]
    print(f"\n[jobs] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        emb_factual   = X_test_emb[sample_idx]
        predict_proba_emb = make_predict_proba_text(sample_idx, x_tab_factual)
        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    other_held = [f"text_{b}" for b in other_branches] + ["tabular"]
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="dice_random", modality=f"text_{branch_name}",
        held_fixed=other_held, embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed early-fusion text embedding for the '{branch_name}' "
            "field, NOT actual text. Other branches and tabular are held fixed. "
            "Proximity/sparsity in embedding space."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_ef_nice_text(
    args, out_root, branch_name,
    X_train_emb, y_train, X_test_static, X_test_emb,
    sample_indices, target_value, k,
    make_predict_proba_text,
):
    """NICE_text_<branch> for jobs early fusion using a factory predict_proba."""
    name = f"NICE_text_{branch_name}"
    other_branches = [b for b in ("description", "company_profile", "requirements")
                      if b != branch_name]
    print(f"\n[jobs] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        emb_factual   = X_test_emb[sample_idx]
        predict_proba_emb = make_predict_proba_text(sample_idx, x_tab_factual)
        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    other_held = [f"text_{b}" for b in other_branches] + ["tabular"]
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="nice", modality=f"text_{branch_name}",
        held_fixed=other_held, embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed early-fusion text embedding for the '{branch_name}' "
            "field, NOT actual text. Other branches and tabular are held fixed. "
            "Proximity/sparsity in embedding space."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_dice_tabular(
    args, out_root, device, model, tokenizer,
    X_train_static, y_train, X_test_static, text_branches,
    tab_col_names, continuous_tab, categorical_tab, tab_lof,
    fixed_proj_embs,
    sample_indices, target_value, label_classes, k,
):
    name = "DiCE_tabular"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]

        # Fixed: pre-projected embeddings of all three text branches
        d_fixed = torch.tensor(fixed_proj_embs["description"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)
        p_fixed = torch.tensor(fixed_proj_embs["company_profile"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)
        r_fixed = torch.tensor(fixed_proj_embs["requirements"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb = model.tab_head(x_t)
                logits  = model.classifier(torch.cat([
                    d_fixed.expand(B, -1),
                    p_fixed.expand(B, -1),
                    r_fixed.expand(B, -1),
                    tab_emb,
                ], dim=1))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_dice_on_features(
            X_train=X_train_static, y_train=y_train,
            col_names=tab_col_names,
            continuous_cols=continuous_tab, categorical_cols=categorical_tab,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="dice_random", modality="tabular",
        held_fixed=["text_description", "text_company_profile", "text_requirements"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_dice_text(
    args, out_root, device, model, branch_name,
    X_train_emb, y_train, X_test_emb,
    emb_col_names,
    tab_embs_test_all,
    fixed_proj_embs,
    proj_module,
    sample_indices, target_value, label_classes, k,
):
    name = f"DiCE_text_{branch_name}"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    other_branches = [b for b in fixed_proj_embs if b != branch_name]

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        tab_emb_fixed = torch.tensor(
            tab_embs_test_all[sample_idx][np.newaxis, :],
            dtype=torch.float32,
        ).to(device)
        other_embs = {
            b: torch.tensor(fixed_proj_embs[b][sample_idx][np.newaxis, :],
                            dtype=torch.float32).to(device)
            for b in other_branches
        }

        def predict_proba_emb(arr: np.ndarray, _branch=branch_name) -> np.ndarray:
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_proj = proj_module(emb_t)  # (B, text_hidden_dim)
                parts = []
                for b in ("description", "company_profile", "requirements"):
                    if b == _branch:
                        parts.append(this_proj)
                    else:
                        parts.append(other_embs[b].expand(B, -1))
                parts.append(tab_emb_fixed.expand(B, -1))
                logits = model.classifier(torch.cat(parts, dim=1))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    other_held = [f"text_{b}" for b in other_branches] + ["tabular"]
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="dice_random", modality=f"text_{branch_name}",
        held_fixed=other_held,
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed DistilBERT CLS embedding for the '{branch_name}' "
            "field, NOT actual text. Other text branches and tabular features are "
            "held fixed at factual values. Proximity/sparsity in embedding space."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_nice_tabular(
    args, out_root, device, model, tokenizer,
    X_train_static, y_train, X_test_static, text_branches,
    tab_lof, fixed_proj_embs,
    sample_indices, target_value, k,
):
    name = "NICE_tabular"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts: Dict[str, int] = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        fixed_embs = {
            b: torch.tensor(fixed_proj_embs[b][sample_idx][np.newaxis, :],
                            dtype=torch.float32).to(device)
            for b in fixed_proj_embs
        }

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb = model.tab_head(x_t)
                parts = [
                    fixed_embs["description"].expand(B, -1),
                    fixed_embs["company_profile"].expand(B, -1),
                    fixed_embs["requirements"].expand(B, -1),
                    tab_emb,
                ]
                logits = model.classifier(torch.cat(parts, dim=1))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_nice(
            X_train=X_train_static, y_train=y_train,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="nice", modality="tabular",
        held_fixed=["text_description", "text_company_profile", "text_requirements"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_jobs_nice_text(
    args, out_root, device, model, branch_name,
    X_train_emb, y_train, X_test_emb,
    tab_embs_test_all,
    fixed_proj_embs,
    proj_module,
    sample_indices, target_value, k,
):
    name = f"NICE_text_{branch_name}"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    other_branches = [b for b in fixed_proj_embs if b != branch_name]

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        tab_emb_fixed = torch.tensor(
            tab_embs_test_all[sample_idx][np.newaxis, :],
            dtype=torch.float32,
        ).to(device)
        other_embs = {
            b: torch.tensor(fixed_proj_embs[b][sample_idx][np.newaxis, :],
                            dtype=torch.float32).to(device)
            for b in other_branches
        }

        def predict_proba_emb(arr: np.ndarray, _branch=branch_name) -> np.ndarray:
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_proj = proj_module(emb_t)
                parts = []
                for b in ("description", "company_profile", "requirements"):
                    if b == _branch:
                        parts.append(this_proj)
                    else:
                        parts.append(other_embs[b].expand(B, -1))
                parts.append(tab_emb_fixed.expand(B, -1))
                logits = model.classifier(torch.cat(parts, dim=1))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    other_held = [f"text_{b}" for b in other_branches] + ["tabular"]
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="nice", modality=f"text_{branch_name}",
        held_fixed=other_held,
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed DistilBERT CLS embedding for the '{branch_name}' "
            "field, NOT actual text. Other text branches and tabular features are "
            "held fixed at factual values. Proximity/sparsity in embedding space."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


# ===========================================================================
# Hateful Memes baselines
# ===========================================================================

def run_memes_baselines(args: argparse.Namespace) -> None:
    dataset_dir = _ROOT / "memes"
    data_dir    = dataset_dir / "data"
    fusion_strategy = getattr(args, "fusion_strategy", "intermediate")
    device      = _resolve_device(args.gpu)

    # Determine output root per fusion strategy
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif fusion_strategy == "intermediate":
        out_root = data_dir / "baselines"
    elif fusion_strategy == "early":
        out_root = data_dir / "baselines_fusion_early"
    else:  # late
        out_root = data_dir / "baselines_fusion_late"

    sys.path.insert(0, str(dataset_dir))
    from encoder_features import (  # type: ignore
        FineTuneMultimodalClassifier,
        TEXT_ENCODERS,
        IMAGE_ENCODERS,
        build_text_backend,
        build_image_backend,
        get_image_transform,
        pool_text_features,
    )

    # ---- Load data ----
    npz = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_train_text  = npz["X_train_text"].tolist()
    X_test_text   = npz["X_test_text"].tolist()
    X_train_img   = npz["X_train_img"]    # (n, 224, 224, 3) uint8
    X_test_img    = npz["X_test_img"]
    y_train       = npz["y_train"].astype(int)
    y_test        = npz["y_test"].astype(int)
    label_classes = npz["label_classes"].tolist()

    # ---- Sample indices (hateful → not hateful) ----
    target_value = args.target_value
    if target_value is None:
        target_value = (label_classes.index("not_hateful")
                        if "not_hateful" in label_classes else 0)

    _memes_pred_file = {
        "intermediate": "y_pred.npy",
        "early":        "y_pred_early_fusion_gbt.npy",
        "late":         "y_pred_late_fusion_nondp.npy",
    }.get(fusion_strategy, "y_pred.npy")
    y_pred_path = data_dir / _memes_pred_file
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else y_test
    if args.sample_indices:
        sample_indices = _parse_csv_indices(args.sample_indices)
    else:
        source_class = 1 - target_value
        sample_indices = _select_source_samples(
            y_pred, source_class, args.max_samples
        )

    k = args.k
    print(f"[memes] {len(sample_indices)} samples → target={label_classes[target_value]}"
          f"  fusion_strategy={fusion_strategy}")

    if fusion_strategy == "intermediate":
        # ---- Existing intermediate-fusion path ----
        ckpt = torch.load(data_dir / "best_model.pt", map_location="cpu")
        cfg  = ckpt["config"]
        model = FineTuneMultimodalClassifier(
            text_encoder_name = cfg["text_encoder"],
            image_encoder_name = cfg["image_encoder"],
            n_classes          = len(ckpt["label_classes"]),
            text_hidden_dim    = cfg["text_hidden_dim"],
            img_hidden_dim     = cfg["img_hidden_dim"],
            dropout            = cfg["dropout"],
            device             = device,
            word2vec_path      = cfg.get("word2vec_path"),
            word2vec_vocab_tokens = cfg.get("word2vec_vocab_tokens"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        text_enc_name = cfg["text_encoder"]
        img_enc_name  = cfg["image_encoder"]
        text_cfg      = TEXT_ENCODERS.get(text_enc_name, {})
        text_dim      = text_cfg.get("feature_dim", 768)
        img_dim       = IMAGE_ENCODERS.get(img_enc_name, {})["feature_dim"]

        if text_enc_name == "word2vec":
            print("[memes] word2vec encoder: embedding-space baseline not supported. "
                  "Re-run with a transformer-based encoder checkpoint.")
            return

        tokenizer, text_encoder, _ = build_text_backend(text_enc_name, device)
        pooling   = text_cfg.get("pooling", "cls")
        normalize = text_cfg.get("normalize", False)
        X_train_text_emb = _encode_texts_batch(
            X_train_text, tokenizer, text_encoder, device,
            max_len=cfg.get("max_len", 128), pooling=pooling, normalize=normalize,
        )
        X_test_text_emb = _encode_texts_batch(
            X_test_text, tokenizer, text_encoder, device,
            max_len=cfg.get("max_len", 128), pooling=pooling, normalize=normalize,
        )
        text_emb_cols = _build_emb_col_names(text_dim)

        img_encoder, img_transform = build_image_backend(img_enc_name, device)
        X_train_img_emb = _encode_images_batch(X_train_img, img_encoder, img_transform, device)
        X_test_img_emb  = _encode_images_batch(X_test_img,  img_encoder, img_transform, device)
        img_emb_cols = _build_emb_col_names(img_dim)

        X_test_txt_t = torch.tensor(X_test_text_emb, dtype=torch.float32).to(device)
        X_test_img_t = torch.tensor(X_test_img_emb,  dtype=torch.float32).to(device)
        with torch.no_grad():
            text_proj_embs_test = model.text_proj(X_test_txt_t).cpu().numpy()
            img_proj_embs_test  = model.img_proj(X_test_img_t).cpu().numpy()

        _run_memes_dice_emb(
            args, out_root, device, model, "meme_text",
            X_train_text_emb, y_train, X_test_text_emb,
            text_emb_cols, img_proj_embs_test,
            model.text_proj, model.img_proj, is_text=True,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
        )
        _run_memes_nice_emb(
            args, out_root, device, model, "meme_text",
            X_train_text_emb, y_train, X_test_text_emb,
            img_proj_embs_test, model.text_proj, is_text=True,
            sample_indices=sample_indices, target_value=target_value, k=k,
        )
        _run_memes_gradient_emb(
            args, out_root, device, model, "meme_text",
            X_train_text_emb, y_train, X_test_text_emb,
            img_proj_embs_test, model.text_proj, is_text=True,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
        )
        _run_memes_dice_emb(
            args, out_root, device, model, "meme_img",
            X_train_img_emb, y_train, X_test_img_emb,
            img_emb_cols, text_proj_embs_test,
            model.img_proj, model.text_proj, is_text=False,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
        )
        _run_memes_nice_emb(
            args, out_root, device, model, "meme_img",
            X_train_img_emb, y_train, X_test_img_emb,
            text_proj_embs_test, model.img_proj, is_text=False,
            sample_indices=sample_indices, target_value=target_value, k=k,
        )
        _run_memes_gradient_emb(
            args, out_root, device, model, "meme_img",
            X_train_img_emb, y_train, X_test_img_emb,
            text_proj_embs_test, model.img_proj, is_text=False,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
        )

    elif fusion_strategy == "early":
        # ---- Early fusion path ----
        # Load early-fusion MLP; use precomputed text+image embeddings
        ef_ckpt = torch.load(data_dir / "best_model_early_fusion_mlp.pt", map_location="cpu")
        ef_cfg   = ef_ckpt["config"]
        # config keys: text_encoder, image_encoder, d_in
        text_enc_name = ef_cfg.get("text_encoder", "bert")
        img_enc_name  = ef_cfg.get("image_encoder", "resnet50")
        ef_d_in   = ef_cfg.get("d_in", ef_ckpt.get("d_in", 2816))
        label_classes = ef_ckpt.get("label_classes", label_classes)
        ef_n_cls  = len(label_classes)
        ef_hidden = ef_cfg.get("hidden", 256) if isinstance(ef_cfg, dict) else 256
        ef_dropout = ef_cfg.get("dropout", 0.3) if isinstance(ef_cfg, dict) else 0.3
        ef_depth = _EarlyFusionMLP.depth_from_state_dict(ef_ckpt["state_dict"])
        ef_model  = _EarlyFusionMLP(ef_d_in, ef_n_cls, ef_hidden, ef_dropout, depth=ef_depth).to(device)
        ef_model.load_state_dict(ef_ckpt["state_dict"])
        ef_model.eval()

        print("[memes] Loading precomputed embeddings for early fusion …")
        emb_data = torch.load(data_dir / "precomputed_embeddings.pt", map_location="cpu")
        # structure: {text: {bert: {train:(n,768), test:(n,768)}},
        #             image: {resnet50: {train:(n,2048), test:(n,2048)}}}
        X_train_text_emb = emb_data["text"][text_enc_name]["train"].numpy().astype(np.float32)
        X_test_text_emb  = emb_data["text"][text_enc_name]["test"].numpy().astype(np.float32)
        X_train_img_emb  = emb_data["image"][img_enc_name]["train"].numpy().astype(np.float32)
        X_test_img_emb   = emb_data["image"][img_enc_name]["test"].numpy().astype(np.float32)
        text_dim = X_train_text_emb.shape[1]
        img_dim  = X_train_img_emb.shape[1]
        text_emb_cols = _build_emb_col_names(text_dim)
        img_emb_cols  = _build_emb_col_names(img_dim)

        # EF predict_proba factories
        def _make_ef_memes_predict_proba_text(sample_idx, x_tab_factual):
            img_fixed = torch.tensor(
                X_test_img_emb[sample_idx][np.newaxis, :], dtype=torch.float32
            ).to(device)
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = emb_t.size(0)
                inp = torch.cat([emb_t, img_fixed.expand(B, -1)], dim=1)
                with torch.no_grad():
                    logits = ef_model(inp)
                    return torch.softmax(logits, dim=1).cpu().numpy()
            return predict_proba_emb

        def _make_ef_memes_predict_proba_image(sample_idx, x_tab_factual):
            txt_fixed = torch.tensor(
                X_test_text_emb[sample_idx][np.newaxis, :], dtype=torch.float32
            ).to(device)
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
                B = emb_t.size(0)
                inp = torch.cat([txt_fixed.expand(B, -1), emb_t], dim=1)
                with torch.no_grad():
                    logits = ef_model(inp)
                    return torch.softmax(logits, dim=1).cpu().numpy()
            return predict_proba_emb

        _run_memes_ef_dice_emb(
            args, out_root, "meme_text", is_text=True,
            X_train_emb=X_train_text_emb, y_train=y_train,
            X_test_emb=X_test_text_emb,
            emb_col_names=text_emb_cols,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
            make_predict_proba=_make_ef_memes_predict_proba_text,
        )
        _run_memes_ef_nice_emb(
            args, out_root, "meme_text", is_text=True,
            X_train_emb=X_train_text_emb, y_train=y_train,
            X_test_emb=X_test_text_emb,
            sample_indices=sample_indices, target_value=target_value, k=k,
            make_predict_proba=_make_ef_memes_predict_proba_text,
        )
        _run_memes_ef_dice_emb(
            args, out_root, "meme_img", is_text=False,
            X_train_emb=X_train_img_emb, y_train=y_train,
            X_test_emb=X_test_img_emb,
            emb_col_names=img_emb_cols,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
            make_predict_proba=_make_ef_memes_predict_proba_image,
        )
        _run_memes_ef_nice_emb(
            args, out_root, "meme_img", is_text=False,
            X_train_emb=X_train_img_emb, y_train=y_train,
            X_test_emb=X_test_img_emb,
            sample_indices=sample_indices, target_value=target_value, k=k,
            make_predict_proba=_make_ef_memes_predict_proba_image,
        )
        print("[skip] GRADIENT_text_meme_text / GRADIENT_image_meme_img — "
              "gradient not supported for early-fusion strategy")

    else:  # late
        # ---- Late fusion path ----
        # RF models for text and image; predict_proba by averaging both branches
        import pickle

        text_rf_path  = data_dir / "best_model_late_fusion_rf_text.pkl"
        image_rf_path = data_dir / "best_model_late_fusion_rf_image.pkl"

        def _load_rf_dict(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[memes] Warning: failed to load {path.name} ({e})")
                return None

        text_rf_dict  = _load_rf_dict(text_rf_path)  if text_rf_path.exists()  else None
        image_rf_dict = _load_rf_dict(image_rf_path) if image_rf_path.exists() else None

        if text_rf_dict is None or image_rf_dict is None:
            print("[memes] late fusion: one or both RF models unavailable; skipping all baselines")
            return

        text_enc_name  = text_rf_dict.get("encoder", "bert")
        image_enc_name = image_rf_dict.get("encoder", "resnet50")
        text_rf  = text_rf_dict["model"]
        image_rf = image_rf_dict["model"]
        label_classes = text_rf_dict.get("label_classes", label_classes)

        # Load precomputed embeddings
        print("[memes] Loading precomputed embeddings for late fusion …")
        emb_data = torch.load(data_dir / "precomputed_embeddings.pt", map_location="cpu")
        X_train_text_emb = emb_data["text"][text_enc_name]["train"].numpy().astype(np.float32)
        X_test_text_emb  = emb_data["text"][text_enc_name]["test"].numpy().astype(np.float32)
        X_train_img_emb  = emb_data["image"][image_enc_name]["train"].numpy().astype(np.float32)
        X_test_img_emb   = emb_data["image"][image_enc_name]["test"].numpy().astype(np.float32)
        text_dim = X_train_text_emb.shape[1]
        img_dim  = X_train_img_emb.shape[1]
        text_emb_cols = _build_emb_col_names(text_dim)
        img_emb_cols  = _build_emb_col_names(img_dim)

        # Average both RF branch predictions
        def _make_late_memes_predict_proba_text(sample_idx, x_tab_factual):
            img_fixed = X_test_img_emb[sample_idx][np.newaxis, :]  # (1, img_dim)
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                p_text = text_rf.predict_proba(arr)
                # replicate image prediction for batch
                p_img  = image_rf.predict_proba(
                    np.tile(img_fixed, (arr.shape[0], 1))
                )
                return ((p_text + p_img) / 2.0).astype(np.float32)
            return predict_proba_emb

        def _make_late_memes_predict_proba_image(sample_idx, x_tab_factual):
            txt_fixed = X_test_text_emb[sample_idx][np.newaxis, :]
            def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
                p_img  = image_rf.predict_proba(arr)
                p_text = text_rf.predict_proba(
                    np.tile(txt_fixed, (arr.shape[0], 1))
                )
                return ((p_text + p_img) / 2.0).astype(np.float32)
            return predict_proba_emb

        _run_memes_ef_dice_emb(
            args, out_root, "meme_text", is_text=True,
            X_train_emb=X_train_text_emb, y_train=y_train,
            X_test_emb=X_test_text_emb,
            emb_col_names=text_emb_cols,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
            make_predict_proba=_make_late_memes_predict_proba_text,
        )
        _run_memes_ef_nice_emb(
            args, out_root, "meme_text", is_text=True,
            X_train_emb=X_train_text_emb, y_train=y_train,
            X_test_emb=X_test_text_emb,
            sample_indices=sample_indices, target_value=target_value, k=k,
            make_predict_proba=_make_late_memes_predict_proba_text,
        )
        _run_memes_ef_dice_emb(
            args, out_root, "meme_img", is_text=False,
            X_train_emb=X_train_img_emb, y_train=y_train,
            X_test_emb=X_test_img_emb,
            emb_col_names=img_emb_cols,
            sample_indices=sample_indices, target_value=target_value,
            label_classes=label_classes, k=k,
            make_predict_proba=_make_late_memes_predict_proba_image,
        )
        _run_memes_ef_nice_emb(
            args, out_root, "meme_img", is_text=False,
            X_train_emb=X_train_img_emb, y_train=y_train,
            X_test_emb=X_test_img_emb,
            sample_indices=sample_indices, target_value=target_value, k=k,
            make_predict_proba=_make_late_memes_predict_proba_image,
        )
        print("[skip] GRADIENT_text_meme_text / GRADIENT_image_meme_img — "
              "gradient not supported for non-deep RF strategy")


def _run_memes_ef_dice_emb(
    args, out_root, modality_name, is_text: bool,
    X_train_emb, y_train, X_test_emb,
    emb_col_names, sample_indices, target_value, label_classes, k,
    make_predict_proba,
    X_test_static=None,
):
    """DiCE embedding baseline for memes early/late fusion using a factory predict_proba."""
    mod_type = "text" if is_text else "image"
    name = f"DiCE_{mod_type}_{modality_name}"
    held = "image_meme_img" if is_text else "text_meme_text"
    print(f"\n[memes] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual = X_test_emb[sample_idx]
        x_tab_factual = (X_test_static[sample_idx]
                         if X_test_static is not None else np.zeros(1, dtype=np.float32))
        predict_proba_emb = make_predict_proba(sample_idx, x_tab_factual)
        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="hateful_memes",
        method="dice_random", modality=f"{mod_type}_{modality_name}",
        held_fixed=[held], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed {mod_type}-encoder embedding, NOT actual "
            f"{'text or image' if mod_type == 'text' else 'image or text'}. "
            f"The {held} branch is held fixed at factual value. "
            "Proximity/sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_memes_ef_nice_emb(
    args, out_root, modality_name, is_text: bool,
    X_train_emb, y_train, X_test_emb,
    sample_indices, target_value, k,
    make_predict_proba,
    X_test_static=None,
):
    """NICE embedding baseline for memes early/late fusion using a factory predict_proba."""
    mod_type = "text" if is_text else "image"
    name = f"NICE_{mod_type}_{modality_name}"
    held = "image_meme_img" if is_text else "text_meme_text"
    print(f"\n[memes] Running {name} (factory) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual = X_test_emb[sample_idx]
        x_tab_factual = (X_test_static[sample_idx]
                         if X_test_static is not None else np.zeros(1, dtype=np.float32))
        predict_proba_emb = make_predict_proba(sample_idx, x_tab_factual)
        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))
        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="hateful_memes",
        method="nice", modality=f"{mod_type}_{modality_name}",
        held_fixed=[held], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed {mod_type}-encoder embedding, NOT actual "
            f"{'text' if mod_type == 'text' else 'image'}. "
            f"The {held} branch is held fixed at factual value. "
            "Proximity/sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_memes_dice_emb(
    args, out_root, device, model, modality_name,
    X_train_emb, y_train, X_test_emb,
    emb_col_names,
    fixed_proj_embs_test,   # (n_test, hidden_dim) — OTHER modality projected
    this_proj, other_proj,
    is_text: bool,
    sample_indices, target_value, label_classes, k,
):
    name = f"DiCE_{'text' if is_text else 'image'}_{modality_name}"
    held = "image_meme_img" if is_text else "text_meme_text"
    print(f"\n[memes] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        other_proj_fixed = torch.tensor(
            fixed_proj_embs_test[sample_idx][np.newaxis, :],
            dtype=torch.float32,
        ).to(device)

        def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_emb  = this_proj(emb_t)
                other_exp = other_proj_fixed.expand(B, -1)
                if is_text:
                    fused = torch.cat([this_emb, other_exp], dim=1)
                else:
                    fused = torch.cat([other_exp, this_emb], dim=1)
                logits = model.classifier(fused)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_dice_embedding_fast(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    mod_type = "text" if is_text else "image"
    payload = make_summary_payload(
        baseline_name=name, dataset="hateful_memes",
        method="dice_random", modality=f"{mod_type}_{modality_name}",
        held_fixed=[held],
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed {mod_type}-encoder embedding, NOT actual "
            f"{'text or image' if mod_type == 'text' else 'image or text'}. "
            f"The {held} branch is held fixed at factual value. "
            "Proximity/sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_memes_nice_emb(
    args, out_root, device, model, modality_name,
    X_train_emb, y_train, X_test_emb,
    fixed_proj_embs_test,   # (n_test, hidden_dim) — OTHER modality projected
    this_proj,
    is_text: bool,
    sample_indices, target_value, k,
):
    name = f"NICE_{'text' if is_text else 'image'}_{modality_name}"
    held = "image_meme_img" if is_text else "text_meme_text"
    print(f"\n[memes] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual = X_test_emb[sample_idx]
        other_proj_fixed = torch.tensor(
            fixed_proj_embs_test[sample_idx][np.newaxis, :],
            dtype=torch.float32,
        ).to(device)

        def predict_proba_emb(arr: np.ndarray) -> np.ndarray:
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_emb  = this_proj(emb_t)
                other_exp = other_proj_fixed.expand(B, -1)
                if is_text:
                    fused = torch.cat([this_emb, other_exp], dim=1)
                else:
                    fused = torch.cat([other_exp, this_emb], dim=1)
                logits = model.classifier(fused)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        cfs, converged = run_nice(
            X_train=X_train_emb, y_train=y_train,
            x_query=emb_factual, target_value=target_value,
            predict_proba_fn=predict_proba_emb, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    mod_type = "text" if is_text else "image"
    payload = make_summary_payload(
        baseline_name=name, dataset="hateful_memes",
        method="nice", modality=f"{mod_type}_{modality_name}",
        held_fixed=[held],
        embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            f"Result is a perturbed {mod_type}-encoder embedding, NOT actual "
            f"{'text' if mod_type == 'text' else 'image'}. "
            f"The {held} branch is held fixed at factual value. "
            "Proximity/sparsity in embedding space. Plausibility is NaN."
        ),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


# ===========================================================================
# Sepsis baselines
# ===========================================================================

def _load_sepsis_model_for_strategy(
    data_dir: Path, fold: int, fusion_strategy: str, device: str
) -> Tuple[nn.Module, bool]:
    """Load the appropriate Sepsis model for the given fusion strategy.

    Returns (model, is_deep) where is_deep indicates whether gradient-based
    baselines are supported.
    """
    if fusion_strategy == "intermediate":
        # Default: existing early-fusion model (called "intermediate/best" historically)
        model_path = data_dir / f"best_model_{fold}.pt"
        model = _load_sepsis_model(model_path, device)
        return model, True

    if fusion_strategy == "early":
        # Same early-fusion SepsisModel — just named best_model_{fold}.pt
        model_path = data_dir / f"best_model_{fold}.pt"
        model = _load_sepsis_model(model_path, device)
        return model, True

    if fusion_strategy == "late":
        # Late fusion: average of TSOnly + StaticOnly
        import pickle
        ts_path     = data_dir / f"best_model_late_fusion_mlp_ts_fold{fold}.pt"
        static_path = data_dir / f"best_model_late_fusion_mlp_static_fold{fold}.pt"
        ts_m = _SepsisTSModel()
        ts_m.load_state_dict(torch.load(ts_path, map_location="cpu"))
        static_m = _SepsisStaticModel()
        static_m.load_state_dict(torch.load(static_path, map_location="cpu"))
        model = _SepsisLateWrapper(ts_m, static_m).to(device)
        model.eval()
        return model, True

    raise ValueError(f"Unknown fusion_strategy: {fusion_strategy!r}")


def run_sepsis_baselines(args: argparse.Namespace) -> None:
    dataset_dir = _ROOT / "sepsis"
    data_dir    = dataset_dir / "data"
    fold        = args.fold
    fold_dir    = data_dir / f"fold_{fold}"
    fusion_strategy = getattr(args, "fusion_strategy", "intermediate")
    device      = _resolve_device(args.gpu)
    ts_name     = "lab_exams_vitals"

    # Determine output root per fusion strategy
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif fusion_strategy == "intermediate":
        # Preserve existing default path (backward compat)
        out_root = data_dir / "baselines" / f"fold_{fold}"
    elif fusion_strategy == "early":
        out_root = data_dir / "baselines_fusion_early" / f"fold_{fold}"
    else:  # late
        out_root = data_dir / "baselines_fusion_late" / f"fold_{fold}"

    print(f"[sepsis] Fold {fold} — device={device}  fusion_strategy={fusion_strategy}")

    # ---- Load data ----
    X_train_static = np.load(fold_dir / "X_train_static.npy").astype(np.float32)
    X_test_static  = np.load(fold_dir / "X_test_static.npy").astype(np.float32)
    X_train_ts     = np.load(fold_dir / "X_train_ts.npy").astype(np.float32)
    X_test_ts      = np.load(fold_dir / "X_test_ts.npy").astype(np.float32)
    y_train        = np.load(fold_dir / "y_train.npy").astype(int)
    y_test         = np.load(fold_dir / "y_test.npy").astype(int)

    # ---- Load strategy-specific model ----
    model, is_deep = _load_sepsis_model_for_strategy(
        data_dir, fold, fusion_strategy, device
    )

    # Feature dimension names for tabular features
    D_static  = X_train_static.shape[1]
    tab_cols  = [f"static_{i}" for i in range(D_static)]

    # ---- Sample indices (death → no_death) ----
    label_classes = ["no_death", "death"]
    target_value  = args.target_value
    if target_value is None:
        target_value = label_classes.index("no_death")

    _sepsis_pred_file = {
        "early":        "y_pred.npy",
        "intermediate": "y_pred_intermediate_rf.npy",
        "late":         "y_pred_late_deep.npy",
    }.get(fusion_strategy, "y_pred.npy")
    y_pred_path = fold_dir / _sepsis_pred_file
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else y_test
    if args.sample_indices:
        sample_indices = _parse_csv_indices(args.sample_indices)
    else:
        source_class = 1 - target_value
        sample_indices = _select_source_samples(
            y_pred, source_class, args.max_samples
        )

    k = args.k
    print(f"[sepsis] {len(sample_indices)} samples → target={label_classes[target_value]}")

    # ---- Feature types + LOF ----
    continuous_tab, categorical_tab = infer_feature_types(X_train_static, tab_cols)
    print(f"[sepsis] Tabular: {len(continuous_tab)} continuous, {len(categorical_tab)} categorical")
    print("[sepsis] Fitting tabular LOF …")
    tab_lof = fit_tabular_lof(X_train_static)

    # ===================================================================
    # Baseline 1: DiCE_tabular
    # ===================================================================
    _run_sepsis_dice_tabular(
        args, out_root, device, model,
        X_train_static, y_train, X_test_static, X_test_ts,
        tab_cols, continuous_tab, categorical_tab, tab_lof,
        ts_name, sample_indices, target_value, label_classes, k, fold,
    )

    # ===================================================================
    # Baseline 2: NICE_tabular
    # ===================================================================
    _run_sepsis_nice_tabular(
        args, out_root, device, model,
        X_train_static, y_train, X_test_static, X_test_ts,
        tab_lof, ts_name, sample_indices, target_value, k, fold,
    )

    # ===================================================================
    # Baseline 3: GRADIENT_tabular
    # ===================================================================
    if is_deep:
        _run_sepsis_gradient_tab(
            args, out_root, device, model,
            X_train_static, y_train, X_test_static, X_test_ts,
            tab_lof, ts_name, sample_indices, target_value, k, fold,
        )
    else:
        print(f"[skip] GRADIENT_tabular — gradient not supported for non-deep "
              f"fusion_strategy={fusion_strategy!r}")

    # ===================================================================
    # Baseline 4: Gradient-based TS perturbation
    # ===================================================================
    if is_deep:
        _run_sepsis_gradient_ts(
            args, out_root, device, model,
            X_test_static, X_test_ts,
            ts_name, sample_indices, target_value, label_classes, k, fold,
        )
    else:
        print(f"[skip] GRADIENT_TS_{ts_name} — gradient not supported for non-deep "
              f"fusion_strategy={fusion_strategy!r}")

    # ===================================================================
    # Baseline 5: Native Guide (NativeGuide-ISW)
    # ===================================================================
    _run_sepsis_native_guide(
        args, out_root, device, model,
        X_train_ts, y_train, X_test_static, X_test_ts,
        ts_name, sample_indices, target_value, label_classes, k, fold,
    )

    # ===================================================================
    # Baseline 6: CoMTE greedy segment swap
    # ===================================================================
    _run_sepsis_comte(
        args, out_root, device, model,
        X_train_ts, y_train, X_test_static, X_test_ts,
        ts_name, sample_indices, target_value, label_classes, k, fold,
    )


def _run_sepsis_dice_tabular(
    args, out_root, device, model,
    X_train_static, y_train, X_test_static, X_test_ts,
    tab_cols, continuous_tab, categorical_tab, tab_lof,
    ts_name, sample_indices, target_value, label_classes, k, fold,
):
    name = "DiCE_tabular"
    print(f"\n[sepsis fold {fold}] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        x_ts_factual  = X_test_ts[sample_idx]   # (T, C)

        # TS is held fixed — pre-expand to match any batch size
        x_ts_t = torch.tensor(x_ts_factual[np.newaxis, :, :],
                              dtype=torch.float32).to(device)  # (1, T, C)

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            # arr: (B, D_static)
            x_static_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_static_t.size(0)
            x_ts_exp = x_ts_t.expand(B, -1, -1)  # (B, T, C)
            with torch.no_grad():
                p = model(x_ts_exp, x_static_t).cpu().numpy()  # (B, 1)
                p = p.squeeze(1)  # (B,)
                probs = np.stack([1 - p, p], axis=1)           # (B, 2)
            return probs

        cfs, converged = run_dice_on_features(
            X_train=X_train_static, y_train=y_train,
            col_names=tab_cols,
            continuous_cols=continuous_tab, categorical_cols=categorical_tab,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab,
            k=k, sample_size=args.dice_sample_size, random_seed=args.dice_seed,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="dice_random", modality="tabular",
        held_fixed=[ts_name],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        extra_meta={"fold": fold, "ts_name": ts_name},
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_sepsis_nice_tabular(
    args, out_root, device, model,
    X_train_static, y_train, X_test_static, X_test_ts,
    tab_lof, ts_name, sample_indices, target_value, k, fold,
):
    name = "NICE_tabular"
    print(f"\n[sepsis] Running {name} (fold {fold}) …")
    started_at = datetime.datetime.now(datetime.timezone.utc)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        x_ts_factual  = X_test_ts[sample_idx]

        x_ts_t = torch.tensor(x_ts_factual[np.newaxis, :, :],
                              dtype=torch.float32).to(device)

        def predict_proba_tab(arr: np.ndarray) -> np.ndarray:
            x_static_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_static_t.size(0)
            x_ts_exp = x_ts_t.expand(B, -1, -1)
            with torch.no_grad():
                p = model(x_ts_exp, x_static_t).cpu().numpy()
                p = p.squeeze(1)
                probs = np.stack([1 - p, p], axis=1)
            return probs

        cfs, converged = run_nice(
            X_train=X_train_static, y_train=y_train,
            x_query=x_tab_factual, target_value=target_value,
            predict_proba_fn=predict_proba_tab, k=k,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="nice", modality="tabular",
        held_fixed=[ts_name],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        extra_meta={"fold": fold, "ts_name": ts_name},
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_sepsis_gradient_ts(
    args, out_root, device, model,
    X_test_static, X_test_ts,
    ts_name, sample_indices, target_value, label_classes, k, fold,
):
    """Gradient-based counterfactual perturbation of the TS branch.

    Methodology note
    ----------------
    No established off-the-shelf optimisation-based TS counterfactual library
    was available at the time of implementation.  This baseline uses gradient-
    based Adam perturbation via ``torch.autograd`` through the SepsisModel as
    the best available alternative for multivariate time-series.  The approach
    minimises a convex combination of:
      (1) Binary cross-entropy between the model prediction and the target
          class, to drive the prediction toward the target; and
      (2) MSE between the perturbed TS and the factual TS, as a proximity
          regulariser.
    Multiple restarts with small random noise initialisation are used to
    encourage diversity among the returned candidates.
    """
    name = f"GRADIENT_TS_{ts_name}"
    print(f"\n[sepsis fold {fold}] Running {name} …")
    print(
        f"  ⚠  Approach: gradient-based Adam (torch.autograd) — no off-the-shelf "
        f"TS CF library was available. See methodology note in docstring."
    )
    started_at = datetime.datetime.now(datetime.timezone.utc)

    n_steps    = args.ts_steps
    lr         = args.ts_lr
    lam        = args.ts_lambda_prox
    target_t   = float(target_value)

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_ts_factual     = X_test_ts[sample_idx]      # (T, C)
        x_static_factual = X_test_static[sample_idx]  # (D_static,)

        x_static_t = torch.tensor(
            x_static_factual[np.newaxis, :], dtype=torch.float32, device=device
        )  # (1, D_static)
        x_ts_fac_t = torch.tensor(
            x_ts_factual, dtype=torch.float32, device=device
        )  # (T, C)
        target_tensor = torch.tensor([[target_t]], dtype=torch.float32, device=device)

        # predict_proba for objective evaluation (2-class softmax of sigmoid score)
        # Called after model.eval() is restored, so dropout/BN use eval behaviour.
        def predict_proba_ts(arr: np.ndarray) -> np.ndarray:
            # arr: (B, T, C)
            x = torch.tensor(arr, dtype=torch.float32).to(device)
            x_s = x_static_t.expand(x.size(0), -1)
            model.eval()
            with torch.no_grad():
                p = model(x, x_s).squeeze(1).cpu().numpy()
            return np.stack([1 - p, p], axis=1)

        candidates_this_sample: List[np.ndarray] = []
        converged_any = False

        # cuDNN's RNN implementation forbids backward in eval mode.
        # Switch to training mode only for gradient computation and restore
        # eval mode immediately after.  Dropout is inside the GRU cell and
        # does not affect other layers during the perturbation loop.
        # cuDNN's RNN forbids backward in eval mode, but train mode fails
        # BatchNorm with batch_size=1.  Solution: keep eval mode and disable
        # cuDNN so PyTorch's own RNN kernel is used, which supports grad in
        # eval mode.
        for restart in range(k):
            # Initialise from factual + small Gaussian noise
            noise_scale = 0.01 * (restart + 1)
            x_ts_var = (x_ts_fac_t + noise_scale * torch.randn_like(x_ts_fac_t)).detach()
            x_ts_var.requires_grad_(True)

            optimizer = torch.optim.Adam([x_ts_var], lr=lr)

            with torch.backends.cudnn.flags(enabled=False):
                for _step in range(n_steps):
                    optimizer.zero_grad()
                    pred = model(x_ts_var.unsqueeze(0), x_static_t)  # (1, 1)
                    loss_outcome = F.binary_cross_entropy(
                        pred.clamp(1e-7, 1 - 1e-7), target_tensor
                    )
                    loss_prox = F.mse_loss(x_ts_var, x_ts_fac_t)
                    loss = loss_outcome + lam * loss_prox
                    loss.backward()
                    optimizer.step()

            x_ts_cf = x_ts_var.detach().cpu().numpy()
            candidates_this_sample.append(x_ts_cf)

        for x_ts_cf in candidates_this_sample:
            with torch.no_grad():
                final_pred = float(model(
                    torch.tensor(x_ts_cf[np.newaxis, :, :],
                                 dtype=torch.float32, device=device),
                    x_static_t,
                ).item())
            if abs(final_pred - target_t) < 0.5:
                converged_any = True
                break  # at least one restart converged

        if converged_any:
            n_converged += 1
        candidate_counts[name] += len(candidates_this_sample)

        for x_ts_cf in candidates_this_sample:
            obj_accum.append(compute_ts_gradient_objectives(
                x_ts_cf, x_ts_factual, predict_proba_ts, target_value
            ))

        if (idx + 1) % 5 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="gradient_adam",
        modality=ts_name,
        held_fixed=["tabular"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "Gradient-based Adam perturbation via torch.autograd (no off-the-shelf "
            "TS CF library available). "
            f"Loss = BCE(pred, target) + {lam} * MSE(x_ts_cf, x_ts_factual). "
            f"k={k} restarts × {n_steps} steps, lr={lr}. "
            "Static tabular features are held fixed at factual values. "
            "Plausibility is NaN (no LOF fitted for TS space). "
            "Proximity = mean |Δx_ts| per element. "
            "Sparsity = fraction of channels with mean absolute change > 1e-6."
        ),
        extra_meta={
            "fold": fold,
            "ts_name": ts_name,
            "n_steps": n_steps,
            "lr": lr,
            "lambda_prox": lam,
        },
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_sepsis_native_guide(
    args, out_root, device, model,
    X_train_ts, y_train, X_test_static, X_test_ts,
    ts_name, sample_indices, target_value, label_classes, k, fold,
):
    """Native Guide counterfactual generation for the Sepsis TS branch.

    Methodology note
    ----------------
    From-scratch implementation following Delaney et al. (2021).  The original
    repo (e-delaney/Instance-Based_CFE_TSC) is notebook-only with no importable
    package.  Implements the NativeGuide-ISW (instance segmentwise) variant:
    find the nearest unlike neighbor via DTW, then greedily swap time-window
    segments from that NUN onto the factual TS until the prediction flips.
    Static tabular features are held fixed at factual values.
    """
    name = f"NATIVEGUIDE_TS_{ts_name}"
    print(f"\n[sepsis fold {fold}] Running {name} …")
    print(
        "  ⚠  From-scratch impl (Delaney et al. 2021): "
        "original repo has no importable package — NativeGuide-ISW via aeon DTW."
    )
    started_at = datetime.datetime.now(datetime.timezone.utc)

    n_segments = args.ts_n_segments

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_ts_factual     = X_test_ts[sample_idx]      # (T, C)
        x_static_factual = X_test_static[sample_idx]  # (D_static,)

        x_static_t = torch.tensor(
            x_static_factual[np.newaxis, :], dtype=torch.float32, device=device
        )

        def predict_proba_ts(arr: np.ndarray) -> np.ndarray:
            x = torch.tensor(arr, dtype=torch.float32).to(device)
            x_s = x_static_t.expand(x.size(0), -1)
            model.eval()
            with torch.no_grad():
                p = model(x, x_s).squeeze(1).cpu().numpy()
            return np.stack([1 - p, p], axis=1)

        cfs, converged = run_native_guide(
            X_train_ts=X_train_ts,
            y_train=y_train,
            x_ts_query=x_ts_factual,
            x_static_factual=x_static_factual,
            target_value=target_value,
            predict_proba_ts_fn=predict_proba_ts,
            k=k,
            n_segments=n_segments,
        )

        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)

        for x_ts_cf in cfs:
            obj_accum.append(compute_ts_gradient_objectives(
                x_ts_cf, x_ts_factual, predict_proba_ts, target_value
            ))

        if (idx + 1) % 5 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="native_guide_isw",
        modality=ts_name,
        held_fixed=["tabular"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "NativeGuide-ISW (Delaney et al. 2021) — from-scratch implementation. "
            "Nearest unlike neighbor retrieval via DTW (aeon), then greedy segment "
            f"swap ({n_segments} equal windows) until prediction flips. "
            "Static tabular features held fixed at factual values. "
            "Plausibility is NaN (no LOF for TS space)."
        ),
        extra_meta={
            "fold": fold,
            "ts_name": ts_name,
            "n_segments": n_segments,
            "distance_metric": "dtw",
        },
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


def _run_sepsis_comte(
    args, out_root, device, model,
    X_train_ts, y_train, X_test_static, X_test_ts,
    ts_name, sample_indices, target_value, label_classes, k, fold,
):
    """CoMTE-style counterfactual generation for the Sepsis TS branch.

    Methodology note
    ----------------
    CoMTE (Ates et al. 2021) requires a Rust-compiled feature extractor
    (fast_features) and a scikit-learn Pipeline — incompatible with the
    PyTorch Sepsis model without substantial re-engineering.  This implements
    CoMTE's greedy (channel, segment) swap logic — equivalent to CoMTE's
    BruteForceSearch fallback — without the mlrose hill-climbing layer and
    without statistical feature extraction:
      1. Find the nearest unlike neighbor via DTW.
      2. Rank (channel, time-window) pairs by absolute mean difference.
      3. Greedily copy pairs from NUN to factual until prediction flips.
    Static tabular features are held fixed at factual values.
    """
    name = f"COMTE_TS_{ts_name}"
    print(f"\n[sepsis fold {fold}] Running {name} …")
    print(
        "  ⚠  Greedy segment-swap impl (Ates et al. 2021): "
        "original CoMTE requires Rust fast_features + sklearn Pipeline — "
        "incompatible with PyTorch model.  Greedy (channel, segment) swap used."
    )
    started_at = datetime.datetime.now(datetime.timezone.utc)

    n_segments = args.ts_n_segments

    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_ts_factual     = X_test_ts[sample_idx]      # (T, C)
        x_static_factual = X_test_static[sample_idx]  # (D_static,)

        x_static_t = torch.tensor(
            x_static_factual[np.newaxis, :], dtype=torch.float32, device=device
        )

        def predict_proba_ts(arr: np.ndarray) -> np.ndarray:
            x = torch.tensor(arr, dtype=torch.float32).to(device)
            x_s = x_static_t.expand(x.size(0), -1)
            model.eval()
            with torch.no_grad():
                p = model(x, x_s).squeeze(1).cpu().numpy()
            return np.stack([1 - p, p], axis=1)

        cfs, converged = run_comte(
            X_train_ts=X_train_ts,
            y_train=y_train,
            x_ts_query=x_ts_factual,
            x_static_factual=x_static_factual,
            target_value=target_value,
            predict_proba_ts_fn=predict_proba_ts,
            k=k,
            n_segments=n_segments,
        )

        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)

        for x_ts_cf in cfs:
            obj_accum.append(compute_ts_gradient_objectives(
                x_ts_cf, x_ts_factual, predict_proba_ts, target_value
            ))

        if (idx + 1) % 5 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="comte_greedy_segment_swap",
        modality=ts_name,
        held_fixed=["tabular"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(
            "CoMTE greedy segment-swap (Ates et al. 2021) — from-scratch BruteForce "
            "variant. Nearest unlike neighbor via DTW (aeon), then greedy "
            f"(channel, segment) copy ({n_segments} windows × C channels) until "
            "prediction flips. No mlrose optimisation or statistical features. "
            "Static tabular features held fixed at factual values. "
            "Plausibility is NaN (no LOF for TS space)."
        ),
        extra_meta={
            "fold": fold,
            "ts_name": ts_name,
            "n_segments": n_segments,
            "distance_metric": "dtw",
        },
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


# ===========================================================================
# Gradient-vector runners  (tabular + embedding; Wachter-style Adam)
# ===========================================================================

def _run_long_covid_gradient_tab(
    args, out_root, device, model, tokenizer,
    X_train_static, y_train, X_test_static, X_test_text,
    tab_lof, sample_indices, target_value, label_classes, k,
):
    name = "GRADIENT_tabular"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    n_classes = len(label_classes)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        text_factual  = X_test_text[sample_idx]

        enc = tokenizer(text_factual, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            cls_fixed = model.text_encoder(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).last_hidden_state[:, 0]
            text_emb_fixed = model.text_proj(cls_fixed)  # (1, text_hidden_dim)

        def forward_fn(x):
            tab_emb = model.tab_head(x.unsqueeze(0))
            return model.classifier(
                torch.cat([text_emb_fixed, tab_emb], dim=1)
            ).squeeze(0)

        def predict_proba_tab(arr):
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb = model.tab_head(x_t)
                logits = model.classifier(
                    torch.cat([text_emb_fixed.expand(B, -1), tab_emb], dim=1)
                )
                return torch.softmax(logits, dim=1).cpu().numpy()

        cfs, converged = run_gradient_vector(
            x_factual=x_tab_factual, target_value=target_value,
            n_classes=n_classes, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_tab,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="gradient_adam", modality="tabular",
        held_fixed=["text_tweet"], embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(f"Wachter-style Adam gradient on tabular features; tweet held fixed. "
               f"Loss = CE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}."),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_long_covid_gradient_text(
    args, out_root, device, model,
    X_train_static, X_train_emb, y_train,
    X_test_static, X_test_emb,
    sample_indices, target_value, label_classes, k,
):
    name = "GRADIENT_text_tweet"
    print(f"\n[long_covid] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    n_classes = len(label_classes)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        x_tab_factual = X_test_static[sample_idx]

        x_tab_t = torch.tensor(x_tab_factual[np.newaxis, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            tab_emb_fixed = model.tab_head(x_tab_t)

        def forward_fn(x):
            text_emb = model.text_proj(x.unsqueeze(0))
            return model.classifier(
                torch.cat([text_emb, tab_emb_fixed], dim=1)
            ).squeeze(0)

        def predict_proba_emb(arr):
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                text_emb = model.text_proj(emb_t)
                logits = model.classifier(
                    torch.cat([text_emb, tab_emb_fixed.expand(B, -1)], dim=1)
                )
                return torch.softmax(logits, dim=1).cpu().numpy()

        cfs, converged = run_gradient_vector(
            x_factual=emb_factual, target_value=target_value,
            n_classes=n_classes, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_emb,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="long_covid_tweets",
        method="gradient_adam", modality="text_tweet",
        held_fixed=["tabular"], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=("Wachter-style Adam gradient on XLM-RoBERTa CLS embedding; "
               "tabular held fixed. Result is a perturbed embedding, NOT actual text. "
               f"Loss = CE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}."),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_jobs_gradient_tab(
    args, out_root, device, model,
    X_train_static, y_train, X_test_static,
    tab_lof, fixed_proj_embs,
    sample_indices, target_value, label_classes, k,
):
    name = "GRADIENT_tabular"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    n_classes = len(label_classes)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]

        d_fixed = torch.tensor(fixed_proj_embs["description"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)
        p_fixed = torch.tensor(fixed_proj_embs["company_profile"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)
        r_fixed = torch.tensor(fixed_proj_embs["requirements"][sample_idx][np.newaxis, :],
                               dtype=torch.float32).to(device)

        def forward_fn(x):
            tab_emb = model.tab_head(x.unsqueeze(0))
            return model.classifier(
                torch.cat([d_fixed, p_fixed, r_fixed, tab_emb], dim=1)
            ).squeeze(0)

        def predict_proba_tab(arr):
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = x_t.size(0)
            with torch.no_grad():
                tab_emb = model.tab_head(x_t)
                logits = model.classifier(torch.cat([
                    d_fixed.expand(B, -1), p_fixed.expand(B, -1),
                    r_fixed.expand(B, -1), tab_emb,
                ], dim=1))
                return torch.softmax(logits, dim=1).cpu().numpy()

        cfs, converged = run_gradient_vector(
            x_factual=x_tab_factual, target_value=target_value,
            n_classes=n_classes, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_tab,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="gradient_adam", modality="tabular",
        held_fixed=["text_description", "text_company_profile", "text_requirements"],
        embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(f"Wachter-style Adam gradient on tabular features; all text branches held fixed. "
               f"Loss = CE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}."),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_jobs_gradient_text(
    args, out_root, device, model, branch_name,
    X_train_emb, y_train, X_test_emb,
    tab_embs_test_all, fixed_proj_embs, proj_module,
    sample_indices, target_value, label_classes, k,
):
    name = f"GRADIENT_text_{branch_name}"
    print(f"\n[jobs] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    n_classes = len(label_classes)
    other_branches = [b for b in ("description", "company_profile", "requirements")
                      if b != branch_name]
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual   = X_test_emb[sample_idx]
        tab_emb_fixed = torch.tensor(
            tab_embs_test_all[sample_idx][np.newaxis, :], dtype=torch.float32
        ).to(device)
        other_embs = {
            b: torch.tensor(fixed_proj_embs[b][sample_idx][np.newaxis, :],
                            dtype=torch.float32).to(device)
            for b in other_branches
        }

        def forward_fn(x, _branch=branch_name):
            this_proj = proj_module(x.unsqueeze(0))
            parts = []
            for b in ("description", "company_profile", "requirements"):
                parts.append(this_proj if b == _branch else other_embs[b])
            parts.append(tab_emb_fixed)
            return model.classifier(torch.cat(parts, dim=1)).squeeze(0)

        def predict_proba_emb(arr, _branch=branch_name):
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_proj = proj_module(emb_t)
                parts = []
                for b in ("description", "company_profile", "requirements"):
                    parts.append(this_proj if b == _branch
                                 else other_embs[b].expand(B, -1))
                parts.append(tab_emb_fixed.expand(B, -1))
                logits = model.classifier(torch.cat(parts, dim=1))
                return torch.softmax(logits, dim=1).cpu().numpy()

        cfs, converged = run_gradient_vector(
            x_factual=emb_factual, target_value=target_value,
            n_classes=n_classes, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_emb,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    other_held = [f"text_{b}" for b in other_branches] + ["tabular"]
    payload = make_summary_payload(
        baseline_name=name, dataset="real_or_fake_jobs",
        method="gradient_adam", modality=f"text_{branch_name}",
        held_fixed=other_held, embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(f"Wachter-style Adam gradient on DistilBERT CLS embedding for '{branch_name}'; "
               "other text branches and tabular held fixed. "
               "Result is a perturbed embedding, NOT actual text. "
               f"Loss = CE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}."),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_memes_gradient_emb(
    args, out_root, device, model, modality_name,
    X_train_emb, y_train, X_test_emb,
    fixed_proj_embs_test,
    this_proj, is_text: bool,
    sample_indices, target_value, label_classes, k,
):
    mod_type = "text" if is_text else "image"
    name = f"GRADIENT_{mod_type}_{modality_name}"
    held = "image_meme_img" if is_text else "text_meme_text"
    print(f"\n[memes] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    n_classes = len(label_classes)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        emb_factual = X_test_emb[sample_idx]
        other_proj_fixed = torch.tensor(
            fixed_proj_embs_test[sample_idx][np.newaxis, :], dtype=torch.float32
        ).to(device)

        def forward_fn(x):
            this_emb = this_proj(x.unsqueeze(0))
            fused = (torch.cat([this_emb, other_proj_fixed], dim=1) if is_text
                     else torch.cat([other_proj_fixed, this_emb], dim=1))
            return model.classifier(fused).squeeze(0)

        def predict_proba_emb(arr):
            emb_t = torch.tensor(arr, dtype=torch.float32).to(device)
            B = emb_t.size(0)
            with torch.no_grad():
                this_emb = this_proj(emb_t)
                other_exp = other_proj_fixed.expand(B, -1)
                fused = (torch.cat([this_emb, other_exp], dim=1) if is_text
                         else torch.cat([other_exp, this_emb], dim=1))
                logits = model.classifier(fused)
                return torch.softmax(logits, dim=1).cpu().numpy()

        cfs, converged = run_gradient_vector(
            x_factual=emb_factual, target_value=target_value,
            n_classes=n_classes, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_emb,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for emb_cf in cfs:
            obj_accum.append(compute_embedding_objectives(
                emb_cf, emb_factual, predict_proba_emb, target_value
            ))

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset="hateful_memes",
        method="gradient_adam", modality=f"{mod_type}_{modality_name}",
        held_fixed=[held], embedding_space=True,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(f"Wachter-style Adam gradient on {mod_type} encoder embedding; "
               f"{held} held fixed. Result is a perturbed embedding, NOT actual "
               f"{'text or image' if mod_type == 'text' else 'image'}. "
               f"Loss = CE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}."),
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}  ({n_converged}/{len(sample_indices)} converged)")


def _run_sepsis_gradient_tab(
    args, out_root, device, model,
    X_train_static, y_train, X_test_static, X_test_ts,
    tab_lof, ts_name, sample_indices, target_value, k, fold,
):
    name = "GRADIENT_tabular"
    print(f"\n[sepsis fold {fold}] Running {name} …")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    obj_accum: List[np.ndarray] = []
    n_converged = 0
    candidate_counts = {name: 0}

    for idx, sample_idx in enumerate(sample_indices):
        x_tab_factual = X_test_static[sample_idx]
        x_ts_factual  = X_test_ts[sample_idx]

        x_ts_fixed = torch.tensor(
            x_ts_factual[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        # Sepsis model outputs sigmoid; keep eval+no-cuDNN for gradient too
        def forward_fn(x):
            with torch.backends.cudnn.flags(enabled=False):
                return model(x_ts_fixed, x.unsqueeze(0)).squeeze()

        def predict_proba_tab(arr):
            x_t = torch.tensor(arr, dtype=torch.float32).to(device)
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=False):
                    p = model(x_ts_fixed.expand(x_t.size(0), -1, -1),
                              x_t).squeeze(1).cpu().numpy()
            return np.stack([1 - p, p], axis=1)

        cfs, converged = run_gradient_vector(
            x_factual=x_tab_factual, target_value=target_value,
            n_classes=1, forward_fn=forward_fn,
            predict_proba_fn=predict_proba_tab,
            k=k, n_steps=args.ts_steps, lr=args.ts_lr,
            lambda_prox=args.ts_lambda_prox, device=device,
        )
        if converged:
            n_converged += 1
        candidate_counts[name] += len(cfs)
        for x_cf in cfs:
            obj_accum.append(compute_tabular_objectives(
                x_cf, x_tab_factual, tab_lof, predict_proba_tab, target_value
            ))

        if (idx + 1) % 5 == 0 or idx + 1 == len(sample_indices):
            print(f"  {idx+1}/{len(sample_indices)} samples  converged={n_converged}/{idx+1}")

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    payload = make_summary_payload(
        baseline_name=name, dataset=f"sepsis_fold_{fold}",
        method="gradient_adam", modality="tabular",
        held_fixed=[ts_name], embedding_space=False,
        sample_indices=sample_indices, target_value=target_value, k=k,
        objectives_by_generator={name: aggregate_objectives(obj_accum)},
        candidate_counts=candidate_counts,
        n_converged=n_converged, n_total=len(sample_indices),
        runtime_sec=(finished_at - started_at).total_seconds(),
        started_at=started_at, finished_at=finished_at,
        notes=(f"Wachter-style Adam gradient on tabular features; TS held fixed. "
               f"Loss = BCE(pred, target) + {args.ts_lambda_prox} * MSE. "
               f"k={k} restarts × {args.ts_steps} steps, lr={args.ts_lr}. "
               "cuDNN disabled for Sepsis GRU backward compatibility (eval mode)."),
        extra_meta={"fold": fold, "ts_name": ts_name},
    )
    path = write_summary(payload, out_root / name)
    print(f"  → {path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    args = _parse_args()
    fusion_strategy = getattr(args, "fusion_strategy", "intermediate")
    print(f"[run_baselines] dataset={args.dataset}  gpu={args.gpu}  k={args.k}"
          f"  fusion_strategy={fusion_strategy}")

    dispatch = {
        "long_covid_tweets": run_long_covid_baselines,
        "real_or_fake_jobs": run_jobs_baselines,
        "hateful_memes":     run_memes_baselines,
        "sepsis":            run_sepsis_baselines,
    }
    dispatch[args.dataset](args)
    print("\n[run_baselines] Done.")


if __name__ == "__main__":
    main()
