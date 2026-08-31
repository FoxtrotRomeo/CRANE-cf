"""
Check whether a distance ablation has been run for the best model in each
fusion strategy, and launch missing ablations.

Reads: data/fusion_model_registry.json  (written by evaluate_fusion_models.py)
Writes: data/ablation_runs/<ablation_run_name>/summary.json per strategy

For each fusion strategy:
  1. Read the registry to find the best model (highest test macro-F1).
  2. Check whether data/ablation_runs/<ablation_run_name>/summary.json exists.
  3. If absent, build the appropriate predict_fn for that model and run the
     full distance ablation (same encoders, metrics and generator set as
     run_cf_ablation.py, minus IntermediateFusion for non-IF models).

Run
---
    cd memes
    python run_cf_for_best_models.py --gpu 0
    python run_cf_for_best_models.py --gpu 0 --no-bert
    python run_cf_for_best_models.py --gpu 0 --dry-run   # check only
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make cf_lib and examples importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

from hateful_memes_cf_factory import build_hateful_memes_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import MultimodalConsensusRetrieval, EarlyFusionNN, ModalityWisePrototypeSynthesis, IntermediateFusionNN
from cf_lib.counterfactual_evaluation_helpers import (
    _make_embed_fn_from_e5_kwargs,
    fit_image_lof,
    fit_text_lof_reference,
)
from cf_lib.counterfactual_helpers import _apply_vision_encoder, _build_vision_encoder
from encoder_features import (
    DEFAULT_CACHE_PATH,
    FineTuneMultimodalClassifier,
    RawMultimodalDataset,
    build_text_backend,
    build_text_tokenizer,
    encode_word2vec_ids,
    get_image_transform,
    load_or_precompute_embeddings,
    load_word2vec_model,
    pool_text_features,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run missing CF ablations for the best model in each fusion strategy."
)
parser.add_argument("--gpu",          type=int,  default=None)
parser.add_argument("--k",            type=int,  default=20)
parser.add_argument("--max-samples",  type=int,  default=None)
parser.add_argument("--n-jobs",       type=int,  default=1)
parser.add_argument("--output-dir",   type=str,  default="data/ablation_runs")
parser.add_argument("--no-bert",      action="store_true")
parser.add_argument("--no-minilm",    action="store_true")
parser.add_argument("--no-word2vec",  action="store_true")
parser.add_argument("--no-efficientnet", action="store_true")
parser.add_argument("--no-vit",       action="store_true")
parser.add_argument("--word2vec-path", type=str,
                    default="../real_or_fake_jobs/data/word2vec_google_news_300.kv")
parser.add_argument("--image-batch-size", type=int, default=32)
parser.add_argument("--source-class", type=str, default="hateful")
parser.add_argument("--target-class", type=str, default="not_hateful")
parser.add_argument("--save-full",    action="store_true")
parser.add_argument("--dry-run",      action="store_true",
                    help="Print which ablations would be launched without running them.")
parser.add_argument("--strategies",   type=str, default=None,
                    help="Comma-separated subset of strategies to process.")
args = parser.parse_args()

DATA_DIR = Path("data")
DEVICE = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Load registry
# ---------------------------------------------------------------------------
REGISTRY_PATH = DATA_DIR / "fusion_model_registry.json"
if not REGISTRY_PATH.exists():
    raise FileNotFoundError(
        f"{REGISTRY_PATH} not found — run evaluate_fusion_models.py first."
    )
with open(REGISTRY_PATH) as fh:
    registry = json.load(fh)

label_classes = registry["label_classes"]
lc_lower      = [c.lower() for c in label_classes]
source_value  = lc_lower.index(args.source_class.lower())
target_value  = lc_lower.index(args.target_class.lower())

# ---------------------------------------------------------------------------
# Resolve strategies to process
# ---------------------------------------------------------------------------
all_strategies = ["intermediate", "early", "late"]
if args.strategies:
    all_strategies = [s.strip() for s in args.strategies.split(",")]

todo = []
for strategy in all_strategies:
    best_key = registry["best_per_strategy"].get(strategy)
    if best_key is None:
        print(f"[{strategy}] No available model in registry — skipping.")
        continue
    entry    = registry["models"][best_key]
    run_name = entry["ablation_run_name"] + "_k50"
    summary_path = Path(args.output_dir) / run_name / "summary.json"
    if summary_path.exists():
        print(f"[{strategy}] Ablation already exists: {summary_path} — skipping.")
        continue
    todo.append((strategy, best_key, entry, run_name))
    print(f"[{strategy}] Will run ablation '{run_name}'  "
          f"(model_type={entry['model_type']}  F1={entry.get('test_macro_f1', 'N/A'):.4f})")

if not todo:
    print("\nAll ablations are up to date.")
    sys.exit(0)

if args.dry_run:
    print("\nDry run — exiting without launching ablations.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Shared setup: load dataset and embedding cache
# ---------------------------------------------------------------------------
print("\nLoading dataset and text backend …")
produced     = build_hateful_memes_dataset(gpu=args.gpu, load_bert=not args.no_bert)
dataset      = produced["dataset"]
text_bk      = produced["text_backend_kwargs"]
y_pred_orig  = produced["y_pred"]

train_texts  = ["" if t is None else str(t) for t in dataset.X_train_text]
test_texts   = ["" if t is None else str(t) for t in dataset.X_test_text]
train_images = dataset.X_train_img
test_images  = dataset.X_test_img

# Source indices (samples to explain)
if y_pred_orig is not None:
    source_indices = [int(i) for i, p in enumerate(y_pred_orig) if int(p) == source_value]
else:
    source_indices = [int(i) for i in range(len(dataset.y_test))
                      if int(dataset.y_test[i]) == source_value]
if args.max_samples is not None:
    source_indices = source_indices[:args.max_samples]
print(f"Source indices: {len(source_indices)} samples predicted as '{args.source_class}'")

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
print("Fitting TF-IDF on meme texts …")
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(train_texts)

def _tfidf_embed_fn(texts):
    return _tfidf_vec.transform(
        ["" if t is None else str(t) for t in texts]
    ).toarray().astype(np.float32)

# ---------------------------------------------------------------------------
# Text / image encoder setup
# ---------------------------------------------------------------------------
text_encoders  = ["tfidf", "raw"]
image_encoders = ["resnet50"]
if not args.no_word2vec:
    text_encoders.insert(0, "word2vec")
if not args.no_bert:
    text_encoders.insert(0, "bert")
if not args.no_minilm:
    text_encoders.insert(0, "minilm")
if not args.no_efficientnet:
    image_encoders.append("efficientnet_b0")
if not args.no_vit:
    image_encoders.append("vit_b_16")

text_vector_metrics  = ["cosine", "euclidean", "manhattan"]
text_direct_metrics  = ["rouge_l", "lcs"]
image_distance_metrics = ["cosine", "euclidean"]

# ---------------------------------------------------------------------------
# Precompute embeddings
# ---------------------------------------------------------------------------
print("\nPreparing cached text/image embeddings …")
embedding_cache = load_or_precompute_embeddings(
    X_train_text=train_texts,
    X_test_text=test_texts,
    X_train_img=train_images,
    X_test_img=test_images,
    device=DEVICE,
    batch_size=args.image_batch_size,
    max_len=128,
    num_workers=2,
    cache_path=DEFAULT_CACHE_PATH,
    word2vec_path=None if args.no_word2vec else args.word2vec_path,
)

def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

_precomputed_text_by_enc: Dict[str, Dict[str, np.ndarray]] = {
    "tfidf": {"train": _tfidf_embed_fn(train_texts), "test": _tfidf_embed_fn(test_texts)}
}
_precomputed_image_by_enc: Dict[str, Dict[str, np.ndarray]] = {}

for enc in ("word2vec", "bert", "minilm"):
    if enc in text_encoders and enc in embedding_cache["text"]:
        d = embedding_cache["text"][enc]
        _precomputed_text_by_enc[enc] = {
            "train": _to_numpy(d["train"]).astype(np.float32),
            "test":  _to_numpy(d["test"]).astype(np.float32),
        }
for enc in image_encoders:
    if enc in embedding_cache["image"]:
        d = embedding_cache["image"][enc]
        _precomputed_image_by_enc[enc] = {
            "train": _to_numpy(d["train"]).astype(np.float32),
            "test":  _to_numpy(d["test"]).astype(np.float32),
        }

# embed_fn objects
_text_embed_fns: Dict[str, object] = {"tfidf": _tfidf_embed_fn, "raw": _tfidf_embed_fn}

if not args.no_bert and "bert" in text_bk:
    _text_embed_fns["bert"] = _make_embed_fn_from_e5_kwargs(
        tokenizer=text_bk["bert_tokenizer"],
        model=text_bk["bert_model"],
        device=text_bk["bert_device"],
    )

if not args.no_word2vec and Path(args.word2vec_path).exists():
    def _tokenize(t):
        return re.findall(r"[A-Za-z0-9']+", str(t).lower())
    _w2v_kv = load_word2vec_model(args.word2vec_path)
    _w2v_wv  = getattr(_w2v_kv, "wv", _w2v_kv)
    def _w2v_embed(texts, _wv=_w2v_wv):
        out = np.zeros((len(texts), int(_wv.vector_size)), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs = [np.asarray(_wv[tok], dtype=np.float32)
                    for tok in _tokenize(t) if tok in _wv]
            if vecs:
                out[i] = np.mean(np.vstack(vecs), axis=0)
        return out
    _text_embed_fns["word2vec"] = _w2v_embed
    text_bk["word2vec_model"] = _w2v_kv

if not args.no_minilm:
    _tok_ml, _model_ml, _cfg_ml = build_text_backend("minilm", DEVICE)
    def _minilm_embed(texts, _tok=_tok_ml, _m=_model_ml, _cfg=_cfg_ml):
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(texts), 32):
                batch = ["" if t is None else str(t) for t in texts[s:s+32]]
                enc = _tok(batch, max_length=128, padding=True,
                            truncation=True, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out = _m(**enc).last_hidden_state
                pooled = pool_text_features(out, enc["attention_mask"],
                                             pooling=_cfg["pooling"],
                                             normalize=_cfg["normalize"])
                rows.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0) if rows else np.empty((0, _cfg["feature_dim"]))
    _text_embed_fns["minilm"] = _minilm_embed

_image_embed_fns: Dict[str, object] = {
    enc: (lambda imgs, _enc=enc: _apply_vision_encoder(
        imgs, _build_vision_encoder(_enc, device=DEVICE), batch_size=args.image_batch_size
    ).astype(np.float32))
    for enc in image_encoders
}

# Text lookup: string → embedding vector (per encoder)
_text_lookup: Dict[str, Dict[str, np.ndarray]] = {}
for enc, splits in _precomputed_text_by_enc.items():
    lk = {}
    lk.update({str(t): v for t, v in zip(train_texts, splits["train"])})
    lk.update({str(t): v for t, v in zip(test_texts,  splits["test"])})
    _text_lookup[enc] = lk

# Set in text_bk for the runner
text_bk["precomputed_text_embeddings_by_encoder"] = _precomputed_text_by_enc

image_bk = {"device": DEVICE, "batch_size": args.image_batch_size}


def _make_text_lookup_embed_fn(encoder: str):
    enc    = "tfidf" if encoder == "raw" else encoder
    lookup = _text_lookup.get(enc, {})
    live   = _text_embed_fns.get(enc, _tfidf_embed_fn)
    def _fn(texts):
        vecs, miss_pos, miss_txt = [None]*len(texts), [], []
        for i, t in enumerate(texts):
            v = lookup.get("" if t is None else str(t))
            if v is not None: vecs[i] = v
            else: miss_pos.append(i); miss_txt.append("" if t is None else str(t))
        if miss_txt:
            embs = np.asarray(live(miss_txt), dtype=np.float32)
            for j, pos in enumerate(miss_pos): vecs[pos] = embs[j]
        return np.asarray(vecs, dtype=np.float32)
    return _fn


def _make_image_lookup_embed_fn(encoder: str):
    tr_emb = _precomputed_image_by_enc[encoder]["train"]
    te_emb = _precomputed_image_by_enc[encoder]["test"]
    live   = _image_embed_fns[encoder]
    def _fn(items):
        vecs, miss_pos, miss_items = [None]*len(items), [], []
        for i, item in enumerate(items):
            if isinstance(item, tuple) and len(item) == 2 and item[0] in {"train","test"}:
                split, idx = item
                vecs[i] = tr_emb[int(idx)] if split == "train" else te_emb[int(idx)]
            else:
                miss_pos.append(i); miss_items.append(item)
        if miss_items:
            embs = np.asarray(live(miss_items), dtype=np.float32)
            for j, pos in enumerate(miss_pos): vecs[pos] = embs[j]
        return np.asarray(vecs, dtype=np.float32)
    return _fn


# ---------------------------------------------------------------------------
# LOF references (per encoder)
# ---------------------------------------------------------------------------
print("Fitting text/image LOF references …")
_text_obj_ctx: Dict[str, dict] = {}
for enc, splits in _precomputed_text_by_enc.items():
    embed_fn = _make_text_lookup_embed_fn(enc)
    lof_ref  = fit_text_lof_reference(
        train_texts, y_train=dataset.y_train,
        target_value=target_value, embed_fn=embed_fn,
    )
    _text_obj_ctx[enc] = {"embed_fn": embed_fn, **lof_ref}

_image_obj_ctx: Dict[str, dict] = {}
for enc in image_encoders:
    embed_fn    = _make_image_lookup_embed_fn(enc)
    train_tokens = [("train", i) for i in range(len(dataset.y_train))]
    lof_ref = fit_image_lof(
        train_tokens, y_train=dataset.y_train,
        target_value=target_value, embed_fn=embed_fn,
    )
    _image_obj_ctx[enc] = {"embed_fn": embed_fn, **lof_ref}


# ---------------------------------------------------------------------------
# Helper: resolve text/image from a candidate dict or raw value
# ---------------------------------------------------------------------------
def _resolve_text(text_or_dict) -> str:
    if isinstance(text_or_dict, dict):
        return str(text_or_dict.get(dataset.primary_text_name, "") or "")
    return "" if text_or_dict is None else str(text_or_dict)


def _resolve_image_token(image_or_dict):
    if isinstance(image_or_dict, dict):
        image_or_dict = image_or_dict.get(dataset.primary_image_name)
    if isinstance(image_or_dict, tuple) and len(image_or_dict) == 2:
        split, idx = image_or_dict
        if split in {"train", "test"}:
            return (split, int(idx)), (train_images[int(idx)] if split == "train"
                                       else test_images[int(idx)])
    return None, image_or_dict


# ---------------------------------------------------------------------------
# Precomputed embedding getter for any encoder
# ---------------------------------------------------------------------------
def _get_text_emb(encoder: str, text: str) -> np.ndarray:
    enc = "tfidf" if encoder == "raw" else encoder
    v   = _text_lookup.get(enc, {}).get(str(text))
    if v is not None:
        return v
    return np.asarray(_text_embed_fns.get(enc, _tfidf_embed_fn)([text]),
                       dtype=np.float32)[0]


def _get_image_emb(encoder: str, token) -> np.ndarray:
    pre = _precomputed_image_by_enc.get(encoder)
    if pre is not None and isinstance(token, tuple) and token[0] in {"train","test"}:
        split, idx = token
        return pre["train"][int(idx)] if split == "train" else pre["test"][int(idx)]
    # fallback: live encode
    raw_img = (train_images[token[1]] if isinstance(token, tuple) and token[0]=="train"
               else test_images[token[1]] if isinstance(token, tuple)
               else token)
    return _image_embed_fns[encoder]([raw_img]).astype(np.float32)[0]


# ===========================================================================
# Predict-fn builders (one per model_type)
# ===========================================================================

class _EarlyFusionMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x): return self.net(x)


class _BranchMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x): return self.net(x)


N_CLASSES = len(label_classes)


def _build_predict_fn_intermediate(entry: dict):
    """predict_fn backed by the FineTuneMultimodalClassifier (best_model.pt)."""
    model_path = Path(entry["model_files"]["main"])
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    ckpt     = torch.load(model_path, map_location=DEVICE)
    cfg      = dict(ckpt["config"])
    t_enc    = cfg["text_encoder"]
    i_enc    = cfg["image_encoder"]
    w2v_path = cfg.get("word2vec_path") or args.word2vec_path

    best_model = FineTuneMultimodalClassifier(
        text_encoder_name=t_enc, image_encoder_name=i_enc,
        n_classes=len(ckpt["label_classes"]),
        text_hidden_dim=cfg["text_hidden_dim"],
        img_hidden_dim=cfg["img_hidden_dim"],
        dropout=cfg["dropout"],
        device=DEVICE,
        word2vec_path=w2v_path,
        word2vec_vocab_tokens=cfg.get("word2vec_vocab_tokens"),
    ).to(DEVICE)
    best_model.load_state_dict(ckpt["state_dict"])
    best_model.eval()

    best_text_backend   = build_text_tokenizer(t_enc, word2vec_path=w2v_path,
                                               vocab_tokens=cfg.get("word2vec_vocab_tokens"))
    best_image_transform = get_image_transform(i_enc)
    max_len = cfg.get("max_len", 128)
    lock    = threading.Lock()

    # Extract latents and build prediction cache from train set
    print("  Extracting model latents for intermediate fusion …")
    buf: Dict[str, torch.Tensor] = {}

    def _hook(m, inp, out): buf["z"] = inp[0].detach().cpu()
    handle = best_model.classifier.register_forward_hook(_hook)

    def _run_latents_preds(images, texts, labels):
        ds = RawMultimodalDataset(
            images=images, texts=[str(t) for t in texts],
            labels=torch.tensor(labels, dtype=torch.long),
            text_encoder_name=t_enc, text_backend=best_text_backend,
            image_transform=best_image_transform, max_len=max_len,
        )
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        lats, preds = [], []
        with torch.no_grad():
            for batch in dl:
                logits = best_model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["image"].to(DEVICE),
                )
                lats.append(buf["z"].clone().numpy().astype(np.float32))
                preds.append(logits.argmax(-1).cpu().numpy())
        return np.vstack(lats), np.concatenate(preds)

    train_latents, train_pred_labels = _run_latents_preds(
        train_images, train_texts, dataset.y_train
    )
    test_latents, _ = _run_latents_preds(
        test_images, test_texts, dataset.y_test
    )
    handle.remove()

    _best_train_texts = np.asarray(train_texts, dtype=object)
    _best_test_texts  = np.asarray(test_texts,  dtype=object)

    def _predict_single_live(text: str, raw_image) -> float:
        if t_enc == "word2vec":
            enc_out = encode_word2vec_ids(text, best_text_backend, max_len)
            input_ids     = enc_out["input_ids"].unsqueeze(0)
            attention_mask = enc_out["attention_mask"].unsqueeze(0)
        else:
            enc_out = best_text_backend(
                text, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids     = enc_out["input_ids"]
            attention_mask = enc_out["attention_mask"]
        img_t = best_image_transform(
            Image.fromarray(np.asarray(raw_image, dtype=np.uint8))
        ).unsqueeze(0)
        with lock:
            with torch.no_grad():
                logits = best_model(input_ids.to(DEVICE),
                                    attention_mask.to(DEVICE),
                                    img_t.to(DEVICE))
        return float(logits.argmax(-1).cpu().item())

    def _predict_fn(x_tab_unused, x_ts_unused, text_or_dict, image_or_dict):
        text  = _resolve_text(text_or_dict)
        token, raw_img = _resolve_image_token(image_or_dict)
        if token is not None:
            split, idx = token
            if split == "train" and idx < len(_best_train_texts):
                if text == str(_best_train_texts[idx]):
                    return float(train_pred_labels[idx])
            if split == "test" and idx < len(_best_test_texts) and y_pred_orig is not None:
                if text == str(_best_test_texts[idx]):
                    return float(y_pred_orig[idx])
        return _predict_single_live(text, raw_img if raw_img is not None else
                                    (train_images[token[1]] if token and token[0]=="train"
                                     else test_images[token[1]] if token else train_images[0]))

    return _predict_fn, train_latents, test_latents


def _build_predict_fn_early(entry: dict):
    """predict_fn for pytorch_early_fusion and sklearn_early_fusion."""
    model_path = Path(entry["model_files"]["main"])
    model_type = entry["model_type"]
    t_enc      = entry.get("text_encoder", "bert")
    i_enc      = entry.get("image_encoder", "resnet50")

    if model_type == "pytorch_early_fusion":
        ckpt = torch.load(model_path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        t_enc = cfg.get("text_encoder", t_enc)
        i_enc = cfg.get("image_encoder", i_enc)
        model = _EarlyFusionMLP(ckpt["d_in"], N_CLASSES, hidden=256, dropout=0.3)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        def _run(ef):
            with torch.no_grad():
                return int(model(torch.tensor(ef, dtype=torch.float32)).argmax(1).item())
    else:
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        cfg   = bundle.get("config", {})
        t_enc = cfg.get("text_encoder", t_enc)
        i_enc = cfg.get("image_encoder", i_enc)
        sk_model = bundle.get("model", bundle)
        def _run(ef):
            return int(sk_model.predict(ef)[0])

    def _predict_fn(x_tab_unused, x_ts_unused, text_or_dict, image_or_dict):
        text  = _resolve_text(text_or_dict)
        token, _ = _resolve_image_token(image_or_dict)
        t_emb = _get_text_emb(t_enc, text)
        i_emb = _get_image_emb(i_enc, token if token else ("train", 0))
        ef    = np.concatenate([t_emb, i_emb]).reshape(1, -1)
        return float(_run(ef))

    return _predict_fn, None, None


def _build_predict_fn_late_deep(entry: dict):
    """predict_fn for late_fusion_deep (text branch MLP + image branch MLP)."""
    txt_path = Path(entry["model_files"]["text"])
    img_path = Path(entry["model_files"]["image"])

    ckpt_t = torch.load(txt_path, map_location="cpu")
    ckpt_i = torch.load(img_path, map_location="cpu")
    t_enc  = ckpt_t["text_encoder"]
    i_enc  = ckpt_i["image_encoder"]
    d_t    = ckpt_t["d_in"]
    d_i    = ckpt_i["d_in"]

    txt_mlp = _BranchMLP(d_t, N_CLASSES, 128)
    txt_mlp.load_state_dict(ckpt_t["state_dict"])
    txt_mlp.eval()
    img_mlp = _BranchMLP(d_i, N_CLASSES, 128)
    img_mlp.load_state_dict(ckpt_i["state_dict"])
    img_mlp.eval()

    def _predict_fn(x_tab_unused, x_ts_unused, text_or_dict, image_or_dict):
        text  = _resolve_text(text_or_dict)
        token, _ = _resolve_image_token(image_or_dict)
        t_emb = _get_text_emb(t_enc, text)
        i_emb = _get_image_emb(i_enc, token if token else ("train", 0))
        with torch.no_grad():
            p_t = torch.softmax(
                txt_mlp(torch.tensor(t_emb).unsqueeze(0)), dim=1
            ).numpy()[0]
            p_i = torch.softmax(
                img_mlp(torch.tensor(i_emb).unsqueeze(0)), dim=1
            ).numpy()[0]
        return float((0.5 * p_t + 0.5 * p_i).argmax())

    return _predict_fn, None, None


def _build_predict_fn_late_nondp(entry: dict):
    """predict_fn for late_fusion_nondp (sklearn text + sklearn image)."""
    txt_path = Path(entry["model_files"]["text"])
    img_path = Path(entry["model_files"]["image"])

    with open(txt_path, "rb") as fh: bndl_t = pickle.load(fh)
    with open(img_path, "rb") as fh: bndl_i = pickle.load(fh)
    t_enc     = bndl_t["encoder"]
    i_enc     = bndl_i["encoder"]
    txt_model = bndl_t["model"]
    img_model = bndl_i["model"]

    def _predict_fn(x_tab_unused, x_ts_unused, text_or_dict, image_or_dict):
        text  = _resolve_text(text_or_dict)
        token, _ = _resolve_image_token(image_or_dict)
        t_emb = _get_text_emb(t_enc, text).reshape(1, -1)
        i_emb = _get_image_emb(i_enc, token if token else ("train", 0)).reshape(1, -1)
        p_t = txt_model.predict_proba(t_emb)[0]
        p_i = img_model.predict_proba(i_emb)[0]
        return float((0.5 * p_t + 0.5 * p_i).argmax())

    return _predict_fn, None, None


_PREDICT_FN_BUILDERS = {
    "pytorch_intermediate":  _build_predict_fn_intermediate,
    "pytorch_early_fusion":  _build_predict_fn_early,
    "sklearn_early_fusion":  _build_predict_fn_early,
    "late_fusion_deep":      _build_predict_fn_late_deep,
    "late_fusion_nondp":     _build_predict_fn_late_nondp,
}

# ===========================================================================
# Generator factory (text + image only, no tabular)
# ===========================================================================
def _fusion_metric(text_encoder, text_metric, image_metric) -> str:
    if (text_encoder != "raw" and text_metric == image_metric
            and text_metric in {"cosine", "euclidean", "manhattan"}):
        return text_metric
    return "cosine"


def _make_generators_factory(k, k_search, include_if, train_latents, test_latents):
    def _factory(tab_cfg, ts_cfg, text_cfg, text_backend_kwargs,
                 image_cfg, image_backend_kwargs):
        if text_cfg is None or image_cfg is None:
            return {}
        t_enc       = (text_cfg or {}).get("encoder", "raw")
        t_metric    = (text_cfg or {}).get("metric", "cosine")
        i_enc       = (image_cfg or {}).get("encoder", image_encoders[0])
        i_metric    = (image_cfg or {}).get("metric", "cosine")
        fusion_m    = _fusion_metric(t_enc, t_metric, i_metric)
        text_efn    = _make_text_lookup_embed_fn(t_enc)
        t_pre       = _precomputed_text_by_enc.get("tfidf" if t_enc == "raw" else t_enc,
                                                    _precomputed_text_by_enc["tfidf"])
        i_pre       = _precomputed_image_by_enc.get(i_enc, {})

        gens = {
            "MPS": ModalityWisePrototypeSynthesis(
                k=k, k_search=k_search,
                e5_embed_fn=text_efn,
                image_encoder=i_enc, img_distance_metric=i_metric,
                img_device=DEVICE, img_batch_size=args.image_batch_size,
            ),
            "MC-R": MultimodalConsensusRetrieval(
                k=k, k_search=k_search,
                e5_embed_fn=text_efn,
                image_encoder=i_enc, img_distance_metric=i_metric,
                img_device=DEVICE, img_batch_size=args.image_batch_size,
            ),
        }
        if i_pre:
            gens["EarlyFusion"] = EarlyFusionNN(
                k=k, distance_metric=fusion_m,
                precomputed_train_text_embeddings=t_pre["train"],
                precomputed_test_text_embeddings=t_pre["test"],
                precomputed_train_image_embeddings_by_name={
                    dataset.primary_image_name: i_pre["train"],
                },
                precomputed_test_image_embeddings_by_name={
                    dataset.primary_image_name: i_pre["test"],
                },
            )
        if include_if and train_latents is not None:
            gens["IntermediateFusion"] = IntermediateFusionNN(
                k=k, distance_metric=fusion_m,
                precomputed_train_latent=train_latents,
                precomputed_test_latent=test_latents,
            )
        return gens
    return _factory


def _candidate_image_token(cand) -> object:
    src_idx = (cand.get("source_indices") or {}).get("image", {})
    if isinstance(src_idx, dict):
        src = src_idx.get(dataset.primary_image_name)
        if src is not None:
            return ("train", int(src))
    src = cand.get("source_train_idx")
    if src is not None:
        return ("train", int(src))
    return cand.get("image") or cand.get("image_input")


def _make_objectives_factory(predict_fn_ref):
    def _factory(text_cfg, image_cfg):
        t_enc      = (text_cfg or {}).get("encoder", "raw")
        i_enc      = (image_cfg or {}).get("encoder", image_encoders[0])
        obj_t_enc  = "tfidf" if t_enc == "raw" else t_enc
        txt_ctx    = _text_obj_ctx.get(obj_t_enc, _text_obj_ctx["tfidf"])
        img_ctx    = _image_obj_ctx.get(i_enc, next(iter(_image_obj_ctx.values())))

        _Xtest_text_arr = np.asarray(test_texts, dtype=object)

        def _image_modalities_fn(sample_idx, cand):
            return {"image_modalities": {
                dataset.primary_image_name: {
                    "candidate": _candidate_image_token(cand),
                    "factual":   ("test", int(sample_idx)),
                    "context":   img_ctx,
                }
            }}

        return {
            "y_target":             target_value,
            "text_objective_context": txt_ctx,
            "_image_modalities_fn": _image_modalities_fn,
            "predict_fn":           predict_fn_ref[0],
        }
    return _factory


# ===========================================================================
# Main loop — run missing ablations
# ===========================================================================
for strategy, best_key, entry, run_name in todo:
    model_type = entry["model_type"]
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}  |  model: {best_key}  |  run: {run_name}")
    print(f"{'='*60}")

    builder = _PREDICT_FN_BUILDERS.get(model_type)
    if builder is None:
        print(f"  [warn] No predict_fn builder for model_type={model_type} — skipping.")
        continue

    predict_fn, train_latents, test_latents = builder(entry)
    include_if = model_type == "pytorch_intermediate"
    k_search   = min(50, args.k * 5)

    predict_fn_ref = [predict_fn]
    gens_factory   = _make_generators_factory(
        args.k, k_search, include_if, train_latents, test_latents
    )
    obj_factory    = _make_objectives_factory(predict_fn_ref)

    run_distance_ablation(
        dataset=dataset,
        model=None,
        sample_indices=source_indices,
        target_value=target_value,
        k=args.k,
        tab_metrics=[],
        ts_metrics=[],
        text_encoders=text_encoders,
        text_vector_metrics=text_vector_metrics,
        text_direct_metrics=text_direct_metrics,
        text_backend_kwargs=text_bk,
        image_encoders=image_encoders,
        image_distance_metrics=image_distance_metrics,
        image_backend_kwargs=image_bk,
        output_dir=args.output_dir,
        run_name=run_name,
        save_full=args.save_full,
        max_combinations=None,
        n_jobs=args.n_jobs,
        objectives_kwargs_factory=obj_factory,
        extra_generators_factory=gens_factory,
    )
    print(f"  Done — results in {Path(args.output_dir) / run_name}/summary.json")
