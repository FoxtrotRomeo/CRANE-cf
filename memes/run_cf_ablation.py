"""Counterfactual ablation: hateful -> not_hateful on Hateful Memes.

Finds all test samples predicted as "hateful", then sweeps text/image encoder
and metric combinations via cf_lib's run_distance_ablation to generate
counterfactuals that flip each sample to "not_hateful".

The dataset is text + image only: there is no static/tabular modality.

Generators
----------
- TextNN on meme text
- ImageNN on meme image
- FrankensteinNN (independent text/image anchors)
- CombinedNN (intersection of text/image neighbor sets)
- EarlyFusionNN (concatenated text+image embedding space)
- IntermediateFusionNN (best-model latent space from the trained classifier)

Run
---
    cd memes
    python run_cf_ablation.py --gpu 0

    # Smoke test
    python run_cf_ablation.py --gpu 0 --max-samples 3 --max-combinations 6
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hateful_memes_cf_factory import build_hateful_memes_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.multimodal import CombinedNN, EarlyFusionNN, FrankensteinNN, IntermediateFusionNN
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


parser = argparse.ArgumentParser(
    description="Hateful->Not-hateful counterfactual ablation on memes."
)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--k", type=int, default=50,
                    help="Number of nearest neighbours per generator.")
parser.add_argument("--max-samples", type=int, default=None,
                    help="Cap on how many predicted hateful samples to process.")
parser.add_argument("--max-combinations", type=int, default=None,
                    help="Cap on how many metric combinations to try.")
parser.add_argument("--n-jobs", type=int, default=1,
                    help="Number of ablation combinations to run in parallel.")
parser.add_argument("--output-dir", type=str, default="data/ablation_runs")
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--k-search-combined", type=int, default=300,
                    help="k_search pool size for FrankensteinNN and CombinedNN (default: 300).")
parser.add_argument("--fusion-strategy", type=str, default="intermediate",
                    choices=["intermediate", "early", "late"],
                    help="Which fusion model to use for predict_fn / latents. "
                         "intermediate=best_model.pt (default); "
                         "early=best_model_early_fusion_gbt.pkl (minilm+resnet50); "
                         "late=best_model_late_fusion_gbt_{text,image}.pkl averaged.")
parser.add_argument("--save-full", action="store_true", default=True,
                    help="Save full candidate dicts as .pkl per combination.")
parser.add_argument(
    "--word2vec-path",
    type=str,
    default="../real_or_fake_jobs/data/word2vec_google_news_300.kv",
    help="Path to a gensim word2vec .kv/.bin file.",
)
parser.add_argument("--source-class", type=str, default="hateful",
                    help="Class to explain (default: hateful).")
parser.add_argument("--target-class", type=str, default="not_hateful",
                    help="Counterfactual target class (default: not_hateful).")
parser.add_argument("--no-bert", action="store_true",
                    help="Skip the BERT text encoder.")
parser.add_argument("--no-minilm", action="store_true",
                    help="Skip the MiniLM text encoder.")
parser.add_argument("--no-word2vec", action="store_true",
                    help="Skip the word2vec text encoder.")
parser.add_argument("--no-efficientnet", action="store_true",
                    help="Skip the EfficientNet-B0 image encoder.")
parser.add_argument("--no-vit", action="store_true",
                    help="Skip the ViT-B/16 image encoder.")
parser.add_argument("--image-batch-size", type=int, default=32,
                    help="Batch size for live image encoding when needed.")
parser.add_argument(
    "--eval-pkls",
    type=str,
    default=None,
    metavar="DIR[,DIR,...]",
    help=(
        "Instead of generating new counterfactuals, re-evaluate saved "
        "combo_*_results.pkl files at multiple k values. "
        "Comma-separated list of run directories (each containing pkls + "
        "summary.jsonl). Results are written to <first-dir>/k_ablation_summary.json."
    ),
)
parser.add_argument(
    "--k-values",
    type=str,
    default="1,5,10,20,50",
    help="Comma-separated k values for --eval-pkls mode (default: 1,5,10,20,50).",
)
args = parser.parse_args()


if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {args.gpu} requested but only {torch.cuda.device_count()} available."
        )
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent / "data"
BEST_MODEL_PATH = DATA_DIR / "best_model.pt"


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def _build_minilm_embed_fn(device: str):
    tokenizer, model, cfg = build_text_backend("minilm", device)

    def _embed(texts):
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = ["" if t is None else str(t) for t in texts[start:start + 32]]
                enc = tokenizer(
                    batch,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc).last_hidden_state
                pooled = pool_text_features(
                    outputs,
                    enc["attention_mask"],
                    pooling=cfg["pooling"],
                    normalize=cfg["normalize"],
                )
                rows.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0) if rows else np.empty((0, cfg["feature_dim"]), dtype=np.float32)

    return _embed


def _build_word2vec_embed_fn(word2vec_path: str):
    kv = load_word2vec_model(word2vec_path)
    wv = getattr(kv, "wv", kv)

    def _embed(texts):
        out = np.zeros((len(texts), int(wv.vector_size)), dtype=np.float32)
        for i, text in enumerate(texts):
            vecs = [
                np.asarray(wv[token], dtype=np.float32)
                for token in _tokenize_text(text)
                if token in wv
            ]
            if vecs:
                out[i] = np.mean(np.vstack(vecs), axis=0)
        return out

    return kv, _embed


def _build_live_image_embed_fn(encoder_name: str, device: str, batch_size: int):
    encode_fn = _build_vision_encoder(encoder_name, device=device)

    def _embed(images):
        return _apply_vision_encoder(images, encode_fn, batch_size=batch_size).astype(np.float32)

    return _embed


def _read_best_row(csv_path: Path) -> Dict[str, str]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"No rows found in {csv_path}")
    return max(rows, key=lambda row: float(row["val_f1"]))


def _load_checkpoint_config(best_model_path: Path) -> Dict[str, object]:
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model.pt not found at {best_model_path}")
    ckpt = torch.load(best_model_path, map_location="cpu")
    return {
        "checkpoint": ckpt,
        "config": ckpt["config"],
        "label_classes": ckpt["label_classes"],
    }


produced = build_hateful_memes_dataset(gpu=args.gpu, load_bert=not args.no_bert,
                                       fusion_strategy=args.fusion_strategy)
dataset = produced["dataset"]
text_bk = produced["text_backend_kwargs"]
image_bk = produced["image_backend_kwargs"]
label_classes = produced["label_classes"]
y_pred = produced["y_pred"]

if y_pred is None:
    raise FileNotFoundError("data/y_pred.npy not found - run evaluate.py first.")

train_texts = ["" if t is None else str(t) for t in dataset.X_train_text]
test_texts = ["" if t is None else str(t) for t in dataset.X_test_text]
train_images = dataset.X_train_img
test_images = dataset.X_test_img

lc_lower = [c.lower() for c in label_classes]
source_class = args.source_class.lower()
target_class = args.target_class.lower()
if source_class not in lc_lower:
    raise ValueError(f"'{source_class}' not found in label classes: {label_classes}.")
if target_class not in lc_lower:
    raise ValueError(f"'{target_class}' not found in label classes: {label_classes}.")
if source_class == target_class:
    raise ValueError("--source-class and --target-class must be different.")

source_value = lc_lower.index(source_class)
target_value = lc_lower.index(target_class)
source_indices = [int(i) for i, pred in enumerate(y_pred) if pred == source_value]
if args.max_samples is not None:
    source_indices = source_indices[: args.max_samples]
if not source_indices:
    raise RuntimeError(f"No test samples predicted as '{label_classes[source_value]}' - nothing to explain.")

from sklearn.feature_extraction.text import TfidfVectorizer

print(f"Using device: {DEVICE}")
print(f"Label classes : {label_classes}")
print(f"source index  : {source_value} ({label_classes[source_value]})  |  "
      f"target index: {target_value} ({label_classes[target_value]})")
print(f"Test samples predicted as '{label_classes[source_value]}': {len(source_indices)}")

print("\nFitting TF-IDF on meme texts ...")
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(train_texts)


def _tfidf_embed_fn(texts):
    return _tfidf_vec.transform(["" if t is None else str(t) for t in texts]).toarray().astype(np.float32)


text_encoders = ["tfidf", "raw"]
if not args.no_word2vec:
    text_encoders.insert(0, "word2vec")
if not args.no_bert:
    text_encoders.insert(0, "bert")
if not args.no_minilm:
    text_encoders.insert(0, "minilm")

image_encoders = ["resnet50"]
if not args.no_efficientnet:
    image_encoders.append("efficientnet_b0")
if not args.no_vit:
    image_encoders.append("vit_b_16")

text_vector_metrics = ["cosine", "euclidean", "manhattan"]
text_direct_metrics = ["rouge_l", "lcs"]
image_distance_metrics = ["cosine", "euclidean"]

print("\nPreparing cached text/image embeddings ...")
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

_precomputed_text_embeddings_by_encoder: Dict[str, Dict[str, np.ndarray]] = {
    "tfidf": {
        "train": _tfidf_embed_fn(train_texts),
        "test": _tfidf_embed_fn(test_texts),
    }
}
_precomputed_image_embeddings_by_encoder: Dict[str, Dict[str, np.ndarray]] = {}

_text_embed_fns: Dict[str, object] = {"tfidf": _tfidf_embed_fn}
_image_embed_fns: Dict[str, object] = {}

_text_lookup_by_encoder: Dict[str, Dict[str, np.ndarray]] = {}

for enc in ("word2vec", "bert", "minilm"):
    if enc not in text_encoders:
        continue
    split_dict = embedding_cache["text"][enc]
    _precomputed_text_embeddings_by_encoder[enc] = {
        "train": _to_numpy(split_dict["train"]).astype(np.float32),
        "test": _to_numpy(split_dict["test"]).astype(np.float32),
    }

for enc in image_encoders:
    split_dict = embedding_cache["image"][enc]
    _precomputed_image_embeddings_by_encoder[enc] = {
        "train": _to_numpy(split_dict["train"]).astype(np.float32),
        "test": _to_numpy(split_dict["test"]).astype(np.float32),
    }

if "bert" in text_encoders:
    _text_embed_fns["bert"] = _make_embed_fn_from_e5_kwargs(
        tokenizer=text_bk["bert_tokenizer"],
        model=text_bk["bert_model"],
        device=text_bk["bert_device"],
    )

if "word2vec" in text_encoders:
    if args.word2vec_path is None:
        raise ValueError("word2vec is enabled but --word2vec-path is not set.")
    w2v_model, w2v_embed_fn = _build_word2vec_embed_fn(args.word2vec_path)
    _text_embed_fns["word2vec"] = w2v_embed_fn
    text_bk["word2vec_model"] = w2v_model
    text_bk["word2vec_embed_fn"] = w2v_embed_fn

if "minilm" in text_encoders:
    minilm_embed_fn = _build_minilm_embed_fn(DEVICE)
    _text_embed_fns["minilm"] = minilm_embed_fn
    text_bk["minilm_embed_fn"] = minilm_embed_fn

text_bk["precomputed_text_embeddings_by_encoder"] = _precomputed_text_embeddings_by_encoder

for enc, splits in _precomputed_text_embeddings_by_encoder.items():
    lookup = {}
    lookup.update({str(t): v for t, v in zip(train_texts, splits["train"])})
    lookup.update({str(t): v for t, v in zip(test_texts, splits["test"])})
    _text_lookup_by_encoder[enc] = lookup

for enc in image_encoders:
    _image_embed_fns[enc] = _build_live_image_embed_fn(enc, DEVICE, args.image_batch_size)

image_bk.update({
    "device": DEVICE,
    "batch_size": args.image_batch_size,
})


def _make_text_lookup_embed_fn(encoder: str):
    lookup = _text_lookup_by_encoder[encoder]
    live_fn = _text_embed_fns[encoder]

    def _fn(texts):
        vecs = [None] * len(texts)
        miss_texts: List[str] = []
        miss_pos: List[int] = []
        for i, text in enumerate(texts):
            key = "" if text is None else str(text)
            val = lookup.get(key)
            if val is None:
                miss_pos.append(i)
                miss_texts.append(key)
            else:
                vecs[i] = val
        if miss_texts:
            miss_embs = np.asarray(live_fn(miss_texts), dtype=np.float32)
            for j, pos in enumerate(miss_pos):
                vecs[pos] = miss_embs[j]
        return np.asarray(vecs, dtype=np.float32)

    return _fn


def _make_image_lookup_embed_fn(encoder: str):
    train_emb = _precomputed_image_embeddings_by_encoder[encoder]["train"]
    test_emb = _precomputed_image_embeddings_by_encoder[encoder]["test"]
    live_fn = _image_embed_fns[encoder]

    def _fn(items):
        vecs = [None] * len(items)
        misses = []
        miss_pos = []
        for i, item in enumerate(items):
            if isinstance(item, tuple) and len(item) == 2 and item[0] in {"train", "test"}:
                split, idx = item
                idx = int(idx)
                vecs[i] = train_emb[idx] if split == "train" else test_emb[idx]
            else:
                miss_pos.append(i)
                misses.append(item)
        if misses:
            miss_embs = np.asarray(live_fn(misses), dtype=np.float32)
            for j, pos in enumerate(miss_pos):
                vecs[pos] = miss_embs[j]
        return np.asarray(vecs, dtype=np.float32)

    return _fn


print("Fitting text/image plausibility references ...")
_text_objective_contexts: Dict[str, Dict[str, object]] = {}
for enc in _precomputed_text_embeddings_by_encoder:
    embed_fn = _make_text_lookup_embed_fn(enc)
    lof_ref = fit_text_lof_reference(
        train_texts,
        y_train=dataset.y_train,
        target_value=target_value,
        embed_fn=embed_fn,
    )
    _text_objective_contexts[enc] = {"embed_fn": embed_fn, **lof_ref}

_image_objective_contexts: Dict[str, Dict[str, object]] = {}
for enc in image_encoders:
    embed_fn = _make_image_lookup_embed_fn(enc)
    train_tokens = [("train", i) for i in range(len(dataset.y_train))]
    lof_ref = fit_image_lof(
        train_tokens,
        y_train=dataset.y_train,
        target_value=target_value,
        embed_fn=embed_fn,
    )
    _image_objective_contexts[enc] = {"embed_fn": embed_fn, **lof_ref}


_fusion_strategy = args.fusion_strategy

print(f"Loading model checkpoint (fusion_strategy={_fusion_strategy}) ...")
best_ckpt_bundle = _load_checkpoint_config(BEST_MODEL_PATH)  # always needed for config
best_ckpt = best_ckpt_bundle["checkpoint"]
best_cfg = dict(best_ckpt_bundle["config"])
if "text_encoder" not in best_cfg or "image_encoder" not in best_cfg:
    best_row = _read_best_row(DATA_DIR / "hparam_results.csv")
    best_cfg.setdefault("text_encoder", best_row["text_encoder"])
    best_cfg.setdefault("image_encoder", best_row["image_encoder"])

best_word2vec_path = best_cfg.get("word2vec_path") or args.word2vec_path

_predict_fn_lock = threading.Lock()
_train_latents: Optional[np.ndarray] = None
_test_latents:  Optional[np.ndarray] = None
_predict_fn = None


def _resolve_text_candidate(text_candidate_or_dict):
    if isinstance(text_candidate_or_dict, dict):
        return str(text_candidate_or_dict.get(dataset.primary_text_name, ""))
    return "" if text_candidate_or_dict is None else str(text_candidate_or_dict)


def _resolve_image_candidate(image_candidate_or_dict):
    if isinstance(image_candidate_or_dict, dict):
        image_candidate_or_dict = image_candidate_or_dict.get(dataset.primary_image_name)
    if isinstance(image_candidate_or_dict, tuple) and len(image_candidate_or_dict) == 2:
        split, idx = image_candidate_or_dict
        idx = int(idx)
        if split == "train":
            return train_images[idx], ("train", idx)
        if split == "test":
            return test_images[idx], ("test", idx)
    return image_candidate_or_dict, None


if _fusion_strategy == "intermediate":
    best_model = FineTuneMultimodalClassifier(
        text_encoder_name=best_cfg["text_encoder"],
        image_encoder_name=best_cfg["image_encoder"],
        n_classes=len(best_ckpt_bundle["label_classes"]),
        text_hidden_dim=best_cfg["text_hidden_dim"],
        img_hidden_dim=best_cfg["img_hidden_dim"],
        dropout=best_cfg["dropout"],
        device=DEVICE,
        word2vec_path=best_word2vec_path,
        word2vec_vocab_tokens=best_cfg.get("word2vec_vocab_tokens"),
    ).to(DEVICE)
    best_model.load_state_dict(best_ckpt["state_dict"])
    best_model.eval()

    best_text_backend = build_text_tokenizer(
        best_cfg["text_encoder"],
        word2vec_path=best_word2vec_path,
        vocab_tokens=best_cfg.get("word2vec_vocab_tokens"),
    )
    best_image_transform = get_image_transform(best_cfg["image_encoder"])

    def _make_loader(images, texts, labels, shuffle=False):
        ds = RawMultimodalDataset(
            images=images,
            texts=[str(t) for t in texts],
            labels=torch.tensor(labels, dtype=torch.long),
            text_encoder_name=best_cfg["text_encoder"],
            text_backend=best_text_backend,
            image_transform=best_image_transform,
            max_len=best_cfg.get("max_len", 128),
        )
        return DataLoader(ds, batch_size=16, shuffle=shuffle, num_workers=2)

    def _extract_latents_and_preds(model, loader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        latents, preds = [], []
        _buf: Dict[str, torch.Tensor] = {}
        def _hook(module, inputs, output):
            _buf["z"] = inputs[0].detach().cpu()
        handle = model.classifier.register_forward_hook(_hook)
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["input_ids"].to(DEVICE),
                               batch["attention_mask"].to(DEVICE),
                               batch["image"].to(DEVICE))
                latents.append(_buf["z"].clone().numpy().astype(np.float32))
                preds.append(logits.argmax(dim=-1).cpu().numpy().astype(np.float32))
        handle.remove()
        return np.vstack(latents), np.concatenate(preds)

    print("Extracting intermediate-fusion latent representations ...")
    _train_latents, _train_pred_labels = _extract_latents_and_preds(
        best_model, _make_loader(train_images, train_texts, dataset.y_train))
    _test_latents, _ = _extract_latents_and_preds(
        best_model, _make_loader(test_images, test_texts, dataset.y_test))

    _best_train_texts = np.asarray(train_texts, dtype=object)
    _best_test_texts  = np.asarray(test_texts,  dtype=object)

    def _predict_single_live(text: str, image) -> float:
        if best_cfg["text_encoder"] == "word2vec":
            encoded = encode_word2vec_ids(text, best_text_backend, best_cfg.get("max_len", 128))
            input_ids, attention_mask = encoded["input_ids"].unsqueeze(0), encoded["attention_mask"].unsqueeze(0)
        else:
            encoded = best_text_backend(text, max_length=best_cfg.get("max_len", 128),
                                        padding="max_length", truncation=True, return_tensors="pt")
            input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
        img_tensor = best_image_transform(
            Image.fromarray(np.asarray(image, dtype=np.uint8))
        ).unsqueeze(0)
        with _predict_fn_lock:
            with torch.no_grad():
                logits = best_model(input_ids.to(DEVICE), attention_mask.to(DEVICE), img_tensor.to(DEVICE))
        return float(logits.argmax(dim=-1).cpu().item())

    def _predict_fn(x_tab_unused, x_ts_unused, text_candidate_or_dict, image_candidate_or_dict):
        text = _resolve_text_candidate(text_candidate_or_dict)
        image, image_token = _resolve_image_candidate(image_candidate_or_dict)
        if image_token is not None:
            split, idx = image_token
            if split == "train" and idx < len(_best_train_texts) and text == str(_best_train_texts[idx]):
                return float(_train_pred_labels[idx])
            if split == "test" and idx < len(y_pred) and text == str(_best_test_texts[idx]):
                return float(y_pred[idx])
        return _predict_single_live(text, image)

elif _fusion_strategy == "early":
    # Early fusion: GBT on concat(minilm_text_emb, resnet50_image_emb)
    import pickle as _pickle
    _ef_gbt_path = DATA_DIR / "best_model_early_fusion_gbt.pkl"
    try:
        with open(_ef_gbt_path, "rb") as _f:
            _ef_obj = _pickle.load(_f)
        _ef_gbt = _ef_obj["model"]   # GradientBoostingClassifier, n_features=2432

        # Load precomputed embeddings (minilm=384, resnet50=2048)
        _emb_cache = torch.load(DATA_DIR / "precomputed_embeddings.pt", map_location="cpu")
        _ef_text_train  = _emb_cache["text"]["minilm"]["train"].numpy().astype(np.float32)
        _ef_text_test   = _emb_cache["text"]["minilm"]["test"].numpy().astype(np.float32)
        _ef_image_train = _emb_cache["image"]["resnet50"]["train"].numpy().astype(np.float32)
        _ef_image_test  = _emb_cache["image"]["resnet50"]["test"].numpy().astype(np.float32)

        # Build lookup: text_str → minilm embedding
        _ef_text_lookup: dict = {}
        for _i, _t in enumerate(train_texts):
            _ef_text_lookup[str(_t)] = _ef_text_train[_i]
        for _i, _t in enumerate(test_texts):
            _ef_text_lookup.setdefault(str(_t), _ef_text_test[_i])

        # Training-sample prediction cache (keyed by (train_idx,))
        _ef_train_inp = np.concatenate([_ef_text_train, _ef_image_train], axis=1)
        _ef_train_preds = _ef_gbt.predict(_ef_train_inp).astype(np.float32)
        _ef_pred_cache: dict = {("train", _i): float(_ef_train_preds[_i])
                                for _i in range(len(_ef_train_preds))}
        _ef_pred_cache.update({("test", _i): float(y_pred[_i]) for _i in range(len(y_pred))})
        print(f"  [early fusion] Cached {len(_ef_pred_cache):,} predictions from GBT.")

        def _predict_fn(x_tab_unused, x_ts_unused, text_candidate_or_dict, image_candidate_or_dict,
                        _gbt=_ef_gbt, _cache=_ef_pred_cache,
                        _txt_lkp=_ef_text_lookup,
                        _img_tr=_ef_image_train, _img_te=_ef_image_test):
            text = _resolve_text_candidate(text_candidate_or_dict)
            _, image_token = _resolve_image_candidate(image_candidate_or_dict)
            if image_token is not None:
                _hit = _cache.get(image_token)
                if _hit is not None:
                    return _hit
                split, idx = image_token
                img_emb = (_img_tr[idx] if split == "train" else _img_te[idx])
            else:
                img_emb = None
            txt_emb = _txt_lkp.get(text)
            if txt_emb is None or img_emb is None:
                return 0.0
            inp = np.concatenate([txt_emb, img_emb])[None, :]
            return float(_gbt.predict(inp)[0])

        print(f"[early fusion] predict_fn ready using {_ef_gbt_path.name} (minilm+resnet50 GBT).")
    except Exception as _e:
        print(f"[warn] Early-fusion GBT failed to load: {_e}")

elif _fusion_strategy == "late":
    # Late fusion: GBT text (minilm, 384-d) + GBT image (resnet50, 2048-d) averaged
    import pickle as _pickle
    _lf_text_path  = DATA_DIR / "best_model_late_fusion_gbt_text.pkl"
    _lf_image_path = DATA_DIR / "best_model_late_fusion_gbt_image.pkl"
    try:
        with open(_lf_text_path, "rb") as _f:
            _lf_text_obj = _pickle.load(_f)
        with open(_lf_image_path, "rb") as _f:
            _lf_image_obj = _pickle.load(_f)
        _lf_text_gbt  = _lf_text_obj["model"]   # n_features=384 (minilm)
        _lf_image_gbt = _lf_image_obj["model"]  # n_features=2048 (resnet50)

        _emb_cache2 = torch.load(DATA_DIR / "precomputed_embeddings.pt", map_location="cpu")
        _lf_text_train  = _emb_cache2["text"]["minilm"]["train"].numpy().astype(np.float32)
        _lf_text_test   = _emb_cache2["text"]["minilm"]["test"].numpy().astype(np.float32)
        _lf_image_train = _emb_cache2["image"]["resnet50"]["train"].numpy().astype(np.float32)
        _lf_image_test  = _emb_cache2["image"]["resnet50"]["test"].numpy().astype(np.float32)

        _lf_text_lookup: dict = {}
        for _i, _t in enumerate(train_texts):
            _lf_text_lookup[str(_t)] = _lf_text_train[_i]
        for _i, _t in enumerate(test_texts):
            _lf_text_lookup.setdefault(str(_t), _lf_text_test[_i])

        # Cache by (split, idx)
        _lf_pred_cache: dict = {}
        for _i in range(len(train_texts)):
            _p_t = _lf_text_gbt.predict_proba(_lf_text_train[_i][None, :])[0]
            _p_i = _lf_image_gbt.predict_proba(_lf_image_train[_i][None, :])[0]
            _lf_pred_cache[("train", _i)] = float(np.argmax((_p_t + _p_i) / 2.0))
        for _i in range(len(test_texts)):
            _lf_pred_cache[("test", _i)] = float(y_pred[_i])
        print(f"  [late fusion] Cached {len(_lf_pred_cache):,} predictions.")

        def _predict_fn(x_tab_unused, x_ts_unused, text_candidate_or_dict, image_candidate_or_dict,
                        _tgbt=_lf_text_gbt, _igbt=_lf_image_gbt,
                        _cache=_lf_pred_cache,
                        _txt_lkp=_lf_text_lookup,
                        _img_tr=_lf_image_train, _img_te=_lf_image_test):
            text = _resolve_text_candidate(text_candidate_or_dict)
            _, image_token = _resolve_image_candidate(image_candidate_or_dict)
            if image_token is not None:
                _hit = _cache.get(image_token)
                if _hit is not None:
                    return _hit
                split, idx = image_token
                img_emb = (_img_tr[idx] if split == "train" else _img_te[idx])
            else:
                img_emb = None
            txt_emb = _txt_lkp.get(text)
            if txt_emb is None or img_emb is None:
                return 0.0
            p_t = _tgbt.predict_proba(txt_emb[None, :])[0]
            p_i = _igbt.predict_proba(img_emb[None, :])[0]
            return float(np.argmax((p_t + p_i) / 2.0))

        print("[late fusion] predict_fn ready (gbt_text + gbt_image averaged).")
    except Exception as _e:
        print(f"[warn] Late-fusion GBT models failed to load: {_e}")


def _text_embed_fn_for_encoder(encoder: str):
    enc = "tfidf" if encoder == "raw" else encoder
    return _text_embed_fns[enc]


def _text_precomputed_for_encoder(encoder: str) -> Dict[str, np.ndarray]:
    enc = "tfidf" if encoder == "raw" else encoder
    return _precomputed_text_embeddings_by_encoder[enc]


def _image_precomputed_for_encoder(encoder: str) -> Dict[str, np.ndarray]:
    return _precomputed_image_embeddings_by_encoder[encoder]


def _fusion_metric(text_cfg, image_cfg) -> str:
    text_encoder = (text_cfg or {}).get("encoder", "raw")
    text_metric = (text_cfg or {}).get("metric", "cosine")
    image_metric = (image_cfg or {}).get("metric", "cosine")
    if text_encoder != "raw" and text_metric == image_metric and text_metric in {"cosine", "euclidean", "manhattan"}:
        return text_metric
    return "cosine"


def _candidate_image_token(cand) -> object:
    source_indices = cand.get("source_indices") or {}
    image_sources = source_indices.get("image")
    if isinstance(image_sources, dict):
        src = image_sources.get(dataset.primary_image_name)
        if src is not None:
            return ("train", int(src))
    src = cand.get("source_train_idx")
    if src is not None:
        return ("train", int(src))
    image_val = cand.get("image")
    if image_val is not None:
        return image_val
    return cand.get("image_input")


def _image_modalities_fn(sample_idx, cand):
    image_encoder = (cand.get("_image_encoder_for_objectives") or "resnet50")
    ctx = _image_objective_contexts[image_encoder]
    return {
        "image_modalities": {
            dataset.primary_image_name: {
                "candidate": _candidate_image_token(cand),
                "factual": ("test", int(sample_idx)),
                "context": ctx,
            }
        }
    }


def _objectives_kwargs_factory(text_cfg, image_cfg):
    text_encoder = (text_cfg or {}).get("encoder", "raw")
    image_encoder = (image_cfg or {}).get("encoder", image_encoders[0])
    obj_text_encoder = "tfidf" if text_encoder == "raw" else text_encoder

    def _wrapped_image_modalities_fn(sample_idx, cand, _enc=image_encoder):
        cand = dict(cand)
        cand["_image_encoder_for_objectives"] = _enc
        return _image_modalities_fn(sample_idx, cand)

    return {
        "y_target": target_value,
        "text_objective_context": _text_objective_contexts[obj_text_encoder],
        "_image_modalities_fn": _wrapped_image_modalities_fn,
        "predict_fn": _predict_fn,
    }


def _multimodal_generators_factory(
    tab_cfg,
    ts_cfg,
    text_cfg,
    text_backend_kwargs,
    image_cfg,
    image_backend_kwargs,
):
    if text_cfg is None or image_cfg is None:
        return {}

    text_encoder = (text_cfg or {}).get("encoder", "raw")
    image_encoder = (image_cfg or {}).get("encoder", image_encoders[0])
    image_metric = (image_cfg or {}).get("metric", "cosine")
    fusion_metric = _fusion_metric(text_cfg, image_cfg)
    text_embed_fn = _text_embed_fn_for_encoder(text_encoder)
    text_pre = _text_precomputed_for_encoder(text_encoder)
    image_pre = _image_precomputed_for_encoder(image_encoder)

    return {
        "Frankenstein": FrankensteinNN(
            k=args.k,
            k_search=args.k_search_combined,
            e5_embed_fn=text_embed_fn,
            image_encoder=image_encoder,
            img_distance_metric=image_metric,
            img_device=DEVICE,
            img_batch_size=args.image_batch_size,
        ),
        "Combined": CombinedNN(
            k=args.k,
            k_search=args.k_search_combined,
            e5_embed_fn=text_embed_fn,
            image_encoder=image_encoder,
            img_distance_metric=image_metric,
            img_device=DEVICE,
            img_batch_size=args.image_batch_size,
        ),
        "EarlyFusion": EarlyFusionNN(
            k=args.k,
            distance_metric=fusion_metric,
            precomputed_train_text_embeddings=text_pre["train"],
            precomputed_test_text_embeddings=text_pre["test"],
            precomputed_train_image_embeddings_by_name={
                dataset.primary_image_name: image_pre["train"],
            },
            precomputed_test_image_embeddings_by_name={
                dataset.primary_image_name: image_pre["test"],
            },
        ),
        "IntermediateFusion": IntermediateFusionNN(
            k=args.k,
            distance_metric=fusion_metric,
            precomputed_train_latent=_train_latents,
            precomputed_test_latent=_test_latents,
        ),
    }


print("\nRunning distance ablation ...")
print(f"  Samples : {len(source_indices)} (predicted '{label_classes[source_value]}')")
print(f"  Target  : {target_value} ({label_classes[target_value]})")
print(f"  k       : {args.k}")
print(f"  n_jobs  : {args.n_jobs}")
print(f"  Text encoders         : {text_encoders}")
print(f"  Text vector metrics   : {text_vector_metrics}")
print(f"  Text direct metrics   : {text_direct_metrics}")
print(f"  Image encoders        : {image_encoders}")
print(f"  Image distance metrics: {image_distance_metrics}")
print(f"  Best-model latent pair: {best_cfg['text_encoder']} + {best_cfg['image_encoder']}")
print()

_default_run_names = {
    "intermediate": "fusion_intermediate_deep_k50",
    "early":        "fusion_early_gbt_k50",
    "late":         "fusion_late_nondp_k50",
}
_run_name = args.run_name or _default_run_names[_fusion_strategy]

if args.eval_pkls:
    from evaluate_k_ablation import evaluate_k_ablation
    _eval_dirs = [Path(d.strip()) for d in args.eval_pkls.split(",")]
    _k_vals = [int(x) for x in args.k_values.split(",")]
    evaluate_k_ablation(
        run_dirs=_eval_dirs,
        dataset=dataset,
        objectives_kwargs_factory=_objectives_kwargs_factory,
        k_values=_k_vals,
        output_path=_eval_dirs[0] / f"k_ablation_summary_k{'_'.join(str(k) for k in _k_vals)}.json",
    )
else:
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
        run_name=_run_name,
        save_full=args.save_full,
        max_combinations=args.max_combinations,
        n_jobs=args.n_jobs,
        objectives_kwargs_factory=_objectives_kwargs_factory,
        extra_generators_factory=_multimodal_generators_factory,
    )
