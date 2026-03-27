"""
Hyperparameter search for the multimodal Hateful Memes classifier.

Architecture
------------
Search stage:
  meme text   -> frozen encoder embedding cache -> projection -> text_emb
  meme image  -> frozen encoder embedding cache -> projection -> img_emb
  [text_emb, img_emb] -> concat -> Linear -> logits (2 classes)

Final retraining:
  rebuild the winning text/image encoder pair from pretrained weights
  and fine-tune the encoders plus the projection heads/classifier end to end.

Strategy
--------
Random search over the defined grid using a held-out validation split.
Embeddings for several text and image encoders are precomputed once at the
beginning of the script and reused across all trials.
Results are appended to data/hparam_results.csv after every trial.
After the sweep, the winning configuration is fine-tuned on the full
training set with the raw text and images.

Run
---
    cd memes
    python hparam_search.py --gpu 7

Outputs in memes/data/
----------------------
  hparam_results.csv         - one row per trial
  best_model.pt              - state dict of the final fine-tuned model
  precomputed_embeddings.pt  - cached search embeddings for all encoders
"""

import argparse
import csv
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from encoder_features import (
    DEFAULT_CACHE_PATH,
    EmbeddingDataset,
    FineTuneMultimodalClassifier,
    IMAGE_ENCODERS,
    PrecomputedMultimodalClassifier,
    RawMultimodalDataset,
    TEXT_ENCODERS,
    build_text_tokenizer,
    collect_word2vec_tokens,
    get_feature_dims,
    get_image_transform,
    load_or_precompute_embeddings,
)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hyperparameter search for Hateful Memes classifier")
parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="GPU index to use (e.g. 7). Defaults to CPU if not specified.",
)
parser.add_argument(
    "--word2vec-path",
    type=str,
    default="../real_or_fake_jobs/data/word2vec_google_news_300.kv",
    help="Path to a gensim word2vec .kv or .bin file used for the word2vec text encoder.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/dataset.npz"
OUT_DIR = "data"
RESULTS_CSV = os.path.join(OUT_DIR, "hparam_results.csv")
EMBEDDING_CACHE = DEFAULT_CACHE_PATH
MAX_TRIALS = 150
VAL_FRAC = 0.15
SEARCH_EPOCHS = 8
FINAL_EPOCHS = 5
SEARCH_BATCH_SIZE = 64
FINAL_BATCH_SIZE = 16
MAX_LEN = 128
FINAL_LR = 2e-5
NUM_WORKERS = 2
SEED = 42

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only {torch.cuda.device_count()} available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "text_encoder": list(TEXT_ENCODERS.keys()),
    "image_encoder": list(IMAGE_ENCODERS.keys()),
    "text_hidden_dim": [64, 128, 256],
    "img_hidden_dim": [128, 256, 512],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "head_lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_img_all = data["X_train_img"]
X_test_img = data["X_test_img"]
X_train_text_all = data["X_train_text"].tolist()
X_test_text = data["X_test_text"].tolist()
y_train_all = torch.tensor(data["y_train"], dtype=torch.long)
y_test = torch.tensor(data["y_test"], dtype=torch.long)
label_classes = data["label_classes"]

N_CLASSES = len(label_classes)
N_TRAIN = len(y_train_all)

print(f"Classes ({N_CLASSES}): {label_classes.tolist()}")
print(f"Train: {N_TRAIN} | Test: {len(y_test)}")

word2vec_vocab_tokens = collect_word2vec_tokens(
    X_train_text_all + X_test_text,
    args.word2vec_path,
)
print(f"Word2Vec corpus vocab size: {len(word2vec_vocab_tokens)}")

# ---------------------------------------------------------------------------
# Validation split (fixed across all trials)
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
all_indices = list(range(N_TRAIN))
rng.shuffle(all_indices)
n_val = int(N_TRAIN * VAL_FRAC)
val_indices = all_indices[:n_val]
train_indices = all_indices[n_val:]
print(f"Search split - train: {len(train_indices)}  val: {len(val_indices)}")

# ---------------------------------------------------------------------------
# Precompute embeddings for the search stage
# ---------------------------------------------------------------------------
embedding_cache = load_or_precompute_embeddings(
    X_train_text=X_train_text_all,
    X_test_text=X_test_text,
    X_train_img=X_train_img_all,
    X_test_img=X_test_img,
    device=DEVICE,
    batch_size=SEARCH_BATCH_SIZE,
    max_len=MAX_LEN,
    num_workers=NUM_WORKERS,
    cache_path=EMBEDDING_CACHE,
    word2vec_path=args.word2vec_path,
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def make_feature_loader(text_features, image_features, labels, shuffle):
    dataset = EmbeddingDataset(text_features, image_features, labels)
    return DataLoader(
        dataset,
        batch_size=SEARCH_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def make_raw_loader(images, texts, labels, text_encoder_name, image_encoder_name, shuffle):
    dataset = RawMultimodalDataset(
        images=images,
        texts=texts,
        labels=labels,
        text_encoder_name=text_encoder_name,
        text_backend=build_text_tokenizer(
            text_encoder_name,
            word2vec_path=args.word2vec_path,
            vocab_tokens=word2vec_vocab_tokens if text_encoder_name == "word2vec" else None,
        ),
        image_transform=get_image_transform(image_encoder_name),
        max_len=MAX_LEN,
    )
    return DataLoader(
        dataset,
        batch_size=FINAL_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def run_feature_epoch(model, loader, optimizer, criterion, train):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            text_features = batch["text_features"].to(DEVICE)
            image_features = batch["image_features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(text_features, image_features)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds_all += logits.argmax(1).cpu().tolist()
            labels_all += labels.cpu().tolist()
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(labels_all), macro_f1


def run_finetune_epoch(model, loader, optimizer, criterion, train):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds_all += logits.argmax(1).cpu().tolist()
            labels_all += labels.cpu().tolist()
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(labels_all), macro_f1


def train_search_model(model, train_loader, val_loader, lr, weight_decay, n_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1 = run_feature_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_f1 = run_feature_epoch(model, val_loader, optimizer, criterion, train=False)
        print(
            f"  ep {epoch}/{n_epochs}  "
            f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f}  "
            f"vl_loss={vl_loss:.4f} vl_f1={vl_f1:.3f}"
        )
        best_val_f1 = max(best_val_f1, vl_f1)
    return best_val_f1


def train_final(model, train_loader, lr, weight_decay, n_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1 = run_finetune_epoch(model, train_loader, optimizer, criterion, train=True)
        print(f"  ep {epoch}/{n_epochs}  tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f}")
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Build trial list
# ---------------------------------------------------------------------------
keys = list(SEARCH_SPACE.keys())
combos = [dict(zip(keys, vals)) for vals in itertools.product(*SEARCH_SPACE.values())]
random.seed(SEED)
random.shuffle(combos)
trials = combos if MAX_TRIALS is None else combos[:MAX_TRIALS]
print(f"\nTotal combinations: {len(combos)}  |  Running: {len(trials)} trials\n")

# ---------------------------------------------------------------------------
# CSV setup
# ---------------------------------------------------------------------------
csv_fields = keys + ["val_f1", "duration_s"]
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------
best_cfg, best_val_f1 = None, -1.0

for trial_idx, cfg in enumerate(trials, 1):
    print(f"[Trial {trial_idx}/{len(trials)}] {cfg}")
    t0 = time.time()

    train_text_features = embedding_cache["text"][cfg["text_encoder"]]["train"][train_indices]
    val_text_features = embedding_cache["text"][cfg["text_encoder"]]["train"][val_indices]
    train_image_features = embedding_cache["image"][cfg["image_encoder"]]["train"][train_indices]
    val_image_features = embedding_cache["image"][cfg["image_encoder"]]["train"][val_indices]
    train_labels = y_train_all[train_indices]
    val_labels = y_train_all[val_indices]

    text_input_dim, img_input_dim = get_feature_dims(cfg["text_encoder"], cfg["image_encoder"])
    model = PrecomputedMultimodalClassifier(
        text_input_dim=text_input_dim,
        img_input_dim=img_input_dim,
        n_classes=N_CLASSES,
        text_hidden_dim=cfg["text_hidden_dim"],
        img_hidden_dim=cfg["img_hidden_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    val_f1 = train_search_model(
        model,
        make_feature_loader(train_text_features, train_image_features, train_labels, shuffle=True),
        make_feature_loader(val_text_features, val_image_features, val_labels, shuffle=False),
        lr=cfg["head_lr"],
        weight_decay=cfg["weight_decay"],
        n_epochs=SEARCH_EPOCHS,
    )
    duration = time.time() - t0
    print(f"  -> val_f1={val_f1:.4f}  ({duration / 60:.1f} min)\n")

    row = {
        **{k: str(v) for k, v in cfg.items()},
        "val_f1": f"{val_f1:.6f}",
        "duration_s": f"{duration:.1f}",
    }
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_cfg = cfg

print(f"\nBest val_f1={best_val_f1:.4f}  config={best_cfg}")

# ---------------------------------------------------------------------------
# Retrain best config on full training data with end-to-end fine-tuning
# ---------------------------------------------------------------------------
print("\nFine-tuning winning encoder pair on the full training data ...")
final_model = FineTuneMultimodalClassifier(
    text_encoder_name=best_cfg["text_encoder"],
    image_encoder_name=best_cfg["image_encoder"],
    n_classes=N_CLASSES,
    text_hidden_dim=best_cfg["text_hidden_dim"],
    img_hidden_dim=best_cfg["img_hidden_dim"],
    dropout=best_cfg["dropout"],
    device=DEVICE,
    word2vec_path=args.word2vec_path,
    word2vec_vocab_tokens=word2vec_vocab_tokens if best_cfg["text_encoder"] == "word2vec" else None,
).to(DEVICE)

final_state = train_final(
    final_model,
    make_raw_loader(
        X_train_img_all,
        X_train_text_all,
        y_train_all,
        best_cfg["text_encoder"],
        best_cfg["image_encoder"],
        shuffle=True,
    ),
    lr=FINAL_LR,
    weight_decay=best_cfg["weight_decay"],
    n_epochs=FINAL_EPOCHS,
)
final_model.load_state_dict({k: v.to(DEVICE) for k, v in final_state.items()})

# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
final_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in make_raw_loader(
        X_test_img,
        X_test_text,
        y_test,
        best_cfg["text_encoder"],
        best_cfg["image_encoder"],
        shuffle=False,
    ):
        logits = final_model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["image"].to(DEVICE),
        )
        all_preds += logits.argmax(1).cpu().tolist()
        all_labels += batch["label"].tolist()

print("\nTest classification report (best config):")
print(classification_report(all_labels, all_preds, target_names=label_classes))

# ---------------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------------
best_model_path = os.path.join(OUT_DIR, "best_model.pt")
torch.save(
    {
        "state_dict": final_state,
        "config": {
            **best_cfg,
            "final_lr": FINAL_LR,
            "final_epochs": FINAL_EPOCHS,
            "max_len": MAX_LEN,
            "word2vec_path": args.word2vec_path,
            "word2vec_vocab_tokens": word2vec_vocab_tokens if best_cfg["text_encoder"] == "word2vec" else None,
        },
        "label_classes": label_classes.tolist(),
    },
    best_model_path,
)
print(f"Best model saved to {best_model_path}")
print(f"Full results log: {RESULTS_CSV}")
