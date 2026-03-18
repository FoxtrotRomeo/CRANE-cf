"""
Hyperparameter search for the multimodal tweet classifier.

Strategy
--------
Random search over the defined grid, using a held-out validation split of
the training data. Results are appended to data/hparam_results.csv after
every trial so progress is never lost. After the sweep, the best config
(highest macro-F1 on validation) is retrained on the full training set and
evaluated on the test set.

Run
---
    cd long_covid_tweets
    python hparam_search.py --gpu 7

Outputs in long_covid_tweets/data/
-----------------------------------
  hparam_results.csv  — one row per trial
  best_model.pt       — state dict of the final retrained model
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hyperparameter search for multimodal tweet classifier")
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index to use (e.g. 7). Defaults to CPU if not specified or unavailable.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH    = "data/dataset.npz"
TEXT_MODEL   = "cardiffnlp/twitter-xlm-roberta-base"
OUT_DIR      = "data"
RESULTS_CSV  = os.path.join(OUT_DIR, "hparam_results.csv")
MAX_TRIALS   = 50    # random search budget; set to None to run all combinations
VAL_FRAC     = 0.15  # fraction of train held out for validation during search
SEARCH_EPOCHS = 12   # transformer is frozen during search, so more epochs are cheap
FINAL_EPOCHS  = 8    # epochs for retraining the best config on full train (unfrozen)
SEARCH_LR     = 1e-3 # higher LR for search: only heads are trained
BATCH_SIZE   = 32
SEED         = 42

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only {torch.cuda.device_count()} GPU(s) available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # list of layer widths for the tabular MLP (variable depth + width)
    "tab_hidden_dims": [[32], [64], [128], [64, 32], [128, 64]],
    "text_hidden_dim": [64, 128, 256],
    "dropout":         [0.1, 0.2, 0.3, 0.4],
    # lr is used only in final retraining (transformer unfrozen); SEARCH_LR is fixed during search
    "lr":              [1e-5, 2e-5, 5e-5],
    "weight_decay":    [0.0, 1e-4, 1e-3],
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_static_all = torch.tensor(data["X_train_static"], dtype=torch.float32)
X_test_static      = torch.tensor(data["X_test_static"],  dtype=torch.float32)
X_train_text_all   = data["X_train_text"].tolist()
X_test_text        = data["X_test_text"].tolist()
y_train_all        = torch.tensor(data["y_train"], dtype=torch.long)
y_test             = torch.tensor(data["y_test"],  dtype=torch.long)
label_classes      = data["label_classes"]

N_CLASSES = len(label_classes)
D_TAB     = X_train_static_all.shape[1]
N_TRAIN   = len(y_train_all)

print(f"Classes ({N_CLASSES}): {label_classes.tolist()}")
print(f"Tabular features: {D_TAB} | Train: {N_TRAIN} | Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# Validation split indices (fixed across all trials)
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
all_indices = list(range(N_TRAIN))
rng.shuffle(all_indices)
n_val = int(N_TRAIN * VAL_FRAC)
val_indices   = all_indices[:n_val]
train_indices = all_indices[n_val:]
print(f"Search split — train: {len(train_indices)}  val: {len(val_indices)}")

# ---------------------------------------------------------------------------
# Tokenizer (shared across all trials)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, X_static, X_text, y, max_len=128):
        self.X_static = X_static
        self.X_text   = X_text
        self.y        = y
        self.max_len  = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.X_text[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "X_static":       self.X_static[idx],
            "label":          self.y[idx],
        }


full_train_ds = TweetDataset(X_train_static_all, X_train_text_all, y_train_all)
test_ds       = TweetDataset(X_test_static,      X_test_text,      y_test)

search_train_ds = Subset(full_train_ds, train_indices)
search_val_ds   = Subset(full_train_ds, val_indices)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TabHead(nn.Module):
    """MLP with variable depth; out_dim is the last hidden width."""
    def __init__(self, d_in, hidden_dims, dropout):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net     = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        text_enc_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_enc_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tab_head  = TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim     = text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, input_ids, attention_mask, X_static):
        cls_emb  = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask).last_hidden_state[:, 0]
        text_emb = self.text_proj(cls_emb)
        tab_emb  = self.tab_head(X_static)
        return self.classifier(torch.cat([text_emb, tab_emb], dim=1))

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)


def run_epoch(model, loader, optimizer, criterion, train):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            tab   = batch["X_static"].to(DEVICE)
            y     = batch["label"].to(DEVICE)
            logits = model(ids, mask, tab)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss  += loss.item() * len(y)
            preds_all   += logits.argmax(1).cpu().tolist()
            labels_all  += y.cpu().tolist()
    n = len(labels_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / n, macro_f1


def train_model(model, train_loader, val_loader, lr, weight_decay, n_epochs):
    """Train with early stopping on val_f1. Used during the search phase."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_f1, best_state = 0.0, None
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader,  optimizer, criterion, train=False)
        print(f"  ep {epoch}/{n_epochs}  "
              f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f}  "
              f"vl_loss={vl_loss:.4f} vl_f1={vl_f1:.3f}")
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_val_f1, best_state


def train_final(model, train_loader, lr, weight_decay, n_epochs):
    """Train on the full training set for a fixed number of epochs, no early stopping.
    The test set is never seen, so the saved weights are unbiased."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        print(f"  ep {epoch}/{n_epochs}  tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f}")
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ---------------------------------------------------------------------------
# Build trial list (random sample if too many combinations)
# ---------------------------------------------------------------------------
keys   = list(SEARCH_SPACE.keys())
combos = [dict(zip(keys, vals))
          for vals in __import__("itertools").product(*SEARCH_SPACE.values())]
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

    model = MultimodalClassifier(
        d_tab           = D_TAB,
        n_classes       = N_CLASSES,
        tab_hidden_dims = cfg["tab_hidden_dims"],
        text_hidden_dim = cfg["text_hidden_dim"],
        dropout         = cfg["dropout"],
    ).to(DEVICE)

    # Freeze transformer — only projection heads and tabular head are trained
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    val_f1, _ = train_model(
        model,
        make_loader(search_train_ds, shuffle=True),
        make_loader(search_val_ds,   shuffle=False),
        lr           = SEARCH_LR,
        weight_decay = cfg["weight_decay"],
        n_epochs     = SEARCH_EPOCHS,
    )
    duration = time.time() - t0
    print(f"  → val_f1={val_f1:.4f}  ({duration/60:.1f} min)\n")

    # Log result
    row = {**{k: str(v) for k, v in cfg.items()},
           "val_f1": f"{val_f1:.6f}", "duration_s": f"{duration:.1f}"}
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_cfg    = cfg

print(f"\nBest val_f1={best_val_f1:.4f}  config={best_cfg}")

# ---------------------------------------------------------------------------
# Retrain best config on full training data
# ---------------------------------------------------------------------------
print("\nRetraining best config on full training data …")
final_model = MultimodalClassifier(
    d_tab           = D_TAB,
    n_classes       = N_CLASSES,
    tab_hidden_dims = best_cfg["tab_hidden_dims"],
    text_hidden_dim = best_cfg["text_hidden_dim"],
    dropout         = best_cfg["dropout"],
).to(DEVICE)

# Unfreeze transformer for full fine-tuning with the best config's LR
for p in final_model.text_encoder.parameters():
    p.requires_grad = True

final_state = train_final(
    final_model,
    make_loader(full_train_ds, shuffle=True),
    lr           = best_cfg["lr"],
    weight_decay = best_cfg["weight_decay"],
    n_epochs     = FINAL_EPOCHS,
)
final_model.load_state_dict({k: v.to(DEVICE) for k, v in final_state.items()})

# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
from sklearn.metrics import classification_report

final_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = final_model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["X_static"].to(DEVICE),
        )
        all_preds  += logits.argmax(1).cpu().tolist()
        all_labels += batch["label"].tolist()

print("\nTest classification report (best config):")
print(classification_report(all_labels, all_preds, target_names=label_classes))

# ---------------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------------
best_model_path = os.path.join(OUT_DIR, "best_model.pt")
torch.save(
    {"state_dict": final_state, "config": best_cfg, "tab_cols": data["tab_cols"].tolist(),
     "label_classes": label_classes.tolist()},
    best_model_path,
)
print(f"Best model saved to {best_model_path}")
print(f"Full results log: {RESULTS_CSV}")
