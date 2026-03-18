"""
Retrain the best hyperparameter config on the full training set.

Reads data/hparam_results.csv to find the best config (highest val_f1),
trains on the full training set with the transformer unfrozen (no early
stopping, so the test set is never seen during training), evaluates on the
test set, and saves the model to data/best_model.pt.

Run
---
    cd real_or_fake_jobs
    python retrain_best.py --gpu 7
"""

import argparse
import ast
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index to use (e.g. 7).")
parser.add_argument("--epochs", type=int, default=8,
                    help="Number of training epochs (default: 8).")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH    = "data/dataset.npz"
RESULTS_CSV  = "data/hparam_results.csv"
OUT_DIR      = "data"
TEXT_MODEL   = "distilbert-base-uncased"
BATCH_SIZE   = 32

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only "
                         f"{torch.cuda.device_count()} GPU(s) available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load best config from CSV
# ---------------------------------------------------------------------------
best_row, best_f1 = None, -1.0
with open(RESULTS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        f1 = float(row["val_f1"])
        if f1 > best_f1:
            best_f1  = f1
            best_row = row

best_cfg = {
    "tab_hidden_dims": ast.literal_eval(best_row["tab_hidden_dims"]),
    "text_hidden_dim": int(best_row["text_hidden_dim"]),
    "dropout":         float(best_row["dropout"]),
    "lr":              float(best_row["lr"]),
    "weight_decay":    float(best_row["weight_decay"]),
}
print(f"\nBest config (val_f1={best_f1:.4f}):")
for k, v in best_cfg.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_static = torch.tensor(data["X_train_static"], dtype=torch.float32)
X_test_static  = torch.tensor(data["X_test_static"],  dtype=torch.float32)
X_train_text   = data["X_train_text"].tolist()
X_test_text    = data["X_test_text"].tolist()
y_train        = torch.tensor(data["y_train"], dtype=torch.long)
y_test         = torch.tensor(data["y_test"],  dtype=torch.long)
label_classes  = data["label_classes"]

N_CLASSES = len(label_classes)
D_TAB     = X_train_static.shape[1]
print(f"\nClasses ({N_CLASSES}): {label_classes.tolist()}")
print(f"Tabular features: {D_TAB} | Train: {len(y_train)} | Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# Tokenizer & Dataset
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)


class JobDataset(Dataset):
    def __init__(self, X_static, X_text, y, max_len=512):
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


train_ds   = JobDataset(X_train_static, X_train_text, y_train)
test_ds    = JobDataset(X_test_static,  X_test_text,  y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TabHead(nn.Module):
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
        text_enc_dim      = self.text_encoder.config.hidden_size
        self.text_proj    = nn.Sequential(
            nn.Linear(text_enc_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tab_head   = TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim      = text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, input_ids, attention_mask, X_static):
        cls_emb  = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask).last_hidden_state[:, 0]
        text_emb = self.text_proj(cls_emb)
        tab_emb  = self.tab_head(X_static)
        return self.classifier(torch.cat([text_emb, tab_emb], dim=1))


model = MultimodalClassifier(
    d_tab           = D_TAB,
    n_classes       = N_CLASSES,
    tab_hidden_dims = best_cfg["tab_hidden_dims"],
    text_hidden_dim = best_cfg["text_hidden_dim"],
    dropout         = best_cfg["dropout"],
).to(DEVICE)

# ---------------------------------------------------------------------------
# Training — full training set, fixed epochs, no early stopping, no test peek
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=best_cfg["lr"],
                              weight_decay=best_cfg["weight_decay"])
criterion = nn.CrossEntropyLoss()

print(f"\nTraining for {args.epochs} epochs …")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in train_loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        tab    = batch["X_static"].to(DEVICE)
        y      = batch["label"].to(DEVICE)
        logits = model(ids, mask, tab)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all  += logits.argmax(1).cpu().tolist()
        labels_all += y.cpu().tolist()
    tr_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    print(f"  ep {epoch}/{args.epochs}  "
          f"loss={total_loss/len(labels_all):.4f}  train_f1={tr_f1:.3f}")

# ---------------------------------------------------------------------------
# Test evaluation (first time the model sees test data)
# ---------------------------------------------------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["X_static"].to(DEVICE),
        )
        all_preds  += logits.argmax(1).cpu().tolist()
        all_labels += batch["label"].tolist()

print("\nTest classification report:")
print(classification_report(all_labels, all_preds, target_names=label_classes))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "best_model.pt")
torch.save(
    {
        "state_dict":    {k: v.cpu() for k, v in model.state_dict().items()},
        "config":        best_cfg,
        "tab_cols":      data["tab_cols"].tolist(),
        "label_classes": label_classes.tolist(),
    },
    out_path,
)
print(f"\nModel saved to {out_path}")