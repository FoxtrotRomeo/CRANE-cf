"""
Evaluate the best saved model on the test set and save outputs.

Outputs in data/
-----------------
  classification_report.txt  — sklearn text report (precision/recall/F1 per class)
  y_true.npy                 — integer class indices, shape (n_test,)
  y_pred.npy                 — predicted class indices, shape (n_test,)
  y_proba.npy                — softmax probabilities,  shape (n_test, n_classes)

Run
---
    cd real_or_fake_jobs
    python evaluate.py --gpu 7
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None)
args = parser.parse_args()

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only "
                         f"{torch.cuda.device_count()} GPU(s) available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = "data/dataset.npz"
MODEL_PATH = "data/best_model.pt"
OUT_DIR    = "data"
TEXT_MODEL = "distilbert-base-uncased"
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Load checkpoint and data
# ---------------------------------------------------------------------------
ckpt          = torch.load(MODEL_PATH, map_location="cpu")
cfg           = ckpt["config"]
label_classes = np.array(ckpt["label_classes"])
N_CLASSES     = len(label_classes)

data                   = np.load(DATA_PATH, allow_pickle=True)
X_test_static          = torch.tensor(data["X_test_static"], dtype=torch.float32)
X_test_description     = data["X_test_description"].tolist()
X_test_company_profile = data["X_test_company_profile"].tolist()
X_test_requirements    = data["X_test_requirements"].tolist()
y_test                 = torch.tensor(data["y_test"], dtype=torch.long)
D_TAB                  = X_test_static.shape[1]

print(f"Test samples: {len(y_test)} | Classes: {label_classes.tolist()}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)


class JobDataset(Dataset):
    def __init__(self, X_static, X_desc, X_profile, X_reqs, y, max_len=512):
        self.X_static  = X_static
        self.X_desc    = X_desc
        self.X_profile = X_profile
        self.X_reqs    = X_reqs
        self.y         = y
        self.max_len   = max_len

    def __len__(self):
        return len(self.y)

    def _enc(self, text):
        return tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        enc_desc    = self._enc(self.X_desc[idx])
        enc_profile = self._enc(self.X_profile[idx])
        enc_reqs    = self._enc(self.X_reqs[idx])
        return {
            "desc_ids":     enc_desc["input_ids"].squeeze(0),
            "desc_mask":    enc_desc["attention_mask"].squeeze(0),
            "profile_ids":  enc_profile["input_ids"].squeeze(0),
            "profile_mask": enc_profile["attention_mask"].squeeze(0),
            "reqs_ids":     enc_reqs["input_ids"].squeeze(0),
            "reqs_mask":    enc_reqs["attention_mask"].squeeze(0),
            "X_static":     self.X_static[idx],
            "label":        self.y[idx],
        }


test_loader = DataLoader(
    JobDataset(X_test_static,
               X_test_description, X_test_company_profile, X_test_requirements,
               y_test),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
)

# ---------------------------------------------------------------------------
# Model  (mirrors the 3-projection architecture from hparam_search.py)
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
        enc_dim           = self.text_encoder.config.hidden_size

        def _proj():
            return nn.Sequential(
                nn.Linear(enc_dim, text_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.proj_description     = _proj()
        self.proj_company_profile = _proj()
        self.proj_requirements    = _proj()

        self.tab_head   = TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim      = 3 * text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, desc_ids, desc_mask, profile_ids, profile_mask,
                reqs_ids, reqs_mask, X_static):
        B        = desc_ids.size(0)
        all_ids  = torch.cat([desc_ids,  profile_ids,  reqs_ids],  dim=0)
        all_mask = torch.cat([desc_mask, profile_mask, reqs_mask], dim=0)
        cls_all  = self.text_encoder(
            input_ids=all_ids, attention_mask=all_mask
        ).last_hidden_state[:, 0]
        desc_emb, profile_emb, reqs_emb = cls_all.split(B, dim=0)
        desc_emb    = self.proj_description(desc_emb)
        profile_emb = self.proj_company_profile(profile_emb)
        reqs_emb    = self.proj_requirements(reqs_emb)
        tab_emb     = self.tab_head(X_static)
        return self.classifier(
            torch.cat([desc_emb, profile_emb, reqs_emb, tab_emb], dim=1)
        )


model = MultimodalClassifier(
    d_tab           = D_TAB,
    n_classes       = N_CLASSES,
    tab_hidden_dims = cfg["tab_hidden_dims"],
    text_hidden_dim = cfg["text_hidden_dim"],
    dropout         = cfg["dropout"],
)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
all_proba, all_pred, all_true = [], [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            batch["desc_ids"].to(DEVICE),     batch["desc_mask"].to(DEVICE),
            batch["profile_ids"].to(DEVICE),  batch["profile_mask"].to(DEVICE),
            batch["reqs_ids"].to(DEVICE),     batch["reqs_mask"].to(DEVICE),
            batch["X_static"].to(DEVICE),
        )
        proba = torch.softmax(logits, dim=1).cpu()
        all_proba.append(proba)
        all_pred  += proba.argmax(1).tolist()
        all_true  += batch["label"].tolist()

y_true  = np.array(all_true,  dtype=np.int32)
y_pred  = np.array(all_pred,  dtype=np.int32)
y_proba = torch.cat(all_proba).numpy().astype(np.float32)

# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------
report = classification_report(y_true, y_pred, target_names=label_classes)
print("\nClassification report:")
print(report)

os.makedirs(OUT_DIR, exist_ok=True)
report_path = os.path.join(OUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

# ---------------------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "y_true.npy"),  y_true)
np.save(os.path.join(OUT_DIR, "y_pred.npy"),  y_pred)
np.save(os.path.join(OUT_DIR, "y_proba.npy"), y_proba)

print(f"\nSaved to {OUT_DIR}/:")
print(f"  classification_report.txt")
print(f"  y_true.npy   shape={y_true.shape}")
print(f"  y_pred.npy   shape={y_pred.shape}")
print(f"  y_proba.npy  shape={y_proba.shape}")
