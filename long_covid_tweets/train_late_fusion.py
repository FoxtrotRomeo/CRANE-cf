"""
Late-fusion classifier for Long COVID Tweets.

Fusion strategy: each modality is processed by its own independent classifier
up to the prediction stage; the final class probability is the average of the
per-modality outputs.

  Text branch   →  classifier_text(tweet_text)    →  p_text   (n_classes,)
  Tabular branch →  classifier_tab(X_static)       →  p_tab    (n_classes,)
  Combined      →  (p_text + p_tab) / 2

NOTE: the existing MultimodalClassifier (hparam_search.py) is an
*intermediate-fusion* model.  This script adds the late-fusion counterpart.

Two classifier families are trained per modality and compared:

  Text branch
  ├── Deep:     frozen XLM-RoBERTa CLS embedding + 2-layer MLP head
  │             Saved as: data/best_model_late_fusion_mlp_text.pt
  └── Non-deep: TF-IDF (50 000 features) + LogisticRegression (class_weight=balanced)
                Saved as: data/best_model_late_fusion_tfidf_logreg_text.pkl

  Tabular branch
  ├── Deep:     TabHead MLP (same architecture as hparam_search.py)
  │             Saved as: data/best_model_late_fusion_mlp_tabular.pt
  ├── RF:       RandomForestClassifier  (class_weight=balanced)
  └── GBT:      GradientBoostingClassifier  (sample_weight balanced)
                Best tabular model selected by val macro-F1.
                Saved as: data/best_model_late_fusion_rf_tabular.pkl
                          data/best_model_late_fusion_gbt_tabular.pkl

Selection metric: macro-F1 on validation split (15 %).
Evaluation:       accuracy, macro-F1, per-class F1 on test set.

Run
---
    cd long_covid_tweets
    python train_late_fusion.py --gpu 7
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",              type=int,   default=None)
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--max-len",          type=int,   default=128)
parser.add_argument("--embed-batch-size", type=int,   default=64)
parser.add_argument("--clf-epochs",       type=int,   default=50)
parser.add_argument("--clf-lr",           type=float, default=1e-3)
parser.add_argument("--rf-n-estimators",  type=int,   default=300)
parser.add_argument("--gbt-n-estimators", type=int,   default=200)
parser.add_argument("--gbt-max-depth",    type=int,   default=5)
parser.add_argument("--tfidf-max-features", type=int, default=50_000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = "data/dataset.npz"
TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
OUT_DIR    = "data"
VAL_FRAC   = 0.15
SEED       = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} not available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_static_all = data["X_train_static"].astype("float32")
X_test_static      = data["X_test_static"].astype("float32")
X_train_text_all   = data["X_train_text"].tolist()
X_test_text        = data["X_test_text"].tolist()
y_train_all        = data["y_train"].astype(int)
y_test             = data["y_test"].astype(int)
label_classes      = data["label_classes"].tolist()

N_CLASSES = len(label_classes)
D_TAB     = X_train_static_all.shape[1]
N_TRAIN   = len(y_train_all)
print(f"Classes ({N_CLASSES}): {label_classes}")
print(f"D_TAB={D_TAB}  |  N_train={N_TRAIN}  |  N_test={len(y_test)}")

# Validation split — identical seed and fraction as hparam_search.py
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val         = int(N_TRAIN * VAL_FRAC)
val_indices   = all_idx[:n_val]
train_indices = all_idx[n_val:]

X_tr_static  = X_train_static_all[train_indices]
X_val_static = X_train_static_all[val_indices]
X_tr_text    = [X_train_text_all[i] for i in train_indices]
X_val_text   = [X_train_text_all[i] for i in val_indices]
y_tr         = y_train_all[train_indices]
y_val        = y_train_all[val_indices]

print(f"Split — train: {len(y_tr)}  val: {len(y_val)}")


# ---------------------------------------------------------------------------
# Utility: extract frozen CLS embeddings (shared with train_early_fusion.py
#          cache if it already exists)
# ---------------------------------------------------------------------------
EMB_CACHE = os.path.join(OUT_DIR, "early_fusion_text_embeddings.pt")

def _load_or_compute_embeddings():
    if os.path.exists(EMB_CACHE):
        print(f"Loading cached CLS embeddings from {EMB_CACHE}")
        cache = torch.load(EMB_CACHE, map_location="cpu")
        return cache["train"], cache["test"]

    print("Extracting frozen XLM-RoBERTa CLS embeddings …")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    enc       = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE)
    enc.eval()

    def _batch_encode(texts):
        out = []
        bs  = args.embed_batch_size
        for i in range(0, len(texts), bs):
            tok = tokenizer(
                texts[i:i+bs],
                max_length=args.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                cls = enc(
                    input_ids=tok["input_ids"].to(DEVICE),
                    attention_mask=tok["attention_mask"].to(DEVICE),
                ).last_hidden_state[:, 0]
            out.append(cls.cpu())
            if (i // bs) % 20 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
        return torch.cat(out, dim=0)

    train_emb = _batch_encode(X_train_text_all)
    test_emb  = _batch_encode(X_test_text)
    torch.save({"train": train_emb, "test": test_emb}, EMB_CACHE)
    del enc
    print(f"\nSaved to {EMB_CACHE}")
    return train_emb, test_emb


# ============================================================== TEXT BRANCH
print("\n" + "=" * 55)
print("TEXT BRANCH")
print("=" * 55)

# ---- 1a. Non-deep text: TF-IDF + Logistic Regression
print("\n-- TF-IDF + LogisticRegression --")
tfidf = TfidfVectorizer(
    max_features=args.tfidf_max_features,
    sublinear_tf=True,
    ngram_range=(1, 2),
)
X_tr_tfidf  = tfidf.fit_transform(X_tr_text)
X_val_tfidf = tfidf.transform(X_val_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Search over a small set of C values
best_lr_c, best_lr_vf1 = None, -1.0
for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
    lr_text = LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000,
        solver="lbfgs", multi_class="multinomial", random_state=SEED,
    )
    lr_text.fit(X_tr_tfidf, y_tr)
    vf1 = f1_score(y_val, lr_text.predict(X_val_tfidf), average="macro", zero_division=0)
    print(f"  C={C:<5}  val macro-F1={vf1:.4f}")
    if vf1 > best_lr_vf1:
        best_lr_vf1  = vf1
        best_lr_c    = C
        best_lr_model = lr_text

print(f"\nBest TF-IDF+LR:  C={best_lr_c}  val macro-F1={best_lr_vf1:.4f}")
tfidf_path = os.path.join(OUT_DIR, "best_model_late_fusion_tfidf_logreg_text.pkl")
with open(tfidf_path, "wb") as fh:
    pickle.dump({"tfidf": tfidf, "model": best_lr_model,
                 "label_classes": label_classes}, fh)

p_text_tfidf_test = best_lr_model.predict_proba(X_test_tfidf).astype("float32")
print(f"Test results — TF-IDF+LR text branch:")
print(classification_report(y_test, p_text_tfidf_test.argmax(1), target_names=label_classes))

# ---- 1b. Deep text: frozen CLS + MLP head
print("\n-- Frozen CLS + MLP head --")
train_emb_all, test_emb = _load_or_compute_embeddings()
train_emb_all = train_emb_all.numpy().astype("float32")
test_emb      = test_emb.numpy().astype("float32")

X_tr_emb  = train_emb_all[train_indices]
X_val_emb = train_emb_all[val_indices]

class TextMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)

text_mlp = TextMLP(768, N_CLASSES, hidden=128, dropout=0.3).to(DEVICE)
opt_text  = torch.optim.AdamW(text_mlp.parameters(), lr=args.clf_lr, weight_decay=1e-4)
crit      = nn.CrossEntropyLoss()

tr_emb_ds  = TensorDataset(torch.tensor(X_tr_emb),  torch.tensor(y_tr,  dtype=torch.long))
val_emb_ds = TensorDataset(torch.tensor(X_val_emb), torch.tensor(y_val, dtype=torch.long))
tr_emb_dl  = DataLoader(tr_emb_ds,  batch_size=256, shuffle=True)
val_emb_dl = DataLoader(val_emb_ds, batch_size=256, shuffle=False)

best_text_vf1, best_text_state = 0.0, None
for epoch in range(1, args.clf_epochs + 1):
    text_mlp.train()
    for xb, yb in tr_emb_dl:
        loss = crit(text_mlp(xb.to(DEVICE)), yb.to(DEVICE))
        opt_text.zero_grad(); loss.backward(); opt_text.step()
    text_mlp.eval()
    preds_v, labs_v = [], []
    with torch.no_grad():
        for xb, yb in val_emb_dl:
            preds_v += text_mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
            labs_v  += yb.tolist()
    vf1 = f1_score(labs_v, preds_v, average="macro", zero_division=0)
    if vf1 > best_text_vf1:
        best_text_vf1   = vf1
        best_text_state = {k: v.cpu().clone() for k, v in text_mlp.state_dict().items()}

print(f"Best val macro-F1 (text MLP): {best_text_vf1:.4f}")
text_mlp_path = os.path.join(OUT_DIR, "best_model_late_fusion_mlp_text.pt")
torch.save({"state_dict": best_text_state, "label_classes": label_classes,
            "hidden": 128, "dropout": 0.3}, text_mlp_path)

text_mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_text_state.items()})
text_mlp.eval()
with torch.no_grad():
    p_text_mlp_test = torch.softmax(
        text_mlp(torch.tensor(test_emb, device=DEVICE)), dim=1
    ).cpu().numpy().astype("float32")

print(f"Test results — MLP text branch:")
print(classification_report(y_test, p_text_mlp_test.argmax(1), target_names=label_classes))


# ============================================================ TABULAR BRANCH
print("\n" + "=" * 55)
print("TABULAR BRANCH")
print("=" * 55)

# ---- 2a. RF
print("\n-- RandomForest --")
rf_tab = RandomForestClassifier(
    n_estimators=args.rf_n_estimators,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
rf_tab.fit(X_tr_static, y_tr)
rf_tab_vf1 = f1_score(y_val, rf_tab.predict(X_val_static), average="macro", zero_division=0)
print(f"Val macro-F1 (RF): {rf_tab_vf1:.4f}")
rf_tab_path = os.path.join(OUT_DIR, "best_model_late_fusion_rf_tabular.pkl")
with open(rf_tab_path, "wb") as fh:
    pickle.dump({"model": rf_tab, "label_classes": label_classes}, fh)
p_tab_rf_test = rf_tab.predict_proba(X_test_static).astype("float32")

# ---- 2b. GBT
print("-- GradientBoosting --")
sw = compute_sample_weight("balanced", y_tr)
gbt_tab = GradientBoostingClassifier(
    n_estimators=args.gbt_n_estimators,
    max_depth=args.gbt_max_depth,
    learning_rate=0.1,
    random_state=SEED,
)
gbt_tab.fit(X_tr_static, y_tr, sample_weight=sw)
gbt_tab_vf1 = f1_score(y_val, gbt_tab.predict(X_val_static), average="macro", zero_division=0)
print(f"Val macro-F1 (GBT): {gbt_tab_vf1:.4f}")
gbt_tab_path = os.path.join(OUT_DIR, "best_model_late_fusion_gbt_tabular.pkl")
with open(gbt_tab_path, "wb") as fh:
    pickle.dump({"model": gbt_tab, "label_classes": label_classes}, fh)
p_tab_gbt_test = gbt_tab.predict_proba(X_test_static).astype("float32")

# ---- 2c. Deep tabular MLP
print("-- Tabular MLP --")
class TabMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x):
        return self.net(x)

tab_mlp   = TabMLP(D_TAB, N_CLASSES, hidden=64, dropout=0.2).to(DEVICE)
opt_tab   = torch.optim.AdamW(tab_mlp.parameters(), lr=args.clf_lr, weight_decay=1e-4)
tr_tab_ds  = TensorDataset(torch.tensor(X_tr_static),  torch.tensor(y_tr,  dtype=torch.long))
val_tab_ds = TensorDataset(torch.tensor(X_val_static), torch.tensor(y_val, dtype=torch.long))
tr_tab_dl  = DataLoader(tr_tab_ds,  batch_size=256, shuffle=True)
val_tab_dl = DataLoader(val_tab_ds, batch_size=256, shuffle=False)

best_tab_vf1, best_tab_state = 0.0, None
for epoch in range(1, args.clf_epochs + 1):
    tab_mlp.train()
    for xb, yb in tr_tab_dl:
        loss = crit(tab_mlp(xb.to(DEVICE)), yb.to(DEVICE))
        opt_tab.zero_grad(); loss.backward(); opt_tab.step()
    tab_mlp.eval()
    preds_v, labs_v = [], []
    with torch.no_grad():
        for xb, yb in val_tab_dl:
            preds_v += tab_mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
            labs_v  += yb.tolist()
    vf1 = f1_score(labs_v, preds_v, average="macro", zero_division=0)
    if vf1 > best_tab_vf1:
        best_tab_vf1   = vf1
        best_tab_state = {k: v.cpu().clone() for k, v in tab_mlp.state_dict().items()}

print(f"Best val macro-F1 (tabular MLP): {best_tab_vf1:.4f}")
tab_mlp_path = os.path.join(OUT_DIR, "best_model_late_fusion_mlp_tabular.pt")
torch.save({"state_dict": best_tab_state, "label_classes": label_classes}, tab_mlp_path)

tab_mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_tab_state.items()})
tab_mlp.eval()
with torch.no_grad():
    p_tab_mlp_test = torch.softmax(
        tab_mlp(torch.tensor(X_test_static, device=DEVICE)), dim=1
    ).cpu().numpy().astype("float32")


# ============================================================== COMBINATIONS
print("\n" + "=" * 55)
print("LATE-FUSION COMBINATIONS (average of branch probabilities)")
print("=" * 55)

# Best non-deep tabular model
best_tab_nd_name, p_tab_nd_test = max(
    [("rf",  rf_tab_vf1,  p_tab_rf_test),
     ("gbt", gbt_tab_vf1, p_tab_gbt_test)],
    key=lambda x: x[1],
)[0::2]   # name, proba

combos = {
    "tfidf_logreg_text + best_tab_nd": (p_text_tfidf_test,  p_tab_nd_test),
    "tfidf_logreg_text + tab_mlp":     (p_text_tfidf_test,  p_tab_mlp_test),
    "mlp_text + best_tab_nd":          (p_text_mlp_test,    p_tab_nd_test),
    "mlp_text + tab_mlp  (all-deep)":  (p_text_mlp_test,    p_tab_mlp_test),
}

print(f"\n  {'Combination':<42}  {'macro-F1':>9}")
print("  " + "-" * 54)
for combo_name, (p_t, p_s) in combos.items():
    p_combined = 0.5 * p_t + 0.5 * p_s
    y_pred_c   = p_combined.argmax(1)
    mf1 = f1_score(y_test, y_pred_c, average="macro", zero_division=0)
    print(f"  {combo_name:<42}  {mf1:>9.4f}")

# Save predictions for both all-non-deep and all-deep combos
for tag, p_t, p_s in [
    ("late_fusion_tfidf_logreg_nondp", p_text_tfidf_test, p_tab_nd_test),
    ("late_fusion_mlp_deep",           p_text_mlp_test,   p_tab_mlp_test),
]:
    p_c = 0.5 * p_t + 0.5 * p_s
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), p_c.astype("float32"))
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  p_c.argmax(1).astype(np.int32))
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(classification_report(y_test, p_c.argmax(1), target_names=label_classes))

print(f"\nSaved models and arrays to {OUT_DIR}/")
