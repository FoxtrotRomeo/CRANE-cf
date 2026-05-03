"""
Late-fusion classifier for Real-or-Fake Jobs.

Fusion strategy: four independent classifiers — one per modality branch — each
produce a class probability vector; the final prediction is the average of all
four vectors.

  description    →  classifier_desc(text)       →  p_desc    (2,)
  company_profile →  classifier_profile(text)    →  p_profile (2,)
  requirements   →  classifier_reqs(text)        →  p_reqs    (2,)
  tabular        →  classifier_tab(X_static)     →  p_tab     (2,)
  Combined       →  (p_desc + p_profile + p_reqs + p_tab) / 4

NOTE: the existing MultimodalClassifier (hparam_search.py) is an
*intermediate-fusion* model.  This script adds the late-fusion counterpart.

Per-branch classifier families (Option A: same text model for each field):

  Text branches (description, company_profile, requirements)
  ├── Deep:    frozen DistilBERT CLS + 2-layer MLP head
  │            Saved as: data/best_model_late_fusion_mlp_{field}.pt
  └── Non-deep: TF-IDF + LogisticRegression  (class_weight=balanced)
               Saved as: data/best_model_late_fusion_tfidf_logreg_{field}.pkl

  Tabular branch
  ├── Deep:    2-layer MLP
  │            Saved as: data/best_model_late_fusion_mlp_tabular.pt
  ├── RF:      RandomForestClassifier
  └── GBT:     GradientBoostingClassifier
               Best tabular model selected by val macro-F1.
               Saved as: data/best_model_late_fusion_rf_tabular.pkl
                         data/best_model_late_fusion_gbt_tabular.pkl

Selection metric: macro-F1 (class imbalance: ~5 % fake).

Run
---
    cd real_or_fake_jobs
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
parser.add_argument("--gpu",               type=int,   default=None)
parser.add_argument("--seed",              type=int,   default=42)
parser.add_argument("--max-len",           type=int,   default=512)
parser.add_argument("--embed-batch-size",  type=int,   default=32)
parser.add_argument("--clf-epochs",        type=int,   default=50)
parser.add_argument("--clf-lr",            type=float, default=1e-3)
parser.add_argument("--rf-n-estimators",   type=int,   default=300)
parser.add_argument("--gbt-n-estimators",  type=int,   default=200)
parser.add_argument("--gbt-max-depth",     type=int,   default=5)
parser.add_argument("--tfidf-max-features", type=int,  default=50_000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "data/dataset.npz"
TEXT_MODEL  = "distilbert-base-uncased"
OUT_DIR     = "data"
VAL_FRAC    = 0.15
SEED        = args.seed
TEXT_FIELDS = ("description", "company_profile", "requirements")

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

X_train_static_all  = data["X_train_static"].astype("float32")
X_test_static       = data["X_test_static"].astype("float32")
X_train_desc_all    = data["X_train_description"].tolist()
X_test_desc         = data["X_test_description"].tolist()
X_train_profile_all = data["X_train_company_profile"].tolist()
X_test_profile      = data["X_test_company_profile"].tolist()
X_train_reqs_all    = data["X_train_requirements"].tolist()
X_test_reqs         = data["X_test_requirements"].tolist()
y_train_all         = data["y_train"].astype(int)
y_test              = data["y_test"].astype(int)
label_classes       = data["label_classes"].tolist()

N_CLASSES = len(label_classes)
D_TAB     = X_train_static_all.shape[1]
N_TRAIN   = len(y_train_all)
print(f"Classes ({N_CLASSES}): {label_classes}  |  D_TAB={D_TAB}  |  N_train={N_TRAIN}")

# Text field lookup
TEXT_DATA = {
    "description":     (X_train_desc_all,    X_test_desc),
    "company_profile": (X_train_profile_all, X_test_profile),
    "requirements":    (X_train_reqs_all,    X_test_reqs),
}

# Validation split (identical to hparam_search.py)
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val         = int(N_TRAIN * VAL_FRAC)
val_indices   = all_idx[:n_val]
train_indices = all_idx[n_val:]

y_tr  = y_train_all[train_indices]
y_val = y_train_all[val_indices]
print(f"Split — train: {len(y_tr)}  val: {len(y_val)}")

# ---------------------------------------------------------------------------
# Frozen CLS embeddings (reuse cache from train_early_fusion.py if present)
# ---------------------------------------------------------------------------
EMB_CACHE = os.path.join(OUT_DIR, "early_fusion_text_embeddings.pt")

def _load_or_compute_cls():
    if os.path.exists(EMB_CACHE):
        print(f"Loading CLS embeddings from {EMB_CACHE}")
        c = torch.load(EMB_CACHE, map_location="cpu")
        return c["train"], c["test"]

    print("Extracting frozen DistilBERT CLS embeddings …")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    enc       = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE)
    enc.eval()

    def _encode(texts):
        out = []
        for i in range(0, len(texts), args.embed_batch_size):
            tok = tokenizer(
                texts[i:i+args.embed_batch_size],
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
        return torch.cat(out, dim=0)

    tr_embs, te_embs = {}, {}
    for field, (tr_t, te_t) in TEXT_DATA.items():
        print(f"  {field} …")
        tr_embs[field] = _encode(tr_t)
        te_embs[field] = _encode(te_t)
    torch.save({"train": tr_embs, "test": te_embs}, EMB_CACHE)
    del enc
    return tr_embs, te_embs

train_cls_embs, test_cls_embs = _load_or_compute_cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class BranchMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp_branch(X_tr, y_tr, X_val, y_val, d_in, epochs, lr):
    tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                            torch.tensor(y_tr,  dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
    tr_dl  = DataLoader(tr_ds,  batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    mlp  = BranchMLP(d_in, N_CLASSES, hidden=128, dropout=0.3).to(DEVICE)
    opt  = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_vf1, best_state = 0.0, None

    for _ in range(epochs):
        mlp.train()
        for xb, yb in tr_dl:
            loss = crit(mlp(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pv += mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
                lv += yb.tolist()
        vf1 = f1_score(lv, pv, average="macro", zero_division=0)
        if vf1 > best_vf1:
            best_vf1   = vf1
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
    return best_vf1, best_state


def predict_mlp(state, X, d_in):
    mlp = BranchMLP(d_in, N_CLASSES, hidden=128, dropout=0.3).to(DEVICE)
    mlp.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(X, dtype=torch.float32, device=DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# TEXT BRANCHES (description, company_profile, requirements)
# ---------------------------------------------------------------------------
p_text_tfidf_val   = {}   # field → (n_val, 2)
p_text_tfidf_test  = {}
p_text_mlp_val     = {}
p_text_mlp_test    = {}

for field in TEXT_FIELDS:
    tr_texts, te_texts = TEXT_DATA[field]
    tr_texts_split = [tr_texts[i] for i in train_indices]
    va_texts_split = [tr_texts[i] for i in val_indices]

    print(f"\n{'='*55}")
    print(f"TEXT BRANCH: {field}")
    print(f"{'='*55}")

    # ---- Non-deep: TF-IDF + LogReg
    tfidf = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        sublinear_tf=True, ngram_range=(1, 2),
    )
    X_tr_tf  = tfidf.fit_transform(tr_texts_split)
    X_val_tf = tfidf.transform(va_texts_split)
    X_te_tf  = tfidf.transform(te_texts)

    best_c, best_lr_m, best_lr_vf1 = None, None, -1.0
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        lr_m = LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000,
            solver="lbfgs", multi_class="multinomial", random_state=SEED,
        )
        lr_m.fit(X_tr_tf, y_tr)
        vf1 = f1_score(y_val, lr_m.predict(X_val_tf), average="macro", zero_division=0)
        if vf1 > best_lr_vf1:
            best_lr_vf1, best_c, best_lr_m = vf1, C, lr_m

    print(f"  TF-IDF+LR (C={best_c}): val macro-F1={best_lr_vf1:.4f}")
    tfidf_path = os.path.join(OUT_DIR, f"best_model_late_fusion_tfidf_logreg_{field}.pkl")
    with open(tfidf_path, "wb") as fh:
        pickle.dump({"tfidf": tfidf, "model": best_lr_m,
                     "label_classes": label_classes}, fh)

    p_text_tfidf_val[field]  = best_lr_m.predict_proba(X_val_tf).astype("float32")
    p_text_tfidf_test[field] = best_lr_m.predict_proba(X_te_tf).astype("float32")

    # ---- Deep: frozen CLS + MLP head
    X_tr_cls  = train_cls_embs[field][train_indices].numpy().astype("float32")
    X_val_cls = train_cls_embs[field][val_indices].numpy().astype("float32")
    X_te_cls  = test_cls_embs[field].numpy().astype("float32")

    vf1_mlp, state_mlp = train_mlp_branch(
        X_tr_cls, y_tr, X_val_cls, y_val, 768,
        args.clf_epochs, args.clf_lr,
    )
    print(f"  CLS+MLP: val macro-F1={vf1_mlp:.4f}")

    mlp_path = os.path.join(OUT_DIR, f"best_model_late_fusion_mlp_{field}.pt")
    torch.save({"state_dict": state_mlp, "label_classes": label_classes,
                "field": field}, mlp_path)

    p_text_mlp_val[field]  = predict_mlp(state_mlp, X_val_cls, 768)
    p_text_mlp_test[field] = predict_mlp(state_mlp, X_te_cls, 768)


# ---------------------------------------------------------------------------
# TABULAR BRANCH
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print("TABULAR BRANCH")
print(f"{'='*55}")

X_tr_tab  = X_train_static_all[train_indices]
X_val_tab = X_train_static_all[val_indices]
sw = compute_sample_weight("balanced", y_tr)

# RF
rf_tab = RandomForestClassifier(
    n_estimators=args.rf_n_estimators,
    class_weight="balanced", n_jobs=-1, random_state=SEED,
)
rf_tab.fit(X_tr_tab, y_tr)
rf_vf1 = f1_score(y_val, rf_tab.predict(X_val_tab), average="macro", zero_division=0)
print(f"  RF:  val macro-F1={rf_vf1:.4f}")
with open(os.path.join(OUT_DIR, "best_model_late_fusion_rf_tabular.pkl"), "wb") as fh:
    pickle.dump({"model": rf_tab, "label_classes": label_classes}, fh)
p_tab_rf_val  = rf_tab.predict_proba(X_val_tab).astype("float32")
p_tab_rf_test = rf_tab.predict_proba(X_test_static).astype("float32")

# GBT
gbt_tab = GradientBoostingClassifier(
    n_estimators=args.gbt_n_estimators,
    max_depth=args.gbt_max_depth,
    learning_rate=0.1, random_state=SEED,
)
gbt_tab.fit(X_tr_tab, y_tr, sample_weight=sw)
gbt_vf1 = f1_score(y_val, gbt_tab.predict(X_val_tab), average="macro", zero_division=0)
print(f"  GBT: val macro-F1={gbt_vf1:.4f}")
with open(os.path.join(OUT_DIR, "best_model_late_fusion_gbt_tabular.pkl"), "wb") as fh:
    pickle.dump({"model": gbt_tab, "label_classes": label_classes}, fh)
p_tab_gbt_val  = gbt_tab.predict_proba(X_val_tab).astype("float32")
p_tab_gbt_test = gbt_tab.predict_proba(X_test_static).astype("float32")

# Deep MLP tabular
vf1_tab_mlp, state_tab_mlp = train_mlp_branch(
    X_tr_tab, y_tr, X_val_tab, y_val, D_TAB,
    args.clf_epochs, args.clf_lr,
)
print(f"  MLP: val macro-F1={vf1_tab_mlp:.4f}")
torch.save({"state_dict": state_tab_mlp, "label_classes": label_classes},
           os.path.join(OUT_DIR, "best_model_late_fusion_mlp_tabular.pt"))
p_tab_mlp_val  = predict_mlp(state_tab_mlp, X_val_tab, D_TAB)
p_tab_mlp_test = predict_mlp(state_tab_mlp, X_test_static, D_TAB)

# Select best non-deep tabular by val F1
if rf_vf1 >= gbt_vf1:
    p_tab_nd_val, p_tab_nd_test, best_tab_nd_name = p_tab_rf_val, p_tab_rf_test, "rf"
else:
    p_tab_nd_val, p_tab_nd_test, best_tab_nd_name = p_tab_gbt_val, p_tab_gbt_test, "gbt"


# ---------------------------------------------------------------------------
# COMBINATIONS  (average of 4 branch probabilities)
# ---------------------------------------------------------------------------
def avg_proba(text_dict, p_tab):
    """Average text probabilities (3 fields) + tabular."""
    stack = list(text_dict.values()) + [p_tab]
    return np.mean(stack, axis=0)


combos = {
    "tfidf_logreg + rf_tab  (all non-deep)":
        (avg_proba(p_text_tfidf_val,  p_tab_nd_val),
         avg_proba(p_text_tfidf_test, p_tab_nd_test)),

    "tfidf_logreg + mlp_tab":
        (avg_proba(p_text_tfidf_val,  p_tab_mlp_val),
         avg_proba(p_text_tfidf_test, p_tab_mlp_test)),

    "mlp_text + rf_tab":
        (avg_proba(p_text_mlp_val,  p_tab_nd_val),
         avg_proba(p_text_mlp_test, p_tab_nd_test)),

    "mlp_text + mlp_tab    (all deep)":
        (avg_proba(p_text_mlp_val,  p_tab_mlp_val),
         avg_proba(p_text_mlp_test, p_tab_mlp_test)),
}

print("\n" + "=" * 55)
print("LATE-FUSION COMBINATIONS  (avg of 4 branch probabilities)")
print("=" * 55)
print(f"\n  {'Combination':<40}  {'val F1':>7}  {'test F1':>8}")
print("  " + "-" * 59)

for combo_name, (p_val_c, p_test_c) in combos.items():
    vf1_c = f1_score(y_val, p_val_c.argmax(1), average="macro", zero_division=0)
    tf1_c = f1_score(y_test, p_test_c.argmax(1), average="macro", zero_division=0)
    print(f"  {combo_name:<40}  {vf1_c:>7.4f}  {tf1_c:>8.4f}")

# Save arrays for the two primary combinations
for tag, p_val_c, p_test_c in [
    ("late_fusion_tfidf_logreg_nondp",
     avg_proba(p_text_tfidf_val,  p_tab_nd_val),
     avg_proba(p_text_tfidf_test, p_tab_nd_test)),
    ("late_fusion_mlp_deep",
     avg_proba(p_text_mlp_val,  p_tab_mlp_val),
     avg_proba(p_text_mlp_test, p_tab_mlp_test)),
]:
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), p_test_c.astype("float32"))
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  p_test_c.argmax(1).astype(np.int32))
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(classification_report(
            y_test, p_test_c.argmax(1), target_names=label_classes
        ))

print(f"\nSaved all models and arrays to {OUT_DIR}/")
