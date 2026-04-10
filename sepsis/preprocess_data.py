"""
preprocess_data.py — Split the original 3-D arrays into time-varying and static parts
for all 5 folds, reading raw data from old_data/ and writing split arrays to fold_N/.

Original layout  : (n, 24, 100)
  features  0–52  : time-varying (53 features, differ across timesteps)
  features 53–99  : static       (47 features, constant across timesteps)

Output per fold:
  fold_N/X_train_ts.npy      (n_train, 24, 53)
  fold_N/X_train_static.npy  (n_train,     47)
  fold_N/X_test_ts.npy       (n_test,  24, 53)
  fold_N/X_test_static.npy   (n_test,      47)

Usage:
    python sepsis/preprocess_data.py
"""

import numpy as np
from pathlib import Path

N_TS_FEATURES = 53   # features 0–52
OLD_DATA_DIR  = Path("sepsis/data/old_data")
FOLD_DIR      = Path("sepsis/data")


def split_and_save(X_raw: np.ndarray, out_ts: Path, out_static: Path, tag: str) -> None:
    X = X_raw.astype(np.float32)
    assert X.shape[1] == 24 and X.shape[2] == 100, f"Unexpected shape: {X.shape}"
    X_ts     = X[:, :, :N_TS_FEATURES]   # (n, 24, 53)
    X_static = X[:, 0,  N_TS_FEATURES:]  # (n, 47) — timestep 0 is representative
    np.save(out_ts,     X_ts)
    np.save(out_static, X_static)
    print(f"  [{tag}]  ts: {X_ts.shape} -> {out_ts}  static: {X_static.shape} -> {out_static}")


for fold in range(5):
    src = OLD_DATA_DIR / f"fold_{fold}"
    dst = FOLD_DIR / f"fold_{fold}"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"fold_{fold}:")

    if fold == 0:
        # fold_0 uses plain X_train.npy / X_test.npy
        for split in ("train", "test"):
            X_raw = np.load(src / f"X_{split}.npy", allow_pickle=True)
            split_and_save(X_raw,
                           dst / f"X_{split}_ts.npy",
                           dst / f"X_{split}_static.npy",
                           split)
            y = np.load(src / f"y_{split}.npy", allow_pickle=True).astype(np.float32)
            np.save(dst / f"y_{split}.npy", y)
            print(f"  [y_{split}]  shape={y.shape}")
    else:
        # folds 1-4 use X_train_fold.npy / X_test_fold.npy
        for split in ("train", "test"):
            X_raw = np.load(src / f"X_{split}_fold.npy", allow_pickle=True)
            split_and_save(X_raw,
                           dst / f"X_{split}_ts.npy",
                           dst / f"X_{split}_static.npy",
                           split)
            y = np.load(src / f"y_{split}_fold.npy", allow_pickle=True).astype(np.float32)
            np.save(dst / f"y_{split}.npy", y)
            print(f"  [y_{split}]  shape={y.shape}")

print("Done.")
