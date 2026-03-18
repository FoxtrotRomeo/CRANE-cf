# %% Imports
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# %% Load dataset
# Source: HuggingFace hub — victor/real-or-fake-fake-jobposting-prediction
# Single "train" split with 17 900 rows; we apply our own 80/20 stratified split.
ds = load_dataset("victor/real-or-fake-fake-jobposting-prediction")
df = ds["train"].to_pandas()
print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

# %% Stratified 80/20 split on the target
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["fraudulent"]
)
df_train = df_train.copy()
df_test  = df_test.copy()
print(f"Train: {len(df_train)}  Test: {len(df_test)}")

# %% Label distribution
print(df_train["fraudulent"].value_counts())

# %% Drop non-informative identifier columns
cols_to_drop = ["job_id"]
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
df_test  = df_test.drop( columns=[c for c in cols_to_drop if c in df_test.columns])

# %% Drop columns with >70% missing
thresh = len(df_train) * 0.7
df_train = df_train.dropna(thresh=thresh, axis=1)
df_test  = df_test[[c for c in df_train.columns if c in df_test.columns]]
print("Columns after missingness filter:", df_train.columns.tolist())

# %% Drop constant columns
df_train = df_train.loc[:, df_train.nunique() > 1]
df_test  = df_test[[c for c in df_train.columns if c in df_test.columns]]

# %% Text columns — fill missing with empty string
TEXT_COLS = [c for c in ["description", "company_profile", "requirements"]
             if c in df_train.columns]
print(f"Text columns: {TEXT_COLS}")

for df in [df_train, df_test]:
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")

# %% Binary/boolean columns — already 0/1 in the raw CSV
binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
binary_cols = [c for c in binary_cols if c in df_train.columns]
print("Binary columns:", binary_cols)
for df in [df_train, df_test]:
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(float)

# %% Split location into country / state / city
# Format: "US, VA, Virginia Beach" — up to 3 comma-separated fields; any can be empty.
def split_location(val):
    if pd.isna(val) or val == "":
        return pd.Series({"location_country": None, "location_state": None, "location_city": None})
    parts = [p.strip() or None for p in str(val).split(",")]
    parts += [None] * (3 - len(parts))   # pad if fewer than 3 fields
    return pd.Series({"location_country": parts[0],
                      "location_state":   parts[1],
                      "location_city":    parts[2]})

for df in [df_train, df_test]:
    df[["location_country", "location_state", "location_city"]] = df["location"].apply(split_location)
    df.drop(columns=["location"], inplace=True)

# %% Head printing
df_train.head(50)

# %% Categorical columns — ordinal-encode, fit on train only
from sklearn.preprocessing import OrdinalEncoder

cat_cols = ["employment_type", "required_experience", "required_education",
            "industry", "function", "location_country", "location_state", "location_city",
            "department"]
cat_cols = [c for c in cat_cols if c in df_train.columns]
print("Categorical columns:", cat_cols)

cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_train[cat_cols] = cat_enc.fit_transform(df_train[cat_cols].fillna("unknown"))
df_test[cat_cols]  = cat_enc.transform(df_test[cat_cols].fillna("unknown"))

# %% Encode title via TF-IDF + TruncatedSVD (LSA) — fit on train only
# Produces 100 dense numeric features named title_svd_0 … title_svd_99.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

TITLE_SVD_DIM = 100

if "title" in df_train.columns:
    title_train = df_train["title"].fillna("").tolist()
    title_test  = df_test["title"].fillna("").tolist()

    tfidf    = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    svd      = TruncatedSVD(n_components=TITLE_SVD_DIM, random_state=42)

    title_train_svd = svd.fit_transform(tfidf.fit_transform(title_train))
    title_test_svd  = svd.transform(tfidf.transform(title_test))

    svd_cols = [f"title_svd_{i}" for i in range(TITLE_SVD_DIM)]
    df_train = pd.concat(
        [df_train.drop(columns=["title"]),
         pd.DataFrame(title_train_svd, columns=svd_cols, index=df_train.index)],
        axis=1,
    ).copy()
    df_test = pd.concat(
        [df_test.drop(columns=["title"]),
         pd.DataFrame(title_test_svd, columns=svd_cols, index=df_test.index)],
        axis=1,
    ).copy()

    print(f"Title encoded → {TITLE_SVD_DIM} SVD components "
          f"(explained variance: {svd.explained_variance_ratio_.sum():.2%})")

# %% Other text-like columns — drop (benefits not used in either branch)
other_text_cols = ["benefits"]
other_text_cols = [c for c in other_text_cols if c in df_train.columns]
for df in [df_train, df_test]:
    df.drop(columns=other_text_cols, inplace=True, errors="ignore")

# %% Salary range — extract midpoint (format "X-Y" or empty)
if "salary_range" in df_train.columns:
    def parse_salary(val):
        if pd.isna(val) or val == "":
            return np.nan
        parts = str(val).split("-")
        try:
            nums = [float(p.strip().replace(",", "")) for p in parts if p.strip()]
            return float(np.mean(nums))
        except ValueError:
            return np.nan

    for df in [df_train, df_test]:
        df["salary_midpoint"] = df["salary_range"].apply(parse_salary)
        df.drop(columns=["salary_range"], inplace=True)

    # Fill missing salary with train median (fit on train only)
    salary_median = df_train["salary_midpoint"].median()
    for df in [df_train, df_test]:
        df["salary_midpoint"] = df["salary_midpoint"].fillna(salary_median)

# %% Scale continuous numeric features — fit on train only
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

label_col = "fraudulent"

continuous_cols = [
    c for c in df_train.select_dtypes(include=[np.number]).columns
    if c != label_col and c not in binary_cols
]
print("Continuous cols to scale:", continuous_cols)

# KNN imputation fit on train only
imputer = KNNImputer()
df_train[continuous_cols] = imputer.fit_transform(df_train[continuous_cols])
df_test[continuous_cols]  = imputer.transform(df_test[continuous_cols])

scaler = StandardScaler()
df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols])
df_test[continuous_cols]  = scaler.transform(df_test[continuous_cols])

# %% Build final arrays
TAB_COLS = [c for c in df_train.columns if c not in TEXT_COLS and c != label_col]

# Verify all tabular columns can be cast to float
for col in TAB_COLS:
    try:
        df_train[col].astype(np.float32)
    except (ValueError, TypeError) as e:
        sample_bad = df_train[col].dropna().head(3).tolist()
        print(f"  BAD column '{col}': {e} | sample values: {sample_bad}")

X_train_static = df_train[TAB_COLS].values.astype(np.float32)
X_test_static  = df_test[TAB_COLS].values.astype(np.float32)

# One numpy string array per text column
X_train_description     = df_train["description"].to_numpy()
X_test_description      = df_test["description"].to_numpy()
X_train_company_profile = df_train["company_profile"].to_numpy()
X_test_company_profile  = df_test["company_profile"].to_numpy()
X_train_requirements    = df_train["requirements"].to_numpy()
X_test_requirements     = df_test["requirements"].to_numpy()

y_train = df_train[label_col].values.astype(np.int64)
y_test  = df_test[label_col].values.astype(np.int64)

label_classes = np.array(["real", "fake"])  # index 0 = real, 1 = fake

print("X_train_static:          ", X_train_static.shape)
print("X_train_description:     ", X_train_description.shape)
print("X_train_company_profile: ", X_train_company_profile.shape)
print("X_train_requirements:    ", X_train_requirements.shape)
print("y_train:                 ", y_train.shape)
print("Class balance (train):", np.bincount(y_train))
print("Tabular feature names:", TAB_COLS)

# %% Save to disk
import os

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    os.path.join(OUT_DIR, "dataset.npz"),
    X_train_static=X_train_static,
    X_test_static=X_test_static,
    X_train_description=X_train_description,
    X_test_description=X_test_description,
    X_train_company_profile=X_train_company_profile,
    X_test_company_profile=X_test_company_profile,
    X_train_requirements=X_train_requirements,
    X_test_requirements=X_test_requirements,
    y_train=y_train,
    y_test=y_test,
    tab_cols=np.array(TAB_COLS),
    label_classes=label_classes,
)
print("Saved to", OUT_DIR + "/dataset.npz")

# To reload later:
# data = np.load("data/dataset.npz", allow_pickle=True)
# X_train_static          = data["X_train_static"]
# X_train_description     = data["X_train_description"]
# X_train_company_profile = data["X_train_company_profile"]
# X_train_requirements    = data["X_train_requirements"]
# y_train                 = data["y_train"]
# %%