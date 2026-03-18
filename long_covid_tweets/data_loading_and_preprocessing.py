# %% Imports
import ast

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% Load dataset
ds = load_dataset("Santarabantoosoo/italian_long_covid_tweets")

# %% Inspect splits
for split, data in ds.items():
    print(f"{split}: {data.num_rows} rows, columns: {data.column_names}")

# %% Convert to pandas and preview
df_train = ds["train"].to_pandas()
df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train["emotion"])
df_train.head()

# %% Missing values (%)
(df_train.isnull().mean() * 100).round(2).rename("missing_%")

# %% Drop columns with >50% missing
df_train = df_train.dropna(thresh=len(df_train) * 0.5, axis=1)
df_train.shape

# %% Unique values per column
df_train.nunique().rename("n_unique")

# %% Drop constant columns (only 1 unique value)
df_train = df_train.loc[:, df_train.nunique() > 1]
df_train.shape

# %% Drop non-informative identifier columns
cols_to_drop = ["Unnamed: 0.1", "Unnamed: 0", "conversation_id",
                'date', 'time', "timezone", "username", "name", "place", "link"]
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
df_train.shape

# %% Remaining columns
df_train.columns.tolist()

# %% Apply same columns to test set
df_test = df_test[df_train.columns]
df_test.shape

# %% Parse created_at as datetime
# Format: "2022-04-12 14:48:06 India Standard Time" — strip the timezone name and parse
for df in [df_train, df_test]:
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"].str[:19], format="%Y-%m-%d %H:%M:%S")

df_train["created_at"].head()

# %% Parse and standardize hashtags
def parse_hashtags(val):
    if not val:
        return []
    if isinstance(val, str):
        val = ast.literal_eval(val)
    return [h.strip().lstrip("#").lower() for h in val if isinstance(h, str)]

for df in [df_train, df_test]:
    if "hashtags" in df.columns:
        df["hashtags"] = df["hashtags"].apply(parse_hashtags)

df_train["hashtags"].head()

# %% Encode hashtags: count + top-5 binary features (fit on train only)
from collections import Counter

TOP_K = 5
all_tags = [tag for tags in df_train["hashtags"] for tag in tags]
top_hashtags = [tag for tag, _ in Counter(all_tags).most_common(TOP_K)]
print("Top hashtags:", top_hashtags)

for df in [df_train, df_test]:
    df["hashtag_count"] = df["hashtags"].apply(len)
    for tag in top_hashtags:
        df[f"hashtag_{tag}"] = df["hashtags"].apply(lambda tags: int(tag in tags))
    df.drop(columns=["hashtags"], inplace=True)

df_train.head(50)

# %% Parse and encode URLs: count + top-5 domain binary features
from urllib.parse import urlparse

def parse_urls(val):
    if not val:
        return []
    if isinstance(val, str):
        val = ast.literal_eval(val)
    return [u for u in val if isinstance(u, str)]

def extract_domain(url):
    return urlparse(url).netloc.removeprefix("www.")

MAX_DOMAINS_PER_ROW = 1

# Fit domain frequency on train only
all_domains = [extract_domain(u) for urls in df_train["urls"].apply(parse_urls) for u in urls]
domain_freq = Counter(all_domains)

def row_domains(urls):
    # unique domains per row, ranked by global frequency, capped at MAX_DOMAINS_PER_ROW
    seen = dict.fromkeys(extract_domain(u) for u in urls if u)
    ranked = sorted(seen, key=lambda d: -domain_freq.get(d, 0))
    return ranked[:MAX_DOMAINS_PER_ROW]

for df in [df_train, df_test]:
    if "urls" not in df.columns:
        continue
    parsed = df["urls"].apply(parse_urls)
    df["url_count"] = parsed.apply(len)
    domains_per_row = parsed.apply(row_domains)
    for i in range(MAX_DOMAINS_PER_ROW):
        df[f"url_domain_{i+1}"] = domains_per_row.apply(lambda ds: ds[i] if i < len(ds) else None)
    df.drop(columns=["urls"], inplace=True)

df_train[["url_count"]].head()

# %% Process reply_to: extract list of ids
def extract_reply_ids(val):
    if not val:
        return []
    if isinstance(val, str):
        val = ast.literal_eval(val)
    return [d["id"] for d in val if isinstance(d, dict) and "id" in d]

for df in [df_train, df_test]:
    for col in ["reply_to", "mentions"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_reply_ids)

# %%
df_train.head(50)

#%% check how many rows have non-empty url_domain_1 column
print("DF train, url 1 not null", df_train["url_domain_1"].notna().mean())

# %% Label distribution
df_train["emotion"].value_counts()

# %% Encode label column
label_col = "emotion"
le = LabelEncoder()
df_train[label_col] = le.fit_transform(df_train[label_col])
df_test[label_col] = le.transform(df_test[label_col])

print(dict(enumerate(le.classes_)))

# %% Identify the text column
# Adjust TEXT_COL if the tweet body is stored under a different name
TEXT_COL = next((c for c in df_train.columns if c in ("tweet", "text", "content", "body")), None)
print("Text column:", TEXT_COL)
print("Remaining columns:", df_train.columns.tolist())

# %% Extract datetime features from created_at
for df in [df_train, df_test]:
    if "created_at" in df.columns:
        df["hour"]        = df["created_at"].dt.hour
        df["day_of_week"] = df["created_at"].dt.dayofweek  # 0=Mon … 6=Sun
        df["month"]       = df["created_at"].dt.month
        df.drop(columns=["created_at"], inplace=True)

# %% Convert list columns to counts
for df in [df_train, df_test]:
    for col in ["reply_to", "mentions"]:
        if col in df.columns:
            df[f"{col}_count"] = df[col].apply(len)
            df.drop(columns=[col], inplace=True)

# %% Encode url_domain_1 (top domain per tweet) — fit on train only
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

if "url_domain_1" in df_train.columns:
    domain_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train["url_domain_1"] = domain_enc.fit_transform(df_train[["url_domain_1"]]).astype(float)
    df_test["url_domain_1"]  = domain_enc.transform(df_test[["url_domain_1"]]).astype(float)

# %% Scale numeric features — fit on train only
from sklearn.preprocessing import StandardScaler

# All numeric columns except the label and binary flags
binary_prefixes = ("hashtag_",)
label_col = "emotion"

numeric_cols = [
    c for c in df_train.select_dtypes(include=[np.number]).columns
    if c != label_col and not any(c.startswith(p) for p in binary_prefixes)
]
print("Numeric cols to scale:", numeric_cols)

scaler = StandardScaler()
df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
df_test[numeric_cols]  = scaler.transform(df_test[numeric_cols])

# %% Build final arrays for the multimodal classifier
for df in [df_train, df_test]:
    df.drop(columns=["photos"], inplace=True, errors="ignore")

TAB_COLS = [c for c in df_train.columns if c not in (label_col, TEXT_COL) and TEXT_COL is not None]

# Identify columns that can't be cast to float
for col in TAB_COLS:
    try:
        df_train[col].astype(np.float32)
    except (ValueError, TypeError) as e:
        sample_bad = df_train[col][~df_train[col].apply(lambda x: pd.isna(x))].head(3).tolist()
        print(f"  BAD column '{col}': {e} | sample values: {sample_bad}")

#%% Final check: ensure all tabular columns can be converted to float
X_train_static = df_train[TAB_COLS].values.astype(np.float32)   # tabular branch
X_test_static  = df_test[TAB_COLS].values.astype(np.float32)

X_train_text   = df_train[TEXT_COL].fillna("").to_numpy()        # text branch
X_test_text    = df_test[TEXT_COL].fillna("").to_numpy()

y_train = df_train[label_col].values
y_test  = df_test[label_col].values

print("X_train_static:", X_train_static.shape)
print("X_train_text:  ", X_train_text.shape)
print("y_train:       ", y_train.shape)
print("Tabular feature names:", TAB_COLS)

# %% Save to disk
import os

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    os.path.join(OUT_DIR, "dataset.npz"),
    X_train_static=X_train_static,
    X_test_static=X_test_static,
    X_train_text=X_train_text,
    X_test_text=X_test_text,
    y_train=y_train,
    y_test=y_test,
    tab_cols=np.array(TAB_COLS),          # feature names for reference
    label_classes=le.classes_,            # int → emotion mapping
)
print("Saved to", OUT_DIR + "/dataset.npz")

# To reload later:
# data = np.load("data/dataset.npz", allow_pickle=True)
# X_train_static = data["X_train_static"]
# X_train_text   = data["X_train_text"]
# y_train        = data["y_train"]
# %%
