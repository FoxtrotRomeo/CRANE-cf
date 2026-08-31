# CRANE

<p align="center">
  <img src="assets/CRANE.png" alt="CRANE-cf logo" width="300"/>
</p>

**Counterfactual Retrieval via Agnostic Nearest-neighbour Explanations**

`CRANE` generates counterfactual explanations for multimodal classifiers by searching for the closest opposite-class training samples in configurable distance spaces. It supports tabular, time-series, text, and image modalities, including multiple named branches of the same modality type, and provides several combination strategies.

## Authors

- **Franco Rugolon** - [ORCID 0000-0002-7693-0576](https://orcid.org/0000-0002-7693-0576)
- **Ioanna Miliou** - [ORCID 0000-0002-1357-1967](https://orcid.org/0000-0002-1357-1967)
- **Panagiotis Papapetrou** - [ORCID 0000-0002-4632-4815](https://orcid.org/0000-0002-4632-4815)

Department of Computer and Systems Sciences, Stockholm University, Sweden.

CRANE is described in *CRANE: Post-Hoc Counterfactual Retrieval for Multimodal Classifiers*. If you use this library in your research, please cite the paper (see [CITATION.cff](CITATION.cff)).

---

## Installation

Clone the repository and install the core dependencies:

```bash
git clone https://github.com/FoxtrotRomeo/CRANE-cf.git
cd CRANE-cf
pip install .
```

For text encoder support (E5, BERT, Word2Vec) and Intermediate-Fusion:

```bash
pip install ".[text,model]"
```

For image encoder support (ResNet-50, EfficientNet-B0, CLIP):

```bash
pip install ".[image]"
# For CLIP: pip install git+https://github.com/openai/CLIP.git
```

> **Note:** the helper modules (`counterfactual_helpers.py`,
> `counterfactual_evaluation_helpers.py`) must remain at the repository root
> alongside the `cf_lib/` package. When running scripts (e.g. from
> `examples/`), do so from the repository root so Python can find them.

---

## Quickstart

```python
from cf_lib import MultimodalDataset, CounterfactualLibrary
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN, ImageNN
from cf_lib.multimodal import FrankensteinNN, CombinedNN

dataset = MultimodalDataset(
    y_train=y_train,                 # np.ndarray (n_train,)  — only required field
    # Optional modalities:
    X_train_static=X_train_static,   # np.ndarray (n_train, D_static)
    X_test_static=X_test_static,     # np.ndarray (n_test,  D_static)
    X_train_ts={"labs": arr_train_labs, "meds": arr_train_meds},
    X_test_ts= {"labs": arr_test_labs,  "meds": arr_test_meds},
    X_train_texts={"report": train_reports, "notes": train_notes},
    X_test_texts={"report": test_reports, "notes": test_notes},
    X_train_images={"cxr": train_cxr},  # PIL list, ndarray, or pre-computed embeddings
    X_test_images={"cxr": test_cxr},
    primary_text_name="report",
    primary_image_name="cxr",
)

lib = CounterfactualLibrary(
    generators={
        "Tabular":       TabularNN(k=50),
        "Labs TS":       TimeSeriesNN("labs", k=50),
        "Meds TS":       TimeSeriesNN("meds", k=50),
        "Notes":         TextNN(text_name="notes", k=50, text_encoder="tfidf"),
        "Image":         ImageNN(image_name="cxr", k=50, image_encoder="resnet50"),
        "Frankenstein":  FrankensteinNN(image_encoder="resnet50"),
        "Combined":      CombinedNN(image_encoder="resnet50"),
    }
)

results = lib.generate(dataset, sample_idx=3, target_value=0)
# results -> {"Tabular": [...], "Labs TS": [...], "Image": [...], ...}
```

---

## Available generators

### Unimodal

| Class | Description |
|---|---|
| `TabularNN` | Nearest neighbours in the static/tabular feature space |
| `TimeSeriesNN` | Nearest neighbours in a named time-series space (DTW / Euclidean / LCSS) |
| `LabsNN` | Alias for `TimeSeriesNN("labs")` |
| `MedsNN` | Alias for `TimeSeriesNN("meds")` |
| `TextNN` | Nearest neighbours in a named text branch (`text_name`) using E5, BERT, MiniLM, TF-IDF, Word2Vec, BLEU, ROUGE, etc. |
| `ImageNN` | Nearest neighbours in a named image branch (`image_name`) using ResNet-50, EfficientNet-B0, ViT-B/16, CLIP ViT-B/32, or pre-computed embeddings |

### Multimodal

| Class | Description |
|---|---|
| `EarlyFusionNN` | NN in the concatenated multimodal feature space across all named tabular, TS, text, and image branches |
| `IntermediateFusionNN` | NN in a latent space provided explicitly via precomputed train/test latents or `latent_fn`; legacy Keras fallback retained |
| `FrankensteinNN` | Independent per-modality searches; assembled by combining the best neighbours from each |
| `CombinedNN` | Per-modality searches intersected by shared candidates, ranked by summed distance |

---

## Dataset format

`MultimodalDataset` is a dataclass. Only `y_train` is required; all other
modalities default to `None`. Modalities absent from the dataset are
automatically skipped by every generator.

```python
MultimodalDataset(
    y_train,                 # REQUIRED — np.ndarray (n_train,), integer labels
    X_train_static=None,     # np.ndarray (n_train, D_static)
    X_test_static=None,      # np.ndarray (n_test,  D_static)
    X_train_ts=None,         # dict[str, np.ndarray (n_train, T, D)]
    X_test_ts=None,          # dict[str, np.ndarray (n_test,  T, D)]
    X_train_tab=None,        # dict[str, np.ndarray (n_train, D)]  — named tabular blocks
    X_test_tab=None,         # dict[str, np.ndarray (n_test,  D)]
    X_train_text=None,       # array-like of strings, length n_train
    X_test_text=None,        # array-like of strings, length n_test
    X_train_texts=None,      # dict[str, array-like], named text branches
    X_test_texts=None,
    X_train_img=None,        # PIL list | ndarray (n,H,W,C) | pre-computed (n,D)
    X_test_img=None,         # same format as X_train_img
    X_train_images=None,     # dict[str, image-like], named image branches
    X_test_images=None,
    X_train_tabular=None,    # optional normalized dict incl. primary static branch
    X_test_tabular=None,
    primary_tabular_name="__primary__",
    primary_text_name="__primary__",
    primary_image_name="__primary__",
    y_test=None,             # np.ndarray (n_test,), optional
)
```

The legacy flat fields (`X_train_text`, `X_train_img`, `X_train_static`, etc.)
are still accepted. Internally the dataset normalizes them into named branch
dicts and keeps the legacy flat attributes pointed at the chosen primary branch
for backward compatibility.

For new code, prefer the named-branch fields (`X_train_texts`,
`X_train_images`, `X_train_tabular`, etc.) as the source of truth and let
`MultimodalDataset` derive the legacy flat compatibility view automatically.
Avoid passing the same primary branch through both the legacy flat field and
the named dict at the same time.

## Candidate payloads

Generators now emit named modality payloads in addition to the legacy flat
single-branch fields.

Typical candidate structure:

```python
{
    "static": ...,            # legacy primary tabular branch when present
    "tab": {...},             # named tabular branches
    "ts": {...},              # named time-series branches
    "text": "...",            # legacy primary text branch
    "text_input": ...,        # legacy primary raw text payload
    "texts": {...},           # named text branches
    "text_inputs": {...},     # named raw text payloads
    "image": ...,             # legacy primary image branch
    "image_input": ...,       # legacy primary raw image payload
    "images": {...},          # named image branches
    "image_inputs": {...},    # named raw image payloads
    "source_indices": {       # source training rows per modality branch
        "tabular": {...},
        "ts": {...},
        "text": {...},
        "image": {...},
    },
}
```

**Image input formats accepted by `ImageNN` and multimodal generators:**
- **Pre-computed embeddings** — 2-D `np.ndarray` of shape `(n, D)`, or a list of
  1-D vectors. Used as-is (no encoding step). Auto-detected when `image_encoder="precomputed"`.
- **Raw images** — list of PIL `Image` objects, 3-D ndarray `(n, H, W)` (grayscale),
  or 4-D ndarray `(n, H, W, C)`. Encoded by the chosen backbone before search.
- **Custom callable** — pass `embed_fn=my_fn` to override backbone encoding entirely.

---

## Experiments

CRANE has been applied to four multimodal classification datasets, each with
its own subdirectory containing a dataset factory, an ablation runner, and a
best-model evaluation script.

| Dataset | Subdirectory | Modalities | Task |
|---|---|---|---|
| Long COVID Tweets | `long_covid_tweets/` | Tabular + Text (Italian, XLM-RoBERTa) | Emotion classification (sadness → joy) |
| Real-or-Fake Jobs | `real_or_fake_jobs/` | Tabular + 3× Text branches (DistilBERT) | Fraud detection (fake → real) |
| Hateful Memes | `memes/` | Text + Image (CLIP / BERT) | Hate detection |
| Sepsis mortality | `sepsis/` | Tabular + Time-series (GRU) | In-hospital mortality (death → no death) |

Each per-dataset subdirectory contains:

- **`*_cf_factory.py`** — builds the `MultimodalDataset` and loads the trained
  classifier. Accepts a `fusion_strategy` argument (`"intermediate"`,
  `"early"`, or `"late"`) to select which model checkpoint and pre-computed
  predictions to use.
- **`run_cf_ablation.py`** — sweeps distance metric combinations for the
  dataset. Key flags: `--fusion-strategy {intermediate,early,late}`,
  `--k` (default 50), `--n-jobs`, `--eval-pkls` (re-evaluate saved pickles at
  multiple k values without re-running search).
- **`run_cf_for_best_models.py`** — generates counterfactuals using the
  best-performing metric combination found in the ablation.
- **`plot_cf_metrics.py`** — bar charts of the four objectives per generator.

The `examples/` directory contains generic tooling:

- **`ablation_factory_template.py`** — factory functions that load a
  `MultimodalDataset` from `.npy` files on disk, or generate a synthetic
  multimodal dataset for smoke-testing. The file-based factory supports both
  legacy single-branch files (`X_train_text.npy`) and named-branch files such
  as `X_train_text_<name>.npy` / `X_train_img_<name>.npy`.
- **`run_distance_ablation.py`** — generic CLI runner that sweeps combinations
  of distance metrics across all modalities and saves results to JSON/pickle.
  When the dataset contains multiple named text or image branches, the runner
  automatically registers one `TextNN` / `ImageNN` per branch unless the
  factory disables that behavior via backend kwargs.
- **`run_baselines.py`** — optimisation-based counterfactual baselines (DiCE,
  NICE, gradient-based TS) for all four datasets. Results are saved in the
  same schema as ablation runs for direct comparison.
- **`evaluate_k_ablation.py`** — re-evaluates saved `combo_*_results.pkl`
  files at multiple k values without re-running NN search, by slicing the
  pre-sorted candidate lists.

Run from the **repository root**:

```bash
# Smoke test with synthetic data (includes image embeddings)
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_synthetic_dataset_and_model \
  --n-jobs 2 \
  --max-combinations 5 \
  --max-samples 3

# Illustrative example: file-based factory with TS + image modalities.
# This shows the full set of CLI flags; adapt the factory kwargs to match
# the modalities actually present in your dataset.
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_dataset_and_model \
  --factory-kwargs-json '{"data_root": "/path/to/fold_0", "ts_modalities": ["labs", "meds"], "load_image": true, "image_encoder": "resnet50"}' \
  --image-encoders resnet50 \
  --image-distance-metrics cosine,euclidean \
  --n-jobs 4 \
  --output-dir ablation_runs \
  --max-samples 25

# Per-dataset ablation (example: Long COVID Tweets, intermediate fusion)
python long_covid_tweets/run_cf_ablation.py --gpu 0 --fusion-strategy intermediate

# Optimisation-based baselines
python examples/run_baselines.py --dataset sepsis --fold 0 --gpu 0
python examples/run_baselines.py --dataset long_covid_tweets --gpu 0
```

The factory functions return a dict with `"dataset"`, `"model"`,
`"text_backend_kwargs"`, and `"image_backend_kwargs"`. The runner uses
`"image_backend_kwargs"` to forward device/batch-size settings to `ImageNN`.
Encoders whose dependencies are missing (e.g. `clip_vit_b32` without the
`clip` package) are skipped automatically.

---

## Evaluation

`counterfactual_evaluation_helpers.py` computes four objectives per
candidate: `outcome`, `proximity`, `sparsity`, and `plausibility`.

- `compute_objectives(...)` now supports named `tabular_modalities`,
  `ts_modalities`, `text_modalities`, and `image_modalities`, so evaluation
  is no longer limited to a single tabular / TS / text input.
- The legacy flat arguments are still supported and are auto-promoted into
  the newer dict-based format for backward compatibility.
- Text objectives use generic embedding callables. **Proximity** is
  `(1 − cos) / 2` in embedding space (range [0, 1]). **Sparsity** is the
  normalised token-edit distance (Levenshtein / factual length); values above
  1 are possible when the counterfactual has more tokens than the original.
- Image **proximity** is cosine distance in embedding space, same formula as
  text. Image **sparsity** is the pixel-level change fraction: the proportion
  of pixel elements whose absolute difference between factual and
  counterfactual exceeds a configurable threshold (default 0, i.e. any
  nonzero change counts). Values are in [0, 1]. Image **plausibility** uses
  LOF in embedding space.
- Plausibility normalizers and LOF references can be fitted on the target
  class only, which makes plausibility scores reflect how typical a
  candidate is within the desired counterfactual class.
- `fit_proximity_normalizer(...)` computes reference pairwise distance
  distributions from the training set. Pass the result as
  `proximity_normalizer` to `compute_objectives` to express proximity
  relative to the typical within-class spread rather than as an absolute
  distance.

`examples/run_distance_ablation.py` can also attach objective summaries to
each ablation row via `objectives_kwargs` or
`objectives_kwargs_factory(text_cfg, image_cfg)`. Use `--n-jobs` to run
multiple ablation combinations concurrently when your backend resources allow it.
If parallel runs use BLAS-backed numeric kernels, prefer
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to avoid nested
thread explosions.

Factories may also provide
`text_backend_kwargs["precomputed_text_embeddings_by_encoder"]` in either of
these formats:

- `{encoder: {"train": ..., "test": ...}}` for the primary text branch
- `{text_name: {encoder: {"train": ..., "test": ...}}}` for named text branches

When present, `TextNN` and downstream multimodal generators reuse those
precomputed matrices instead of re-embedding the full train/test text corpus
for every sample or combo.

Before the combo loop, the runner also precomputes all unique unimodal NN
searches via `_prepare_unimodal_search_cache`. For each distinct
`(modality, metric/encoder)` signature, the neighbor indices, distances, and
materialized candidate dicts are computed once for the full sample batch and
stored. Each combo receives only the relevant entries as a `precomputed_seed`
passed to `CounterfactualLibrary`, so combos that share a tabular metric or
text encoder never repeat that search. This is on by default and can be
disabled with `--no-precompute-unimodal-searches`.

The runner also builds string→vector lookup tables from those precomputed
embeddings once before the combo loop. Each combo's `embed_fn` (used during
objective evaluation) is automatically wrapped to serve vectors from the lookup
with the live callable as a fallback for cache misses. This eliminates GPU
inference during the parallel objective-evaluation phase.

Factories may also set:

- `text_backend_kwargs["auto_text_branch_generators"] = False`
- `image_backend_kwargs["auto_image_branch_generators"] = False`

to keep the generic ablation runner from auto-registering per-branch text or
image generators when a project provides custom branch-aware generators.

---

## Dependencies

| Package | Role |
|---|---|
| `numpy` | Core array operations |
| `scikit-learn` | Pairwise distances, TF-IDF |
| `aeon` | DTW and other time-series distances |
| `transformers` + `torch` | E5 / BERT text encoders *(optional)* |
| `gensim` | Word2Vec text encoder *(optional)* |
| `rouge-score` + `nltk` | ROUGE / BLEU text distance metrics *(optional)* |
| `torch` + `torchvision` + `Pillow` | Image encoding with ResNet-50 / EfficientNet-B0 / ViT-B/16 *(optional)* |
| `clip` (OpenAI) | CLIP ViT-B/32 image encoder *(optional)* |
| `keras` | Loading models for `IntermediateFusionNN` *(optional)* |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Derivatives and redistributions must retain the copyright notice and cite the
original work (see [CITATION.cff](CITATION.cff)).
