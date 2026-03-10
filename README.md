# cf-lib

**Nearest-neighbour multimodal counterfactual generation library.**

`cf-lib` generates counterfactual explanations for multimodal classifiers by
searching for the closest opposite-class training samples in configurable
distance spaces. It supports tabular, time-series, and text modalities, and
provides several combination strategies.

> If you use this library in your research, please cite the associated paper
> (see [CITATION.cff](CITATION.cff)).

---

## Installation

Clone the repository and install the core dependencies:

```bash
git clone https://github.com/FoxtrotRomeo/cf-lib.git
cd cf-lib
pip install .
```

For text encoder support (E5, BERT, Word2Vec) and Intermediate-Fusion:

```bash
pip install ".[text,model]"
```

> **Note:** the helper modules (`counterfactual_helpers.py`,
> `counterfactual_evaluation_helpers.py`) must remain at the repository root
> alongside the `cf_lib/` package. When running scripts (e.g. from
> `examples/`), do so from the repository root so Python can find them.

---

## Quickstart

```python
from cf_lib import MultimodalDataset, CounterfactualLibrary
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN
from cf_lib.multimodal import FrankensteinNN, CombinedNN

dataset = MultimodalDataset(
    X_train_static=X_train_static,   # np.ndarray (n_train, D_static)
    y_train=y_train,                 # np.ndarray (n_train,)
    X_test_static=X_test_static,     # np.ndarray (n_test,  D_static)
    # Optional modalities:
    X_train_ts={"labs": arr_train_labs, "meds": arr_train_meds},
    X_test_ts= {"labs": arr_test_labs,  "meds": arr_test_meds},
    X_train_text=train_texts,
    X_test_text=test_texts,
)

lib = CounterfactualLibrary(
    generators={
        "Tabular":       TabularNN(k=50),
        "Labs TS":       TimeSeriesNN("labs", k=50),
        "Meds TS":       TimeSeriesNN("meds", k=50),
        "Frankenstein":  FrankensteinNN(),
        "Combined":      CombinedNN(),
    }
)

results = lib.generate(dataset, sample_idx=3, target_value=0)
# results -> {"Tabular": [...], "Labs TS": [...], ...}
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
| `TextNN` | Nearest neighbours in text space (E5, BERT, TF-IDF, Word2Vec, BLEU, ROUGE) |

### Multimodal

| Class | Description |
|---|---|
| `EarlyFusionNN` | NN in the concatenated multimodal feature space |
| `IntermediateFusionNN` | NN in the classifier's latent (penultimate-layer) space — requires a Keras model |
| `FrankensteinNN` | Independent per-modality searches; assembled by combining the best neighbours from each |
| `CombinedNN` | Per-modality searches intersected by shared candidates, ranked by summed distance |

---

## Dataset format

`MultimodalDataset` is a dataclass. Only static features and labels are
required; all other modalities default to `None`.

```python
MultimodalDataset(
    X_train_static,          # required — np.ndarray (n_train, D_static)
    y_train,                 # required — np.ndarray (n_train,), integer labels
    X_test_static,           # required — np.ndarray (n_test,  D_static)
    X_train_ts=None,         # dict[str, np.ndarray (n_train, T, D)]
    X_test_ts=None,          # dict[str, np.ndarray (n_test,  T, D)]
    X_train_tab=None,        # dict[str, np.ndarray (n_train, D)]  — extra tabular modalities
    X_test_tab=None,         # dict[str, np.ndarray (n_test,  D)]
    X_train_text=None,       # array-like of strings, length n_train
    X_test_text=None,        # array-like of strings, length n_test
    y_test=None,             # np.ndarray (n_test,), optional
)
```

---

## Experiments

The `examples/` directory contains two scripts:

- **`ablation_factory_template.py`** — factory functions that load a
  `MultimodalDataset` from `.npy` files on disk, or generate a synthetic
  dataset for smoke-testing.
- **`run_distance_ablation.py`** — CLI runner that sweeps combinations of
  distance metrics across all modalities and saves results to JSON/pickle.

Run from the **repository root**:

```bash
# Smoke test with synthetic data (no real data needed)
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_synthetic_dataset_and_model \
  --max-combinations 5 \
  --max-samples 3

# Real data
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_dataset_and_model \
  --factory-kwargs-json '{"data_root": "/path/to/fold_0", "ts_modalities": ["labs", "meds"]}' \
  --output-dir ablation_runs \
  --max-samples 25
```

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
| `keras` | Loading models for `IntermediateFusionNN` *(optional)* |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Derivatives and redistributions must retain the copyright notice and cite the
original work (see [CITATION.cff](CITATION.cff)).
