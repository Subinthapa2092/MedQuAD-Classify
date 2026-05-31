# MedQuAD-Classify

A Naive Bayes text classifier that takes a raw medical question and predicts two labels at once.

| Task | Classes | Test Accuracy | Weighted F1 |
|:---|:---:|:---:|:---:|
| Question Type | 9 | **97.5%** | **97.5%** |
| Medical Department | 20 | **93.5%** | **93.4%** |

**Algorithm:** `CalibratedClassifierCV(ComplementNB, cv=5)` + TF-IDF (10,000 features, unigrams + bigrams)

**Dataset:** [MedQuAD on Kaggle](https://www.kaggle.com/datasets/jpmiller/layoutlm/data) — 16,412 medical Q&A pairs from the U.S. National Institutes of Health (NIH)

## Quick Start

```bash
# Clone and set up
git clone <repo-url>
cd MedQuAD-Classify
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

# Train both models
python main.py --train

# Predict
python main.py --predict "What are the symptoms of diabetes?"
```

## Entry Point

All tasks are accessible through `main.py`:

| Command | What it does |
|:---|:---|
| `python main.py` | Interactive menu |
| `python main.py --train` | Train both classifiers |
| `python main.py --predict "..."` | Predict for a single question |
| `python main.py --batch questions.txt` | Predict for every line in a file |
| `python main.py --evaluate` | Print the saved evaluation report |
| `python main.py --pipeline` | Run full pipeline: clean → features → train |

## Project Structure

```
MedQuAD-Classify/
├── data/
│   ├── raw/
│   │   └── medquad.csv                   original dataset (16,412 rows)
│   └── processed/
│       ├── medquad_model_ready.csv        cleaned and labelled (14,964 rows)
│       └── splits/
│           ├── X_train.npz               TF-IDF matrix, training split
│           ├── X_test.npz                TF-IDF matrix, test split
│           ├── y_qtype_train.npy
│           ├── y_qtype_test.npy
│           ├── y_dept_train.npy
│           └── y_dept_test.npy
│
├── models/
│   ├── tfidf_vectorizer.pkl              fitted TfidfVectorizer
│   ├── qtype_label_encoder.pkl           LabelEncoder — question type
│   ├── dept_label_encoder.pkl            LabelEncoder — department
│   ├── qtype_class_mapping.json          integer to class name (question type)
│   ├── dept_class_mapping.json           integer to class name (department)
│   ├── qtype_model.pkl                   trained CalibratedComplementNB
│   └── dept_model.pkl                    trained CalibratedComplementNB
│
├── notebooks/
│   ├── 01_eda.ipynb                      EDA                    (person 1)
│   ├── 02_preprocessing.ipynb            feature engineering    (person 2)
│   └── 03_modeling.ipynb                 model training + eval  (person 3)
│
├── reports/
│   └── figures/
│       ├── eda_plots.png
│       ├── qtype_confusion_matrix.png
│       ├── dept_confusion_matrix.png
│       ├── qtype_f1_per_class.png
│       ├── dept_f1_per_class.png
│       ├── qtype_top_features.png
│       └── confidence_distribution.png
│
├── results/
│   └── evaluation_report.txt             full per-class classification reports
│
├── scripts/
│   └── generate_figures.py              regenerate all evaluation figures
│
├── src/
│   ├── __init__.py
│   ├── cleaning_eda.py                   data cleaning and EDA       (person 1)
│   ├── feature_engineering.py            TF-IDF pipeline             (person 2)
│   ├── model.py                          ComplementNB model builders  (person 3)
│   ├── train.py                          training script              (person 3)
│   └── predict.py                        inference module             (person 3)
│
├── streamlit/
│   └── app.py                            Streamlit UI (placeholder)
│
├── tests/
│   └── test_predict.py                   unit and integration tests
│
├── conftest.py                           pytest configuration
├── main.py                               unified CLI entry point
├── setup.py                              package setup (pip install -e .)
├── requirements.txt
├── QUICKSTART.md
└── .gitignore
```

## Data Pipeline

| Step | Script | Input | Output |
|:---:|:---|:---|:---|
| 1 | `src/cleaning_eda.py` | `data/raw/medquad.csv` | `data/processed/medquad_model_ready.csv` |
| 2 | `src/feature_engineering.py` | cleaned CSV | `data/processed/splits/*.npz / *.npy`, encoder PKLs |
| 3 | `src/train.py` | splits + encoders | `models/qtype_model.pkl`, `models/dept_model.pkl`, evaluation report |
| 4 | `src/predict.py` | trained models | `{"qtype": "...", "department": "...", "qtype_proba": ..., "dept_proba": ...}` |

Run all four steps in one command:

```bash
python main.py --pipeline
```

## Algorithm

**Why ComplementNB?**

The question-type label distribution is heavily skewed — `definition` has ~4,600 samples while `prevention` has only ~186 (a 25x ratio). ComplementNB addresses this by computing class scores from the complement distribution (all *other* classes), which reduces bias toward the majority class. It consistently outperforms MultinomialNB on imbalanced text classification tasks.

**Why CalibratedClassifierCV?**

Wrapping ComplementNB in `CalibratedClassifierCV(cv=5, method="sigmoid")` provides two benefits:

1. **Probability calibration** — Platt scaling maps raw log-scores to well-calibrated probabilities. A reported confidence of 88% means the model is correct approximately 88% of the time.
2. **Ensemble effect** — `cv=5` trains five models on different data folds and averages their predictions, improving both accuracy and robustness.

**TF-IDF configuration:**

| Parameter | Value | Reason |
|:---|:---:|:---|
| `max_features` | 10,000 | Keeps vocabulary manageable |
| `ngram_range` | (1, 2) | Captures intent phrases like "what causes", "how is" |
| `min_df` | 2 | Removes terms appearing in only one document |
| `max_df` | 0.95 | Removes near-universal terms |
| `sublinear_tf` | True | Log-scales term frequency, standard for Naive Bayes |

## Results

**Question Type — 9 classes**

| Class | Precision | Recall | F1 | Support |
|:---|:---:|:---:|:---:|:---:|
| causes | 0.99 | 0.97 | 0.98 | 132 |
| definition | 0.94 | 0.98 | 0.96 | 928 |
| diagnosis | 1.00 | 0.98 | 0.99 | 123 |
| epidemiology | 1.00 | 1.00 | 1.00 | 223 |
| genetic | 0.96 | 0.92 | 0.94 | 497 |
| prevention | 1.00 | 0.92 | 0.96 | 37 |
| prognosis | 1.00 | 0.99 | 0.99 | 70 |
| symptoms | 1.00 | 1.00 | 1.00 | 537 |
| treatment | 1.00 | 0.99 | 1.00 | 446 |
| **weighted avg** | **0.98** | **0.97** | **0.97** | **2993** |

**Medical Department — 20 classes, weighted avg F1: 0.93**

Low-support departments (Geriatrics n=8, Psychiatry n=21) show lower F1 due to limited test samples, not model failure.

## Python API

Install as an editable package from the project root:

```bash
pip install -e .
```

Then import directly from anywhere:

```python
from src.predict import predict, predict_batch

result = predict("What are the symptoms of diabetes?")
# {
#   "qtype":       "symptoms",
#   "department":  "Endocrinology & Metabolism",
#   "qtype_proba": 0.967,
#   "dept_proba":  0.878
# }

questions = [
    "How is cancer treated?",
    "What causes kidney stones?",
    "Is Huntington disease genetic?",
]
for r in predict_batch(questions):
    print(r["qtype"], "|", r["department"])
```

## Running Tests

```bash
pytest tests/ -v
```

Covers: output schema, confidence range, known-input regression, batch/single consistency, and artifact presence checks.

## Team

| Person | Responsibility | Key Files |
|:---|:---|:---|
| Person 1 | Data cleaning and EDA | `src/cleaning_eda.py`, `notebooks/01_eda.ipynb` |
| Person 2 | Feature engineering | `src/feature_engineering.py`, `notebooks/02_preprocessing.ipynb` |
| Person 3 | Model, inference, packaging | `src/model.py`, `src/train.py`, `src/predict.py`, `main.py`, `notebooks/03_modeling.ipynb` |
