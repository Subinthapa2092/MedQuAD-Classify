# Quick Start Guide

Step-by-step instructions to set up, train, and run the MedQuAD classifier.

---

## Requirements

- Python 3.10 or higher
- Git

---

## 1. Clone the Repository

```bash
git clone <repo-url>
cd MedQuAD-Classify
```

---

## 2. Create a Virtual Environment

**Windows (PowerShell / Git Bash)**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: scikit-learn, pandas, numpy, scipy, joblib, matplotlib, seaborn.

---

## 4. (Optional) Re-run Data Cleaning and Feature Engineering

The cleaned dataset and pre-computed features are already in the repository.
Skip this step unless you want to regenerate them from the raw CSV.

```bash
# Step 1 — clean raw data, extract labels, save EDA plots
python src/cleaning_eda.py

# Step 2 — encode labels, split data, build TF-IDF matrices
python src/feature_engineering.py
```

Output locations:
- `data/processed/medquad_model_ready.csv`
- `data/processed/splits/`  (X_train.npz, X_test.npz, y_*.npy)
- `models/tfidf_vectorizer.pkl`, `models/*_label_encoder.pkl`
- `reports/figures/eda_plots.png`

---

## 5. Train the Models

```bash
python src/train.py
```

Expected output:
```
============================================================
MedQuAD Classifier -- Training
============================================================

Loading features ...
  Train : 11,971 samples  |  8,607 features
  Test  : 2,993 samples

[1/2] Training question-type classifier (ComplementNB, alpha=0.3) ...
  Saved: models/qtype_model.pkl
  Accuracy     : 0.9749
  F1 weighted  : 0.9749

[2/2] Training department classifier (ComplementNB, alpha=0.5) ...
  Saved: models/dept_model.pkl
  Accuracy     : 0.9352
  F1 weighted  : 0.9339

Evaluation report  -> results/evaluation_report.txt
Training complete.
```

Trained models are saved to `models/`. The full per-class report is at
`results/evaluation_report.txt`.

---

## 6. Predict on New Questions

**Command line**

```bash
python src/predict.py "What are the symptoms of diabetes?"
```

Output:
```
Question   : What are the symptoms of diabetes?
Type       : symptoms         (confidence: 96.7%)
Department : Endocrinology & Metabolism      (confidence: 87.8%)
```

**Python API**

```python
import sys
sys.path.insert(0, "src")
from predict import predict, predict_batch

# Single question
result = predict("What causes kidney stones?")
print(result)
# {'qtype': 'causes', 'department': 'Nephrology & Urology',
#  'qtype_proba': 0.9580, 'dept_proba': 0.9720}

# Batch
questions = [
    "How is asthma treated?",
    "Can diabetes be prevented?",
    "What is the prognosis for multiple sclerosis?",
]
for r in predict_batch(questions):
    print(f"{r['qtype']:<15} | {r['department']}")
```

---

## 7. Regenerate Report Figures

```bash
python scripts/generate_figures.py
```

Saves 6 PNG files to `reports/figures/`:
- `qtype_confusion_matrix.png`
- `dept_confusion_matrix.png`
- `qtype_f1_per_class.png`
- `dept_f1_per_class.png`
- `qtype_top_features.png`
- `confidence_distribution.png`

---

## 8. Run the Modeling Notebook

Open `notebooks/03_modeling.ipynb` in Jupyter for an interactive walkthrough
of training, evaluation, and visualizations.

```bash
pip install notebook        # if not already installed
jupyter notebook
```

---

## File Output Reference

| Command | Files produced |
|---|---|
| `src/cleaning_eda.py` | `data/processed/medquad_model_ready.csv`, `reports/figures/eda_plots.png` |
| `src/feature_engineering.py` | `data/processed/splits/*.npz`, `data/processed/splits/*.npy`, `models/tfidf_vectorizer.pkl`, `models/*_label_encoder.pkl`, `models/*_class_mapping.json` |
| `src/train.py` | `models/qtype_model.pkl`, `models/dept_model.pkl`, `results/evaluation_report.txt` |
| `scripts/generate_figures.py` | `reports/figures/*.png` (6 files) |

---

## Project Documentation

- Algorithm rationale: [docs/algorithm_documentation.md](docs/algorithm_documentation.md)
- Full evaluation report: [results/evaluation_report.txt](results/evaluation_report.txt)
