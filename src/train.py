"""
train.py
src/

Entry-point for training both Naive Bayes classifiers.

Loads pre-computed TF-IDF features from data/processed/, trains two
ComplementNB models (question type and department), evaluates them on the
held-out test split, and writes results to:
  - models/qtype_model.pkl
  - models/dept_model.pkl
  - results/evaluation_report.txt

Usage:
    python src/train.py
"""

import sys
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import load_features
from model import build_qtype_model, build_dept_model, save_model

RESULTS_DIR = ROOT / "results"


def train_and_evaluate() -> None:
    print("=" * 60)
    print("MedQuAD Classifier -- Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load pre-computed features
    # ------------------------------------------------------------------
    print("\nLoading features ...")
    data = load_features()
    X_train   = data["X_train"]
    X_test    = data["X_test"]
    yq_train  = data["yq_train"]
    yq_test   = data["yq_test"]
    yd_train  = data["yd_train"]
    yd_test   = data["yd_test"]
    qtype_enc = data["qtype_enc"]
    dept_enc  = data["dept_enc"]

    print(f"  Train : {X_train.shape[0]:,} samples  |  {X_train.shape[1]:,} features")
    print(f"  Test  : {X_test.shape[0]:,} samples")

    report_lines = [
        "=" * 60,
        "MedQuAD Classifier - Evaluation Report",
        "=" * 60,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Dataset   : data/processed/medquad_model_ready.csv",
        f"Train     : {X_train.shape[0]:,} samples",
        f"Test      : {X_test.shape[0]:,} samples",
        f"Features  : {X_train.shape[1]:,} TF-IDF features (unigrams + bigrams)",
        "Algorithm : ComplementNB  (Naive Bayes, handles class imbalance)",
        "Split     : 80 / 20  stratified on question type",
    ]

    # ------------------------------------------------------------------
    # Model 1: Question Type  (9 classes)
    # ------------------------------------------------------------------
    print("\n[1/2] Training question-type classifier (ComplementNB, alpha=0.3) ...")
    qtype_model = build_qtype_model(alpha=0.3)
    qtype_model.fit(X_train, yq_train)
    save_model(qtype_model, "qtype_model.pkl")

    yq_pred  = qtype_model.predict(X_test)
    qtype_acc = accuracy_score(yq_test, yq_pred)
    qtype_f1  = f1_score(yq_test, yq_pred, average="weighted")
    print(f"  Accuracy     : {qtype_acc:.4f}")
    print(f"  F1 weighted  : {qtype_f1:.4f}")

    report_lines += [
        "",
        "=" * 60,
        "TASK 1 - Question Type Classification (9 classes)",
        "=" * 60,
        f"Model    : ComplementNB(alpha=0.3)",
        f"Accuracy : {qtype_acc:.4f}",
        f"F1 Score : {qtype_f1:.4f}  (weighted average)",
        "",
        "Per-class Classification Report:",
        classification_report(yq_test, yq_pred, target_names=qtype_enc.classes_),
    ]

    # ------------------------------------------------------------------
    # Model 2: Department  (20 classes)
    # ------------------------------------------------------------------
    print("\n[2/2] Training department classifier (ComplementNB, alpha=0.5) ...")
    dept_model = build_dept_model(alpha=0.5)
    dept_model.fit(X_train, yd_train)
    save_model(dept_model, "dept_model.pkl")

    yd_pred   = dept_model.predict(X_test)
    dept_acc  = accuracy_score(yd_test, yd_pred)
    dept_f1   = f1_score(yd_test, yd_pred, average="weighted")
    print(f"  Accuracy     : {dept_acc:.4f}")
    print(f"  F1 weighted  : {dept_f1:.4f}")

    report_lines += [
        "",
        "=" * 60,
        "TASK 2 - Medical Department Classification (20 classes)",
        "=" * 60,
        f"Model    : ComplementNB(alpha=0.5)",
        f"Accuracy : {dept_acc:.4f}",
        f"F1 Score : {dept_f1:.4f}  (weighted average)",
        "",
        "Per-class Classification Report:",
        classification_report(yd_test, yd_pred, target_names=dept_enc.classes_),
    ]

    # ------------------------------------------------------------------
    # Save evaluation report
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nEvaluation report  -> results/evaluation_report.txt")
    print("Training complete.")


if __name__ == "__main__":
    train_and_evaluate()
