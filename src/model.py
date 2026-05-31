"""
model.py
src/

Defines and builds the Naive Bayes classifiers for medical question
classification. Two separate models are trained:
  - qtype model  : classifies question type (9 classes)
  - dept model   : classifies medical department (20 classes)

Algorithm: ComplementNB wrapped in CalibratedClassifierCV
  - ComplementNB handles class imbalance better than MultinomialNB by
    computing class scores from the complement distribution (all other classes)
  - CalibratedClassifierCV (Platt scaling) corrects the raw log-sum-exp
    probabilities so that a confidence of 0.9 means the model is right ~90%
    of the time, making the output probabilities interpretable and reliable
"""

import joblib
from pathlib import Path
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def build_qtype_model(alpha: float = 0.3) -> CalibratedClassifierCV:
    """
    Return an untrained calibrated ComplementNB for question-type
    classification (9 classes).

    alpha=0.3 uses tighter additive smoothing to improve recall on minority
    classes (prevention ~186 samples, prognosis ~350 samples).
    cv=5 trains 5 folds for calibration — gives reliable probability estimates.
    """
    base = ComplementNB(alpha=alpha)
    return CalibratedClassifierCV(base, cv=5, method="sigmoid")


def build_dept_model(alpha: float = 0.5) -> CalibratedClassifierCV:
    """
    Return an untrained calibrated ComplementNB for medical-department
    classification (20 classes).

    alpha=0.5 (Lidstone smoothing) balances well across 20 classes.
    Calibration is critical here: a 20-class raw ComplementNB tends to spread
    probability mass too thinly, giving misleadingly low confidence values.
    After calibration, confidence aligns with actual model accuracy (~93%).
    """
    base = ComplementNB(alpha=alpha)
    return CalibratedClassifierCV(base, cv=5, method="sigmoid")


def save_model(model: CalibratedClassifierCV, filename: str) -> None:
    """Persist a trained sklearn model to models/<filename>."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"  Saved: models/{filename}")


def load_model(filename: str) -> CalibratedClassifierCV:
    """Load a trained model from models/<filename>."""
    return joblib.load(MODELS_DIR / filename)
