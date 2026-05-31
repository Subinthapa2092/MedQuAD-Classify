"""
predict.py
src/

Inference module for the trained MedQuAD classifiers.

Loads the TF-IDF vectorizer, label encoders, and both trained Naive Bayes
models from models/ and predicts the question type and medical department
for any input text.  Models are lazy-loaded on first call and cached.

Usage - command line:
    python src/predict.py "What are the symptoms of diabetes?"

Usage - Python import:
    from predict import predict, predict_batch

    result = predict("What causes high blood pressure?")
    # {'qtype': 'causes', 'department': 'Cardiology',
    #  'qtype_proba': 0.92, 'dept_proba': 0.87}
"""

import sys
import joblib
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_artifacts = None  # lazy-loaded on first call


def _load_artifacts() -> tuple:
    tfidf       = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    qtype_enc   = joblib.load(MODELS_DIR / "qtype_label_encoder.pkl")
    dept_enc    = joblib.load(MODELS_DIR / "dept_label_encoder.pkl")
    qtype_model = joblib.load(MODELS_DIR / "qtype_model.pkl")
    dept_model  = joblib.load(MODELS_DIR / "dept_model.pkl")
    return tfidf, qtype_enc, dept_enc, qtype_model, dept_model


def predict(question: str) -> dict:
    """
    Predict question type and medical department for a single question.

    Parameters
    ----------
    question : str
        Raw medical question text. No preprocessing required.

    Returns
    -------
    dict
        qtype       : str   predicted question type  (e.g. 'symptoms')
        department  : str   predicted department      (e.g. 'Cardiology')
        qtype_proba : float calibrated confidence for qtype  (0-1)
        dept_proba  : float calibrated confidence for dept   (0-1)
    """
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()

    tfidf, qtype_enc, dept_enc, qtype_model, dept_model = _artifacts

    X = tfidf.transform([question])

    qi = qtype_model.predict(X)[0]
    di = dept_model.predict(X)[0]

    return {
        "qtype":       qtype_enc.inverse_transform([qi])[0],
        "department":  dept_enc.inverse_transform([di])[0],
        "qtype_proba": round(float(qtype_model.predict_proba(X)[0][qi]), 4),
        "dept_proba":  round(float(dept_model.predict_proba(X)[0][di]), 4),
    }


def predict_batch(questions: list) -> list:
    """
    Predict question type and department for a list of questions.

    Parameters
    ----------
    questions : list of str

    Returns
    -------
    list of dicts (same structure as predict())
    """
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()

    tfidf, qtype_enc, dept_enc, qtype_model, dept_model = _artifacts

    X            = tfidf.transform(questions)
    qtype_idxs   = qtype_model.predict(X)
    dept_idxs    = dept_model.predict(X)
    qtype_probas = qtype_model.predict_proba(X)
    dept_probas  = dept_model.predict_proba(X)

    return [
        {
            "qtype":       qtype_enc.inverse_transform([qi])[0],
            "department":  dept_enc.inverse_transform([di])[0],
            "qtype_proba": round(float(qtype_probas[i][qi]), 4),
            "dept_proba":  round(float(dept_probas[i][di]), 4),
        }
        for i, (qi, di) in enumerate(zip(qtype_idxs, dept_idxs))
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"Your medical question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result   = predict(question)

    print(f"\nQuestion   : {question}")
    print(f"Type       : {result['qtype']:<15}  (confidence: {result['qtype_proba']:.1%})")
    print(f"Department : {result['department']:<30}  (confidence: {result['dept_proba']:.1%})")
