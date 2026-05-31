"""
feature_engineering.py
src/

Reusable functions for label encoding, train/test splitting, and TF-IDF
vectorization. Called from 02_preprocessing.ipynb and any future notebooks
that need to reload processed features.
"""

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Paths relative to this file's location (src/)
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_clean_data():
    """
    Load medquad_model_ready.csv from data/processed/ and drop focus_area.
    Returns a DataFrame with columns: question_clean, qtype, department.
    """
    path = DATA_PROCESSED / "medquad_model_ready.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["focus_area"])
    return df


# ---------------------------------------------------------------------------
# 2. Encode labels
# ---------------------------------------------------------------------------

def encode_labels(df):
    """
    Fit LabelEncoder on qtype and department columns.
    Saves qtype_mapping.json and dept_mapping.json to models/.

    Parameters
    ----------
    df : DataFrame with qtype and department columns

    Returns
    -------
    df        : DataFrame with qtype_encoded and dept_encoded columns added
    qtype_enc : fitted LabelEncoder for qtype
    dept_enc  : fitted LabelEncoder for department
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    qtype_enc = LabelEncoder()
    dept_enc  = LabelEncoder()

    df = df.copy()
    df["qtype_encoded"] = qtype_enc.fit_transform(df["qtype"])
    df["dept_encoded"]  = dept_enc.fit_transform(df["department"])

    qtype_map = {int(i): c for i, c in enumerate(qtype_enc.classes_)}
    dept_map  = {int(i): c for i, c in enumerate(dept_enc.classes_)}

    with open(MODELS_DIR / "qtype_mapping.json", "w") as f:
        json.dump(qtype_map, f, indent=2)

    with open(MODELS_DIR / "dept_mapping.json", "w") as f:
        json.dump(dept_map, f, indent=2)

    return df, qtype_enc, dept_enc


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------

def split_data(X, y_qtype, y_dept, test_size=0.2, random_state=42):
    """
    Stratified 80/20 split. Stratifies on y_qtype because it has the most
    class imbalance (definition ~4600 vs prevention ~186).

    Parameters
    ----------
    X         : Series of question text
    y_qtype   : Series of encoded qtype labels
    y_dept    : Series of encoded department labels
    test_size : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    X_train, X_test, yq_train, yq_test, yd_train, yd_test
    """
    X_train, X_test, yq_train, yq_test, yd_train, yd_test = train_test_split(
        X, y_qtype, y_dept,
        test_size=test_size,
        random_state=random_state,
        stratify=y_qtype
    )
    return X_train, X_test, yq_train, yq_test, yd_train, yd_test


# ---------------------------------------------------------------------------
# 4. TF-IDF vectorization
# ---------------------------------------------------------------------------

def build_tfidf(X_train, X_test):
    """
    Fit TF-IDF on X_train only, then transform both splits.
    Never fits on X_test to prevent data leakage.

    Parameters
    ----------
    X_train : Series or array of training question text
    X_test  : Series or array of test question text

    Returns
    -------
    X_train_tfidf : sparse matrix
    X_test_tfidf  : sparse matrix
    tfidf         : fitted TfidfVectorizer
    """
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # unigrams + bigrams: captures "what causes", "how is"
        min_df=2,             # drop terms appearing in only 1 document
        max_df=0.95,          # drop terms appearing in 95%+ of documents
        sublinear_tf=True,    # log-scale TF, works better with Naive Bayes
        strip_accents="unicode",
        analyzer="word"
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf


# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------

def save_features(X_train_tfidf, X_test_tfidf,
                  yq_train, yq_test,
                  yd_train, yd_test,
                  tfidf, qtype_enc, dept_enc):
    """
    Save all processed features and model artifacts.

    data/processed/  <- X/y splits (data artifacts)
    models/          <- vectorizer and encoders (inference artifacts)
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sparse feature matrices
    sp.save_npz(str(DATA_PROCESSED / "X_train_tfidf.npz"), X_train_tfidf)
    sp.save_npz(str(DATA_PROCESSED / "X_test_tfidf.npz"),  X_test_tfidf)

    # Labels as numpy arrays
    np.save(str(DATA_PROCESSED / "yq_train.npy"), np.array(yq_train))
    np.save(str(DATA_PROCESSED / "yq_test.npy"),  np.array(yq_test))
    np.save(str(DATA_PROCESSED / "yd_train.npy"), np.array(yd_train))
    np.save(str(DATA_PROCESSED / "yd_test.npy"),  np.array(yd_test))

    # Vectorizer and label encoders
    joblib.dump(tfidf,     MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(qtype_enc, MODELS_DIR / "qtype_encoder.pkl")
    joblib.dump(dept_enc,  MODELS_DIR / "dept_encoder.pkl")

    print("Saved to data/processed/:")
    print("  X_train_tfidf.npz, X_test_tfidf.npz")
    print("  yq_train.npy, yq_test.npy, yd_train.npy, yd_test.npy")
    print("Saved to models/:")
    print("  tfidf_vectorizer.pkl, qtype_encoder.pkl, dept_encoder.pkl")
    print("  qtype_mapping.json, dept_mapping.json")


# ---------------------------------------------------------------------------
# 6. Load artifacts  (used in modeling/inference notebooks)
# ---------------------------------------------------------------------------

def load_features():
    """
    Load all saved features and artifacts from disk.
    Call this from modeling notebooks instead of re-running the pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test       : sparse matrices
        yq_train, yq_test     : numpy arrays, encoded qtype labels
        yd_train, yd_test     : numpy arrays, encoded department labels
        tfidf                 : fitted TfidfVectorizer
        qtype_enc, dept_enc   : fitted LabelEncoders
    """
    return {
        "X_train":   sp.load_npz(str(DATA_PROCESSED / "X_train_tfidf.npz")),
        "X_test":    sp.load_npz(str(DATA_PROCESSED / "X_test_tfidf.npz")),
        "yq_train":  np.load(str(DATA_PROCESSED / "yq_train.npy")),
        "yq_test":   np.load(str(DATA_PROCESSED / "yq_test.npy")),
        "yd_train":  np.load(str(DATA_PROCESSED / "yd_train.npy")),
        "yd_test":   np.load(str(DATA_PROCESSED / "yd_test.npy")),
        "tfidf":     joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl"),
        "qtype_enc": joblib.load(MODELS_DIR / "qtype_encoder.pkl"),
        "dept_enc":  joblib.load(MODELS_DIR / "dept_encoder.pkl"),
    }


# ---------------------------------------------------------------------------
# 7. Full pipeline  (called from 02_preprocessing.ipynb)
# ---------------------------------------------------------------------------

def run_pipeline():
    """
    Runs the full feature engineering pipeline end to end.
    Saves all artifacts and returns a dict of processed data.

    Usage in 02_preprocessing.ipynb:
        import sys
        sys.path.append("../src")
        from feature_engineering import run_pipeline
        data = run_pipeline()
    """
    print("Loading data ...")
    df = load_clean_data()
    print(f"  {df.shape[0]} rows loaded")

    print("Encoding labels ...")
    df, qtype_enc, dept_enc = encode_labels(df)
    print(f"  qtype classes : {list(qtype_enc.classes_)}")
    print(f"  dept classes  : {df['department'].nunique()} departments")

    print("Splitting data ...")
    X       = df["question_clean"]
    y_qtype = df["qtype_encoded"]
    y_dept  = df["dept_encoded"]

    X_train, X_test, yq_train, yq_test, yd_train, yd_test = split_data(
        X, y_qtype, y_dept
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("Building TF-IDF features ...")
    X_train_tfidf, X_test_tfidf, tfidf = build_tfidf(X_train, X_test)
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")

    print("Saving artifacts ...")
    save_features(
        X_train_tfidf, X_test_tfidf,
        yq_train, yq_test,
        yd_train, yd_test,
        tfidf, qtype_enc, dept_enc
    )

    print("\nPipeline complete.")

    return {
        "X_train":   X_train_tfidf,
        "X_test":    X_test_tfidf,
        "yq_train":  np.array(yq_train),
        "yq_test":   np.array(yq_test),
        "yd_train":  np.array(yd_train),
        "yd_test":   np.array(yd_test),
        "tfidf":     tfidf,
        "qtype_enc": qtype_enc,
        "dept_enc":  dept_enc,
    }


# ---------------------------------------------------------------------------
# Run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()