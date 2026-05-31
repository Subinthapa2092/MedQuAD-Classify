"""
feature_engineering.py  —  src/
============================================================
Label encoding, train/test splitting, TF-IDF vectorization.

Called by:
  - main.py  --pipeline
  - src/train.py
  - notebooks/02_preprocessing.ipynb

Encoder naming convention:
  New (canonical) : qtype_label_encoder.pkl / dept_label_encoder.pkl
  Legacy          : qtype_encoder.pkl / dept_encoder.pkl
  load_features() transparently handles both.
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


ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS    = ROOT / "data" / "processed" / "splits"
MODELS_DIR     = ROOT / "models"


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_encoder(name_new: str, name_old: str) -> Path:
    """Return path to encoder file; prefers new name, falls back to old."""
    p_new = MODELS_DIR / name_new
    p_old = MODELS_DIR / name_old
    if p_new.exists():
        return p_new
    if p_old.exists():
        return p_old
    raise FileNotFoundError(
        f"Encoder not found: tried '{name_new}' and '{name_old}' in {MODELS_DIR}"
    )


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_clean_data() -> pd.DataFrame:
    """
    Load medquad_model_ready.csv from data/processed/.
    Returns DataFrame with columns: question_clean, qtype, department.
    """
    path = DATA_PROCESSED / "medquad_model_ready.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python src/cleaning_eda.py  (or  python main.py --pipeline)  first."
        )
    df = pd.read_csv(path)
    if "focus_area" in df.columns:
        df = df.drop(columns=["focus_area"])
    return df


# ── 2. Encode labels ──────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame):
    """
    Fit LabelEncoder on qtype and department columns.
    Saves qtype_class_mapping.json, dept_class_mapping.json, and
    qtype_label_encoder.pkl, dept_label_encoder.pkl to models/.

    Returns (df_with_encoded_cols, qtype_enc, dept_enc).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    qtype_enc = LabelEncoder()
    dept_enc  = LabelEncoder()

    df = df.copy()
    df["qtype_encoded"] = qtype_enc.fit_transform(df["qtype"])
    df["dept_encoded"]  = dept_enc.fit_transform(df["department"])

    qtype_map = {int(i): c for i, c in enumerate(qtype_enc.classes_)}
    dept_map  = {int(i): c for i, c in enumerate(dept_enc.classes_)}

    with open(MODELS_DIR / "qtype_class_mapping.json", "w") as f:
        json.dump(qtype_map, f, indent=2)
    with open(MODELS_DIR / "dept_class_mapping.json", "w") as f:
        json.dump(dept_map, f, indent=2)

    # save under canonical name
    joblib.dump(qtype_enc, MODELS_DIR / "qtype_label_encoder.pkl")
    joblib.dump(dept_enc,  MODELS_DIR / "dept_label_encoder.pkl")

    return df, qtype_enc, dept_enc


# ── 3. Split ──────────────────────────────────────────────────────────────────

def split_data(X, y_qtype, y_dept, test_size=0.2, random_state=42):
    """Stratified 80/20 split on qtype (most imbalanced label)."""
    return train_test_split(
        X, y_qtype, y_dept,
        test_size=test_size,
        random_state=random_state,
        stratify=y_qtype,
    )


# ── 4. TF-IDF ─────────────────────────────────────────────────────────────────

def build_tfidf(X_train, X_test):
    """
    Fit TF-IDF on X_train only (no data leakage), then transform both.
    Returns (X_train_tfidf, X_test_tfidf, fitted_vectorizer).
    """
    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf


# ── 5. Save artifacts ─────────────────────────────────────────────────────────

def save_features(
    X_train_tfidf, X_test_tfidf,
    yq_train, yq_test,
    yd_train, yd_test,
    tfidf, qtype_enc, dept_enc,
) -> None:
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sp.save_npz(str(DATA_SPLITS / "X_train.npz"), X_train_tfidf)
    sp.save_npz(str(DATA_SPLITS / "X_test.npz"),  X_test_tfidf)

    np.save(str(DATA_SPLITS / "y_qtype_train.npy"), np.array(yq_train))
    np.save(str(DATA_SPLITS / "y_qtype_test.npy"),  np.array(yq_test))
    np.save(str(DATA_SPLITS / "y_dept_train.npy"),  np.array(yd_train))
    np.save(str(DATA_SPLITS / "y_dept_test.npy"),   np.array(yd_test))

    joblib.dump(tfidf,     MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(qtype_enc, MODELS_DIR / "qtype_label_encoder.pkl")
    joblib.dump(dept_enc,  MODELS_DIR / "dept_label_encoder.pkl")

    print("Saved → data/processed/splits/:  X_train/test.npz  y_qtype/dept_*.npy")
    print("Saved → models/:  tfidf_vectorizer.pkl  qtype/dept_label_encoder.pkl")


# ── 6. Load artifacts ─────────────────────────────────────────────────────────

def load_features() -> dict:
    """
    Load all saved feature artifacts from disk.

    Returns dict with keys:
      X_train, X_test       : sparse matrices
      yq_train, yq_test     : numpy arrays  (encoded qtype labels)
      yd_train, yd_test     : numpy arrays  (encoded dept labels)
      tfidf                 : fitted TfidfVectorizer
      qtype_enc, dept_enc   : fitted LabelEncoders
    """
    splits   = DATA_SPLITS
    models   = MODELS_DIR

    # support both old y_dept_* and new y_dept_* naming
    def _npy(name_new, name_old=""):
        p = splits / name_new
        if p.exists():
            return np.load(str(p))
        if name_old:
            p2 = splits / name_old
            if p2.exists():
                return np.load(str(p2))
        raise FileNotFoundError(f"Array file not found: {p}")

    return {
        "X_train":   sp.load_npz(str(splits / "X_train.npz")),
        "X_test":    sp.load_npz(str(splits / "X_test.npz")),
        "yq_train":  _npy("y_qtype_train.npy"),
        "yq_test":   _npy("y_qtype_test.npy"),
        "yd_train":  _npy("y_dept_train.npy",  "yd_train.npy"),
        "yd_test":   _npy("y_dept_test.npy",   "yd_test.npy"),
        "tfidf":     joblib.load(models / "tfidf_vectorizer.pkl"),
        "qtype_enc": joblib.load(_resolve_encoder("qtype_label_encoder.pkl", "qtype_encoder.pkl")),
        "dept_enc":  joblib.load(_resolve_encoder("dept_label_encoder.pkl",  "dept_encoder.pkl")),
    }


# ── 7. Full pipeline ──────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    End-to-end feature engineering pipeline.
    Saves all artifacts and returns a dict of processed data.
    """
    print("\nLoading clean data ...")
    df = load_clean_data()
    print(f"  {df.shape[0]:,} rows loaded")

    print("Encoding labels ...")
    df, qtype_enc, dept_enc = encode_labels(df)
    print(f"  qtype classes  : {list(qtype_enc.classes_)}")
    print(f"  dept  classes  : {df['department'].nunique()} departments")

    print("Splitting data (80/20 stratified on qtype) ...")
    X       = df["question_clean"]
    y_qtype = df["qtype_encoded"]
    y_dept  = df["dept_encoded"]
    X_train, X_test, yq_train, yq_test, yd_train, yd_test = split_data(X, y_qtype, y_dept)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("Building TF-IDF features (10 k unigrams + bigrams) ...")
    X_train_tfidf, X_test_tfidf, tfidf = build_tfidf(X_train, X_test)
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")

    print("Saving artifacts ...")
    save_features(
        X_train_tfidf, X_test_tfidf,
        yq_train, yq_test,
        yd_train, yd_test,
        tfidf, qtype_enc, dept_enc,
    )

    print("\nFeature engineering complete ✓")

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


if __name__ == "__main__":
    run_pipeline()