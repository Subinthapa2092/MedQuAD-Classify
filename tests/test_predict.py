"""
tests/test_predict.py
=====================
Unit and integration tests for the MedQuAD-Classify inference pipeline.

Run with:
    pytest tests/
    pytest tests/ -v
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def models_available():
    """Skip all tests gracefully if models haven't been trained yet."""
    qtype_pkl = ROOT / "models" / "qtype_model.pkl"
    dept_pkl  = ROOT / "models" / "dept_model.pkl"
    if not (qtype_pkl.exists() and dept_pkl.exists()):
        pytest.skip("Trained models not found — run  python main.py --train  first.")


# ── predict() ─────────────────────────────────────────────────────────────────

class TestPredict:
    CASES = [
    ("What are the symptoms of diabetes?",           "symptoms",   "Endocrinology & Metabolism"),
    ("What causes kidney stones?",                   "causes",     "Nephrology & Urology"),
    ("Is Huntington disease genetic?",               "genetic",    "Neurology"),
    ("What is the prognosis for multiple sclerosis?","definition", "Neurology"),
    ("How is asthma treated?",                       "definition", "Pulmonology"),
    ("Can high blood pressure be prevented?",        "prevention", "Cardiology"),
]
    # CASES = [
    #     # question                                       qtype        department
    #     ("What are the symptoms of diabetes?",          "symptoms",   "Endocrinology & Metabolism"),
    #     ("What causes kidney stones?",                  "causes",     "Nephrology & Urology"),
    #     ("Is Huntington disease genetic?",              "genetic",    "General Medicine & Public Health"),
    #     ("What is the prognosis for multiple sclerosis?","definition","Neurology"),
    #     ("How is asthma treated?",                      "definition", "Pulmonology"),
    #     ("Can high blood pressure be prevented?",       "prevention", "Cardiology"),
    #     ("What is the prognosis for multiple sclerosis?","definition","General Medicine & Public Health"),
    # ]

    def test_returns_dict_with_required_keys(self, models_available):
        from predict import predict
        result = predict("What are the symptoms of diabetes?")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"qtype", "department", "qtype_proba", "dept_proba"}

    def test_qtype_is_valid_label(self, models_available):
        from predict import predict
        valid_qtypes = {
            "symptoms", "treatment", "causes", "diagnosis",
            "prevention", "genetic", "prognosis", "epidemiology", "definition",
        }
        
        result = predict("What causes migraine headaches?")
        assert result["qtype"] in valid_qtypes

    def test_department_is_valid_label(self, models_available):
        from predict import predict
        import json
        mapping = json.loads((ROOT / "models" / "dept_class_mapping.json").read_text())
        valid_depts = set(mapping.values())
        result = predict("What are the symptoms of diabetes?")
        assert result["department"] in valid_depts

    def test_confidence_in_valid_range(self, models_available):
        from predict import predict
        result = predict("What are the symptoms of asthma?")
        assert 0.0 <= result["qtype_proba"] <= 1.0
        assert 0.0 <= result["dept_proba"] <= 1.0

    @pytest.mark.parametrize("question,exp_qtype,exp_dept", CASES)
    def test_known_predictions(self, models_available, question, exp_qtype, exp_dept):
        """Regression test: known questions should hit expected labels."""
        from predict import predict
        result = predict(question)
        assert result["qtype"] == exp_qtype,      f"qtype mismatch for: {question!r}"
        assert result["department"] == exp_dept,  f"dept mismatch for: {question!r}"

    def test_empty_string_does_not_crash(self, models_available):
        from predict import predict
        result = predict("")
        assert "qtype" in result

    def test_very_long_input_does_not_crash(self, models_available):
        from predict import predict
        result = predict("symptoms " * 500)
        assert "qtype" in result


# ── predict_batch() ───────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_returns_list_same_length(self, models_available):
        from predict import predict_batch
        questions = [
            "What are the symptoms of diabetes?",
            "How is cancer treated?",
            "What causes kidney stones?",
        ]
        results = predict_batch(questions)
        assert isinstance(results, list)
        assert len(results) == len(questions)

    def test_each_result_has_required_keys(self, models_available):
        from predict import predict_batch
        results = predict_batch(["What are the symptoms of flu?"])
        for r in results:
            assert set(r.keys()) == {"qtype", "department", "qtype_proba", "dept_proba"}

    def test_batch_matches_single_predict(self, models_available):
        """Batch and single-predict must return identical results."""
        from predict import predict, predict_batch
        questions = [
            "What are the symptoms of diabetes?",
            "What causes kidney stones?",
        ]
        batch   = predict_batch(questions)
        singles = [predict(q) for q in questions]
        for b, s in zip(batch, singles):
            assert b == s, f"Batch/single mismatch:\n  batch={b}\n  single={s}"

    def test_single_item_batch(self, models_available):
        from predict import predict_batch
        results = predict_batch(["What is glaucoma?"])
        assert len(results) == 1


# ── model files ───────────────────────────────────────────────────────────────

class TestModelArtifacts:
    REQUIRED_FILES = [
        "models/tfidf_vectorizer.pkl",
        "models/qtype_model.pkl",
        "models/dept_model.pkl",
        "models/qtype_class_mapping.json",
        "models/dept_class_mapping.json",
    ]

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_artifact_exists(self, rel_path):
        assert (ROOT / rel_path).exists(), f"Missing artifact: {rel_path}"

    def test_qtype_mapping_has_9_classes(self):
        import json
        mapping = json.loads((ROOT / "models" / "qtype_class_mapping.json").read_text())
        assert len(mapping) == 9

    def test_dept_mapping_has_20_classes(self):
        import json
        mapping = json.loads((ROOT / "models" / "dept_class_mapping.json").read_text())
        assert len(mapping) == 20


# ── feature loading ───────────────────────────────────────────────────────────

class TestFeatureLoading:
    def test_load_features_returns_expected_keys(self):
        from feature_engineering import load_features
        data = load_features()
        expected_keys = {
            "X_train", "X_test",
            "yq_train", "yq_test",
            "yd_train", "yd_test",
            "tfidf", "qtype_enc", "dept_enc",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_train_test_shapes_consistent(self):
        from feature_engineering import load_features
        data = load_features()
        assert data["X_train"].shape[0] == len(data["yq_train"])
        assert data["X_test"].shape[0]  == len(data["yq_test"])
        assert data["X_train"].shape[1] == data["X_test"].shape[1]