"""
main.py  —  MedQuAD-Classify  unified entry point
============================================================
Usage
-----
  python main.py                          # interactive menu
  python main.py --train                  # train both models
  python main.py --predict "question"     # predict single question
  python main.py --evaluate               # print evaluation report
  python main.py --pipeline               # full pipeline (clean → features → train)
  python main.py --batch questions.txt    # predict from a text file (one per line)
"""

import sys
import argparse
from pathlib import Path

# ── make src/ importable no matter where main.py lives ─────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _models_exist() -> bool:
    """Return True only if both trained model files are present."""
    models_dir = ROOT / "models"
    return (
        (models_dir / "qtype_model.pkl").exists()
        and (models_dir / "dept_model.pkl").exists()
    )


def _features_exist() -> bool:
    """Return True if the pre-computed TF-IDF splits exist."""
    splits = ROOT / "data" / "processed" / "splits"
    return (
        (splits / "X_train.npz").exists()
        and (splits / "y_qtype_train.npy").exists()
    )


# ────────────────────────────────────────────────────────────────────────────
# actions
# ────────────────────────────────────────────────────────────────────────────

def run_train():
    """Train both classifiers (requires pre-computed features)."""
    if not _features_exist():
        print("[!] Pre-computed features not found.")
        print("    Run  python main.py --pipeline  to build them first,")
        print("    or   python src/feature_engineering.py")
        sys.exit(1)

    _banner("Training MedQuAD Classifiers")
    from train import train_and_evaluate
    train_and_evaluate()


def run_predict(question: str):
    """Predict question type and department for a single question."""
    if not _models_exist():
        print("[!] Trained models not found — run  python main.py --train  first.")
        sys.exit(1)

    from predict import predict
    result = predict(question)

    print(f"\nQuestion   : {question}")
    print(f"Type       : {result['qtype']:<18}  (confidence: {result['qtype_proba']:.1%})")
    print(f"Department : {result['department']:<35}  (confidence: {result['dept_proba']:.1%})")
    return result


def run_batch(filepath: str):
    """Predict for every line in a text file."""
    if not _models_exist():
        print("[!] Trained models not found — run  python main.py --train  first.")
        sys.exit(1)

    path = Path(filepath)
    if not path.exists():
        print(f"[!] File not found: {filepath}")
        sys.exit(1)

    questions = [q.strip() for q in path.read_text().splitlines() if q.strip()]
    if not questions:
        print("[!] No questions found in the file.")
        sys.exit(1)

    from predict import predict_batch
    results = predict_batch(questions)

    _banner(f"Batch Predictions  ({len(questions)} questions)")
    print(f"\n{'#':<4} {'Q-Type':<15} {'Conf':>6}  {'Department':<35} {'Conf':>6}")
    print("-" * 75)
    for i, (q, r) in enumerate(zip(questions, results), 1):
        print(
            f"{i:<4} {r['qtype']:<15} {r['qtype_proba']:>6.1%}  "
            f"{r['department']:<35} {r['dept_proba']:>6.1%}"
        )
        print(f"     Q: {q[:80]}")
    print()


def run_evaluate():
    """Print the saved evaluation report."""
    report_path = ROOT / "results" / "evaluation_report.txt"
    if not report_path.exists():
        print("[!] No evaluation report found — run  python main.py --train  first.")
        sys.exit(1)
    print(report_path.read_text())


def run_pipeline():
    """
    Full end-to-end pipeline:
      1. Data cleaning + EDA  (src/cleaning_eda.py)
      2. Feature engineering  (src/feature_engineering.py)
      3. Model training       (src/train.py)
    """
    _banner("Full Pipeline: clean → features → train")

    # Step 1 — cleaning / EDA
    print("\n[Step 1/3] Running data cleaning and EDA ...")
    from cleaning_eda import run_cleaning          # noqa: F401 — runs on import
    print("  Done.")

    # Step 2 — feature engineering
    print("\n[Step 2/3] Running feature engineering ...")
    from feature_engineering import run_pipeline as _fe
    _fe()

    # Step 3 — training
    print("\n[Step 3/3] Training classifiers ...")
    from train import train_and_evaluate
    train_and_evaluate()

    _banner("Pipeline complete ✓")


def interactive_menu():
    """Simple interactive menu for users who just run  python main.py."""
    _banner("MedQuAD-Classify  —  Medical Question Classifier")
    print("""
  1. Predict a single question
  2. Batch predict from a file
  3. Train models
  4. View evaluation report
  5. Run full pipeline (clean → features → train)
  6. Exit
""")
    choice = input("Select an option [1-6]: ").strip()

    if choice == "1":
        q = input("\nEnter your medical question: ").strip()
        if q:
            run_predict(q)
        else:
            print("[!] Empty question — aborting.")

    elif choice == "2":
        fp = input("\nPath to text file (one question per line): ").strip()
        run_batch(fp)

    elif choice == "3":
        run_train()

    elif choice == "4":
        run_evaluate()

    elif choice == "5":
        run_pipeline()

    elif choice == "6":
        print("Bye!")
        sys.exit(0)

    else:
        print("[!] Invalid choice.")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MedQuAD-Classify — medical question type & department predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --train
  python main.py --predict "What are the symptoms of diabetes?"
  python main.py --evaluate
  python main.py --batch questions.txt
  python main.py --pipeline
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train",    action="store_true", help="Train both classifiers")
    group.add_argument("--predict",  metavar="QUESTION",  help="Predict for a single question")
    group.add_argument("--batch",    metavar="FILE",       help="Predict for every line in FILE")
    group.add_argument("--evaluate", action="store_true", help="Print the evaluation report")
    group.add_argument("--pipeline", action="store_true", help="Run full pipeline end to end")
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.train:
        run_train()
    elif args.predict:
        run_predict(args.predict)
    elif args.batch:
        run_batch(args.batch)
    elif args.evaluate:
        run_evaluate()
    elif args.pipeline:
        run_pipeline()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()