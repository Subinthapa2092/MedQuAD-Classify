"""
generate_figures.py
scripts/

Generates all evaluation figures for the MedQuAD classification project
and saves them to reports/figures/.

Figures produced:
  qtype_confusion_matrix.png    Confusion matrix - question type (9 classes)
  dept_confusion_matrix.png     Confusion matrix - department (20 classes)
  qtype_f1_per_class.png        Per-class F1 bar chart - question type
  dept_f1_per_class.png         Per-class F1 bar chart - department
  qtype_top_features.png        Top 10 TF-IDF features per question-type class
  confidence_distribution.png   Prediction confidence histogram (both tasks)

Usage:
    python scripts/generate_figures.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB

from src.feature_engineering import load_features
from src.model import load_model

FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10})


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
print("Loading features and models ...")
data        = load_features()
qtype_model = load_model("qtype_model.pkl")
dept_model  = load_model("dept_model.pkl")

qtype_enc  = data["qtype_enc"]
dept_enc   = data["dept_enc"]
tfidf      = data["tfidf"]
X_test     = data["X_test"]
X_train    = data["X_train"]
yq_test    = data["yq_test"]
yd_test    = data["yd_test"]
yq_train   = data["yq_train"]
yd_train   = data["yd_train"]

yq_pred = qtype_model.predict(X_test)
yd_pred = dept_model.predict(X_test)
yq_prob = qtype_model.predict_proba(X_test)
yd_prob = dept_model.predict_proba(X_test)

qtype_classes = qtype_enc.classes_
dept_classes  = dept_enc.classes_

print(f"  Test samples : {X_test.shape[0]}")


# ---------------------------------------------------------------------------
# 1. Confusion matrix - Question Type
# ---------------------------------------------------------------------------
print("Generating qtype_confusion_matrix.png ...")
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(yq_test, yq_pred)
sns.heatmap(
    cm, annot=True, fmt="d", ax=ax,
    xticklabels=qtype_classes, yticklabels=qtype_classes,
    cmap="Blues", linewidths=0.5
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Question Type Classification - Confusion Matrix", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "qtype_confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 2. Confusion matrix - Department (20 classes)
# ---------------------------------------------------------------------------
print("Generating dept_confusion_matrix.png ...")
fig, ax = plt.subplots(figsize=(17, 15))
cm = confusion_matrix(yd_test, yd_pred)
sns.heatmap(
    cm, annot=True, fmt="d", ax=ax,
    xticklabels=dept_classes, yticklabels=dept_classes,
    cmap="Blues", linewidths=0.3, annot_kws={"size": 7}
)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Medical Department Classification - Confusion Matrix", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "dept_confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 3. Per-class F1 - Question Type
# ---------------------------------------------------------------------------
print("Generating qtype_f1_per_class.png ...")
qtype_f1 = f1_score(yq_test, yq_pred, average=None, labels=range(len(qtype_classes)))
mean_f1  = np.mean(qtype_f1)

colors = ["#2196F3" if v >= mean_f1 else "#90CAF9" for v in qtype_f1]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(qtype_classes, qtype_f1, color=colors, edgecolor="white", height=0.6)
ax.set_xlim(0, 1.12)
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title("Per-class F1 Score - Question Type Classification", fontsize=13, fontweight="bold")
for bar, val in zip(bars, qtype_f1):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
            va="center", fontsize=10)
ax.axvline(x=mean_f1, color="crimson", linestyle="--", alpha=0.8, label=f"Mean F1: {mean_f1:.3f}")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "qtype_f1_per_class.png"), dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 4. Per-class F1 - Department
# ---------------------------------------------------------------------------
print("Generating dept_f1_per_class.png ...")
dept_f1 = f1_score(yd_test, yd_pred, average=None, labels=range(len(dept_classes)))
mean_f1 = np.mean(dept_f1)

colors = ["#009688" if v >= mean_f1 else "#80CBC4" for v in dept_f1]
fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(dept_classes, dept_f1, color=colors, edgecolor="white", height=0.6)
ax.set_xlim(0, 1.15)
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title("Per-class F1 Score - Medical Department Classification", fontsize=13, fontweight="bold")
for bar, val in zip(bars, dept_f1):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
            va="center", fontsize=9)
ax.axvline(x=mean_f1, color="crimson", linestyle="--", alpha=0.8, label=f"Mean F1: {mean_f1:.3f}")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "dept_f1_per_class.png"), dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 5. Top TF-IDF features per question-type class
#    Fit a plain ComplementNB (no calibration) for direct feature_log_prob_ access
# ---------------------------------------------------------------------------
print("Generating qtype_top_features.png ...")
vis_nb = ComplementNB(alpha=0.3)
vis_nb.fit(X_train, yq_train)
feature_names = np.array(tfidf.get_feature_names_out())

n_cols, n_rows = 3, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
TOP_N = 10

for idx, (class_name, ax) in enumerate(zip(qtype_classes, axes.flatten())):
    log_probs    = vis_nb.feature_log_prob_[idx]
    top_idx      = np.argsort(log_probs)[-TOP_N:][::-1]
    top_features = feature_names[top_idx]
    top_scores   = log_probs[top_idx]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, TOP_N))
    ax.barh(range(TOP_N), top_scores[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_title(class_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("log P(feature | class)", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Top 10 TF-IDF Features per Question Type Class", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "qtype_top_features.png"), dpi=120, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 6. Confidence distribution
# ---------------------------------------------------------------------------
print("Generating confidence_distribution.png ...")
qtype_confidence = np.max(yq_prob, axis=1)
dept_confidence  = np.max(yd_prob, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(qtype_confidence, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].axvline(np.mean(qtype_confidence), color="crimson", linestyle="--",
                label=f"Mean: {np.mean(qtype_confidence):.2f}")
axes[0].set_title("Confidence Distribution - Question Type", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted Probability (max class)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[0].legend()

axes[1].hist(dept_confidence, bins=30, color="#009688", edgecolor="white", alpha=0.85)
axes[1].axvline(np.mean(dept_confidence), color="crimson", linestyle="--",
                label=f"Mean: {np.mean(dept_confidence):.2f}")
axes[1].set_title("Confidence Distribution - Medical Department", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted Probability (max class)", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[1].legend()

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "confidence_distribution.png"), dpi=120)
plt.close()


print("\nAll figures saved to reports/figures/:")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}")
