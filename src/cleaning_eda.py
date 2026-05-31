"""
cleaning_eda.py  —  src/
============================================================
Step 1 of the MedQuAD pipeline.

Loads raw medquad.csv, cleans question text, extracts qtype labels,
maps focus_area → medical department, runs basic EDA, and saves:
  - data/processed/medquad_model_ready.csv
  - reports/figures/eda_plots.png

Also exposes run_cleaning() so main.py --pipeline can call it directly.
"""

import os
import re
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
# src/cleaning_eda.py lives inside src/ — go one level up to reach project root
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw", "medquad.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "reports", "figures")
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "medquad_model_ready.csv")

# ── department mapping (focus_area → department) ─────────────────────────────
DEPT_MAP = {
    # Cardiology
    "Heart Attack":                       "Cardiology",
    "Coronary Artery Disease":            "Cardiology",
    "Arrhythmia":                         "Cardiology",
    "Heart Failure":                      "Cardiology",
    "High Blood Pressure":                "Cardiology",
    "Cholesterol":                        "Cardiology",
    "Peripheral Arterial Disease":        "Cardiology",
    "Atrial Fibrillation":                "Cardiology",
    # Neurology
    "Alzheimer's Disease":                "Neurology",
    "Multiple Sclerosis":                 "Neurology",
    "Parkinson's Disease":                "Neurology",
    "Huntington's Disease":               "Neurology",
    "Stroke":                             "Neurology",
    "Epilepsy":                           "Neurology",
    "Migraine":                           "Neurology",
    "ALS":                                "Neurology",
    "Traumatic Brain Injury":             "Neurology",
    # Oncology
    "Breast Cancer":                      "Oncology",
    "Lung Cancer":                        "Oncology",
    "Prostate Cancer":                    "Oncology",
    "Colorectal Cancer":                  "Oncology",
    "Leukemia":                           "Oncology",
    "Lymphoma":                           "Oncology",
    "Melanoma":                           "Oncology",
    "Ovarian Cancer":                     "Oncology",
    "Bladder Cancer":                     "Oncology",
    "Thyroid Cancer":                     "Oncology",
    "Pancreatic Cancer":                  "Oncology",
    # Endocrinology
    "Diabetes Mellitus":                  "Endocrinology & Metabolism",
    "Diabetes":                           "Endocrinology & Metabolism",
    "Thyroid Diseases":                   "Endocrinology & Metabolism",
    "Obesity":                            "Endocrinology & Metabolism",
    "Metabolic Syndrome":                 "Endocrinology & Metabolism",
    # Gastroenterology
    "Crohn's Disease":                    "Gastroenterology & Hepatology",
    "Irritable Bowel Syndrome":           "Gastroenterology & Hepatology",
    "Liver Diseases":                     "Gastroenterology & Hepatology",
    "Hepatitis":                          "Gastroenterology & Hepatology",
    "Celiac Disease":                     "Gastroenterology & Hepatology",
    "Gallstones":                         "Gastroenterology & Hepatology",
    "GERD":                               "Gastroenterology & Hepatology",
    "Pancreatitis":                       "Gastroenterology & Hepatology",
    # Pulmonology
    "Asthma":                             "Pulmonology",
    "COPD":                               "Pulmonology",
    "Pneumonia":                          "Pulmonology",
    "Sleep Apnea":                        "Pulmonology",
    "Cystic Fibrosis":                    "Pulmonology",
    "Tuberculosis":                       "Pulmonology",
    # Nephrology
    "Kidney Diseases":                    "Nephrology & Urology",
    "Kidney Failure":                     "Nephrology & Urology",
    "Kidney Stones":                      "Nephrology & Urology",
    "Bladder Diseases":                   "Nephrology & Urology",
    "Prostate Diseases":                  "Nephrology & Urology",
    # Orthopedics
    "Arthritis":                          "Orthopedics & Rheumatology",
    "Osteoporosis":                       "Orthopedics & Rheumatology",
    "Lupus":                              "Orthopedics & Rheumatology",
    "Gout":                               "Orthopedics & Rheumatology",
    "Scoliosis":                          "Orthopedics & Rheumatology",
    "Fibromyalgia":                       "Orthopedics & Rheumatology",
    # Dermatology
    "Acne":                               "Dermatology",
    "Eczema":                             "Dermatology",
    "Psoriasis":                          "Dermatology",
    "Skin Cancer":                        "Dermatology",
    "Rosacea":                            "Dermatology",
    "Alopecia":                           "Dermatology",
    # Psychiatry
    "Depression":                         "Psychiatry & Mental Health",
    "Anxiety":                            "Psychiatry & Mental Health",
    "Bipolar Disorder":                   "Psychiatry & Mental Health",
    "Schizophrenia":                      "Psychiatry & Mental Health",
    "PTSD":                               "Psychiatry & Mental Health",
    "Autism":                             "Psychiatry & Mental Health",
    "ADHD":                               "Psychiatry & Mental Health",
    # Infectious Disease
    "HIV/AIDS":                           "Infectious Disease",
    "Influenza":                          "Infectious Disease",
    "Lyme Disease":                       "Infectious Disease",
    "COVID-19":                           "Infectious Disease",
    "Malaria":                            "Infectious Disease",
    "Sexually Transmitted Diseases":      "Infectious Disease",
    # ENT
    "Hearing Disorders":                  "ENT",
    "Tinnitus":                           "ENT",
    "Sinusitis":                          "ENT",
    "Ear Infections":                     "ENT",
    # Ophthalmology
    "Glaucoma":                           "Ophthalmology",
    "Cataracts":                          "Ophthalmology",
    "Macular Degeneration":               "Ophthalmology",
    "Diabetic Eye Problems":              "Ophthalmology",
    "Retinal Disorders":                  "Ophthalmology",
    # Hematology
    "Anemia":                             "Hematology",
    "Sickle Cell Anemia":                 "Hematology",
    "Hemophilia":                         "Hematology",
    "Blood Clots":                        "Hematology",
    "Thalassemia":                        "Hematology",
    # Immunology
    "Allergies":                          "Immunology & Allergy",
    "Food Allergy":                       "Immunology & Allergy",
    "Immune System Diseases":             "Immunology & Allergy",
    # Genetics
    "Down Syndrome":                      "Genetics & Rare Diseases",
    "Turner Syndrome":                    "Genetics & Rare Diseases",
    "Marfan Syndrome":                    "Genetics & Rare Diseases",
    "Rare Diseases":                      "Genetics & Rare Diseases",
    # Pediatrics
    "Childhood Obesity":                  "Pediatrics",
    "Developmental Disabilities":         "Pediatrics",
    # Geriatrics
    "Dementia":                           "Geriatrics & Aging",
    "Falls":                              "Geriatrics & Aging",
    # Gynecology
    "Menopause":                          "Gynecology & Obstetrics",
    "Pregnancy":                          "Gynecology & Obstetrics",
    "Endometriosis":                      "Gynecology & Obstetrics",
    "Polycystic Ovary Syndrome":          "Gynecology & Obstetrics",
}

KEYWORD_DEPT = [
    # check substrings in focus_area (lower-cased)
    ("cancer",           "Oncology"),
    ("tumor",            "Oncology"),
    ("leukemia",         "Oncology"),
    ("lymphoma",         "Oncology"),
    ("heart",            "Cardiology"),
    ("cardiac",          "Cardiology"),
    ("coronary",         "Cardiology"),
    ("blood pressure",   "Cardiology"),
    ("diabetes",         "Endocrinology & Metabolism"),
    ("thyroid",          "Endocrinology & Metabolism"),
    ("liver",            "Gastroenterology & Hepatology"),
    ("hepat",            "Gastroenterology & Hepatology"),
    ("bowel",            "Gastroenterology & Hepatology"),
    ("kidney",           "Nephrology & Urology"),
    ("renal",            "Nephrology & Urology"),
    ("urology",          "Nephrology & Urology"),
    ("lung",             "Pulmonology"),
    ("pulmo",            "Pulmonology"),
    ("asthma",           "Pulmonology"),
    ("brain",            "Neurology"),
    ("neuro",            "Neurology"),
    ("stroke",           "Neurology"),
    ("alzheimer",        "Neurology"),
    ("parkinson",        "Neurology"),
    ("multiple sclerosis","Neurology"),
    ("skin",             "Dermatology"),
    ("derma",            "Dermatology"),
    ("eye",              "Ophthalmology"),
    ("vision",           "Ophthalmology"),
    ("glaucoma",         "Ophthalmology"),
    ("ear",              "ENT"),
    ("hearing",          "ENT"),
    ("sinus",            "ENT"),
    ("blood",            "Hematology"),
    ("anemia",           "Hematology"),
    ("allerg",           "Immunology & Allergy"),
    ("immune",           "Immunology & Allergy"),
    ("genetic",          "Genetics & Rare Diseases"),
    ("rare disease",     "Genetics & Rare Diseases"),
    ("down syndrome",    "Genetics & Rare Diseases"),
    ("child",            "Pediatrics"),
    ("infant",           "Pediatrics"),
    ("aging",            "Geriatrics & Aging"),
    ("dementia",         "Geriatrics & Aging"),
    ("pregnancy",        "Gynecology & Obstetrics"),
    ("menopause",        "Gynecology & Obstetrics"),
    ("ovarian",          "Gynecology & Obstetrics"),
    ("joint",            "Orthopedics & Rheumatology"),
    ("arthritis",        "Orthopedics & Rheumatology"),
    ("bone",             "Orthopedics & Rheumatology"),
    ("depression",       "Psychiatry & Mental Health"),
    ("anxiety",          "Psychiatry & Mental Health"),
    ("mental",           "Psychiatry & Mental Health"),
    ("schizophrenia",    "Psychiatry & Mental Health"),
    ("bipolar",          "Psychiatry & Mental Health"),
    ("hiv",              "Infectious Disease"),
    ("aids",             "Infectious Disease"),
    ("infect",           "Infectious Disease"),
    ("virus",            "Infectious Disease"),
    ("bacteria",         "Infectious Disease"),
]


def map_department(focus_area: str) -> str:
    """Map a focus_area string to a standardised medical department."""
    if pd.isna(focus_area):
        return "General Medicine & Public Health"
    # exact lookup first
    if focus_area in DEPT_MAP:
        return DEPT_MAP[focus_area]
    # substring search (case-insensitive)
    fa_lower = focus_area.lower()
    for keyword, dept in KEYWORD_DEPT:
        if keyword in fa_lower:
            return dept
    return "General Medicine & Public Health"


# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\(are\)|\(s\)|\(es\)", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── qtype extraction ──────────────────────────────────────────────────────────

def extract_qtype(q: str) -> str:
    q = q.lower()
    if re.search(r"symptom",           q): return "symptoms"
    if re.search(r"treatment|therapy|treat", q): return "treatment"
    if re.search(r"cause",             q): return "causes"
    if re.search(r"diagnos|how to test", q): return "diagnosis"
    if re.search(r"prevent",           q): return "prevention"
    if re.search(r"genetic|inherit",   q): return "genetic"
    if re.search(r"risk factor",       q): return "risk_factors"
    if re.search(r"prognosis|outlook", q): return "prognosis"
    if re.search(r"how many|statistic",q): return "epidemiology"
    return "definition"


# ── main cleaning function ────────────────────────────────────────────────────

def run_cleaning(raw_csv: str = RAW_CSV) -> pd.DataFrame:
    """
    Full cleaning pipeline.  Returns the cleaned DataFrame and saves
    data/processed/medquad_model_ready.csv and reports/figures/eda_plots.png.
    """
    print("=" * 60)
    print("  Step 1 — Data Cleaning & EDA")
    print("=" * 60)

    # 1. Load
    df = pd.read_csv(raw_csv)
    print(f"\nRaw shape  : {df.shape}")
    print(f"Columns    : {df.columns.tolist()}")

    # 2. Missing values
    print("\nMissing values:\n", df.isnull().sum())
    df = df.dropna(subset=["question", "answer", "focus_area"])
    df = df.reset_index(drop=True)

    # 3. Deduplicate
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="question", keep="first")
    df = df.reset_index(drop=True)
    print(f"After dedup: {df.shape}")

    # 4. Clean text
    df["question_clean"] = df["question"].apply(clean_text)

    # 5. qtype label
    df["qtype"] = df["question_clean"].apply(extract_qtype)

    # 6. department label
    df["department"] = df["focus_area"].apply(map_department)

    # 7. Drop rare qtype
    df = df[df["qtype"] != "risk_factors"].reset_index(drop=True)

    print(f"\nqtype distribution:\n{df['qtype'].value_counts()}")
    print(f"\ndepartment distribution:\n{df['department'].value_counts()}")

    # 8. EDA plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df["q_words"] = df["question_clean"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["qtype"].value_counts().plot(
        kind="barh", ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Label distribution (qtype)")
    axes[0].set_xlabel("Count")
    df["q_words"].hist(bins=25, ax=axes[1], color="teal", edgecolor="white")
    axes[1].set_title("Question word count")
    axes[1].set_xlabel("Words per question")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    eda_path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(eda_path, dpi=120)
    plt.close()
    print(f"\nEDA plot saved → {eda_path}")

    # 9. Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_out = df[["question_clean", "qtype", "focus_area", "department"]]
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")
    print(f"Final shape: {df_out.shape}")

    return df_out


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cleaning()