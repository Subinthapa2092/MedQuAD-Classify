import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')



# STEP 1: Loading the Dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'raw', 'medquad.csv'))

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()
print(df.dtypes)
print()
print(df.head(3))


# STEP 2: Checking Missing Values


print("Missing values:")
print(df.isnull().sum())

print()
print("Missing %:")
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct.round(2))

print()
print("Rows with missing answer:")
print(df[df['answer'].isnull()][['question', 'source']])


# STEP 3: Removing Duplicates


print("Before:", df.shape)
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate questions:", df['question'].duplicated().sum())

df = df.drop_duplicates()
df = df.drop_duplicates(subset='question', keep='first')
df = df.reset_index(drop=True)

print("After:", df.shape)



# STEP 4: Drop Missing Rows


df = df.dropna(subset=['answer'])
df = df.dropna(subset=['focus_area'])
df = df.reset_index(drop=True)

print("Shape after dropping nulls:", df.shape)
print("Any nulls remaining?", df.isnull().any().any())
print(df.isnull().sum())



# STEP 5: Cleaning the Question Text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\(are\)|\(s\)|\(es\)', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['question_clean'] = df['question'].apply(clean_text)

for i in range(3):
    print(f"BEFORE: {df['question'][i]}")
    print(f"AFTER : {df['question_clean'][i]}")
    print()



# STEP 6: Creating the qtype Label


def extract_qtype(q):
    q = q.lower()

    if re.search(r'symptom', q):
        return 'symptoms'
    if re.search(r'treatment|therapy|treat', q):
        return 'treatment'
    if re.search(r'cause', q):
        return 'causes'
    if re.search(r'diagnos|how to test', q):
        return 'diagnosis'
    if re.search(r'prevent', q):
        return 'prevention'
    if re.search(r'genetic|inherit', q):
        return 'genetic'
    if re.search(r'risk factor', q):
        return 'risk_factors'
    if re.search(r'prognosis|outlook', q):
        return 'prognosis'
    if re.search(r'how many|statistic', q):
        return 'epidemiology'

    return 'definition'


df['qtype'] = df['question_clean'].apply(extract_qtype)

print(df['qtype'].value_counts())



# STEP 7: EDA Visualise Distributions


df['q_words'] = df['question_clean'].str.split().str.len()

print("Word count stats:")
print(df['q_words'].describe().round(1))
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = df['qtype'].value_counts()
counts.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Label distribution (qtype)')
axes[0].set_xlabel('Count')

df['q_words'].hist(bins=25, ax=axes[1], color='teal', edgecolor='white')
axes[1].set_title('Question word count')
axes[1].set_xlabel('Words per question')
axes[1].set_ylabel('Count')

plt.tight_layout()

figures_dir = os.path.join(BASE_DIR, '..', 'reports', 'figures')
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, 'eda_plots.png'), dpi=120)
plt.show()
print("Plot saved to reports/figures/eda_plots.png")



# STEP 8: Save the Clean Dataset


df_clean = df[['question_clean', 'qtype', 'source', 'focus_area']]
df_clean = df_clean[df_clean['qtype'] != 'risk_factors']

processed_dir = os.path.join(BASE_DIR, '..', 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)
df_clean.to_csv(os.path.join(processed_dir, 'medquad_cleaned.csv'), index=False)

print("Saved: medquad_cleaned.csv")
print(f"Final shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()}")
print()
print("Final qtype counts:")
print(df_clean['qtype'].value_counts())