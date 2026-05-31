# streamlit/app.py
# Run from project root:  streamlit run streamlit/app.py

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Set CWD to project root so model file paths inside predict.py resolve correctly
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import streamlit as st

st.set_page_config(
    page_title="MedQuAD-Classify",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Figtree:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background-color: #090909;
    font-family: 'Figtree', sans-serif;
}

.main .block-container {
    background-color: #090909;
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

#MainMenu,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer {
    visibility: hidden;
    height: 0;
    padding: 0;
    margin: 0;
}

/* ── Header ─────────────────────────────────────── */
.app-header {
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #131313;
}

.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f0f0f0;
    margin: 0 0 0.55rem 0;
    line-height: 1;
}

.app-desc {
    font-size: 0.88rem;
    color: #585858;
    line-height: 1.65;
    margin: 0 0 1.1rem 0;
    max-width: 54ch;
}

.stat-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: #3c3c3c;
    background: #0e0e0e;
    border: 1px solid #1c1c1c;
    border-radius: 20px;
    padding: 0.22rem 0.65rem;
}

.stat-pill .accent { color: #4a7eff; }

/* ── Input ──────────────────────────────────────── */
.field-label {
    display: block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #404040;
    margin-bottom: 0.55rem;
}

.stTextArea > label { display: none !important; }

.stTextArea textarea {
    background-color: #0d0d0d !important;
    color: #d0d0d0 !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-family: 'Figtree', sans-serif !important;
    line-height: 1.65 !important;
    padding: 0.85rem 1rem !important;
    caret-color: #4a7eff !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.stTextArea textarea:focus {
    border-color: #4a7eff !important;
    box-shadow: 0 0 0 3px rgba(74, 126, 255, 0.08) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder { color: #2e2e2e !important; }

/* ── Button ─────────────────────────────────────── */
.stButton { margin-top: 0.75rem; }

.stButton > button {
    background-color: #4a7eff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.62rem 1.5rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    font-family: 'Figtree', sans-serif !important;
    letter-spacing: 0.2px !important;
    width: 100% !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
}

.stButton > button:hover {
    background-color: #3b6ef0 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    background-color: #2f5ed8 !important;
    transform: translateY(0) !important;
}

.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(74, 126, 255, 0.2) !important;
    outline: none !important;
}

.stButton > button:disabled {
    background-color: #1a1a1a !important;
    color: #333333 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Alerts ─────────────────────────────────────── */
[data-testid="stAlert"] {
    background-color: #0d0d0d !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 8px !important;
    color: #888888 !important;
    font-family: 'Figtree', sans-serif !important;
    font-size: 0.87rem !important;
}

/* ── Result cards ───────────────────────────────── */
.results-row {
    margin-top: 1.75rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}

.result-card {
    background-color: #0d0d0d;
    border: 1px solid #181818;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
}

.result-tag {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #4a7eff;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #ededed;
    line-height: 1.3;
    margin-bottom: 1.1rem;
    min-height: 1.8rem;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.conf-track {
    flex: 1;
    height: 3px;
    background-color: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #3b6ef0, #6090ff);
}

.conf-pct {
    font-size: 0.77rem;
    font-weight: 600;
    color: #4a7eff;
    min-width: 3rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ── Footer ─────────────────────────────────────── */
.site-footer {
    margin-top: 4rem;
    padding-top: 1.2rem;
    border-top: 1px solid #111111;
    text-align: center;
    font-size: 0.76rem;
    color: #333333;
    letter-spacing: 0.1px;
}

.site-footer a {
    color: #4a7eff;
    text-decoration: none;
    transition: color 0.15s ease;
}

.site-footer a:hover { color: #6090ff; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classifier...")
def load_predictor():
    try:
        from predict import predict as _predict
        return _predict, None
    except Exception as exc:
        return None, str(exc)


predict_fn, model_error = load_predictor()
model_ready = predict_fn is not None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1 class="app-title">MedQuAD-Classify</h1>
    <p class="app-desc">
        Enter a medical question. The model identifies the question category and routes
        it to the relevant department, using a Complement Naive Bayes classifier trained
        on 16,412 NIH medical Q&amp;A pairs.
    </p>
    <div class="stat-row">
        <span class="stat-pill">Question Type <span class="accent">97.5% acc</span></span>
        <span class="stat-pill">Department <span class="accent">93.5% acc</span></span>
        <span class="stat-pill">9 types &middot; 20 departments</span>
    </div>
</div>
""", unsafe_allow_html=True)


if not model_ready:
    st.error(
        f"Could not load the classifier. "
        f"Run `python src/train.py` from the project root first.\n\n"
        f"Details: {model_error}"
    )


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<span class="field-label">Medical Question</span>', unsafe_allow_html=True)

question = st.text_area(
    label="Medical Question",
    placeholder="e.g. What are the symptoms of type 2 diabetes?",
    height=130,
    label_visibility="collapsed",
    disabled=not model_ready,
)

classify_btn = st.button(
    "Classify",
    use_container_width=True,
    disabled=not model_ready,
)


# ── Prediction ────────────────────────────────────────────────────────────────
if classify_btn and model_ready:
    q = question.strip()
    if not q:
        st.warning("Enter a question before classifying.")
    else:
        try:
            result     = predict_fn(q)
            qtype_disp = result.get("qtype", "").replace("_", " ").title()
            dept_disp  = result.get("department", "")
            qtype_conf = float(result.get("qtype_proba", 0))
            dept_conf  = float(result.get("dept_proba", 0))

            st.markdown(f"""
            <div class="results-row">
                <div class="result-card">
                    <div class="result-tag">Question Type</div>
                    <div class="result-value">{qtype_disp}</div>
                    <div class="conf-row">
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{qtype_conf * 100:.1f}%"></div>
                        </div>
                        <span class="conf-pct">{qtype_conf * 100:.1f}%</span>
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-tag">Medical Department</div>
                    <div class="result-value">{dept_disp}</div>
                    <div class="conf-row">
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{dept_conf * 100:.1f}%"></div>
                        </div>
                        <span class="conf-pct">{dept_conf * 100:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    Built by Subin Thapa, Sachin KC and Bibek Subedi
    &nbsp;&middot;&nbsp;
    <a href="https://github.com/Subinthapa2092/MedQuAD-Classify" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)