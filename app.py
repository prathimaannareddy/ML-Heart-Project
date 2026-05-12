import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
#  HEART DISEASE PREDICTION - STREAMLIT WEB APP
#  Course: DSE2220 | Manipal University Jaipur | Jan-May 2026
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import urllib.request
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0F111A;
        color: #E8EDF5;
    }

    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8892A4;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: #141928;
        border-left: 4px solid #E63946;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8892A4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E63946;
    }

    /* Result box - Disease */
    .result-disease {
        background: linear-gradient(135deg, #3D0A0F, #1A0508);
        border: 2px solid #E63946;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    /* Result box - No Disease */
    .result-healthy {
        background: linear-gradient(135deg, #0A3D1F, #051A0D);
        border: 2px solid #00B4D8;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00B4D8;
        border-bottom: 2px solid #E63946;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #141928;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #E63946, #9B1C2A);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF6B7A, #E63946);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(230,57,70,0.4);
    }

    /* Divider */
    hr {
        border-color: #1C2236;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD AND TRAIN MODEL (cached so it only runs once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    # Download dataset
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    urllib.request.urlretrieve(url, "heart.csv")
    df = pd.read_csv("heart.csv")

    # Preprocessing
    df.fillna(df.median(numeric_only=True), inplace=True)
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_enc.drop('target', axis=1)
    y = df_enc['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train all 3 models
    lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

    lr.fit(X_train_s,  y_train)
    rf.fit(X_train_s,  y_train)
    svm.fit(X_train_s, y_train)

    # Accuracy on test set
    acc = {
        'Logistic Regression': round(accuracy_score(y_test, lr.predict(X_test_s))  * 100, 1),
        'Random Forest':       round(accuracy_score(y_test, rf.predict(X_test_s))  * 100, 1),
        'SVM (RBF)':           round(accuracy_score(y_test, svm.predict(X_test_s)) * 100, 1),
    }
    auc = {
        'Logistic Regression': round(roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1]),  3),
        'Random Forest':       round(roc_auc_score(y_test, rf.predict_proba(X_test_s)[:,1]),  3),
        'SVM (RBF)':           round(roc_auc_score(y_test, svm.predict_proba(X_test_s)[:,1]), 3),
    }

    return {
        'models':  {'Logistic Regression': lr, 'Random Forest': rf, 'SVM (RBF)': svm},
        'scaler':  scaler,
        'columns': X.columns.tolist(),
        'cat_cols': cat_cols,
        'acc':     acc,
        'auc':     auc,
        'df':      df,
    }


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">❤️ Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">DSE2220 · Machine Learning · Manipal University Jaipur · Jan–May 2026</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (with spinner)
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading dataset and training models... (first run takes ~15 seconds)"):
    data = load_and_train()

models  = data['models']
scaler  = data['scaler']
columns = data['columns']
df_raw  = data['df']


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Model selection + info
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    selected_model = st.selectbox(
        "Choose ML Algorithm:",
        ['Random Forest', 'Logistic Regression', 'SVM (RBF)'],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    for name in ['Logistic Regression', 'Random Forest', 'SVM (RBF)']:
        icon = "🏆" if name == 'Random Forest' else "  "
        st.markdown(f"""
        <div style='background:#0F111A; border-left:3px solid {"#E63946" if name=="Random Forest" else "#00B4D8"};
                    padding:8px 12px; border-radius:4px; margin-bottom:8px;'>
            <div style='font-size:0.75rem; color:#8892A4;'>{icon} {name}</div>
            <div style='color:#FFFFFF; font-weight:700;'>
                Acc: {data['acc'][name]}% &nbsp;|&nbsp; AUC: {data['auc'][name]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Dataset Info")
    st.info(f"""
    **UCI Heart Disease Dataset**
    - Records: {len(df_raw)}
    - Features: 13 clinical inputs
    - Source: Kaggle / UCI ML Repo
    """)

    st.markdown("---")
    st.markdown("### 📖 Feature Guide")
    with st.expander("Click to see what each field means"):
        st.markdown("""
        | Feature | Meaning |
        |---------|---------|
        | **cp** | Chest pain type (0–3) |
        | **thalach** | Max heart rate |
        | **oldpeak** | ST depression |
        | **ca** | Blocked vessels (0–3) |
        | **thal** | Thalassemia type |
        | **exang** | Exercise angina |
        | **slope** | ST slope |
        """)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — Input Form
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Patient Clinical Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age    = st.slider("Age (years)", 20, 80, 55)
    sex    = st.radio("Sex", ["Male", "Female"], horizontal=True)
    sex_val = 1 if sex == "Male" else 0
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol   = st.slider("Cholesterol (mg/dl)", 100, 600, 220)

with col2:
    st.markdown("**Cardiac Measurements**")
    cp     = st.selectbox("Chest Pain Type", [
        "0 — Typical Angina",
        "1 — Atypical Angina",
        "2 — Non-Anginal Pain",
        "3 — Asymptomatic (Silent)"
    ])
    cp_val = int(cp[0])
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 210, 150)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.5, step=0.1)
    slope  = st.selectbox("ST Slope", ["0 — Upsloping", "1 — Flat", "2 — Downsloping"])
    slope_val = int(slope[0])

with col3:
    st.markdown("**Test Results**")
    fbs    = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No (0)", "Yes (1)"], horizontal=True)
    fbs_val = 1 if "Yes" in fbs else 0
    restecg = st.selectbox("Resting ECG", [
        "0 — Normal",
        "1 — ST-T Wave Abnormality",
        "2 — Left Ventricular Hypertrophy"
    ])
    restecg_val = int(restecg[0])
    exang  = st.radio("Exercise-Induced Angina?", ["No (0)", "Yes (1)"], horizontal=True)
    exang_val = 1 if "Yes" in exang else 0
    ca     = st.selectbox("Major Vessels Blocked (0–3)", [0, 1, 2, 3])
    thal   = st.selectbox("Thalassemia", [
        "1 — Normal",
        "2 — Fixed Defect",
        "3 — Reversible Defect"
    ])
    thal_val = int(thal[0])


# ─────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🔍  PREDICT HEART DISEASE RISK")


# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────
if predict_clicked:
    # Build raw patient dataframe
    patient_raw = pd.DataFrame([{
        'age': age, 'sex': sex_val, 'cp': cp_val,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs_val,
        'restecg': restecg_val, 'thalach': thalach,
        'exang': exang_val, 'oldpeak': oldpeak,
        'slope': slope_val, 'ca': ca, 'thal': thal_val
    }])

    # Encode same as training
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    patient_enc = pd.get_dummies(patient_raw, columns=cat_cols, drop_first=True)
    patient_enc = patient_enc.reindex(columns=columns, fill_value=0)

    # Scale
    patient_scaled = scaler.transform(patient_enc)

    # Get prediction from selected model
    model      = models[selected_model]
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    st.markdown("---")
    st.markdown("### 🩺 Prediction Result")

    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-disease">
                <div style="font-size:3rem;">⚠️</div>
                <div style="font-size:1.8rem; font-weight:800; color:#E63946; margin:0.5rem 0;">
                    HEART DISEASE DETECTED
                </div>
                <div style="font-size:1rem; color:#C5CCDA;">
                    The model predicts a <b style="color:#E63946;">high risk</b> of heart disease.<br>
                    Please consult a cardiologist immediately.
                </div>
                <div style="margin-top:1.2rem; font-size:2rem; font-weight:700; color:#FF6B7A;">
                    Risk Probability: {probability*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <div style="font-size:3rem;">✅</div>
                <div style="font-size:1.8rem; font-weight:800; color:#00B4D8; margin:0.5rem 0;">
                    NO HEART DISEASE DETECTED
                </div>
                <div style="font-size:1rem; color:#C5CCDA;">
                    The model predicts a <b style="color:#00B4D8;">low risk</b> of heart disease.<br>
                    Continue maintaining a healthy lifestyle.
                </div>
                <div style="margin-top:1.2rem; font-size:2rem; font-weight:700; color:#00B4D8;">
                    Risk Probability: {probability*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("**All Models' Predictions:**")
        for name, mdl in models.items():
            p    = mdl.predict(patient_scaled)[0]
            prob = mdl.predict_proba(patient_scaled)[0][1]
            icon = "🔴" if p == 1 else "🟢"
            active = "→ " if name == selected_model else "   "
            st.markdown(f"""
            <div style='background:#141928; border-radius:6px; padding:10px 14px;
                        margin-bottom:8px; border-left:3px solid {"#E63946" if p==1 else "#00B4D8"}'>
                <span style='color:#8892A4; font-size:0.8rem;'>{active}{name}</span><br>
                <span style='font-weight:700; color:{"#E63946" if p==1 else "#00B4D8"};'>
                    {icon} {"Disease" if p==1 else "No Disease"} — {prob*100:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style='background:#0F111A; border-radius:8px; padding:12px;
                    border:1px solid #1C2236; font-size:0.85rem; color:#8892A4;'>
            <b style='color:#F4A261;'>⚠️ Disclaimer</b><br>
            This tool is for educational purposes only and is
            <b>not a substitute for professional medical advice.</b>
            Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    # Show patient summary
    st.markdown("---")
    st.markdown("**Patient Summary (Input Values):**")
    summary_cols = st.columns(7)
    labels = ["Age", "Sex", "Chest Pain", "Max HR", "ST Dep.", "Vessels", "Thalassemia"]
    values = [age, sex, f"Type {cp_val}", thalach, oldpeak, ca, f"Type {thal_val}"]
    for col, lbl, val in zip(summary_cols, labels, values):
        col.metric(lbl, val)


# ─────────────────────────────────────────────────────────────
# DATASET PREVIEW (expandable)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Raw Dataset (first 10 rows)"):
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Total: {len(df_raw)} rows × {len(df_raw.columns)} columns")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8892A4; font-size:0.85rem; padding:1rem;'>
    Heart Disease Prediction &nbsp;·&nbsp; DSE2220 Machine Learning &nbsp;·&nbsp;
    B.Tech CSE (Data Science) &nbsp;·&nbsp; Manipal University Jaipur &nbsp;·&nbsp; Jan–May 2026
</div>
""", unsafe_allow_html=True)
