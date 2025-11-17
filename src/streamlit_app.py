# src/streamlit_app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
from pathlib import Path
import joblib
from income_analysis import load_data, clean_data

# PATHS
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "adult_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.json"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"

st.set_page_config(page_title="Income >$50K", layout="wide")

# LOAD
@st.cache_data
def load_full_dataset():
    if not DATA_PATH.exists():
        st.error(f"Data not found: `{DATA_PATH}`")
        st.stop()
    return clean_data(load_data(DATA_PATH))

@st.cache_resource
def load_xgboost_model():
    if not XGBOOST_MODEL_PATH.exists():
        st.error(f"Model not found: `{XGBOOST_MODEL_PATH}`\nRun: `python src/model.py`")
        st.stop()
    model = xgb.Booster()
    model.load_model(str(XGBOOST_MODEL_PATH))
    return model

@st.cache_resource
def load_logistic_pipeline():
    if not LOGISTIC_MODEL_PATH.exists():
        st.error(f"Model not found: `{LOGISTIC_MODEL_PATH}`")
        st.stop()
    return joblib.load(LOGISTIC_MODEL_PATH)

df = load_full_dataset()
xgb_model = load_xgboost_model()
logistic_pipe = load_logistic_pipeline()

# UI
st.title("Income >$50K Predictor")
st.sidebar.header("Input")

with st.sidebar.form("form"):
    age = st.slider("Age", 17, 90, 35)
    education = st.selectbox("Education", df["education"].unique())
    education_num = st.slider("Education Years", 1, 16, 13)
    marital_status = st.selectbox("Marital Status", df["marital_status"].unique())
    occupation = st.selectbox("Occupation", df["occupation"].unique())
    hours_per_week = st.slider("Hours/Week", 1, 99, 40)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
    workclass = st.selectbox("Work Class", df["workclass"].unique())
    race = st.selectbox("Race", df["race"].unique())
    sex = st.selectbox("Sex", df["sex"].unique())
    native_country = st.selectbox("Country", df["native_country"].unique())
    submitted = st.form_submit_button("Predict")

# PREDICTION
if submitted:
    input_data = {
        "age": age,
        "workclass": workclass,
        # fnlwgt REMOVED — not in training
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": "Other",
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }
    input_df = pd.DataFrame([input_data])

    # Convert to category
    cat_cols = input_df.select_dtypes(include='object').columns
    input_df[cat_cols] = input_df[cat_cols].astype('category')

    # Predict
    dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
    xgb_prob = xgb_model.predict(dmatrix)[0]
    log_prob = logistic_pipe.predict_proba(input_df)[0, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("XGBoost (>50K)", f"{xgb_prob:.1%}")
    with col2:
        st.metric("Logistic (>50K)", f"{log_prob:.1%}")

    verdict = "Likely >$50K" if xgb_prob > 0.5 else "Likely ≤$50K"
    st.success(f"**{verdict}**")