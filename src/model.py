# src/model.py
# Updated Nov 25, 2025
"""
ML Models: Logistic Regression & XGBoost
Target: income (>50K = 1)
RUN THIS FILE DIRECTLY: python src/model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = Path("data/adult_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. LOAD & PREP
# -------------------------------------------------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # Drop fnlwgt (sampling weight)
    X = df.drop(columns=["income", "fnlwgt"])
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )
    return X, y, preprocessor

# -------------------------------------------------
# 2. TRAIN LOGISTIC
# -------------------------------------------------
def train_logistic(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    model_path = MODEL_DIR / "logistic_model.pkl"
    joblib.dump(pipe, model_path)
    print(f"Saved: {model_path}")
    return pipe

# -------------------------------------------------
# 3. TRAIN XGBOOST
# -------------------------------------------------
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FIX: Convert object (string) columns to category dtype
    cat_cols = X_train.select_dtypes(include='object').columns
    X_train[cat_cols] = X_train[cat_cols].astype('category')
    X_test[cat_cols] = X_test[cat_cols].astype('category')

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum()
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    y_pred = (model.predict(dtest) > 0.5).astype(int)
    y_prob = model.predict(dtest)

    print("\nXGBoost")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    model_path = MODEL_DIR / "xgboost_model.json"
    model.save_model(model_path)
    print(f"Saved: {model_path}")
    return model

# -------------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------------
def main():
    print("Starting ML Training Pipeline".center(60, "="))
    X, y, preprocessor = load_and_prepare()

    print("\nTraining Logistic Regression...")
    train_logistic(X, y, preprocessor)

    print("\nTraining XGBoost...")
    train_xgboost(X, y)

    print("\nAll models trained and saved!".center(60, "="))

# -------------------------------------------------
# RUN WHEN EXECUTED
# -------------------------------------------------
if __name__ == "__main__":
    main()