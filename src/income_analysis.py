# src/income_analysis.py
"""
Shared data module: load + clean
Uses: adult_clean.csv (cleaned once)
"""
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = Path("data/adult_clean.csv")
COLS = [  # Keep for reference
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# -------------------------------------------------
# 1. LOAD CLEANED DATA
# -------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load pre-cleaned CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}\nRun data cleaning step first.")
    
    df = pd.read_csv(path)  # ← CLEAN CSV = NO header=None!
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df

# -------------------------------------------------
# 2. CLEAN (minimal)
# -------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df.dropna(inplace=True)
    return df

# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    df = clean_data(load_data())
    print(f"Final: {df.shape}")
    print(df["income"].value_counts(normalize=True).round(3))