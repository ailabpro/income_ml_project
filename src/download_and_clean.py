#!/usr/bin/env python
"""
Download and clean UCI Adult dataset once and for all.
Run: python scripts/download_and_clean.py
"""
import pandas as pd
from pathlib import Path
import urllib.request
import os

# Paths
RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
RAW_PATH = Path("data/adult.data")
CLEAN_PATH = Path("data/adult_clean.csv")
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

def download_data():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {RAW_URL}...")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    print(f"Saved to {RAW_PATH}")

def clean_data():
    print("Cleaning data...")
    df = pd.read_csv(
        RAW_PATH,
        header=None,
        names=COLUMNS,
        na_values=' ?',
        skipinitialspace=True
    )
    # Drop rows with missing values (as per original)
    df.dropna(inplace=True)
    # Strip income
    df['income'] = df['income'].str.strip()
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved: {CLEAN_PATH} ({len(df):,} rows)")

if __name__ == "__main__":
    if not RAW_PATH.exists():
        download_data()
    if not CLEAN_PATH.exists():
        clean_data()
    else:
        print(f"Clean data already exists: {CLEAN_PATH}")