# src/eda_exploration.py
"""
Full Exploratory Data Analysis (EDA) for Adult Income Dataset
7 Sections | Uses: adult_clean.csv | Output: plots, stats, insights
Run: python src/eda_exploration.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
DATA_PATH = Path("data/adult_clean.csv")
OUTPUT_DIR = Path("reports/eda_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD & CLEAN
# -------------------------------------------------
def load_and_clean():
    print(f"Loading data from: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    # Encode target
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df.dropna(inplace=True)
    print(f"Cleaned shape: {df.shape}")

    # Feature engineering
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    return df

# -------------------------------------------------
# 3. EDA SUMMARY
# -------------------------------------------------
def eda_summary(df):
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*60)

    high_income_rate = df["income"].mean() * 100
    print(f"High Earners (>50K): {high_income_rate:.2f}%")

    print("\nNumeric Features:")
    print(df[["age", "education_num", "hours_per_week", "capital_net"]].describe().round(2))

    print("\nTop Occupations:")
    print(df["occupation"].value_counts().head(5))

    print("\nEducation vs Income:")
    edu_income = df.groupby("education")["income"].mean().sort_values(ascending=False)
    print(edu_income.round(3))

# -------------------------------------------------
# 4. STATIC VISUALIZATIONS (matplotlib / seaborn)
# -------------------------------------------------
def plot_static(df):
    print("\nGenerating static plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4.1 Income Distribution
    sns.countplot(data=df, x="income", ax=axes[0,0], palette="Set2")
    axes[0,0].set_title("Income Class Distribution")
    axes[0,0].set_xticklabels(["≤50K", ">50K"])

    # 4.2 Age vs Hours
    sns.boxplot(data=df, x="income", y="age", ax=axes[0,1], palette="Set3")
    axes[0,1].set_title("Age by Income")
    axes[0,1].set_xticklabels(["≤50K", ">50K"])

    # 4.3 Education Level
    edu_order = df["education"].value_counts().index
    sns.countplot(data=df, y="education", order=edu_order, ax=axes[1,0], palette="viridis")
    axes[1,0].set_title("Education Level Frequency")

    # 4.4 Capital Net
    sns.histplot(data=df, x="capital_net", hue="income", bins=50, ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title("Net Capital by Income")
    axes[1,1].set_xlim(-5000, 50000)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "static_eda.png", dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------------------------
# 5. INTERACTIVE PLOTLY VISUALS
# -------------------------------------------------
def plot_interactive(df):
    print("Generating interactive plots...")

    # 5.1 % >50K by Education
    edu = (df.groupby("education")["income"]
             .mean()
             .mul(100)
             .reset_index()
             .sort_values("income", ascending=False))

    fig1 = px.bar(
        edu, x="education", y="income",
        title="%>50K by Education Level",
        labels={"income": "% Earning >$50K"},
        color="income", color_continuous_scale="Plasma"
    )
    fig1.update_layout(xaxis_tickangle=-45)
    fig1.write_html(OUTPUT_DIR / "edu_income.html")
    fig1.show()

    # 5.2 Age vs Hours (scatter)
    sample = df.sample(5000, random_state=42)
    fig2 = px.scatter(
        sample, x="age", y="hours_per_week", color="income",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        title="Age vs Hours/Week (Interactive Sample)",
        opacity=0.6, marginal_x="histogram", marginal_y="histogram"
    )
    fig2.write_html(OUTPUT_DIR / "age_hours_scatter.html")
    fig2.show()

# -------------------------------------------------
# 6. CORRELATION HEATMAP
# -------------------------------------------------
def correlation_heatmap(df):
    print("Generating correlation heatmap...")

    # Select numeric + encode key categorical
    num_cols = ["age", "education_num", "capital_net", "hours_per_week", "income"]
    cat_cols = ["workclass", "marital_status", "occupation", "sex", "race"]
    df_enc = pd.get_dummies(df[cat_cols], drop_first=True)
    df_corr = pd.concat([df[num_cols], df_enc], axis=1)

    corr = df_corr.corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False, fmt=".2f")
    plt.title("Feature Correlation Heatmap (Lower Triangle)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------------------------
# 7. MAIN PIPELINE
# -------------------------------------------------
def main():
    print("Adult Income EDA Exploration".center(70, "="))
    df = load_and_clean()

    eda_summary(df)
    plot_static(df)
    plot_interactive(df)
    correlation_heatmap(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("EDA Complete!".center(70, "="))

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    main()