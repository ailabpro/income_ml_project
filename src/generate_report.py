# src/generate_report.py
"""
Generate PDF + Interactive HTML Executive Report
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import mpld3
from pathlib import Path
from income_analysis import load_data, clean_data, DATA_PATH

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("Generating Executive Report".center(50, "="))
    
    df = clean_data(load_data(DATA_PATH))
    
    # PDF Report
    pdf_path = REPORT_DIR / "income_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # FIX 1: Use color instead of palette
        sns.countplot(data=df, x="income", ax=ax1, color="#66c2a5")
        ax1.set_title("Income Distribution")
        
        # FIX 2: Proper tick labels
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["≤50K", ">50K"])
        
        df["education"].value_counts().head(10).plot(kind="barh", ax=ax2, color="skyblue")
        ax2.set_title("Top 10 Education Levels")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.select_dtypes(include="number").corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        pdf.savefig(fig)
        plt.close()
    
    print(f"PDF saved: {pdf_path}")
    
    # Interactive HTML (mpld3)
    html_path = REPORT_DIR / "income_report.html"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df.sample(1000, random_state=42),
        x="age", y="hours_per_week", hue="income",
        palette={0: "#636EFA", 1: "#EF553B"}, ax=ax, alpha=0.7
    )
    ax.set_title("Age vs Hours/Week (Interactive)")
    
    # FIX 3: Save with file object
    with open(html_path, "w") as f:
        mpld3.save_html(fig, f)
    
    print(f"HTML saved: {html_path}")

if __name__ == "__main__":
    main()