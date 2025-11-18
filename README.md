# Income >$50K Predictor  
**End-to-End ML Pipeline | XGBoost | Streamlit | dbt | CI/CD**  
*November 17, 2025*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-brightgreen?logo=streamlit)](https://income-predictor-ailabpro.streamlit.app) [![GitHub](https://img.shields.io/badge/GitHub-ailabpro-blue?logo=github)](https://github.com/ailabpro/income_ml_project) [![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict whether an individual earns **> $50K/year** using the **UCI Adult Dataset**.  
Built with **production-grade MLOps**: ETL (dbt), modeling (XGBoost), deployment (Streamlit), and **CI/CD (GitHub Actions)**.

---

## Live Demo
[Try it now: Income Predictor](https://income-predictor-ailabpro.streamlit.app)

---

## Key Results
| Metric | Value |
|-------|-------|
| **XGBoost AUC** | **0.93** |
| **Accuracy** | 0.84 |
| **Top Predictor** | `education_num`, `capital_gain` |
| **Deployment** | Streamlit Cloud + CI/CD |

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/ailabpro/income_ml_project.git
cd income_ml_project

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python src/download_and_clean.py

# 5. Train models
python src/model.py

# 6. Launch dashboard
streamlit run src/streamlit_app.py
```

---

## Generate Data Visualization Reports

```bash
# 1. Generate EDA reports
python src/eda_exploration.py

# 2. Generate reports (PDF + Interactive HTML report)
python src/generate_report.py
```

---

## Run dbt Pipeline (ETL)

```bash
# 1. Load CSV into DuckDB
cd dbt_project/income_dbt
cp profiles.yml ~/.dbt/
dbt seed

# 2. Build models
dbt run

# 3. View results (Interactive dbt Docs)
dbt docs generate
dbt docs serve
```
