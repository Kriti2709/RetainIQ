<div align="center">

# RetainIQ

### Churn Intelligence Platform for B2B SaaS

**Predict churn 30 days early · Recommend retention actions · Analyse cohort health**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-orange?style=flat-square&logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?style=flat-square&logo=docker)](https://docker.com)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-black?style=flat-square&logo=githubactions)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is RetainIQ?

RetainIQ is a **plug-and-play churn intelligence platform** that any B2B SaaS company can adopt. It connects to your product's event data through a standardised connector SDK and delivers three things:

| Engine | What it does |
|---|---|
| **Early Warning System** | Scores every customer account daily with a 30-day churn probability using XGBoost + SHAP |
| **Intervention Engine** | Recommends the right retention action per account — discount, CSM outreach, or feature nudge |
| **Cohort Analytics** | Surfaces retention curves and churn trends segmented by plan tier, company size, and acquisition month |

> Built as a reference implementation for a **B2B project management SaaS** (Jira/Asana-style), but the connector SDK is designed to work with any platform.

---

## Architecture

```
Raw Data Sources          ETL Layer              ML Core              Serving Layer
─────────────────    ─────────────────    ─────────────────    ─────────────────
 Workspace events  →  Airflow DAGs      →  XGBoost model    →  FastAPI REST API
 Subscription data →  dbt transforms   →  MLflow registry  →  Streamlit dashboard
 Support tickets   →  Great Expectations→  SHAP explainer   →  Webhook alerts
                       validation           Intervention rules
                                            Evidently monitoring
```

---

## Key Features

- **Standardised event schema** — any platform maps their data once and plugs in
- **Automated ETL** — Airflow DAGs ingest, transform, and validate data daily
- **Experiment tracking** — every model training run logged in MLflow with full reproducibility
- **Per-account explainability** — SHAP waterfall plots show exactly *why* an account is at risk
- **Intervention recommendations** — bridges prediction to action with a rule-based + ML engine
- **Data drift monitoring** — Evidently AI detects when live data drifts from training distribution
- **Auto-retraining** — model automatically retrains and promotes itself when drift exceeds threshold
- **CI/CD pipeline** — GitHub Actions runs lint → test → build → deploy on every push

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data generation | Python, Faker, pandas |
| ETL pipeline | Apache Airflow, dbt, Great Expectations |
| Feature engineering | pandas, NumPy, scikit-learn |
| ML training | XGBoost, Optuna, scikit-learn |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Containerisation | Docker, Docker Compose |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |
| Cloud deployment | Render |

---

## Project Structure

```
RetainIQ/
├── data_pipeline/
│   ├── generate_data.py          # Synthetic SaaS data generator
│   ├── airflow_dags/             # ETL DAG definitions
│   ├── dbt_models/               # SQL transformation models
│   └── expectations/             # Great Expectations suites
├── feature_store/
│   ├── engineer_features.py      # Feature engineering logic
│   └── feature_registry.md       # Feature definitions & rationale
├── models/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation + SHAP analysis
│   ├── intervention_engine.py    # Retention action recommender
│   └── mlflow_config.py          # MLflow experiment setup
├── api/
│   ├── main.py                   # FastAPI app
│   ├── schemas.py                # Pydantic request/response models
│   └── Dockerfile                # Container definition
├── dashboard/
│   ├── app.py                    # Streamlit entrypoint
│   ├── pages/
│   │   ├── 01_risk_radar.py      # Account-level churn heatmap
│   │   ├── 02_account_deepdive.py# SHAP waterfall per account
│   │   └── 03_cohort_health.py   # Retention curves by segment
├── monitoring/
│   ├── drift_report.py           # Evidently drift detection
│   └── retrain_trigger.py        # Auto-retraining logic
├── connector_sdk/
│   ├── schema.py                 # Standardised event schema
│   ├── ingest_api.py             # REST ingestion endpoint
│   └── connectors/
│       └── project_mgmt.py       # Reference connector (Jira/Asana style)
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── tests/
│   ├── test_features.py          # Unit tests for feature engineering
│   └── test_api.py               # API endpoint tests
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/RetainIQ.git
cd RetainIQ

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic dataset
python data_pipeline/generate_data.py

# 5. Run feature engineering
python feature_store/engineer_features.py

# 6. Train the model
python models/train.py

# 7. Start the API
uvicorn api.main:app --reload

# 8. Launch the dashboard
streamlit run dashboard/app.py
```

---

## The Connector SDK

RetainIQ is not tied to one platform. Any B2B SaaS product can integrate by mapping their events to the standardised schema:

```python
# Example event payload (any platform)
{
  "account_id": "acct_abc123",
  "event_type": "feature_used",
  "timestamp": "2024-03-01T10:00:00Z",
  "properties": {
    "feature_name": "export",
    "user_count": 3,
    "session_duration_mins": 12
  }
}
```

See `connector_sdk/` for the full schema definition and the reference connector implementation.

---

## Live Demo

| | Link |
|---|---|
| API docs | `https://retainiq-api.onrender.com/docs` |
| Dashboard | `https://retainiq-dashboard.onrender.com` |

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | ~0.91 |
| F1 Score (churn class) | ~0.84 |
| Precision | ~0.87 |
| Recall | ~0.81 |
| Prediction horizon | 30 days before renewal |

> Trained on synthetic data representing ~2,000 B2B SaaS accounts across 18 months.

---

## Roadmap

- [ ] Kafka real-time event streaming connector
- [ ] LLM-generated CSM email drafts per at-risk account
- [ ] Slack bot for instant churn alerts
- [ ] Multi-tenant support for SaaS-on-SaaS deployment

---

## Author

Built by **Kriti Saini** as a portfolio project demonstrating end-to-end MLOps.

<div align="center">
<sub>RetainIQ · MIT License · Built with Python, FastAPI, MLflow, and Streamlit</sub>
</div>
