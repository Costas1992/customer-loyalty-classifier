---
title: Customer Loyalty Classifier
emoji: ✂️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: true
---

# Customer Loyalty Classifier

A machine learning app that classifies barbershop customers into **Loyal**, **At Risk**, or **Lost** segments using RFM analysis.

# Live Demo

[View live app](https://costas92-customer-loyalty-classifier.hf.space/dashboard)

- **Dashboard** → `/dashboard`
- **API Docs** → `/docs`

## Tech Stack

Python | FastAPI | Scikit-learn | MLflow | Docker | GitHub Actions

## How it works

1. Calculates RFM scores (Recency, Frequency, Monetary) from visit data
2. Uses KMeans clustering to group customers into 3 segments
3. Serves predictions via a REST API with a frontend dashboard

## MLOps: Monitoring & Automated Retraining

The classifier ships with a drift-monitoring and retraining layer on top of the base model, so it doesn't silently go stale as customer behavior changes.

**Drift detection** — `model/monitor.py` compares incoming customer data against the training distribution using the Population Stability Index (PSI) on each RFM feature. PSI uses equal-width binning for these narrow-range integer features (0–13), since standard quantile binning collapses under the heavy value ties typical of visit-count data and loses resolution exactly where drift shows up.

**Synthetic drift testing** — since real "new month" data isn't available yet, `data/simulate_new_batch.py` generates two synthetic batches from the existing customer base: a clean batch (random noise, should NOT trigger drift — the false-positive control) and a churn batch (a subset of customers' recent months zeroed out, simulating real attrition). This validates the detector actually works, not just that it runs.

**Conditional retraining** — `model/retrain_if_drifted.py` checks for drift and, if flagged, merges new observations into the training set (upserting by `client_id` — new data supersedes old for returning customers, new customers are appended) and retrains. Retrain output writes to a separate file rather than the canonical `customers.csv`, so runs are safe to repeat without risking the source dataset.

**Experiment tracking** — every training and drift-check run is logged to MLflow (params, RFM summary metrics, segment distributions, drift PSI scores, model artifacts), so runs are comparable and reproducible over time.

```bash
python model/train.py                  # train + log to MLflow
python data/simulate_new_batch.py       # generate synthetic drift scenarios
python model/monitor.py                 # check drift
python model/retrain_if_drifted.py      # conditional retrain
mlflow ui                               # view tracked runs at localhost:5000
```

_Note: GitHub Actions currently runs the core train/test/build pipeline on every push. The drift-check and retrain steps above are run manually — wiring them into CI is a natural next step._

## Run locally

```bash
git clone https://github.com/Costas1992/customer-loyalty-classifier
cd customer-loyalty-classifier
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload
```

Built by Kostas — ML Student at Noroff, Trondheim 🇳🇴
h