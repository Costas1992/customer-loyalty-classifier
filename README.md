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
- **Live drift check** → `/drift-status?scenario=drifted` or `/drift-status?scenario=clean`

## Tech Stack

Python | FastAPI | Scikit-learn | MLflow | Docker | GitHub Actions

## How it works

1. Calculates RFM scores (Recency, Frequency, Monetary) from visit data
2. Uses KMeans clustering to group customers into 3 segments
3. Serves predictions via a REST API with a frontend dashboard

## MLOps: Monitoring & Automated Retraining

Added a monitoring and retraining layer on top of the base model, so it doesn't just quietly go stale as customer behavior changes.

**Drift detection** — `model/monitor.py` compares new customer data against the training distribution using PSI (Population Stability Index) on each RFM feature. Started with standard quantile binning, but it collapsed on these features — most customers share the same low recency value, so the buckets lost resolution exactly where drift shows up. Switched to equal-width binning for narrow integer ranges, verified it against real data.

**Synthetic drift testing** — no real "new month" data yet, so `data/simulate_new_batch.py` generates two batches to test against: one with random noise (should NOT trigger drift the false-positive check) and one simulating churn, where a chunk of customers' recent months get zeroed out. This is what actually proves the detector works, not just that it runs without errors.

**Live drift endpoint** — `GET /drift-status` runs the same check on demand against the deployed model. Try `?scenario=drifted` or `?scenario=clean` directly, no local setup needed. Example:

```json
{
  "scenario": "drifted",
  "features": {
    "recency": { "psi": 0.5734, "status": "significant" },
    "frequency": { "psi": 0.1525, "status": "moderate" },
    "monetary": { "psi": 0.1702, "status": "moderate" }
  },
  "overall_drift": true
}
```

**Conditional retraining** — `model/retrain_if_drifted.py` checks for drift, and if it's flagged, merges the new data into the training set (matched by `client_id` — new data overwrites old for returning customers, new customers get appended) and retrains. Writes to a separate file instead of overwriting `customers.csv` directly, so it's safe to run more than once without risking the real dataset.

**Experiment tracking** — every training and drift-check run logs to MLflow: params, RFM stats, segment counts, PSI scores, model artifacts. Makes runs comparable over time instead of just trusting whatever the last run happened to produce.

```bash
python model/train.py                  # train + log to MLflow
python data/simulate_new_batch.py       # generate synthetic drift scenarios
python model/monitor.py                 # check drift
python model/retrain_if_drifted.py      # conditional retrain
mlflow ui                               # view tracked runs at localhost:5000
```

_GitHub Actions currently runs the core train/test/build pipeline on every push. Retraining is still manual wiring it into CI as a scheduled job is the natural next step._

## Run locally

```bash
git clone https://github.com/Costas1992/customer-loyalty-classifier
cd customer-loyalty-classifier
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload
```

Built by Kostas — ML Student at Noroff, Trondheim 🇳🇴
