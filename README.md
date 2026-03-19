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

Python | FastAPI | Scikit-learn | Docker | GitHub Actions

## How it works

1. Calculates RFM scores (Recency, Frequency, Monetary) from visit data
2. Uses KMeans clustering to group customers into 3 segments
3. Serves predictions via a REST API with a frontend dashboard

## Run locally

```bash
git clone https://github.com/Costas1992/customer-loyalty-classifier
cd customer-loyalty-classifier
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload
```

Built by Kostas ML Student at Noroff, Trondheim 🇳🇴
