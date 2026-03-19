from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# LOAD THE TRAINED MODEL 
# These files were created when we ran train.py
kmeans    = joblib.load('model/kmeans_model.pkl')
scaler    = joblib.load('model/scaler.pkl')
label_map = joblib.load('model/label_map.pkl')

# create the FastAPI app
app = FastAPI(
    tittle="Customer Loyalty Classifier",
    description="Predicts is a cliant is Loyal, at risk, or Lost",
    version="1.0.0"

)    

#  SERVE FRONTEND 
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/dashboard")
def dashboard():
    return FileResponse("app/static/index.html")

# DEFINE INPUT STRUCTURE 
# This is what the API expects to receive
# Pydantic validates the data automatically
class CustomerInput(BaseModel):
    client_id:   str
    client_name: str
    visits: list[int]  # list of 12 numbers, one per month

    # Example data shown in the API docs
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "client_id": "C00001",
                "client_name": "Jonas Larsen",
                "visits": [1,0,0,1,0,1,0,1,0,1,0,0]
            }]
        }
    }

# DEFINE OUTPUT STRUCTURE 
class PredictionOutput(BaseModel):
    client_id:   str
    client_name: str
    recency:     int
    frequency:   int
    monetary:    int
    segment:     str
    message:     str

# HELPER: CALCULATE RFM 
def calculate_rfm(visits: list[int]):
    # Recency: how many months since last visit
    recency = 12
    for i in reversed(range(len(visits))):
        if visits[i] > 0:
            recency = 12 - i
            break

    frequency = sum(1 for v in visits if v > 0)  # months with visits
    monetary  = sum(visits)                        # total visits

    return recency, frequency, monetary

# SEGMENT MESSAGE 
def get_message(segment: str) -> str:
    messages = {
        "Loyal":   "⭐ VIP client! Keep them happy with loyalty rewards.",
        "At Risk": "⚠️ Slowing down. Send them a comeback offer soon!",
        "Lost":    "❌ Haven't seen them in a while. Time for a win-back campaign."
    }
    return messages.get(segment, "Unknown segment")

# ENDPOINTS 
# Root endpoint, just a welcome message
@app.get("/")
def root():
    return {
        "message": "Welcome to the Customer Loyalty Classifier ",
        "docs": "Visit /docs to try the API interactively"
    }

# Predict endpoint — classify a single customer
@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):

    # Validate we have exactly 12 months of data
    if len(customer.visits) != 12:
        raise HTTPException(
            status_code=400,
            detail="visits must contain exactly 12 values (one per month)"
        )

    # Calculate RFM
    recency, frequency, monetary = calculate_rfm(customer.visits)

    # Scale and predict
    rfm_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster    = kmeans.predict(rfm_scaled)[0]
    segment    = label_map[cluster]

    return PredictionOutput(
        client_id=customer.client_id,
        client_name=customer.client_name,
        recency=recency,
        frequency=frequency,
        monetary=monetary,
        segment=segment,
        message=get_message(segment)
    )

# Customers endpoint — return all segmented customers
@app.get("/customers")
def get_customers(segment: str = None):
    path = 'data/customers_segmented.csv'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No segmented data found. Run train.py first.")

    df = pd.read_csv(path)

    # Optional filter by segment
    if segment:
        df = df[df['segment'].str.lower() == segment.lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No customers found for segment: {segment}")

    return {
        "total": len(df),
        "customers": df.to_dict(orient='records')
    }