
# These tests run automatically in GitHub Actions
# They check your API works correctly

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test 1: Root endpoint returns welcome message
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# Test 2: Loyal customer is classified correctly
def test_predict_loyal():
    response = client.post("/predict", json={
        "client_id": "TEST001",
        "client_name": "Test Loyal",
        "visits": [2,2,2,3,2,2,2,3,2,3,2,3]
    })
    assert response.status_code == 200
    assert response.json()["segment"] == "Loyal"

# Test 3: Lost customer is classified correctly
def test_predict_lost():
    response = client.post("/predict", json={
        "client_id": "TEST002",
        "client_name": "Test Lost",
        "visits": [1,0,0,0,0,0,0,0,0,0,0,0]
    })
    assert response.status_code == 200
    assert response.json()["segment"] == "Lost"

# Test 4: Invalid input returns error
def test_predict_invalid():
    response = client.post("/predict", json={
        "client_id": "TEST003",
        "client_name": "Test Invalid",
        "visits": [1,0,0]  # only 3 months, should fail
    })
    assert response.status_code == 400