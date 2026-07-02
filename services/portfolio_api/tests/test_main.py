# =============================================================
# FILE: services/portfolio_api/tests/test_main.py
#
# WHAT IS UNIT TESTING?
#   Testing individual pieces of code in isolation
#   Without connecting to real Vertex AI or Redis
#
# WHY TEST WITHOUT REAL SERVICES?
#   Tests must run in CI/CD (GitHub's servers)
#   GitHub servers have no access to our Vertex AI endpoint
#   Solution: "mock" external services
#   Mock = fake version that returns predictable responses
#   Tests verify OUR logic, not Google's infrastructure
# =============================================================

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# TestClient = synchronous test client for FastAPI
# Lets you make HTTP requests without running a real server

# We need to mock Vertex AI before importing main
# Otherwise import fails (no GCP credentials in CI)
with patch("google.cloud.aiplatform.init"), \
     patch("google.cloud.aiplatform.Endpoint"):
    from main import app

client = TestClient(app)

def test_health_check():
    """Health endpoint should always return 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_root_endpoint():
    """Root endpoint should return service info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "WealthWise Portfolio API"

def test_predict_invalid_equity_pct():
    """Should reject equity_pct > 100"""
    response = client.post("/predict", json={
        "client_id": "CLT001",
        "equity_pct": 150.0,   # INVALID — over 100%
        "debt_pct": 10.0,
        "gold_pct": 5.0,
        "total_num_funds": 5,
        "years_invested": 3.0,
        "aum_lakh": 85.0,
        "age_group": "Young"
    })
    assert response.status_code == 422
    # 422 = Unprocessable Entity = validation failed

def test_predict_invalid_age_group():
    """Should reject invalid age_group values"""
    response = client.post("/predict", json={
        "client_id": "CLT001",
        "equity_pct": 50.0,
        "debt_pct": 30.0,
        "gold_pct": 20.0,
        "total_num_funds": 3,
        "years_invested": 5.0,
        "aum_lakh": 50.0,
        "age_group": "Baby"    # INVALID — not in allowed list
    })
    assert response.status_code == 422

def test_predict_missing_field():
    """Should reject requests with missing required fields"""
    response = client.post("/predict", json={
        "client_id": "CLT001",
        "equity_pct": 50.0,
        # Missing: debt_pct, gold_pct, etc.
    })
    assert response.status_code == 422

def test_encode_features():
    """Feature encoding should produce correct list"""
    from main import encode_features, PredictionRequest
    req = PredictionRequest(
        client_id="CLT001",
        equity_pct=85.0,
        debt_pct=10.0,
        gold_pct=5.0,
        total_num_funds=5,
        years_invested=3.0,
        aum_lakh=85.0,
        age_group="Young"
    )
    features = encode_features(req)
    assert len(features) == 7          # 7 features
    assert features[0] == 85.0        # equity_pct
    assert features[6] == 0.0         # Young = 0

def test_cache_key_consistency():
    """Same input should always produce same cache key"""
    from main import make_cache_key, PredictionRequest
    req = PredictionRequest(
        client_id="CLT001",
        equity_pct=85.0,
        debt_pct=10.0,
        gold_pct=5.0,
        total_num_funds=5,
        years_invested=3.0,
        aum_lakh=85.0,
        age_group="Young"
    )
    key1 = make_cache_key(req)
    key2 = make_cache_key(req)
    assert key1 == key2   # deterministic
    assert "CLT001" in key1
    assert "wealthwise:risk:" in key1