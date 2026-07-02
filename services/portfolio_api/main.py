# =============================================================
# FILE: services/portfolio_api/main.py
#
# WHAT IT IS:
#   A FastAPI microservice — a small focused web server
#   that does ONE thing: predict client risk profiles
#
# WHY FASTAPI AND NOT CALLING VERTEX AI DIRECTLY?
#   Mobile app → directly calls Vertex AI = BAD because:
#     - Vertex AI credentials exposed to mobile app
#     - No caching → same client asked 100 times = 100 Vertex calls = $$$
#     - No validation → garbage input crashes Vertex AI
#     - No audit trail → who called what when?
#
#   Mobile app → FastAPI → Vertex AI = GOOD because:
#     - FastAPI holds credentials securely
#     - Redis cache → same client = instant response
#     - Pydantic validates input before it reaches Vertex AI
#     - Every request logged for compliance (SEBI requirement)
#
# WHAT IS A MICROSERVICE?
#   A small independent application that does ONE job.
#   Instead of one giant application doing everything,
#   you have many small services each doing one thing:
#     portfolio-api    → risk predictions
#     anomaly-api      → fraud detection
#     rag-api          → AI document search
#     notification-api → send emails/SMS
#   Each runs independently, scales independently, fails independently
#
# COMPANY USE CASE:
#   10,000 advisors use the mobile app simultaneously
#   Each advisor clicks "Get Risk Profile" for their client
#   FastAPI receives request → checks Redis cache first
#   If cached → returns in 2ms (no Vertex AI call)
#   If not cached → calls Vertex AI → caches result → returns
#   Result: 90% of requests served from cache = 90% cost reduction
# =============================================================

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

# ── FASTAPI IMPORTS ───────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# FastAPI  = the web framework (like Flask but faster, modern)
# HTTPException = raise HTTP errors (400 Bad Request, 404 Not Found)
# Request = access raw request data (headers, IP address)
# CORSMiddleware = allows browser apps to call this API
#   CORS = Cross-Origin Resource Sharing
#   Without it: browser blocks requests from different domains
#   With it: React dashboard on port 3000 can call API on port 8000

from pydantic import BaseModel, Field, validator
# Pydantic = data validation library
# BaseModel = base class for defining data shapes
# Field = add constraints to fields (min value, max value, description)
# validator = custom validation logic

# ── GOOGLE CLOUD IMPORTS ──────────────────────────────────────
from google.cloud import aiplatform
# aiplatform = Vertex AI Python SDK
# We use it to call our deployed endpoint

# ── REDIS IMPORT ──────────────────────────────────────────────
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
# redis = in-memory cache
# try/except = if redis not installed, continue without cache
# This pattern = graceful degradation
# API still works without Redis, just slower

# ── LOGGING SETUP ─────────────────────────────────────────────
# logging = Python's built-in tool for recording events
# In production: logs go to Loki (our log aggregation system)
# For now: logs print to console

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
    # asctime = timestamp, levelname = INFO/ERROR/WARNING, message = text
)
logger = logging.getLogger(__name__)
# __name__ = name of current module = "main"
# This lets you filter logs by module in production

# ── CONFIG ────────────────────────────────────────────────────
PROJECT_ID   = os.getenv("PROJECT_ID",   "project-0ef73ac4-7a69-455a-987")
REGION       = os.getenv("REGION",       "us-central1")
ENDPOINT_ID  = os.getenv("ENDPOINT_ID",  "1972638759195246592")
REDIS_HOST   = os.getenv("REDIS_HOST",   "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL    = int(os.getenv("CACHE_TTL",  "3600"))
# TTL = Time To Live = how long cache entry stays valid (seconds)
# 3600 seconds = 1 hour
# After 1 hour Redis automatically deletes the entry
# Next request recalculates fresh prediction

# os.getenv("KEY", "default") = read environment variable
# If not set → use default value
# This is how 12-factor apps work:
#   Development: use defaults
#   Production: set real values via environment variables
#   No hardcoded secrets in code

# ── FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(
    title="WealthWise Portfolio API",
    description="Predicts client risk profiles using ML model on Vertex AI",
    version="1.0.0",
    # These appear in the auto-generated API documentation
    # Visit http://localhost:8000/docs to see interactive docs
)

# CORS middleware — allows browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production: specify exact domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STARTUP: Connect to Vertex AI and Redis ───────────────────
# These run ONCE when the server starts, not on every request
# Connecting on every request = slow (100ms per connection)
# Connecting once at startup = fast (reuse same connection)

vertex_endpoint = None
redis_client    = None

@app.on_event("startup")
async def startup():
    # async = asynchronous function
    # FastAPI uses async for non-blocking I/O
    # While waiting for Vertex AI response, server handles other requests
    global vertex_endpoint, redis_client

    # Connect to Vertex AI
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        vertex_endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        logger.info(f"Connected to Vertex AI endpoint: {ENDPOINT_ID}")
    except Exception as e:
        logger.error(f"Failed to connect to Vertex AI: {e}")

    # Connect to Redis
    if REDIS_AVAILABLE:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
                # decode_responses=True = return strings not bytes
            )
            redis_client.ping()  # test connection
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Running without cache.")
            redis_client = None

# ── REQUEST/RESPONSE MODELS ───────────────────────────────────
# Pydantic models define the SHAPE of data
# FastAPI automatically validates incoming JSON against these models
# If validation fails → 422 Unprocessable Entity response automatically

class PredictionRequest(BaseModel):
    # This defines what the API expects in the request body
    client_id:       str   = Field(..., description="Unique client identifier")
    equity_pct:      float = Field(..., ge=0, le=100, description="% in equity funds")
    debt_pct:        float = Field(..., ge=0, le=100, description="% in debt funds")
    gold_pct:        float = Field(..., ge=0, le=100, description="% in gold funds")
    total_num_funds: int   = Field(..., ge=1, le=50,  description="Number of funds")
    years_invested:  float = Field(..., ge=0, le=50,  description="Years investing")
    aum_lakh:        float = Field(..., ge=0,          description="AUM in lakhs")
    age_group:       str   = Field(..., description="Young/Middle-Aged/Senior")

    # Field(...) = required field (... means no default)
    # ge=0 = greater than or equal to 0
    # le=100 = less than or equal to 100
    # If equity_pct=-5 is sent → automatic 422 error

    @validator("age_group")
    def validate_age_group(cls, v):
        # Custom validator for age_group field
        # cls = the class itself (Pydantic style)
        # v = the value being validated
        allowed = ["Young", "Middle-Aged", "Senior"]
        if v not in allowed:
            raise ValueError(f"age_group must be one of {allowed}")
        return v

    @validator("equity_pct", "debt_pct", "gold_pct")
    def validate_percentages(cls, v):
        # Round to 2 decimal places
        return round(v, 2)

class PredictionResponse(BaseModel):
    # This defines what the API returns
    client_id:        str
    risk_profile:     str
    confidence:       str        # High/Medium/Low
    from_cache:       bool       # was this served from Redis cache?
    prediction_time:  str        # timestamp
    model_version:    str

class HealthResponse(BaseModel):
    status:       str
    vertex_ai:    str
    redis:        str
    timestamp:    str

# ── HELPER: Cache Key ─────────────────────────────────────────
def make_cache_key(request: PredictionRequest) -> str:
    # Create a unique cache key from the features
    # Same features = same key = same cached result
    # We hash the features so the key is short and consistent

    features_str = (
        f"{request.equity_pct}_{request.debt_pct}_{request.gold_pct}_"
        f"{request.total_num_funds}_{request.years_invested}_"
        f"{request.aum_lakh}_{request.age_group}"
    )
    # hashlib.md5 = creates a short fixed-length hash from any string
    # "equity=85_debt=10..." → "a3f9b2c1..." (32 characters)
    feature_hash = hashlib.md5(features_str.encode()).hexdigest()
    return f"wealthwise:risk:{request.client_id}:{feature_hash}"

# ── HELPER: Encode Features ───────────────────────────────────
def encode_features(request: PredictionRequest) -> list:
    # Convert request to feature list for Vertex AI
    # Order MUST match training: [equity_pct, debt_pct, gold_pct,
    # total_num_funds, years_invested, aum_lakh, age_group_encoded]

    age_map = {"Young": 0, "Middle-Aged": 1, "Senior": 2}
    return [
        request.equity_pct,
        request.debt_pct,
        request.gold_pct,
        float(request.total_num_funds),
        request.years_invested,
        request.aum_lakh,
        float(age_map.get(request.age_group, 1))
    ]

# ── HELPER: Decode Prediction ─────────────────────────────────
def decode_prediction(raw_prediction) -> tuple:
    label_map = {0: "Aggressive", 1: "Conservative", 2: "Moderate"}
    label = label_map.get(int(raw_prediction), str(raw_prediction))
    # Confidence based on how clear the prediction is
    # In production: use predict_proba() for actual probabilities
    confidence = "High"
    return label, confidence

# ── ROUTES ────────────────────────────────────────────────────
# Routes = URL paths the API responds to
# @app.get("/path")  = responds to GET requests
# @app.post("/path") = responds to POST requests

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Health check endpoint — called by Kubernetes every 10 seconds
    # If this returns error → Kubernetes restarts the container
    # This is how Kubernetes knows if our service is alive
    return HealthResponse(
        status    = "healthy",
        vertex_ai = "connected" if vertex_endpoint else "disconnected",
        redis     = "connected" if redis_client else "unavailable",
        timestamp = datetime.utcnow().isoformat()
    )

@app.get("/")
async def root():
    return {
        "service": "WealthWise Portfolio API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest, http_request: Request):
    # Main prediction endpoint
    # POST /predict with JSON body → returns risk profile

    logger.info(f"Prediction request for client: {request.client_id}")

    # ── CHECK CACHE FIRST ─────────────────────────────────────
    cache_key  = make_cache_key(request)
    from_cache = False

    if redis_client:
        cached = redis_client.get(cache_key)
        # redis.get(key) = returns value if exists, None if not
        if cached:
            logger.info(f"Cache HIT for {request.client_id}")
            cached_data = json.loads(cached)
            # Update from_cache flag and return cached response
            cached_data["from_cache"] = True
            return PredictionResponse(**cached_data)

    # ── CALL VERTEX AI ────────────────────────────────────────
    if not vertex_endpoint:
        raise HTTPException(
            status_code=503,
            detail="Vertex AI endpoint not available"
        )
        # 503 = Service Unavailable
        # Client knows to retry later

    try:
        features   = encode_features(request)
        response   = vertex_endpoint.predict(instances=[features])
        raw_pred   = response.predictions[0]
        risk_label, confidence = decode_prediction(raw_pred)

        logger.info(
            f"Vertex AI prediction for {request.client_id}: "
            f"{risk_label} (confidence: {confidence})"
        )

    except Exception as e:
        logger.error(f"Vertex AI prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # ── BUILD RESPONSE ────────────────────────────────────────
    result = PredictionResponse(
        client_id       = request.client_id,
        risk_profile    = risk_label,
        confidence      = confidence,
        from_cache      = False,
        prediction_time = datetime.utcnow().isoformat(),
        model_version   = "v1.0-random-forest"
    )

    # ── STORE IN CACHE ────────────────────────────────────────
    if redis_client:
        redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(result.dict())
            # setex = SET with EXpiry
            # After CACHE_TTL seconds → Redis deletes automatically
        )
        logger.info(f"Cached result for {request.client_id} ({CACHE_TTL}s TTL)")

    return result

@app.get("/stats")
async def get_stats():
    # Returns API statistics — useful for monitoring dashboard
    stats = {
        "service":    "portfolio-api",
        "version":    "1.0.0",
        "vertex_ai":  "connected" if vertex_endpoint else "disconnected",
        "cache":      "enabled"   if redis_client    else "disabled",
        "timestamp":  datetime.utcnow().isoformat(),
    }
    if redis_client:
        try:
            info = redis_client.info()
            stats["cache_memory_mb"] = round(
                info["used_memory"] / 1024 / 1024, 2
            )
            stats["cache_hits"]   = info.get("keyspace_hits", 0)
            stats["cache_misses"] = info.get("keyspace_misses", 0)
        except Exception:
            pass
    return stats