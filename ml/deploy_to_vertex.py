# =============================================================
# FILE: ml/deploy_to_vertex.py
#
# WHAT IT DOES:
#   1. Uploads model files to GCS
#   2. Registers model in Vertex AI Model Registry
#   3. Creates a Vertex AI Endpoint (the URL)
#   4. Deploys model to endpoint (connects model to URL)
#   5. Saves endpoint ID for future predictions
#
# WHAT IS VERTEX AI?
#   Google's managed ML platform.
#   You give it your model file → it handles:
#     - Serving infrastructure (servers, load balancing)
#     - Auto-scaling (more traffic → more servers automatically)
#     - Health checks (restart if model crashes)
#     - Logging (every prediction logged)
#   You pay only for prediction time, not idle time.
#
# COMPANY USE CASE:
#   Model trained by data scientist on laptop
#   Deployed to Vertex AI → becomes production API
#   10,000 advisors across India hit the same endpoint
#   Vertex AI scales automatically — no DevOps work needed
# =============================================================

import os
import json
from google.cloud import storage, aiplatform

PROJECT_ID  = "project-0ef73ac4-7a69-455a-987"
REGION      = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-wealthwise"
GCS_PREFIX  = "models/risk-model-v1"

# sklearn 1.5 serving container — Google provides this pre-built
# It knows how to load a sklearn pickle and serve predictions
SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

LOCAL_MODEL_DIR = "ml/model_artifacts"

# ── INIT ──────────────────────────────────────────────────────
aiplatform.init(project=PROJECT_ID, location=REGION)
gcs_client = storage.Client(project=PROJECT_ID)

# ── STEP 1: Upload model files to GCS ────────────────────────
# Vertex AI needs the model files in GCS — it can't read
# from your laptop directly
# We upload 3 files:
#   model.pkl        — the trained Random Forest
#   label_encoder.pkl — converts 0/1/2 back to text
#   metadata.json    — model info (not needed by Vertex but good practice)

print(f"[1/4] Uploading model files to gs://{BUCKET_NAME}/{GCS_PREFIX}/")
bucket = gcs_client.bucket(BUCKET_NAME)

files_to_upload = {
    "risk_model.pkl":    "model.pkl",       # Vertex AI expects "model.pkl"
    "label_encoder.pkl": "label_encoder.pkl",
    "metadata.json":     "metadata.json",
}

for local_name, gcs_name in files_to_upload.items():
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)
    if not os.path.exists(local_path):
        print(f"  Skipping {local_name} — not found")
        continue
    blob = bucket.blob(f"{GCS_PREFIX}/{gcs_name}")
    blob.upload_from_filename(local_path)
    print(f"  Uploaded {local_name} → gs://{BUCKET_NAME}/{GCS_PREFIX}/{gcs_name}")

model_gcs_uri = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/"
print(f"  Model URI: {model_gcs_uri}")

# ── STEP 2: Register Model in Vertex AI Model Registry ───────
# Model Registry = catalogue of all your models
# Each model can have multiple versions (v1, v2, v3...)
# You can tag a version as "champion" (production) or "challenger" (testing)
# If new version performs worse → roll back to previous version in one click

print("\n[2/4] Registering model in Vertex AI Model Registry...")

model = aiplatform.Model.upload(
    display_name="wealthwise-risk-model-v1",
    artifact_uri=model_gcs_uri,
    serving_container_image_uri=SERVING_CONTAINER,
    # Vertex AI uses this container to serve predictions
    # The sklearn container knows how to:
    #   1. Load model.pkl on startup
    #   2. Accept POST requests with feature values
    #   3. Call model.predict() internally
    #   4. Return prediction as JSON response
)

print(f"  Model registered: {model.resource_name}")

# ── STEP 3: Create Endpoint ───────────────────────────────────
# Endpoint = the stable URL that never changes
# You can swap models behind the endpoint without changing the URL
# Old model → new model: clients calling the endpoint notice nothing
# This enables zero-downtime model updates

print("\n[3/4] Creating Vertex AI endpoint...")
endpoint = aiplatform.Endpoint.create(
    display_name="wealthwise-risk-endpoint-v1"
)
print(f"  Endpoint created: {endpoint.resource_name}")

# ── STEP 4: Deploy Model to Endpoint ─────────────────────────
# This step takes 5-10 minutes
# Vertex AI is:
#   - Pulling the sklearn Docker container
#   - Starting a VM with the container
#   - Loading model.pkl into memory
#   - Starting the prediction server
#   - Running health checks
#   - Connecting endpoint URL to the server

print("\n[4/4] Deploying model to endpoint (takes 5-10 min)...")

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="risk-classifier-v1",
    machine_type="n1-standard-2",
    # n1-standard-2 = 2 vCPUs, 7.5GB RAM
    # Enough for sklearn Random Forest predictions
    # For deep learning models you'd use GPU machines
    min_replica_count=1,   # always keep at least 1 server running
    max_replica_count=3,   # scale up to 3 servers under high traffic
    traffic_percentage=100 # send 100% of traffic to this model version
    # traffic_percentage enables canary deployments:
    # new model v2: traffic_percentage=10 (test with 10% of users)
    # if good → increase to 50% → then 100%
    # if bad  → rollback to 0% immediately
)

print("  Model deployed successfully!")

# ── STEP 5: Save Endpoint Info ────────────────────────────────
# Save endpoint ID locally so test script can use it
# without hardcoding

endpoint_info = {
    "endpoint_id":   endpoint.name,
    "endpoint_name": endpoint.resource_name,
    "model_id":      model.name,
    "model_uri":     model_gcs_uri,
    "region":        REGION,
    "project":       PROJECT_ID,
}

out_path = os.path.join(LOCAL_MODEL_DIR, "vertex_endpoint.json")
with open(out_path, "w") as f:
    json.dump(endpoint_info, f, indent=2)

print(f"\nDeployment complete!")
print(f"  Endpoint ID : {endpoint.name}")
print(f"  Saved to    : {out_path}")
print(f"\nNext: python ml/test_vertex_endpoint.py")