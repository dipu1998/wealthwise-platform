# =============================================================
# FILE: ml/test_vertex_endpoint.py
#
# WHAT IT DOES:
#   Sends real prediction requests to our live Vertex AI endpoint
#   Tests 3 different client profiles
#   Shows predicted risk profile for each
#
# THIS IS THE MOMENT OF TRUTH:
#   Model trained on laptop → deployed to Google's servers
#   Now we call it like any real application would
#   Mobile app, dashboard, Airflow — all call this same way
# =============================================================

import json
from google.cloud import aiplatform

PROJECT_ID  = "project-0ef73ac4-7a69-455a-987"
REGION      = "us-central1"

# Load endpoint ID we saved during deployment
with open("ml/model_artifacts/vertex_endpoint.json") as f:
    info = json.load(f)

endpoint_id = info["endpoint_id"]
print(f"Calling endpoint: {endpoint_id}")

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_id)

# ── TEST CASES ────────────────────────────────────────────────
# Feature order MUST match training:
# [equity_pct, debt_pct, gold_pct, total_num_funds,
#  years_invested, aum_lakh, age_group_encoded]

test_clients = [
    {
        "name": "Young aggressive investor",
        "features": [85.0, 10.0, 5.0, 6, 3.0, 25.0, 0],
        "expected": "Aggressive"
    },
    {
        "name": "Senior conservative investor",
        "features": [5.0, 85.0, 10.0, 3, 15.0, 200.0, 2],
        "expected": "Conservative"
    },
    {
        "name": "Middle-aged moderate investor",
        "features": [45.0, 40.0, 15.0, 4, 7.0, 80.0, 1],
        "expected": "Moderate"
    },
]

print("\nTesting live Vertex AI predictions...\n")
print("-" * 55)

correct = 0
for client in test_clients:
    response   = endpoint.predict(instances=[client["features"]])
    prediction = response.predictions[0]

    # LabelEncoder encoded: Aggressive=0, Conservative=1, Moderate=2
    label_map  = {0: "Aggressive", 1: "Conservative", 2: "Moderate"}
    predicted_label = label_map.get(int(prediction), str(prediction))

    match = predicted_label == client["expected"]
    if match:
        correct += 1

    status = "CORRECT" if match else "WRONG"
    print(f"Client  : {client['name']}")
    print(f"Expected: {client['expected']}")
    print(f"Got     : {predicted_label}  [{status}]")
    print("-" * 55)

print(f"\nResult: {correct}/{len(test_clients)} correct")
print("\nVertex AI endpoint is live and working!")