# =============================================================
# FILE: ml/train_risk_model.py
#
# WHAT IT DOES:
#   Loads the training CSV we just exported
#   Trains a Random Forest classifier
#   Tracks everything with MLflow (parameters, metrics, model)
#   Saves the trained model locally
#
# WHAT IS RANDOM FOREST?
#   Imagine you want to decide: "Is this client Conservative?"
#   One decision tree asks: "Is equity_pct < 30%? Yes → Conservative"
#   But one tree makes mistakes.
#   Random Forest = 100 decision trees, each trained slightly differently
#   Final answer = majority vote of all 100 trees
#   Much more accurate than one tree alone
#
# WHAT IS MLFLOW?
#   Every time you train a model you change things:
#     - Try 100 trees vs 200 trees
#     - Try different data splits
#     - Try different features
#   Without MLflow: you forget what you tried, lose track of results
#   With MLflow: every experiment logged automatically
#   You can compare runs and pick the best one
#
# COMPANY USE CASE:
#   Data scientist tries 10 different model configurations
#   MLflow records all 10 runs with their accuracy scores
#   Team reviews MLflow UI and picks the best model
#   Best model gets promoted to production on Vertex AI
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH      = "ml/data/training_features.csv"
MODEL_DIR      = "ml/model_artifacts"
MLFLOW_DIR     = "ml/mlruns"

# Hyperparameters — settings that control how the model trains
# We'll try these values and MLflow records which works best
N_ESTIMATORS   = 100    # number of trees in the forest
MAX_DEPTH      = 10     # how deep each tree can grow
                        # deeper = learns more detail but risks overfitting
MIN_SAMPLES    = 5      # minimum rows needed to split a tree node
TEST_SIZE      = 0.2    # 20% of data used for testing, 80% for training
RANDOM_STATE   = 42     # fixed seed = reproducible results every run

os.makedirs(MODEL_DIR, exist_ok=True)

# ── MLFLOW SETUP ──────────────────────────────────────────────
# MLflow tracking URI = where to save experiment logs
# Using local folder for now (in production: MLflow on GKE)

mlflow.set_tracking_uri("sqlite:///ml/mlflow.db")
mlflow.set_experiment("wealthwise-risk-model")
# An "experiment" in MLflow = a group of related runs
# Like a project folder that holds all your attempts

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading training data...")
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df):,}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# ── FEATURE COLUMNS ───────────────────────────────────────────
# These are the inputs to our model
# The model learns: given these numbers → predict risk profile

FEATURE_COLS = [
    "equity_pct",        # % in equity — most important feature
    "debt_pct",          # % in debt
    "gold_pct",          # % in gold
    "total_num_funds",   # number of funds held
    "years_invested",    # investing experience
    "aum_lakh",          # wealth level
    "age_group_encoded", # 0=Young, 1=Middle-Aged, 2=Senior
]

TARGET_COL = "target"  # what we're predicting

# ── HANDLE MISSING VALUES ─────────────────────────────────────
# years_invested has NaN (missing values)
# Strategy: fill with median (middle value of all non-null values)
# Why median not mean? Mean is affected by extreme values.
# If most clients invested 3 years but one invested 50 years,
# median stays around 3, mean gets pulled towards 50.

for col in FEATURE_COLS:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {null_count} nulls in '{col}' with median={median_val:.2f}")

# ── ENCODE TARGET ─────────────────────────────────────────────
# ML models need numbers, not text
# LabelEncoder converts: Aggressive=0, Conservative=1, Moderate=2
# (alphabetical order by default)

le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df[TARGET_COL])

print(f"\nLabel encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")
# le.classes_ = ['Aggressive', 'Conservative', 'Moderate']
# So predictions of 0=Aggressive, 1=Conservative, 2=Moderate

# ── TRAIN / TEST SPLIT ────────────────────────────────────────
# We split data into two parts:
#   Training set (80%): model LEARNS from this
#   Test set (20%): model is EVALUATED on this (never seen during training)
#
# WHY SPLIT?
# If you test on the same data you trained on,
# the model memorises answers → 100% accuracy → useless in real world
# Test set simulates "new clients the model has never seen"

X = df[FEATURE_COLS].values   # features matrix — shape: (49701, 7)
y = df["target_encoded"].values # target array  — shape: (49701,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y   # ensures same % of each class in train and test
                 # without this: test might have too many Conservatives
)

print(f"\nTraining set: {len(X_train):,} rows")
print(f"Test set:     {len(X_test):,} rows")

# ── TRAIN WITH MLFLOW TRACKING ────────────────────────────────
# Everything inside "with mlflow.start_run()" is tracked
# MLflow records: parameters, metrics, model file, artifacts

print("\nStarting MLflow run...")

with mlflow.start_run(run_name="random-forest-v1"):

    # ── LOG PARAMETERS ────────────────────────────────────────
    # Parameters = settings you chose before training
    # MLflow saves these so you can compare runs later
    # "I used 100 trees in run 1, 200 trees in run 2 — which was better?"

    mlflow.log_param("n_estimators",  N_ESTIMATORS)
    mlflow.log_param("max_depth",     MAX_DEPTH)
    mlflow.log_param("min_samples",   MIN_SAMPLES)
    mlflow.log_param("test_size",     TEST_SIZE)
    mlflow.log_param("random_state",  RANDOM_STATE)
    mlflow.log_param("features",      FEATURE_COLS)
    mlflow.log_param("training_rows", len(X_train))

    # ── TRAIN THE MODEL ───────────────────────────────────────
    print(f"Training Random Forest ({N_ESTIMATORS} trees, max_depth={MAX_DEPTH})...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,      # use ALL CPU cores — faster training
        class_weight="balanced"
        # class_weight="balanced" handles imbalanced classes
        # We have: Moderate=22383, Conservative=14800, Aggressive=12518
        # Without balancing: model learns to always predict Moderate
        # balanced = automatically gives more weight to minority classes
    )

    model.fit(X_train, y_train)
    print("Training complete!")

    # ── EVALUATE ──────────────────────────────────────────────
    # Make predictions on the TEST set (data model never saw)
    y_pred = model.predict(X_test)

    # ACCURACY = how many predictions were correct / total predictions
    # If 8500 out of 10000 correct → accuracy = 0.85 = 85%
    accuracy = accuracy_score(y_test, y_pred)

    # F1 SCORE = balance between precision and recall
    # Better metric than accuracy when classes are imbalanced
    # F1 = 1.0 is perfect, F1 = 0.0 is worst
    # macro = calculate F1 for each class separately, then average
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTest Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"F1 Score      : {f1:.4f}")

    # Classification report = precision, recall, F1 per class
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_
    )
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix = table showing right vs wrong predictions
    # Row = actual class, Column = predicted class
    # Diagonal = correct predictions
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # ── LOG METRICS ───────────────────────────────────────────
    # Metrics = results after training
    # MLflow saves these to compare across runs

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    # Random Forest tells you which features mattered most
    # This is very useful: "equity_pct is 60% responsible for predictions"
    # If a feature has 0% importance → remove it, simplifies model

    importances = model.feature_importances_
    # feature_importances_ = array of numbers, one per feature
    # Higher = more important for predictions
    # All values sum to 1.0

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLS, importances),
        key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    for feat, imp in zip(FEATURE_COLS, importances):
        mlflow.log_metric(f"importance_{feat}", imp)

    # ── SAVE MODEL ────────────────────────────────────────────
    # Log the trained model to MLflow
    # This saves the model file inside the MLflow run folder
    # Later we'll push this to Vertex AI Model Registry

    mlflow.sklearn.log_model(
        model,
        artifact_path="risk-model",
        registered_model_name="wealthwise-risk-model"
        # registered_model_name = adds to MLflow Model Registry
        # Model Registry = catalogue of all your production models
        # Each model has versions: v1, v2, v3...
        # You can tag a version as "Production" or "Staging"
    )

    # ── SAVE ARTIFACTS LOCALLY ────────────────────────────────
    import pickle

    # Save model as pickle (Python's way of saving objects)
    model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

    # Save label encoder (needed to decode predictions back to text)
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to: {le_path}")

    # Save metadata (feature names, classes, version info)
    metadata = {
        "model_type":    "RandomForestClassifier",
        "features":      FEATURE_COLS,
        "classes":       list(le.classes_),
        "n_estimators":  N_ESTIMATORS,
        "max_depth":     MAX_DEPTH,
        "accuracy":      round(accuracy, 4),
        "f1_score":      round(f1, 4),
        "training_rows": len(X_train),
        "test_rows":     len(X_test),
    }
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Get the MLflow run ID for reference
    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow Run ID: {run_id}")

print("\nTraining complete!")
print(f"\nTo view MLflow UI run:")
print(f"  cd {os.getcwd()}")
print(f"  mlflow ui --backend-store-uri file://ml/mlruns")
print(f"  Then open: http://localhost:5000")