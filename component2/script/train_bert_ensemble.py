import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import torch
from transformers import AutoTokenizer, AutoModel


# CONFIG

DATA_PATH = "../data/processed/sentiment/reviews_labeled.csv"
OUTPUT_DIR = "../model/BertModels"
METRICS_DIR = "../metrics/bert"

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# LOAD DATA

print("Loading data...")
df = pd.read_csv(DATA_PATH)

texts = df["Text"].astype(str).tolist()
y = ((df["sentiment_score"] + 1) / 2).values  # normalize to 0–1


# LOAD BERT (FROZEN)

print("Loading BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
bert_model.to(device)
bert_model.eval()


# BERT EMBEDDING FUNCTION

@torch.no_grad()
def bert_encode(texts):
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        outputs = bert_model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


# EXTRACT EMBEDDINGS

print("Extracting BERT embeddings...")
X = bert_encode(texts)


# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# TRAIN BASE MODELS

print("Training Ridge...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)
xgb_model.fit(X_train, y_train)


# META-LEARNER (STACKING)

print("Training meta-learner...")

ridge_pred = ridge.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

meta_X = np.column_stack([ridge_pred, rf_pred, xgb_pred])
meta_learner = LinearRegression()
meta_learner.fit(meta_X, y_test)

ensemble_pred = meta_learner.predict(meta_X)
mae = mean_absolute_error(y_test, ensemble_pred)

print(f"Ensemble MAE: {mae:.4f}")


# SAVE MODELS

print("Saving models...")

joblib.dump(ridge, os.path.join(OUTPUT_DIR, "ridge.joblib"))
joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf.joblib"))
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb.joblib"))
joblib.dump(meta_learner, os.path.join(OUTPUT_DIR, "meta_learner.joblib"))

config = {
    "bert_model": BERT_MODEL_NAME,
    "max_len": MAX_LEN,
    "ensemble_features": ["ridge", "rf", "xgb"],
    "output_scale": "0_to_1"
}

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("All models saved to:", OUTPUT_DIR)


# METRICS & EVALUATION

ridge_mae = mean_absolute_error(y_test, ridge_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

#  SAVE METRICS TXT 
metrics_txt = os.path.join(METRICS_DIR, "performance.txt")
with open(metrics_txt, "w") as f:
    f.write("BERT ENSEMBLE SENTIMENT MODEL PERFORMANCE\n")
    f.write("=" * 50 + "\n")
    f.write(f"Timestamp: {datetime.now()}\n\n")
    f.write(f"Dataset size: {len(X)}\n")
    f.write(f"Train size: {len(X_train)}\n")
    f.write(f"Test size: {len(X_test)}\n\n")

    f.write("MAE Scores (lower is better)\n")
    f.write("-" * 30 + "\n")
    f.write(f"Ridge MAE:        {ridge_mae:.4f}\n")
    f.write(f"Random Forest MAE:{rf_mae:.4f}\n")
    f.write(f"XGBoost MAE:      {xgb_mae:.4f}\n")
    f.write(f"Ensemble MAE:     {mae:.4f}\n")

print("Saved metrics →", metrics_txt)

errors = ensemble_pred - y_test

plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Ensemble Error Distribution")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "error_distribution.png"))
plt.close()


models = ["Ridge", "Random Forest", "XGBoost", "Ensemble"]
maes = [ridge_mae, rf_mae, xgb_mae, mae]

plt.figure(figsize=(6, 4))
plt.bar(models, maes)
plt.ylabel("MAE")
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "model_mae_comparison.png"))
plt.close()



