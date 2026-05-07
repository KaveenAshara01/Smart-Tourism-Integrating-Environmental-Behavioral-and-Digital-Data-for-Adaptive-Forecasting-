import os
import json
import joblib
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel


# CONFIG

MODEL_DIR = Path("../model/BertModels")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded sample sentences
SAMPLES = [
    "Absolutely amazing experience! The beaches were clean and people were friendly.",
    "The hotel was okay, nothing special but not terrible either.",
    "Terrible service. The staff was rude and the place was dirty.",
    "Good value for money. Would recommend to friends.",
    "Overcrowded and overpriced. Not worth the hype."
]


# LOAD CONFIG

with open(MODEL_DIR / "config.json", "r") as f:
    config = json.load(f)

BERT_MODEL_NAME = config["bert_model"]
MAX_LEN = config["max_len"]


# LOAD BERT

print("Loading BERT...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
bert_model.to(DEVICE)
bert_model.eval()


# LOAD ENSEMBLE MODELS

print("Loading ensemble models...")
ridge = joblib.load(MODEL_DIR / "ridge.joblib")
rf = joblib.load(MODEL_DIR / "rf.joblib")
xgb_model = joblib.load(MODEL_DIR / "xgb.joblib")
meta_learner = joblib.load(MODEL_DIR / "meta_learner.joblib")


# BERT EMBEDDING FUNCTION

@torch.no_grad()
def encode_texts(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = bert_model(**encoded)

    # CLS token embedding
    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings.cpu().numpy()


# SENTIMENT LABEL FUNCTION

def sentiment_label(score):
    if score >= 0.7:
        return "Positive"
    elif score >= 0.4:
        return "Neutral"
    else:
        return "Negative"


# RUN INFERENCE

print("\nBERT ENSEMBLE SENTIMENT INFERENCE")
print("=" * 60)

X = encode_texts(SAMPLES)

ridge_pred = ridge.predict(X)
rf_pred = rf.predict(X)
xgb_pred = xgb_model.predict(X)

meta_X = np.column_stack([ridge_pred, rf_pred, xgb_pred])
final_scores = np.clip(meta_learner.predict(meta_X), 0, 1)


# DISPLAY RESULTS

for i, text in enumerate(SAMPLES):
    raw_score = float(final_scores[i])
    normalized_score = raw_score * 100
    label = sentiment_label(raw_score)

    print(f"\nSample {i + 1}")
    print("-" * 60)
    print(f"Text: {text}")
    print(f"Raw sentiment score (0–1): {raw_score:.4f}")
    print(f"Normalized score (0–100): {normalized_score:.1f}")
    print(f"Sentiment label: {label}")

print("\nInference completed successfully.")
