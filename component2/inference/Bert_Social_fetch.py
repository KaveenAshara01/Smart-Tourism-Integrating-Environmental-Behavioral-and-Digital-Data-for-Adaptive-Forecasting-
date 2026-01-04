import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timezone
import warnings
import os
import json
import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


# CONFIG

YEAR = 2025
MONTH = 12
MONTH_STR = "2025-12"
RAW_OUTPUT = "202512Reviews.csv"
FINAL_OUTPUT = "202512.csv"

YOUTUBE_API_KEY = "AIzaSyBfm3vopeBDKbvT1QBOQughc-7wQXh49SA"
GOOGLE_MAPS_API_KEY = "AIzaSyCy-RXNKcFXjFzxuQcBNKTtBUaE9wg8AYo"

YOUTUBE_LIMIT = 50
MAPS_LIMIT = 5


# DISTRICTS

DISTRICTS = {
    "ampara": {"coords": [(6.8741, 81.0537)], "yt": ["Arugam Bay Sri Lanka travel"]},
    "anuradhapura": {"coords": [(8.3114, 80.4037)], "yt": ["Anuradhapura Sri Lanka travel"]},
    "badulla": {"coords": [(6.9934, 81.0550)], "yt": ["Ella Sri Lanka travel"]},
    "colombo": {"coords": [(6.9271, 79.8612)], "yt": ["Colombo Sri Lanka travel"]},
    "galle": {"coords": [(6.0535, 80.2210)], "yt": ["Galle Fort Sri Lanka travel"]},
    "gampaha": {"coords": [(7.0917, 79.9940)], "yt": ["Negombo Sri Lanka travel"]},
    "jaffna": {"coords": [(9.6615, 80.0255)], "yt": ["Jaffna Sri Lanka travel"]},
    "kalutara": {"coords": [(6.5854, 79.9607)], "yt": ["Bentota Sri Lanka travel"]},
    "kandy": {"coords": [(7.2906, 80.6337)], "yt": ["Kandy Sri Lanka travel"]},
    "matale": {"coords": [(7.4675, 80.6234)], "yt": ["Matale Sri Lanka travel"]},
    "nuwara_eliya": {"coords": [(6.9497, 80.7891)], "yt": ["Nuwara Eliya Sri Lanka travel"]},
    "polonnaruwa": {"coords": [(7.9403, 81.0188)], "yt": ["Polonnaruwa Sri Lanka travel"]},
    "trincomalee": {"coords": [(8.5874, 81.2152)], "yt": ["Trincomalee Sri Lanka travel"]},
}


# LOAD BERT ENSEMBLE

MODEL_DIR = Path("../model/BertModels")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading BERT ensemble...")

with open(MODEL_DIR / "config.json", "r") as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
bert_model = AutoModel.from_pretrained(config["bert_model"])
bert_model.to(DEVICE)
bert_model.eval()

ridge = joblib.load(MODEL_DIR / "ridge.joblib")
rf = joblib.load(MODEL_DIR / "rf.joblib")
xgb_model = joblib.load(MODEL_DIR / "xgb.joblib")
meta_learner = joblib.load(MODEL_DIR / "meta_learner.joblib")

MAX_LEN = config["max_len"]


# BERT ENCODER

@torch.no_grad()
def encode_text(text):
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    output = bert_model(**encoded)
    return output.last_hidden_state[:, 0, :].cpu().numpy()


# SENTIMENT PREDICTION (NEW)

def predict_sentiment(text):
    X = encode_text(text)

    r = ridge.predict(X)[0]
    f = rf.predict(X)[0]
    x = xgb_model.predict(X)[0]

    meta_X = np.array([[r, f, x]])
    return float(np.clip(meta_learner.predict(meta_X)[0], 0, 1))


# FETCH YOUTUBE COMMENTS

def fetch_youtube_comments(query, limit):
    comments = []

    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": 3
    }

    videos = requests.get(search_url, params=params).json().get("items", [])
    for v in videos:
        video_id = v["id"]["videoId"]

        comment_url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "key": YOUTUBE_API_KEY,
            "videoId": video_id,
            "part": "snippet",
            "maxResults": 50
        }

        res = requests.get(comment_url, params=params).json()
        for item in res.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            if snip["publishedAt"].startswith(MONTH_STR):
                comments.append({
                    "source": "youtube",
                    "text": snip["textDisplay"],
                    "created_at": snip["publishedAt"]
                })

            if len(comments) >= limit:
                return comments

        time.sleep(1)

    return comments


# FETCH GOOGLE MAPS REVIEWS

def fetch_google_maps_reviews(lat, lng, limit):
    reviews = []

    nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 5000,
        "type": "tourist_attraction",
        "key": GOOGLE_MAPS_API_KEY
    }

    places = requests.get(nearby_url, params=params).json().get("results", [])

    for place in places:
        if len(reviews) >= limit:
            break

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        d_params = {
            "place_id": place["place_id"],
            "fields": "reviews",
            "key": GOOGLE_MAPS_API_KEY
        }

        details = requests.get(details_url, params=d_params).json().get("result", {})
        for r in details.get("reviews", []):
            ts = datetime.fromtimestamp(r["time"], tz=timezone.utc)
            if ts.strftime("%Y-%m") == MONTH_STR:
                reviews.append({
                    "source": "google_maps",
                    "text": r["text"],
                    "created_at": ts.isoformat()
                })

            if len(reviews) >= limit:
                break

        time.sleep(0.5)

    return reviews


# MAIN PIPELINE

raw_rows = []
sentiment_rows = []

for district, info in DISTRICTS.items():
    texts = []

    for q in info["yt"]:
        texts.extend(fetch_youtube_comments(q, YOUTUBE_LIMIT))
        if len(texts) >= YOUTUBE_LIMIT:
            break

    for lat, lng in info["coords"]:
        texts.extend(fetch_google_maps_reviews(lat, lng, MAPS_LIMIT))
        if len(texts) >= YOUTUBE_LIMIT + MAPS_LIMIT:
            break

    for item in texts[:10]:
        score = predict_sentiment(item["text"])
        category = "positive" if score >= 0.7 else "neutral" if score >= 0.4 else "negative"

        raw_rows.append({
            "month": MONTH_STR,
            "district": district,
            "source": item["source"],
            "text": item["text"],
            "created_at": item["created_at"],
            "sentiment_score": score,
            "category": category
        })

        sentiment_rows.append({
            "district": district,
            "score": score,
            "category": category
        })


# SAVE OUTPUTS

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(RAW_OUTPUT, index=False)

df = pd.DataFrame(sentiment_rows)
overall = df["score"].mean()

row = {
    "month": MONTH_STR,
    "sentiment_score": overall,
    "sentiment_score_normalized": overall * 100,
    "review_count": len(df),
    "positive_count": (df["category"] == "positive").sum(),
    "neutral_count": (df["category"] == "neutral").sum(),
    "negative_count": (df["category"] == "negative").sum()
}

for d in DISTRICTS:
    subset = df[df["district"] == d]
    row[f"{d}_sentiment"] = subset["score"].mean() if not subset.empty else overall
    row[f"{d}_reviews"] = len(subset)

row["other_sentiment"] = overall
row["other_reviews"] = 0

final_df = pd.DataFrame([row])
final_df.to_csv(FINAL_OUTPUT, index=False)

print(f"Raw saved → {RAW_OUTPUT}")
print(f"Monthly saved → {FINAL_OUTPUT}")
