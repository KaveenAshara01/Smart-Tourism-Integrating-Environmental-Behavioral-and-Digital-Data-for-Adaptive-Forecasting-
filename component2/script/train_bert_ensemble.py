import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)



# HELPERS

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_bar_comparison(model_names, values, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_grouped_bar(models, train_vals, val_vals, test_vals, ylabel, title, save_path):
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, train_vals, width=width, label="Train")
    plt.bar(x, val_vals, width=width, label="Validation")
    plt.bar(x + width, test_vals, width=width, label="Test")

    plt.xticks(x, models)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_distribution(errors, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=40)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Ensemble Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_pred, residuals, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Sentiment Score")
    plt.ylabel("Predicted Sentiment Score")
    plt.title("Actual vs Predicted (Ensemble)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cv_scores(model_names, cv_means, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, cv_means)
    plt.ylabel("5-Fold CV MAE")
    plt.title("Cross-Validation Performance Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_metric_block(name, metric_dict):
    print(f"{name}")
    for k, v in metric_dict.items():
        print(f"  {k}: {v:.4f}")


def write_text_report(report_path, report_data):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BERT ENSEMBLE SENTIMENT MODEL - VALIDATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {report_data['dataset']['total_samples']}\n")
        f.write(f"Train samples: {report_data['dataset']['train_samples']}\n")
        f.write(f"Validation samples: {report_data['dataset']['val_samples']}\n")
        f.write(f"Test samples: {report_data['dataset']['test_samples']}\n\n")

        f.write("BASE MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for model_name, splits in report_data["base_models"].items():
            f.write(f"{model_name}\n")
            for split_name, metrics in splits.items():
                f.write(f"  {split_name} -> ")
                f.write(
                    f"MAE={metrics['MAE']:.4f}, "
                    f"RMSE={metrics['RMSE']:.4f}, "
                    f"R2={metrics['R2']:.4f}\n"
                )
            f.write("\n")

        f.write("ENSEMBLE PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for split_name, metrics in report_data["ensemble"].items():
            f.write(
                f"{split_name} -> "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}\n"
            )
        f.write("\n")

        f.write("5-FOLD CROSS-VALIDATION MAE\n")
        f.write("-" * 30 + "\n")
        for model_name, value in report_data["cross_validation"].items():
            f.write(f"{model_name}: {value:.4f}\n")
        f.write("\n")

        f.write("OVERFITTING CHECK\n")
        f.write("-" * 30 + "\n")
        for model_name, gap_info in report_data["overfitting_check"].items():
            f.write(
                f"{model_name}: "
                f"Train MAE={gap_info['train_mae']:.4f}, "
                f"Validation MAE={gap_info['val_mae']:.4f}, "
                f"Gap={gap_info['gap']:.4f}, "
                f"Status={gap_info['status']}\n"
            )
        f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        f.write(
            "A model is considered more likely to be overfitting when its "
            "training MAE is much lower than its validation MAE.\n"
        )
        f.write(
            "The ensemble was trained correctly using validation predictions "
            "for the meta-learner, avoiding test-set leakage.\n"
        )



# LOAD DATA

print("Loading data...")
df = pd.read_csv(DATA_PATH)

texts = df["Text"].astype(str).tolist()
y = ((df["sentiment_score"] + 1) / 2).values.astype(np.float32)  # normalize to 0-1

print(f"Dataset size: {len(texts)}")



# LOAD BERT (FROZEN)

print("Loading BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
bert_model.to(device)
bert_model.eval()



# BERT EMBEDDING FUNCTION

@torch.no_grad()
def bert_encode(text_list):
    embeddings = []

    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="Encoding text"):
        batch = text_list[i:i + BATCH_SIZE]

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
print("Embedding shape:", X.shape)



# TRAIN / VALIDATION / TEST SPLIT

print("Splitting dataset into train / validation / test...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE
)

relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=relative_test_ratio,
    random_state=RANDOM_STATE
)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")



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
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)



# PREDICTIONS FOR EACH SPLIT

print("Generating predictions...")

# Ridge
ridge_train_pred = ridge.predict(X_train)
ridge_val_pred = ridge.predict(X_val)
ridge_test_pred = ridge.predict(X_test)

# Random Forest
rf_train_pred = rf.predict(X_train)
rf_val_pred = rf.predict(X_val)
rf_test_pred = rf.predict(X_test)

# XGBoost
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)



# CORRECT STACKING (NO TEST LEAKAGE)

print("Training meta-learner on validation predictions only...")

meta_X_val = np.column_stack([ridge_val_pred, rf_val_pred, xgb_val_pred])
meta_learner = LinearRegression()
meta_learner.fit(meta_X_val, y_val)

meta_X_train = np.column_stack([ridge_train_pred, rf_train_pred, xgb_train_pred])
meta_X_test = np.column_stack([ridge_test_pred, rf_test_pred, xgb_test_pred])

ensemble_train_pred = meta_learner.predict(meta_X_train)
ensemble_val_pred = meta_learner.predict(meta_X_val)
ensemble_test_pred = meta_learner.predict(meta_X_test)



# METRICS

print("Calculating metrics...")

ridge_metrics = {
    "Train": calculate_metrics(y_train, ridge_train_pred),
    "Validation": calculate_metrics(y_val, ridge_val_pred),
    "Test": calculate_metrics(y_test, ridge_test_pred)
}

rf_metrics = {
    "Train": calculate_metrics(y_train, rf_train_pred),
    "Validation": calculate_metrics(y_val, rf_val_pred),
    "Test": calculate_metrics(y_test, rf_test_pred)
}

xgb_metrics = {
    "Train": calculate_metrics(y_train, xgb_train_pred),
    "Validation": calculate_metrics(y_val, xgb_val_pred),
    "Test": calculate_metrics(y_test, xgb_test_pred)
}

ensemble_metrics = {
    "Train": calculate_metrics(y_train, ensemble_train_pred),
    "Validation": calculate_metrics(y_val, ensemble_val_pred),
    "Test": calculate_metrics(y_test, ensemble_test_pred)
}

print_metric_block("Ridge - Train", ridge_metrics["Train"])
print_metric_block("Ridge - Validation", ridge_metrics["Validation"])
print_metric_block("Ridge - Test", ridge_metrics["Test"])

print_metric_block("Random Forest - Train", rf_metrics["Train"])
print_metric_block("Random Forest - Validation", rf_metrics["Validation"])
print_metric_block("Random Forest - Test", rf_metrics["Test"])

print_metric_block("XGBoost - Train", xgb_metrics["Train"])
print_metric_block("XGBoost - Validation", xgb_metrics["Validation"])
print_metric_block("XGBoost - Test", xgb_metrics["Test"])

print_metric_block("Ensemble - Train", ensemble_metrics["Train"])
print_metric_block("Ensemble - Validation", ensemble_metrics["Validation"])
print_metric_block("Ensemble - Test", ensemble_metrics["Test"])



# CROSS-VALIDATION

print("Running 5-fold cross-validation for base models...")

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

ridge_cv_scores = cross_val_score(
    Ridge(alpha=1.0),
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

rf_cv_scores = cross_val_score(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

xgb_cv_scores = cross_val_score(
    xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

ridge_cv_mae = float(-ridge_cv_scores.mean())
rf_cv_mae = float(-rf_cv_scores.mean())
xgb_cv_mae = float(-xgb_cv_scores.mean())

print(f"Ridge CV MAE: {ridge_cv_mae:.4f}")
print(f"Random Forest CV MAE: {rf_cv_mae:.4f}")
print(f"XGBoost CV MAE: {xgb_cv_mae:.4f}")

# SAVE MODELS

print("Saving models...")

joblib.dump(ridge, os.path.join(OUTPUT_DIR, "ridge.joblib"))
joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf.joblib"))
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb.joblib"))
joblib.dump(meta_learner, os.path.join(OUTPUT_DIR, "meta_learner.joblib"))

config = {
    "bert_model": BERT_MODEL_NAME,
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "random_state": RANDOM_STATE,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "ensemble_features": ["ridge", "rf", "xgb"],
    "output_scale": "0_to_1",
    "stacking_strategy": "base models trained on training set, meta-learner trained on validation predictions"
}
save_json(config, os.path.join(OUTPUT_DIR, "config.json"))



# SAVE METRICS JSON

report_data = {
    "dataset": {
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test))
    },
    "base_models": {
        "Ridge": ridge_metrics,
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics
    },
    "ensemble": ensemble_metrics,
    "cross_validation": {
        "Ridge_CV_MAE": ridge_cv_mae,
        "RandomForest_CV_MAE": rf_cv_mae,
        "XGBoost_CV_MAE": xgb_cv_mae
    },

}

save_json(report_data, os.path.join(METRICS_DIR, "validation_metrics.json"))



# WRITE TEXT REPORT

write_text_report(
    os.path.join(METRICS_DIR, "validation_report.txt"),
    report_data
)



# PLOTS

print("Saving plots...")

# Error distribution
test_errors = ensemble_test_pred - y_test
plot_error_distribution(
    test_errors,
    os.path.join(METRICS_DIR, "error_distribution.png")
)

# Residual plot
residuals = y_test - ensemble_test_pred
plot_residuals(
    ensemble_test_pred,
    residuals,
    os.path.join(METRICS_DIR, "residual_plot.png")
)

# Actual vs Predicted
plot_actual_vs_predicted(
    y_test,
    ensemble_test_pred,
    os.path.join(METRICS_DIR, "actual_vs_predicted.png")
)

# Base + ensemble test MAE comparison
model_names = ["Ridge", "Random Forest", "XGBoost", "Ensemble"]
test_maes = [
    ridge_metrics["Test"]["MAE"],
    rf_metrics["Test"]["MAE"],
    xgb_metrics["Test"]["MAE"],
    ensemble_metrics["Test"]["MAE"]
]
plot_bar_comparison(
    model_names,
    test_maes,
    ylabel="Test MAE",
    title="Model Test Performance Comparison",
    save_path=os.path.join(METRICS_DIR, "model_test_mae_comparison.png")
)

# Overfitting comparison plot
plot_grouped_bar(
    models=["Ridge", "Random Forest", "XGBoost", "Ensemble"],
    train_vals=[
        ridge_metrics["Train"]["MAE"],
        rf_metrics["Train"]["MAE"],
        xgb_metrics["Train"]["MAE"],
        ensemble_metrics["Train"]["MAE"]
    ],
    val_vals=[
        ridge_metrics["Validation"]["MAE"],
        rf_metrics["Validation"]["MAE"],
        xgb_metrics["Validation"]["MAE"],
        ensemble_metrics["Validation"]["MAE"]
    ],
    test_vals=[
        ridge_metrics["Test"]["MAE"],
        rf_metrics["Test"]["MAE"],
        xgb_metrics["Test"]["MAE"],
        ensemble_metrics["Test"]["MAE"]
    ],
    ylabel="MAE",
    title="Train vs Validation vs Test Error",
    save_path=os.path.join(METRICS_DIR, "overfitting_check.png")
)

# Cross-validation plot
plot_cv_scores(
    ["Ridge", "Random Forest", "XGBoost"],
    [ridge_cv_mae, rf_cv_mae, xgb_cv_mae],
    os.path.join(METRICS_DIR, "cross_validation_mae.png")
)



# FINAL CONSOLE SUMMARY

print("\n" + "=" * 70)
print("VALIDATION PIPELINE COMPLETED")
print("=" * 70)
print(f"Models saved to: {OUTPUT_DIR}")
print(f"Metrics and plots saved to: {METRICS_DIR}")
print("\nFinal Ensemble Test Metrics:")
print(f"MAE  : {ensemble_metrics['Test']['MAE']:.4f}")
print(f"RMSE : {ensemble_metrics['Test']['RMSE']:.4f}")
print(f"R2   : {ensemble_metrics['Test']['R2']:.4f}")
print("=" * 70)