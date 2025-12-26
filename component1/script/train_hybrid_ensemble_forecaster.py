import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pickle

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

print("HYBRID ENSEMBLE - TOURIST FORECASTING WITH AUTO PCA (OPTIMIZED)")

# Configuration
DISTRICTS = ['colombo', 'kandy', 'galle', 'badulla', 'gampaha', 'matale',
             'nuwara_eliya', 'kalutara', 'matara', 'anuradhapura',
             'hambantota', 'polonnaruwa']

TARGET_COLS = [f'{d}_tourists' for d in DISTRICTS]

PCA_VARIANCE_THRESHOLD = 0.95
TEST_SIZE = 10

os.makedirs('../models', exist_ok=True)
os.makedirs('../metrics', exist_ok=True)

# Load data
print("\n Loading data...")
df = pd.read_csv('../data/processed/final_training_dataset.csv')
print(f"  Dataset shape: {df.shape}")
print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")

# Feature engineering improvements
print("\nCreating advanced features...")

# Remove existing lag columns
cols_to_drop = [col for col in df.columns if '_lag' in col.lower()]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"  Removed {len(cols_to_drop)} existing lag columns")

# Create lag features for tourists (1 month back)
lag_features = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    if col in df.columns:
        lag_col = f'{district}_lag1'
        df[lag_col] = df[col].shift(1)
        lag_features.append(lag_col)

# NEW: Create rolling averages (3-month and 6-month)
rolling_features = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    if col in df.columns:
        # 3-month rolling average
        roll3_col = f'{district}_roll3'
        df[roll3_col] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
        rolling_features.append(roll3_col)

        # 6-month rolling average
        roll6_col = f'{district}_roll6'
        df[roll6_col] = df[col].shift(1).rolling(window=6, min_periods=1).mean()
        rolling_features.append(roll6_col)

print(f" Created {len(rolling_features)} rolling average features")

# Weather lag features
weather_lag_features = []
weather_base_features = ['temp_mean', 'rain_sum', 'humidity_mean', 'wind_mean']

for district in DISTRICTS:
    for feat in weather_base_features:
        col = f'{district}_{feat}'
        if col in df.columns:
            lag_col = f'{district}_{feat}_lag1'
            df[lag_col] = df[col].shift(1)
            weather_lag_features.append(lag_col)

# Sentiment lag features
sentiment_lag_features = []
if 'sentiment_score' in df.columns:
    df['sentiment_score_lag1'] = df['sentiment_score'].shift(1)
    sentiment_lag_features.append('sentiment_score_lag1')

# Crisis lag features
crisis_lag_features = []
crisis_base_cols = ['composite_crisis_score', 'terror_score', 'economic_score',
                    'unrest_score', 'diplomacy_score', 'tone_crisis_score',
                    'disease_score', 'crime_score', 'disaster_score']

for col in crisis_base_cols:
    if col in df.columns:
        lag_col = f'{col}_lag1'
        df[lag_col] = df[col].shift(1)
        crisis_lag_features.append(lag_col)

# Time features
time_features = ['year', 'month_num', 'quarter', 'month_sin', 'month_cos']
time_features = [col for col in time_features if col in df.columns]

# Peak season indicators (Dec, Jan, Feb, Jul, Aug)
df['is_peak_season'] = df['month_num'].isin([12, 1, 2, 7, 8]).astype(int)
df['is_monsoon'] = df['month_num'].isin([4, 5, 10, 11]).astype(int)
time_features.extend(['is_peak_season', 'is_monsoon'])

print(f" Added peak season and monsoon indicators")

# Drop rows with NaN
rows_before = len(df)
all_feature_cols = lag_features + rolling_features + weather_lag_features + sentiment_lag_features + crisis_lag_features
df = df.dropna(subset=all_feature_cols)
rows_after = len(df)

print(f"  Dropped {rows_before - rows_after} rows with NaN")
print(f"  Final dataset: {rows_after} samples")

print(f"\n  Feature breakdown:")
print(f"    Tourist lags: {len(lag_features)}")
print(f"    Rolling averages: {len(rolling_features)}")
print(f"    Weather lags: {len(weather_lag_features)}")
print(f"    Sentiment lags: {len(sentiment_lag_features)}")
print(f"    Crisis lags: {len(crisis_lag_features)}")
print(f"    Time features: {len(time_features)}")

# Apply PCA to weather features
print("\n Applying automatic PCA to weather features...")

# Non-PCA features
non_pca_features = lag_features + rolling_features + sentiment_lag_features + crisis_lag_features + time_features

X_non_pca = df[non_pca_features].copy()
X_weather = df[weather_lag_features].copy()

X_non_pca = X_non_pca.fillna(X_non_pca.mean())
X_weather = X_weather.fillna(X_weather.mean())

# Scale weather features
weather_scaler = StandardScaler()
X_weather_scaled = weather_scaler.fit_transform(X_weather)

# Apply PCA
pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
X_weather_pca = pca.fit_transform(X_weather_scaled)

n_components = X_weather_pca.shape[1]
explained_variance = pca.explained_variance_ratio_.sum()

print(f"  PCA Results:")
print(f"    Original weather features: {len(weather_lag_features)}")
print(f"    Optimal PCA components: {n_components}")
print(f"    Explained variance: {explained_variance:.2%}")

# Combine features
pca_feature_names = [f'weather_PC{i + 1}' for i in range(n_components)]
X_weather_pca_df = pd.DataFrame(X_weather_pca, columns=pca_feature_names, index=X_weather.index)

X_final = pd.concat([X_non_pca, X_weather_pca_df], axis=1)
y = df[TARGET_COLS].copy()

print(f"\n  Final feature count: {X_final.shape[1]}")

# Split data
print("\n Splitting data (time series)...")

X_train = X_final.iloc[:-TEST_SIZE]
X_test = X_final.iloc[-TEST_SIZE:]
y_train = y.iloc[:-TEST_SIZE]
y_test = y.iloc[-TEST_SIZE:]

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# Scale features
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train)
X_test_scaled = final_scaler.transform(X_test)

# XGBoost
print("\n Training XGBoost model...")

xgb_models = {}
xgb_predictions_test = np.zeros((len(X_test), len(DISTRICTS)))

for i, district in enumerate(DISTRICTS):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train.iloc[:, i])
    xgb_predictions_test[:, i] = model.predict(X_test)
    xgb_models[district] = model

xgb_mae = mean_absolute_error(y_test, xgb_predictions_test)
xgb_r2 = r2_score(y_test, xgb_predictions_test)
epsilon = 1e-10
xgb_mape = np.mean(np.abs((y_test - xgb_predictions_test) / (y_test + epsilon))) * 100

print(f"  XGBoost - MAE: {xgb_mae:.2f}, R²: {xgb_r2:.4f}, MAPE: {xgb_mape:.2f}%")

# LSTM
print("\n Training LSTM model...")

# Set additional seeds for full reproducibility
import os

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Force CPU-only execution for determinism (comment out if too slow)
# tf.config.set_visible_devices([], 'GPU')

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


def build_lstm_model(input_shape, output_dim, seed=42):
    # Set seed for weight initialization
    tf.random.set_seed(seed)

    inputs = layers.Input(shape=input_shape)

    # Use seed in initializers for full reproducibility
    x = layers.LSTM(128, return_sequences=True, dropout=0.3,
                    kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                    recurrent_initializer=keras.initializers.Orthogonal(seed=seed))(inputs)
    x = layers.LSTM(64, dropout=0.3,
                    kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                    recurrent_initializer=keras.initializers.Orthogonal(seed=seed))(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)
    x = layers.Dropout(0.3, seed=seed)(x)
    x = layers.Dense(32, activation='relu',
                     kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)

    outputs = layers.Dense(output_dim, activation='linear',
                           kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


lstm_model = build_lstm_model((1, X_train_scaled.shape[1]), len(DISTRICTS), seed=42)

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)

# Train with deterministic settings
history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=300,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    shuffle=False,  # Disable shuffling for reproducibility
    verbose=0
)

lstm_predictions_test = lstm_model.predict(X_test_lstm, verbose=0)
lstm_mae = mean_absolute_error(y_test, lstm_predictions_test)
lstm_r2 = r2_score(y_test, lstm_predictions_test)
lstm_mape = np.mean(np.abs((y_test - lstm_predictions_test) / (y_test + epsilon))) * 100

print(f"  LSTM - MAE: {lstm_mae:.2f}, R²: {lstm_r2:.4f}, MAPE: {lstm_mape:.2f}%")
print(f"  Training stopped at epoch: {len(history.history['loss'])}")

# Optimize ensemble weights based on MAPE (not R²)
print("\n Creating ensemble (optimized for MAPE)...")

if xgb_mape < lstm_mape:
    xgb_weight = 0.7
    lstm_weight = 0.3
else:
    xgb_weight = 0.3
    lstm_weight = 0.7

ensemble_predictions_test = xgb_weight * xgb_predictions_test + lstm_weight * lstm_predictions_test

ensemble_mae = mean_absolute_error(y_test, ensemble_predictions_test)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions_test))
ensemble_r2 = r2_score(y_test, ensemble_predictions_test)
ensemble_mape = np.mean(np.abs((y_test - ensemble_predictions_test) / (y_test + epsilon))) * 100

print(f"\n  Ensemble Weights (MAPE-optimized):")
print(f"    XGBoost: {xgb_weight:.2f}")
print(f"    LSTM: {lstm_weight:.2f}")
print(f"\n  Ensemble Performance:")
print(f"    MAE: {ensemble_mae:.2f}")
print(f"    RMSE: {ensemble_rmse:.2f}")
print(f"    R²: {ensemble_r2:.4f}")
print(f"    MAPE: {ensemble_mape:.2f}%")

# Per-district performance
district_metrics = []
for i, district in enumerate(DISTRICTS):
    district_mae = mean_absolute_error(y_test.iloc[:, i], ensemble_predictions_test[:, i])
    district_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], ensemble_predictions_test[:, i]))
    district_r2 = r2_score(y_test.iloc[:, i], ensemble_predictions_test[:, i])
    district_mape = np.mean(np.abs((y_test.iloc[:, i] - ensemble_predictions_test[:, i]) /
                                   (y_test.iloc[:, i] + epsilon))) * 100

    district_metrics.append({
        'District': district.replace('_', ' ').title(),
        'MAE': district_mae,
        'RMSE': district_rmse,
        'R2': district_r2,
        'MAPE': district_mape
    })

district_df = pd.DataFrame(district_metrics)

print("\n  Per-District Performance:")
print(district_df.to_string(index=False))

# Feature importance
print("\n Analyzing feature importance...")

feature_importance_dict = {}
for district, model in xgb_models.items():
    importance = model.feature_importances_
    for feat, imp in zip(X_final.columns, importance):
        if feat not in feature_importance_dict:
            feature_importance_dict[feat] = []
        feature_importance_dict[feat].append(imp)

feature_importance_df = pd.DataFrame([
    {'Feature': feat, 'Importance': np.mean(imps)}
    for feat, imps in feature_importance_dict.items()
]).sort_values('Importance', ascending=False)

top_20_features = feature_importance_df.head(20)

print("\n  Top 20 Most Important Features:")
print(top_20_features.to_string(index=False))

# Save models
print("\n Saving models and artifacts...")

with open('../models/xgb_models.pkl', 'wb') as f:
    pickle.dump(xgb_models, f)

lstm_model.save('../models/lstm_model.h5')

with open('../models/weather_scaler.pkl', 'wb') as f:
    pickle.dump(weather_scaler, f)

with open('../models/pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('../models/final_scaler.pkl', 'wb') as f:
    pickle.dump(final_scaler, f)

# Save metadata
metadata = {
    'districts': DISTRICTS,
    'target_cols': TARGET_COLS,
    'n_pca_components': int(n_components),
    'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
    'explained_variance': float(explained_variance),
    'non_pca_features': non_pca_features,
    'weather_lag_features': weather_lag_features,
    'pca_feature_names': pca_feature_names,
    'final_feature_names': X_final.columns.tolist(),
    'ensemble_weights': {
        'xgboost': float(xgb_weight),
        'lstm': float(lstm_weight)
    },
    'test_performance': {
        'mae': float(ensemble_mae),
        'rmse': float(ensemble_rmse),
        'r2': float(ensemble_r2),
        'mape': float(ensemble_mape)
    }
}

with open('../models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  All models saved")

# Generate visualizations (same as before)
print("\nGenerating visualizations...")

# Graph 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.tab20(np.linspace(0, 1, 12))
for i, (district, color) in enumerate(zip(DISTRICTS, colors)):
    ax.scatter(y_test.iloc[:, i], ensemble_predictions_test[:, i],
               alpha=0.6, s=80, color=color, label=district.replace('_', ' ').title())

min_val = min(y_test.min().min(), ensemble_predictions_test.min())
max_val = max(y_test.max().max(), ensemble_predictions_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual Tourist Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Tourist Count', fontsize=12, fontweight='bold')
ax.set_title('Actual vs Predicted Tourist Arrivals (All 12 Districts)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../metrics/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("  Saved: actual_vs_predicted.png")
plt.close()

# Graph 2: Residual Plot
residuals = y_test.values - ensemble_predictions_test

fig, ax = plt.subplots(figsize=(12, 8))
for i, (district, color) in enumerate(zip(DISTRICTS, colors)):
    ax.scatter(ensemble_predictions_test[:, i], residuals[:, i],
               alpha=0.6, s=60, color=color, label=district.replace('_', ' ').title())

ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Tourist Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
ax.set_title('Residual Plot - Error Distribution', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../metrics/residual_plot.png', dpi=300, bbox_inches='tight')
print("  Saved: residual_plot.png")
plt.close()

# Graph 3: Feature Importance
fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(range(len(top_20_features)), top_20_features['Importance'], alpha=0.8, color='steelblue')
ax.set_yticks(range(len(top_20_features)))
ax.set_yticklabels(top_20_features['Feature'], fontsize=9)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Feature Importance (XGBoost Average)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../metrics/feature_importance.png', dpi=300, bbox_inches='tight')
print("  Saved: feature_importance.png")
plt.close()

# Graph 4: Per-District Performance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.bar(range(len(district_df)), district_df['MAE'], color='steelblue', alpha=0.8)
ax1.set_xticks(range(len(district_df)))
ax1.set_xticklabels(district_df['District'], rotation=45, ha='right')
ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax1.set_title('MAE by District', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[0, 1]
ax2.bar(range(len(district_df)), district_df['RMSE'], color='coral', alpha=0.8)
ax2.set_xticks(range(len(district_df)))
ax2.set_xticklabels(district_df['District'], rotation=45, ha='right')
ax2.set_ylabel('Root Mean Squared Error', fontsize=11, fontweight='bold')
ax2.set_title('RMSE by District', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes[1, 0]
ax3.bar(range(len(district_df)), district_df['R2'], color='mediumseagreen', alpha=0.8)
ax3.set_xticks(range(len(district_df)))
ax3.set_xticklabels(district_df['District'], rotation=45, ha='right')
ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax3.set_title('R² Score by District', fontsize=12, fontweight='bold')
ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='Target (0.7)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes[1, 1]
ax4.bar(range(len(district_df)), district_df['MAPE'], color='mediumpurple', alpha=0.8)
ax4.set_xticks(range(len(district_df)))
ax4.set_xticklabels(district_df['District'], rotation=45, ha='right')
ax4.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
ax4.set_title('MAPE by District', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Performance Metrics by District', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../metrics/district_performance.png', dpi=300, bbox_inches='tight')
print("  Saved: district_performance.png")
plt.close()

# Save performance metrics
with open('../metrics/performance_metrics.txt', 'w') as f:
    f.write("OPTIMIZED HYBRID ENSEMBLE - TOURIST FORECASTING\n")

    f.write("Model Architecture:\n")
    f.write("  Ensemble Components:\n")
    f.write("    - XGBoost (n_estimators=300, max_depth=6, gamma=0.1)\n")
    f.write("    - LSTM (128 64 units with dropout 0.3 and L2 regularization)\n")
    f.write("  MAPE-optimized weighted ensemble\n\n")

    f.write("Feature Engineering:\n")
    f.write(
        f"  Automatic PCA: {len(weather_lag_features)}  {n_components} components ({explained_variance:.2%} variance)\n")
    f.write(f"  Rolling averages: 3-month and 6-month windows\n")
    f.write(f"  Peak season indicators: December, January, February, July, August\n")
    f.write(f"  Total features: {X_final.shape[1]}\n\n")

    f.write("Dataset:\n")
    f.write(f"  Training samples: {len(X_train)} months\n")
    f.write(f"  Test samples: {len(X_test)} months\n\n")

    f.write("OVERALL PERFORMANCE (Test Set)\n")

    f.write(f"Mean Absolute Error (MAE): {ensemble_mae:,.2f} tourists\n")
    f.write(f"Root Mean Squared Error (RMSE): {ensemble_rmse:,.2f} tourists\n")
    f.write(f"Mean Absolute Percentage Error (MAPE): {ensemble_mape:.2f}%\n")
    f.write(f"R² Score: {ensemble_r2:.4f}\n\n")

    f.write("PER-DISTRICT PERFORMANCE\n")

    f.write(f"{'District':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'R²':<10}\n")
    f.write("-" * 80 + "\n")
    for _, row in district_df.iterrows():
        f.write(f"{row['District']:<20} {row['MAE']:<12,.2f} {row['RMSE']:<12,.2f} "
                f"{row['MAPE']:<10.2f} {row['R2']:<10.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("BASE MODEL PERFORMANCE\n")

    f.write(f"XGBoost: MAE={xgb_mae:,.2f}, R²={xgb_r2:.4f}, MAPE={xgb_mape:.2f}%\n")
    f.write(f"LSTM: MAE={lstm_mae:,.2f}, R²={lstm_r2:.4f}, MAPE={lstm_mape:.2f}%\n")
    f.write(f"Ensemble: MAE={ensemble_mae:,.2f}, R²={ensemble_r2:.4f}, MAPE={ensemble_mape:.2f}%\n")
    f.write(f"Weights: XGBoost={xgb_weight:.2f}, LSTM={lstm_weight:.2f}\n\n")

    f.write("TOP 20 FEATURES\n")

    for idx, (_, row) in enumerate(top_20_features.iterrows(), 1):
        f.write(f"{idx:2d}. {row['Feature']:<45} {row['Importance']:.4f}\n")

    f.write("\n" + "=" * 80 + "\n")

print("  Saved: performance_metrics.txt")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")

print(f"\nFinal Performance:")
print(f"  MAE: {ensemble_mae:,.2f} tourists")
print(f"  RMSE: {ensemble_rmse:,.2f} tourists")
print(f"  R²: {ensemble_r2:.4f}")
print(f"  MAPE: {ensemble_mape:.2f}%")

print("\nAll artifacts saved to:")
print("  Models: component1/models/")
print("  Metrics: component1/metrics/")
