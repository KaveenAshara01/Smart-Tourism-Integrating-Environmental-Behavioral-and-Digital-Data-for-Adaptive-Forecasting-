import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Deterministic training
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'


print("ADAPTIVE MODEL TRAINING")


# Configuration
DISTRICTS = ['colombo', 'kandy', 'galle', 'badulla', 'gampaha', 'matale',
             'nuwara_eliya', 'kalutara', 'matara', 'anuradhapura',
             'hambantota', 'polonnaruwa']

TARGET_COLS = [f'{d}_tourists' for d in DISTRICTS]
PCA_VARIANCE_THRESHOLD = 0.95
TEST_SIZE = 10

# Parse command line arguments
TRAINING_MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'  # 'full', 'incremental', 'finetune'
WINDOW_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 24  # For incremental mode
EXISTING_MODELS_PATH = sys.argv[3] if len(sys.argv) > 3 else None  # For fine-tuning

print(f"\nTraining Mode: {TRAINING_MODE.upper()}")
print(f"Window Size: {WINDOW_SIZE} months" if TRAINING_MODE == 'incremental' else "")
print(f"Base Models: {EXISTING_MODELS_PATH}" if TRAINING_MODE == 'finetune' else "")

os.makedirs('../models/temp', exist_ok=True)

# Load data
print("\n Loading data...")
df = pd.read_csv('../data/training/final_training_dataset.csv')
print(f"  Dataset shape: {df.shape}")
print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")

# Apply window for incremental learning
if TRAINING_MODE == 'incremental':
    df = df.tail(WINDOW_SIZE + TEST_SIZE + 1)  # +1 for lag creation
    print(f"  Using last {WINDOW_SIZE + TEST_SIZE} months for incremental training")

# Feature engineering (same as Component 1)
print("\n Creating features...")

# Remove existing lag columns
cols_to_drop = [col for col in df.columns if '_lag' in col.lower()]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Create lag features
lag_features = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    if col in df.columns:
        lag_col = f'{district}_lag1'
        df[lag_col] = df[col].shift(1)
        lag_features.append(lag_col)

# Rolling averages
rolling_features = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    if col in df.columns:
        roll3_col = f'{district}_roll3'
        df[roll3_col] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
        rolling_features.append(roll3_col)

        roll6_col = f'{district}_roll6'
        df[roll6_col] = df[col].shift(1).rolling(window=6, min_periods=1).mean()
        rolling_features.append(roll6_col)

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

# Sentiment and crisis lags
sentiment_lag_features = []
if 'sentiment_score' in df.columns:
    df['sentiment_score_lag1'] = df['sentiment_score'].shift(1)
    sentiment_lag_features.append('sentiment_score_lag1')

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

# Peak season indicators
df['is_peak_season'] = df['month_num'].isin([12, 1, 2, 7, 8]).astype(int)
df['is_monsoon'] = df['month_num'].isin([4, 5, 10, 11]).astype(int)
time_features.extend(['is_peak_season', 'is_monsoon'])

# Drop NaN
all_feature_cols = lag_features + rolling_features + weather_lag_features + sentiment_lag_features + crisis_lag_features
df = df.dropna(subset=all_feature_cols)

print(f"  Final dataset: {len(df)} samples")

# Apply PCA to weather features
print("\n Applying PCA...")

non_pca_features = lag_features + rolling_features + sentiment_lag_features + crisis_lag_features + time_features

X_non_pca = df[non_pca_features].copy().fillna(0)
X_weather = df[weather_lag_features].copy().fillna(0)

# Scale weather
weather_scaler = StandardScaler()
X_weather_scaled = weather_scaler.fit_transform(X_weather)

# Apply PCA
pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
X_weather_pca = pca.fit_transform(X_weather_scaled)

n_components = X_weather_pca.shape[1]
pca_feature_names = [f'weather_PC{i + 1}' for i in range(n_components)]
X_weather_pca_df = pd.DataFrame(X_weather_pca, columns=pca_feature_names, index=X_weather.index)

X_final = pd.concat([X_non_pca, X_weather_pca_df], axis=1)
y = df[TARGET_COLS].copy()

print(f"  PCA: {len(weather_lag_features)} → {n_components} components ({pca.explained_variance_ratio_.sum():.2%})")

# Split data
print("\n Splitting data...")
X_train = X_final.iloc[:-TEST_SIZE]
X_test = X_final.iloc[-TEST_SIZE:]
y_train = y.iloc[:-TEST_SIZE]
y_test = y.iloc[-TEST_SIZE:]

print(f"  Training: {len(X_train)}, Test: {len(X_test)}")

# Scale features
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train)
X_test_scaled = final_scaler.transform(X_test)

# XGBoost Training
print("\n Training XGBoost...")

xgb_models = {}
xgb_predictions_test = np.zeros((len(X_test), len(DISTRICTS)))

if TRAINING_MODE == 'finetune' and EXISTING_MODELS_PATH:
    print("  Loading existing XGBoost models for warm-start...")
    with open(os.path.join(EXISTING_MODELS_PATH, 'xgb_models.pkl'), 'rb') as f:
        existing_xgb = pickle.load(f)

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

    # Warm-start if fine-tuning
    if TRAINING_MODE == 'finetune' and EXISTING_MODELS_PATH and district in existing_xgb:
        print(f"    Warm-starting {district}...")
        # XGBoost warm-start: train with fewer iterations
        model.set_params(n_estimators=100)  # Fewer iterations for fine-tuning

    model.fit(X_train, y_train.iloc[:, i])
    xgb_predictions_test[:, i] = model.predict(X_test)
    xgb_models[district] = model

xgb_mae = mean_absolute_error(y_test, xgb_predictions_test)
xgb_r2 = r2_score(y_test, xgb_predictions_test)
xgb_mape = np.mean(np.abs((y_test - xgb_predictions_test) / (y_test + 1e-10))) * 100

print(f"  XGBoost - MAE: {xgb_mae:.2f}, R²: {xgb_r2:.4f}, MAPE: {xgb_mape:.2f}%")

# LSTM Training
print("\n Training LSTM...")

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


def build_lstm_model(input_shape, output_dim, seed=42):
    tf.random.set_seed(seed)
    inputs = layers.Input(shape=input_shape)

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

    return Model(inputs=inputs, outputs=outputs)


# Fine-tuning mode: Load existing model and freeze layers
if TRAINING_MODE == 'finetune' and EXISTING_MODELS_PATH:
    print("  Loading existing LSTM for transfer learning...")
    lstm_model = keras.models.load_model(os.path.join(EXISTING_MODELS_PATH, 'lstm_model.h5'))

    # Freeze early layers
    for layer in lstm_model.layers[:-3]:
        layer.trainable = False
    print(f"  Froze {len(lstm_model.layers) - 3} layers, fine-tuning top 3")

    # Lower learning rate
    lstm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    epochs = 50
else:
    # Full training
    lstm_model = build_lstm_model((1, X_train_scaled.shape[1]), len(DISTRICTS))
    lstm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    epochs = 300

early_stop = EarlyStopping(monitor='loss', patience=30 if TRAINING_MODE != 'finetune' else 15,
                           restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)

history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=epochs,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    shuffle=False,
    verbose=0
)

lstm_predictions_test = lstm_model.predict(X_test_lstm, verbose=0)
lstm_mae = mean_absolute_error(y_test, lstm_predictions_test)
lstm_r2 = r2_score(y_test, lstm_predictions_test)
lstm_mape = np.mean(np.abs((y_test - lstm_predictions_test) / (y_test + 1e-10))) * 100

print(f"  LSTM - MAE: {lstm_mae:.2f}, R²: {lstm_r2:.4f}, MAPE: {lstm_mape:.2f}%")
print(f"  Stopped at epoch: {len(history.history['loss'])}")

# Ensemble
print("\n Creating ensemble...")

if xgb_mape < lstm_mape:
    xgb_weight, lstm_weight = 0.7, 0.3
else:
    xgb_weight, lstm_weight = 0.3, 0.7

ensemble_predictions_test = xgb_weight * xgb_predictions_test + lstm_weight * lstm_predictions_test

ensemble_mae = mean_absolute_error(y_test, ensemble_predictions_test)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions_test))
ensemble_r2 = r2_score(y_test, ensemble_predictions_test)
ensemble_mape = np.mean(np.abs((y_test - ensemble_predictions_test) / (y_test + 1e-10))) * 100

print(f"\n  Ensemble Performance:")
print(f"    MAE: {ensemble_mae:.2f}")
print(f"    RMSE: {ensemble_rmse:.2f}")
print(f"    R²: {ensemble_r2:.4f}")
print(f"    MAPE: {ensemble_mape:.2f}%")

# Save models
print("\n Saving models...")

with open('../models/temp/xgb_models.pkl', 'wb') as f:
    pickle.dump(xgb_models, f)

lstm_model.save('../models/temp/lstm_model.h5')

with open('../models/temp/weather_scaler.pkl', 'wb') as f:
    pickle.dump(weather_scaler, f)

with open('../models/temp/pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('../models/temp/final_scaler.pkl', 'wb') as f:
    pickle.dump(final_scaler, f)

# Metadata
metadata = {
    'districts': DISTRICTS,
    'target_cols': TARGET_COLS,
    'training_mode': TRAINING_MODE,
    'window_size': WINDOW_SIZE if TRAINING_MODE == 'incremental' else len(df),
    'training_samples': len(X_train),
    'n_pca_components': int(n_components),
    'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
    'explained_variance': float(pca.explained_variance_ratio_.sum()),
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
    },
    'timestamp': datetime.now().isoformat()
}

with open('../models/temp/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Models saved to ../models/temp/")


print(f"\nMode: {TRAINING_MODE.upper()}")
print(f"Performance: MAE={ensemble_mae:.2f}, R²={ensemble_r2:.4f}, MAPE={ensemble_mape:.2f}%")