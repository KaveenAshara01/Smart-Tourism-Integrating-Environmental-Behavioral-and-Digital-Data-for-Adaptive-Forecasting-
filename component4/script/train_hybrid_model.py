import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import xgboost as xgb

# TensorFlow for Fusion NN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not found. Will use XGBoost only.")

import warnings

warnings.filterwarnings('ignore')

print("COMPONENT 4: HYBRID PREFERENCE MODEL TRAINING")
print("XGBoost + Fusion Neural Network with Negative Sampling")

# LOAD DATA

print("\n Loading data...")

attractions = pd.read_csv('../data/tourist_attractions.csv')
training_data = pd.read_csv('../data/itinerary_training_data_v2.csv')

print(f"  Attractions: {len(attractions)} loaded")
print(f"  Training samples: {len(training_data)} loaded")

# PREPARE FEATURES

print("\n Preparing user features...")

# User preference features
user_feature_cols = ['budget', 'available_days', 'distance_preference', 'num_travelers']

# Activity type encoding (multi-label)
activity_types = ['beach', 'historical', 'temple', 'national_park', 'waterfall',
                  'mountain', 'cultural', 'city', 'adventure', 'wildlife']

for activity in activity_types:
    training_data[f'pref_{activity}'] = training_data['activity_type'].apply(
        lambda x: 1 if activity in str(x).lower() else 0
    )
    user_feature_cols.append(f'pref_{activity}')

# Season encoding
training_data['season_spring'] = (training_data['season'] == 1).astype(int)
training_data['season_summer'] = (training_data['season'] == 2).astype(int)
training_data['season_autumn'] = (training_data['season'] == 3).astype(int)
training_data['season_winter'] = (training_data['season'] == 4).astype(int)
user_feature_cols.extend(['season_spring', 'season_summer', 'season_autumn', 'season_winter'])

print(f"  User features: {len(user_feature_cols)} dimensions")

# PREPARE ATTRACTION FEATURES


print("\n Preparing attraction features...")

# Encode attraction categories
for activity in activity_types:
    attractions[f'is_{activity}'] = (attractions['category'] == activity).astype(int)

attraction_feature_cols = [
                              'avg_duration_hours', 'avg_cost', 'outdoor', 'popularity_score',
                              'accessibility', 'tourist_density', 'safety_rating'
                          ] + [f'is_{activity}' for activity in activity_types]

print(f"  Attraction features: {len(attraction_feature_cols)} dimensions")

# CREATE TRAINING DATA WITH NEGATIVE SAMPLING


print("\n Creating training data with negative sampling...")

X_train_list = []
y_train_list = []

negative_per_positive = 3  # 3 negative samples per positive

for idx, row in training_data.iterrows():
    # Get selected attractions
    selected = eval(row['selected_attractions']) if isinstance(row['selected_attractions'], str) else row[
        'selected_attractions']
    selected = [int(x) for x in selected]

    # User features
    user_features = row[user_feature_cols].values

    # Positive samples (selected attractions)
    for attr_id in selected:
        attr_data = attractions[attractions['attraction_id'] == attr_id]
        if len(attr_data) == 0:
            continue

        attr_features = attr_data[attraction_feature_cols].iloc[0].values

        # Combine user + attraction features
        combined_features = np.concatenate([user_features, attr_features])
        X_train_list.append(combined_features)
        y_train_list.append(1)  # Positive

    # Negative samples (random non-selected attractions)
    non_selected = attractions[~attractions['attraction_id'].isin(selected)]['attraction_id'].values

    if len(non_selected) > 0:
        neg_samples = np.random.choice(non_selected,
                                       size=min(len(selected) * negative_per_positive, len(non_selected)),
                                       replace=False)

        for attr_id in neg_samples:
            attr_data = attractions[attractions['attraction_id'] == attr_id]
            attr_features = attr_data[attraction_feature_cols].iloc[0].values

            combined_features = np.concatenate([user_features, attr_features])
            X_train_list.append(combined_features)
            y_train_list.append(0)  # Negative

X = np.array(X_train_list)
y = np.array(y_train_list)

print(f"  Total samples: {len(X)}")
print(f"  Positive samples: {np.sum(y)}")
print(f"  Negative samples: {len(y) - np.sum(y)}")
print(f"  Positive rate: {np.mean(y):.3f}")

# SPLIT DATA


print("\n Splitting train/validation/test...")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# SCALE FEATURES


print("\n Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# TRAIN XGBOOST BASE MODEL


print("\n Training XGBoost base model...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train) - np.sum(y_train)) / np.sum(y_train),  # Handle imbalance
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

# Predictions
y_val_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

auc_xgb_val = roc_auc_score(y_val, y_val_proba_xgb)
auc_xgb_test = roc_auc_score(y_test, y_test_proba_xgb)

print(f"  XGBoost Validation AUC: {auc_xgb_val:.4f}")
print(f"  XGBoost Test AUC: {auc_xgb_test:.4f}")

# TRAIN FUSION NEURAL NETWORK


if HAS_TF:
    print("\n Training Fusion Neural Network...")

    # Add XGBoost predictions as feature
    X_train_fusion = np.concatenate([
        X_train_scaled,
        xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
    ], axis=1)

    X_val_fusion = np.concatenate([
        X_val_scaled,
        y_val_proba_xgb.reshape(-1, 1)
    ], axis=1)

    X_test_fusion = np.concatenate([
        X_test_scaled,
        y_test_proba_xgb.reshape(-1, 1)
    ], axis=1)

    # Build neural network
    fusion_model = keras.Sequential([
        layers.Input(shape=(X_train_fusion.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    fusion_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Train
    history = fusion_model.fit(
        X_train_fusion, y_train,
        validation_data=(X_val_fusion, y_val),
        epochs=50,
        batch_size=128,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # Evaluate
    y_val_proba_fusion = fusion_model.predict(X_val_fusion, verbose=0).flatten()
    y_test_proba_fusion = fusion_model.predict(X_test_fusion, verbose=0).flatten()

    auc_fusion_val = roc_auc_score(y_val, y_val_proba_fusion)
    auc_fusion_test = roc_auc_score(y_test, y_test_proba_fusion)

    print(f"  Fusion NN Validation AUC: {auc_fusion_val:.4f}")
    print(f"  Fusion NN Test AUC: {auc_fusion_test:.4f}")
    print(f"  Improvement: {(auc_fusion_test - auc_xgb_test) * 100:.2f}%")
else:
    fusion_model = None
    history = None
    auc_fusion_val = auc_xgb_val
    auc_fusion_test = auc_xgb_test

# SAVE MODELS


print("\n Saving models...")

os.makedirs('../models', exist_ok=True)

# Save XGBoost
with open('../models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save Fusion NN
if fusion_model is not None:
    fusion_model.save('../models/fusion_model.h5')

# Save scaler
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata
metadata = {
    'model_type': 'XGBoost + Fusion Neural Network',
    'negative_sampling_ratio': negative_per_positive,
    'user_feature_columns': user_feature_cols,
    'attraction_feature_columns': attraction_feature_cols,
    'activity_types': activity_types,
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'test_samples': len(X_test),
    'performance': {
        'xgboost_val_auc': float(auc_xgb_val),
        'xgboost_test_auc': float(auc_xgb_test),
        'fusion_val_auc': float(auc_fusion_val),
        'fusion_test_auc': float(auc_fusion_test),
        'improvement': float((auc_fusion_test - auc_xgb_test) * 100) if fusion_model else 0
    },
    'timestamp': datetime.now().isoformat()
}

with open('../models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("   XGBoost saved: ../models/xgboost_model.pkl")
if fusion_model:
    print("   Fusion NN saved: ../models/fusion_model.h5")
print("   Scaler saved: ../models/scaler.pkl")
print("   Metadata saved: ../models/model_metadata.json")

# SUMMARY

print("TRAINING COMPLETE!")

print("\nModel Performance Summary:")
print(f"  XGBoost Test AUC: {auc_xgb_test:.1%}")
if fusion_model:
    print(f"  Fusion NN Test AUC: {auc_fusion_test:.1%}")
    print(f"  Improvement: +{(auc_fusion_test - auc_xgb_test) * 100:.2f}%")

print(f"\nTraining Statistics:")
print(f"  Total samples: {len(X)}")
print(f"  Positive rate: {np.mean(y):.1%}")
print(f"  Negative sampling ratio: 1:{negative_per_positive}")
