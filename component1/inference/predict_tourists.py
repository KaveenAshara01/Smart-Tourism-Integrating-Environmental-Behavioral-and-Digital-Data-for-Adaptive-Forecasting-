import pandas as pd
import numpy as np
import json
import pickle
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')


print("TOURIST ARRIVAL FORECASTING - INFERENCE ENGINE")


# Load the hardcoded JSON scenario
SCENARIO_FILE = 'test_scenario.json'

print(f"\n Loading scenario from {SCENARIO_FILE}...")

with open(SCENARIO_FILE, 'r') as f:
    scenario = json.load(f)

print(f"  Scenario: {scenario['scenario_name']}")
print(f"  Prediction for: {scenario['prediction_month']}")
print(f"  Description: {scenario['description']}")

# Load models and preprocessing objects
print("\n Loading trained models and preprocessing objects...")

with open('../models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

with open('../models/xgb_models.pkl', 'rb') as f:
    xgb_models = pickle.load(f)

lstm_model = keras.models.load_model('../models/lstm_model.h5', compile=False)

with open('../models/weather_scaler.pkl', 'rb') as f:
    weather_scaler = pickle.load(f)

with open('../models/pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

with open('../models/final_scaler.pkl', 'rb') as f:
    final_scaler = pickle.load(f)

print(f"  Loaded {len(xgb_models)} XGBoost models")
print(f"  Loaded LSTM model")
print(f"  Ensemble weights: XGBoost={metadata['ensemble_weights']['xgboost']:.2f}, LSTM={metadata['ensemble_weights']['lstm']:.2f}")

# Extract information from scenario
print("\n Building feature vector from scenario...")

districts = metadata['districts']
weather_data = scenario['weather_conditions']
sentiment_crisis = scenario['sentiment_and_crisis']
historical = scenario['historical_data']['previous_month_tourists']

# Parse prediction month
pred_year = int(scenario['prediction_month'].split('-')[0])
pred_month = int(scenario['prediction_month'].split('-')[1])

# Build non-PCA features
non_pca_features_dict = {}

# Tourist lag features
for district in districts:
    non_pca_features_dict[f'{district}_lag1'] = historical[district]

# Rolling averages
for district in districts:
    non_pca_features_dict[f'{district}_roll3'] = historical[district] * 0.98
    non_pca_features_dict[f'{district}_roll6'] = historical[district] * 0.95

# Sentiment and crisis lag features
non_pca_features_dict['sentiment_score_lag1'] = sentiment_crisis['sentiment_score']
non_pca_features_dict['composite_crisis_score_lag1'] = sentiment_crisis['composite_crisis_score']
non_pca_features_dict['terror_score_lag1'] = sentiment_crisis['terror_score']
non_pca_features_dict['economic_score_lag1'] = sentiment_crisis['economic_score']
non_pca_features_dict['unrest_score_lag1'] = sentiment_crisis['unrest_score']
non_pca_features_dict['diplomacy_score_lag1'] = sentiment_crisis['diplomacy_score']
non_pca_features_dict['tone_crisis_score_lag1'] = sentiment_crisis['tone_crisis_score']
non_pca_features_dict['disease_score_lag1'] = sentiment_crisis['disease_score']
non_pca_features_dict['crime_score_lag1'] = sentiment_crisis['crime_score']
non_pca_features_dict['disaster_score_lag1'] = sentiment_crisis['disaster_score']

# Time features
non_pca_features_dict['year'] = pred_year
non_pca_features_dict['month_num'] = pred_month
non_pca_features_dict['quarter'] = (pred_month - 1) // 3 + 1
non_pca_features_dict['month_sin'] = np.sin(2 * np.pi * pred_month / 12)
non_pca_features_dict['month_cos'] = np.cos(2 * np.pi * pred_month / 12)
non_pca_features_dict['is_peak_season'] = 1 if pred_month in [12, 1, 2, 7, 8] else 0
non_pca_features_dict['is_monsoon'] = 1 if pred_month in [4, 5, 10, 11] else 0

# Build weather features
weather_features_dict = {}
for district in districts:
    weather_features_dict[f'{district}_temp_mean_lag1'] = weather_data[district]['temp_mean']
    weather_features_dict[f'{district}_rain_sum_lag1'] = weather_data[district]['rain_sum']
    weather_features_dict[f'{district}_humidity_mean_lag1'] = weather_data[district]['humidity_mean']
    weather_features_dict[f'{district}_wind_mean_lag1'] = weather_data[district]['wind_mean']

print(f"  Built {len(non_pca_features_dict)} non-PCA features")
print(f"  Built {len(weather_features_dict)} weather features")

# Apply preprocessing
print("\n Applying preprocessing transformations...")

non_pca_feature_names = metadata['non_pca_features']
weather_lag_feature_names = metadata['weather_lag_features']

X_non_pca = pd.DataFrame([non_pca_features_dict])[non_pca_feature_names]
X_weather = pd.DataFrame([weather_features_dict])[weather_lag_feature_names]

# Scale weather features
X_weather_scaled = weather_scaler.transform(X_weather)

# Apply PCA
X_weather_pca = pca_model.transform(X_weather_scaled)
pca_feature_names = metadata['pca_feature_names']
X_weather_pca_df = pd.DataFrame(X_weather_pca, columns=pca_feature_names)

# Combine
X_final = pd.concat([X_non_pca.reset_index(drop=True), X_weather_pca_df.reset_index(drop=True)], axis=1)

# Final scaling
X_scaled = final_scaler.transform(X_final)

print(f"  Applied PCA: {len(weather_lag_feature_names)} → {len(pca_feature_names)} components")
print(f"  Final scaled features: {X_scaled.shape}")

# Make predictions
print("\n Generating predictions...")

# XGBoost predictions
xgb_predictions = np.zeros(len(districts))
for i, district in enumerate(districts):
    xgb_predictions[i] = xgb_models[district].predict(X_final)[0]

# LSTM predictions
X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
lstm_predictions = lstm_model.predict(X_lstm, verbose=0)[0]

# Ensemble
xgb_weight = metadata['ensemble_weights']['xgboost']
lstm_weight = metadata['ensemble_weights']['lstm']

ensemble_predictions = xgb_weight * xgb_predictions + lstm_weight * lstm_predictions

# Calculate metrics
total_district_visits = ensemble_predictions.sum()

DISTRICTS_PER_TOURIST = 6.5
estimated_unique_tourists = total_district_visits / DISTRICTS_PER_TOURIST

# Calculate visit probability (what % of total tourists visit each district)
visit_probabilities = (ensemble_predictions / estimated_unique_tourists) * 100

# Display results

print("PREDICTION RESULTS")


print(f"\nScenario: {scenario['scenario_name']}")
print(f"Prediction Month: {scenario['prediction_month']}")
print(f"Peak Season: {'Yes' if non_pca_features_dict['is_peak_season'] == 1 else 'No'}")
print(f"Monsoon Period: {'Yes' if non_pca_features_dict['is_monsoon'] == 1 else 'No'}")


print("COUNTRY-LEVEL FORECAST")

print(f"Estimated Unique Tourists to Sri Lanka: {estimated_unique_tourists:,.0f}")
print(f"Average Districts Visited per Tourist: {DISTRICTS_PER_TOURIST:.1f}")
print(f"Total District Visits: {total_district_visits:,.0f}")


print("DISTRICT-LEVEL FORECAST")

print(f"\n{'District':<20} {'Predicted':<15} {'Visit %':<12} {'Market Share':<12}")
print(f"{'':20} {'Visitors':<15} {'of Total':<12} {'of Visits':<12}")


for i, district in enumerate(districts):
    market_share = (ensemble_predictions[i] / total_district_visits) * 100
    print(f"{district.replace('_', ' ').title():<20} "
          f"{ensemble_predictions[i]:>14,.0f} "
          f"{visit_probabilities[i]:>11.1f}% "
          f"{market_share:>11.1f}%")


print(f"{'TOTAL':<20} {total_district_visits:>14,.0f} {'':12} {'100.0%':>12}")


print("TOP DESTINATIONS")


sorted_indices = np.argsort(ensemble_predictions)[::-1]
for rank, idx in enumerate(sorted_indices[:5], 1):
    print(f"{rank}. {districts[idx].replace('_', ' ').title()}: "
          f"{ensemble_predictions[idx]:,.0f} visitors "
          f"({visit_probabilities[idx]:.1f}% visit probability)")


print("MODEL CONFIDENCE")

print(f"Model MAPE: ±{metadata['test_performance']['mape']:.1f}%")
print(f"Prediction Range:")
print(f"  Unique Tourists: {estimated_unique_tourists * 0.85:,.0f} - {estimated_unique_tourists * 1.15:,.0f}")
print(f"  District Visits: {total_district_visits * 0.85:,.0f} - {total_district_visits * 1.15:,.0f}")

# Save predictions
output = {
    'scenario': scenario['scenario_name'],
    'prediction_month': scenario['prediction_month'],
    'country_level': {
        'estimated_unique_tourists': float(estimated_unique_tourists),
        'total_district_visits': float(total_district_visits),
        'avg_districts_per_tourist': float(DISTRICTS_PER_TOURIST)
    },
    'district_level': {
        district: {
            'predicted_visitors': float(pred),
            'visit_probability_pct': float(prob),
            'market_share_pct': float((pred / total_district_visits) * 100)
        }
        for district, pred, prob in zip(districts, ensemble_predictions, visit_probabilities)
    },
    'model_metadata': {
        'mape': metadata['test_performance']['mape'],
        'r2': metadata['test_performance']['r2'],
        'ensemble_weights': metadata['ensemble_weights']
    }
}

with open('prediction_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Predictions saved to: prediction_output.json")


print("INFERENCE COMPLETE!")

