"""
Run monitoring pipeline on new month of data
Auto-generates realistic data based on historical patterns
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.abspath(os.path.join(current_dir, '..', 'script'))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from monitoring_pipeline import MonitoringPipeline

print("ADAPTIVE ML MONITORING SYSTEM")

# Configuration
districts = ['colombo', 'kandy', 'galle', 'badulla', 'gampaha', 'matale',
             'nuwara_eliya', 'kalutara', 'matara', 'anuradhapura',
             'hambantota', 'polonnaruwa']

num_samples = 10  # Number of samples per district (simulates daily/weekly data points)

#  AUTO-DETECT NEXT MONTH TO PROCESS


print("\n Auto-detecting next month to process...")

# Load historical dataset
historical_data_path = '../data/training/final_training_dataset.csv'

if not os.path.exists(historical_data_path):
    print(f"  Error: Training dataset not found at {historical_data_path}")
    print("  Please copy final_training_dataset.csv to component3/data/training/")
    sys.exit(1)

df_historical = pd.read_csv(historical_data_path)

# Get last month in dataset
last_month_str = df_historical['year_month'].iloc[-1]
last_year, last_month = map(int, last_month_str.split('-'))

# Calculate next month
if last_month == 12:
    next_year = last_year + 1
    next_month = 1
else:
    next_year = last_year
    next_month = last_month + 1

prediction_month = f'{next_year}-{next_month:02d}'

print(f"  Last month in dataset: {last_month_str}")
print(f"  Next month to process: {prediction_month}")

#  GENERATE REALISTIC DATA BASED ON HISTORICAL PATTERNS


print(f"\n Generating realistic data for {prediction_month}...")

# Get historical data for this specific month (e.g., all previous Augusts)
same_month_historical = df_historical[df_historical['month_num'] == next_month]

if len(same_month_historical) == 0:
    print(f"  Warning: No historical data for month {next_month}")
    print("  Using overall averages instead")
    same_month_historical = df_historical

print(f"  Using {len(same_month_historical)} historical samples from month {next_month}")

# Calculate tourism growth rate from historical trend
years_in_data = df_historical['year'].max() - df_historical['year'].min()
if years_in_data > 0:
    # Calculate average growth from first year to last year
    first_year_total = df_historical[df_historical['year'] == df_historical['year'].min()]['Total'].mean()
    last_year_total = df_historical[df_historical['year'] == df_historical['year'].max()]['Total'].mean()

    if first_year_total > 0:
        compound_growth = (last_year_total / first_year_total) ** (1 / years_in_data) - 1
        annual_growth_rate = max(0.02, min(0.10, compound_growth))  # Clip between 2% and 10%
    else:
        annual_growth_rate = 0.05
else:
    annual_growth_rate = 0.05

print(f"  Detected annual growth rate: {annual_growth_rate * 100:.1f}%")

# Years since last data point
years_diff = next_year - df_historical['year'].max()
if next_month <= df_historical[df_historical['year'] == df_historical['year'].max()]['month_num'].max():
    years_diff = max(0, years_diff - 1)

growth_factor = (1 + annual_growth_rate) ** years_diff

print(f"  Applying growth factor: {growth_factor:.3f} ({years_diff} years)")

# GENERATE PREDICTIONS (Simulated Component 1 outputs)


predictions_dict = {}

for d in districts:
    pred_col = f'{d}_predicted'
    tourist_col = f'{d}_tourists'

    if tourist_col in same_month_historical.columns:
        # Get historical average for this month
        historical_mean = same_month_historical[tourist_col].mean()
        historical_std = same_month_historical[tourist_col].std()

        if np.isnan(historical_std) or historical_std == 0:
            historical_std = historical_mean * 0.1  # 10% variation

        # Project to next year with growth
        projected_mean = historical_mean * growth_factor

        # Generate samples with realistic variation
        # Use normal distribution around mean (±1 std dev covers ~68% of data)
        samples = np.random.normal(projected_mean, historical_std * 0.3, num_samples)

        # Ensure no negative values
        samples = np.maximum(samples, projected_mean * 0.8)

        predictions_dict[pred_col] = samples
    else:
        # Fallback if column missing
        print(f"    Warning: {tourist_col} not found, using defaults")
        predictions_dict[pred_col] = np.random.randint(50000, 150000, num_samples)

predictions_df = pd.DataFrame(predictions_dict)
predictions_file = f'../data/predictions_{prediction_month}.csv'
predictions_df.to_csv(predictions_file, index=False)

print(f"   Generated predictions (mean values):")
for i, d in enumerate(districts[:3]):  # Show first 3
    print(f"    {d.title()}: {predictions_dict[f'{d}_predicted'].mean():,.0f} tourists")
print(f"    ... (12 districts total)")

# GENERATE ACTUALS (Simulated real tourist data with realistic forecast error)


actuals_dict = {}

# Realistic forecast error based on Component 1 MAPE (~15%)
# Use normal distribution: most errors within ±10%, some up to ±20%
for d in districts:
    pred_col = f'{d}_predicted'
    actual_col = f'{d}_tourists'

    # Get predictions
    predictions = predictions_dict[pred_col]

    # Add realistic forecast error
    # MAPE of 15% means errors are typically within ±10-15%
    # Use normal distribution with mean=0, std=0.08 (gives ~15% MAPE)
    errors = np.random.normal(0, 0.08, num_samples)

    # Apply errors
    actuals = predictions * (1 + errors)

    # Ensure realistic bounds
    actuals = np.maximum(actuals, predictions * 0.70)  # At least 70% of prediction
    actuals = np.minimum(actuals, predictions * 1.30)  # At most 130% of prediction

    actuals_dict[actual_col] = actuals.astype(int)

actuals_df = pd.DataFrame(actuals_dict)
actuals_file = f'../data/actuals_{prediction_month}.csv'
actuals_df.to_csv(actuals_file, index=False)

print(f"\n   Generated actuals with realistic forecast errors")
print(f"  Sample MAPE per district:")
for i, d in enumerate(districts[:3]):
    pred = predictions_dict[f'{d}_predicted']
    act = actuals_dict[f'{d}_tourists']
    mape = np.mean(np.abs((act - pred) / (act + 1e-10))) * 100
    print(f"    {d.title()}: {mape:.2f}%")

#  INITIALIZE AND RUN MONITORING PIPELINE


print(f"\n Running monitoring pipeline for {prediction_month}...")

pipeline = MonitoringPipeline(
    component1_model_path='../../component1/models',
    data_path='../data'
)

print(f"PROCESSING MONTH: {prediction_month}")

result = pipeline.process_new_month(
    prediction_month=prediction_month,
    predictions_file=predictions_file,
    actuals_file=actuals_file
)

#  DISPLAY RESULTS


print("MONITORING COMPLETE")

print("\nResults Summary:")

avg_mape = np.mean([m['mape'] for m in result['metrics']])
print(f"  Average MAPE: {avg_mape:.2f}%")

drift_detected = result['drift']['drift_detected']
print(f"  Drift Detected: {drift_detected}")
print(f"  Severity: {result['drift']['severity']}")
print(f"  Consensus: {result['drift']['votes']}/3 methods")

action = result['action']
print(f"  Action Taken: {action}")

xgb_weight = result['weights']['xgboost_weight']
lstm_weight = result['weights']['lstm_weight']
print(f"  Ensemble Weights: XGBoost={xgb_weight:.3f}, LSTM={lstm_weight:.3f}")

print("OUTPUTS")

print(f"  Database: ../data/monitoring.db")
print(f"  Reports: ../metrics/monitoring_reports/report_{prediction_month}.txt")
print(f"  Comparison: ../metrics/comparisons/comparison_{prediction_month}.csv")
if action in ['retrain_immediately', 'retrain_poor_performance']:
    print(f"  New Models: ../models/v{int(pipeline.current_version.replace('v', ''))}/")
