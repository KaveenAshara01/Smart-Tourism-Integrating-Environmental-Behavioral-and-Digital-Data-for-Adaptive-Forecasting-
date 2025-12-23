import pandas as pd
import numpy as np
import os


print("CREATING FINAL TRAINING DATASET - COMPONENT 1")


print("\nMerging 4 data sources:")
print("  1. Tourist Arrivals (with district distribution)")
print("  2. Weather Data (12 districts, comprehensive)")
print("  3. Sentiment Scores (Component 2)")
print("  4. Crisis Scores (Component 2)")


# LOAD ALL DATA SOURCES



print(" LOADING DATA SOURCES")


# 1. Tourist Arrivals
print("\n1. Loading tourist arrivals...")
tourists_path = '../data/processed/tourist_arrivals_with_districts.csv'
if not os.path.exists(tourists_path):
    tourists_path = '../data/raw/tourist_arrivals_with_districts.csv'
tourists = pd.read_csv(tourists_path)
print(f"    Shape: {tourists.shape}")
print(f"    Date range: {tourists['year_month'].min()} to {tourists['year_month'].max()}")

# 2. Weather Data
print("\n2. Loading weather data...")
weather_path = '../data/processed/weather_data_all_districts_comprehensive.csv'
weather = pd.read_csv(weather_path)
print(f"    Shape: {weather.shape}")
print(f"    Date range: {weather['year_month'].min()} to {weather['year_month'].max()}")

# 3. Sentiment & Crisis Data (from Component 2)
print("\n3. Loading sentiment & crisis scores (Component 2)...")
# Try multiple possible paths
sentiment_paths = [
    '../../component2/data/processed/final/district_sentiment_crisis_complete_2015_2023.csv',
    '../data/processed/district_sentiment_crisis_complete_2015_2023.csv',
    '../data/raw/district_sentiment_crisis_complete_2015_2023.csv'
]

sentiment_crisis = None
for path in sentiment_paths:
    if os.path.exists(path):
        sentiment_crisis = pd.read_csv(path)
        print(f"    Loaded from: {path}")
        break

if sentiment_crisis is None:
    print(" ERROR: Could not find sentiment & crisis data!")
    print(" Please copy from Component 2:")
    print(" component2/data/processed/final/district_sentiment_crisis_complete_2015_2023.csv")
    print(" to")
    print(" component1/data/processed/")
    exit(1)

print(f" Shape: {sentiment_crisis.shape}")
print(f" Date range: {sentiment_crisis['month'].min()} to {sentiment_crisis['month'].max()}")

# Rename 'month' to 'year_month' for consistency
sentiment_crisis = sentiment_crisis.rename(columns={'month': 'year_month'})


# DATA ALIGNMENT CHECK



print(" DATA ALIGNMENT CHECK")


print("\nDate ranges:")
print(f"  Tourists:  {tourists['year_month'].min()} to {tourists['year_month'].max()} ({len(tourists)} months)")
print(f"  Weather:   {weather['year_month'].min()} to {weather['year_month'].max()} ({len(weather)} months)")
print(f"  Sentiment: {sentiment_crisis['year_month'].min()} to {sentiment_crisis['year_month'].max()} ({len(sentiment_crisis)} months)")

# Find common date range
all_dates = set(tourists['year_month']) & set(weather['year_month']) & set(sentiment_crisis['year_month'])
print(f"\n Common months across all sources: {len(all_dates)}")
if len(all_dates) > 0:
    print(f"  Range: {min(all_dates)} to {max(all_dates)}")
else:
    print("  ERROR: No overlapping dates found!")
    exit(1)


# MERGE DATA SOURCES



print(" MERGING DATA SOURCES")


print("\nStep 1: Merge tourists + weather...")
df = tourists.merge(weather, on='year_month', how='inner')
print(f"   Shape after merge: {df.shape}")

print("\nStep 2: Merge with sentiment & crisis...")
df = df.merge(sentiment_crisis, on='year_month', how='inner')
print(f"   Shape after merge: {df.shape}")

# Sort by date
df = df.sort_values('year_month').reset_index(drop=True)

print(f"\n Final merged dataset:")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")
print(f"  Total months: {len(df)}")


# ADD TIME-BASED FEATURES



print(" ADDING TIME-BASED FEATURES")


# Convert year_month to datetime for feature extraction
df['date'] = pd.to_datetime(df['year_month'] + '-01')

# Extract time features
df['year'] = df['date'].dt.year
df['month_num'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Cyclical encoding for month (captures seasonality better)
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Season indicators (Sri Lanka has 2 monsoons + 2 inter-monsoons)
def get_season(month):
    if month in [12, 1, 2, 3]:  # Dec-Mar: Main tourist season (dry in west/south)
        return 'peak_season'
    elif month in [4, 5]:  # Apr-May: Inter-monsoon
        return 'inter_monsoon_1'
    elif month in [6, 7, 8, 9]:  # Jun-Sep: Southwest monsoon
        return 'southwest_monsoon'
    else:  # Oct-Nov: Inter-monsoon
        return 'inter_monsoon_2'

df['season'] = df['month_num'].apply(get_season)

# Create dummy variables for season
season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)

# Trend feature (months since start)
df['month_index'] = range(len(df))

# Lag features for tourist arrivals (previous month)
tourist_cols = [col for col in df.columns if '_tourists' in col]
for col in tourist_cols:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag3'] = df[col].shift(3)

# Fill first lag with current value
for col in tourist_cols:
    df[f'{col}_lag1'].fillna(df[col], inplace=True)
    df[f'{col}_lag3'].fillna(df[col], inplace=True)

print(" Added time features:")
print("  - Year, Month, Quarter")
print("  - Cyclical month encoding (sin/cos)")
print("  - Season indicators (4 seasons)")
print("  - Trend index")
print("  - Lag features (1 & 3 months)")


# FEATURE SUMMARY



print(" FINAL DATASET SUMMARY")


# Categorize columns
tourist_cols = [col for col in df.columns if '_tourists' in col and 'lag' not in col]
tourist_lag_cols = [col for col in df.columns if '_tourists' in col and 'lag' in col]
weather_cols = [col for col in df.columns if any(x in col for x in ['temp', 'rain', 'wind', 'humidity', 'pressure', 'cloud', 'solar', 'sunshine', 'evap', 'precip', 'comfort', 'dry_month'])]
sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and '_tourists' not in col]
crisis_cols = [col for col in df.columns if 'score' in col.lower() and 'sentiment' not in col.lower()]
time_cols = ['year', 'month_num', 'quarter', 'month_sin', 'month_cos', 'month_index'] + [col for col in df.columns if 'season_' in col]
country_cols = [col for col in df.columns if col in ['India', 'China', 'Russia', 'UK', 'Germany', 'France', 'Australia', 'Netherland', 'Bangladesh', 'Maldives', 'Canada', 'US', 'Poland', 'Italy', 'Spain', 'Others']]

print(f"\nFeature Categories:")
print(f"  Tourist Districts:      {len(tourist_cols)} columns")
print(f"  Tourist Lag Features:   {len(tourist_lag_cols)} columns")
print(f"  Weather Variables:      {len(weather_cols)} columns")
print(f"  Sentiment Scores:       {len(sentiment_cols)} columns")
print(f"  Crisis Scores:          {len(crisis_cols)} columns")
print(f"  Time Features:          {len(time_cols)} columns")
print(f"  Origin Countries:       {len(country_cols)} columns")
print(f"  TOTAL:                  {df.shape[1]} columns")


# SAVE FINAL DATASET



print("SAVING FINAL DATASET")


output_file = '../data/processed/final_training_dataset.csv'
df.to_csv(output_file, index=False)

print(f"\n Saved: {output_file}")
print(f"  Shape: {df.shape}")
print(f"  Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Save feature lists for reference
feature_info = {
    'tourist_districts': tourist_cols,
    'tourist_lag_features': tourist_lag_cols,
    'weather_features': weather_cols[:50],  # Save first 50 (too many to list all)
    'sentiment_features': sentiment_cols,
    'crisis_features': crisis_cols,
    'time_features': time_cols,
    'country_features': country_cols,
    'total_features': df.shape[1],
    'total_months': len(df),
    'date_range': f"{df['year_month'].min()} to {df['year_month'].max()}"
}

import json
with open('../data/processed/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"\n Saved feature info: feature_info.json")


# SAMPLE DATA DISPLAY



print("SAMPLE DATA (First 3 Months)")


sample_cols = ['year_month', 'Total',
               'colombo_tourists', 'kandy_tourists', 'galle_tourists', 'badulla_tourists',
               'colombo_temp_mean', 'colombo_rain_sum',
               'sentiment_score', 'composite_crisis_score',
               'month_num', 'season']

print("\n", df[sample_cols].head(3).to_string(index=False))


# DATA QUALITY CHECKS



print("DATA QUALITY CHECKS")


# Check for missing values
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    print(f"\n⚠ Missing values found in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
else:
    print("\n No missing values")

# Check for duplicates
duplicates = df['year_month'].duplicated().sum()
if duplicates > 0:
    print(f"\n⚠ Duplicate months found: {duplicates}")
else:
    print(" No duplicate months")
