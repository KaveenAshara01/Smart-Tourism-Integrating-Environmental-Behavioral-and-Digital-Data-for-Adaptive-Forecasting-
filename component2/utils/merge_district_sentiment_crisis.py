import pandas as pd
import numpy as np


print("MERGING DISTRICT SENTIMENT + CRISIS SCORES")


# Load district-level sentiment
print("\n1. Loading district sentiment data...")
sentiment = pd.read_csv('../data/processed/sentiment/monthly_sentiment_combined_2015_2023.csv')
print(f"   Sentiment data: {sentiment.shape[0]} months, {sentiment.shape[1]} columns")
print(f"   Date range: {sentiment['month'].min()} to {sentiment['month'].max()}")

# Load crisis scores
print("\n2. Loading crisis scores...")
crisis = pd.read_csv('../data/processed/crisis/monthly_crisis_scores_2015_2025.csv')
print(f"   Crisis data: {crisis.shape[0]} months, {crisis.shape[1]} columns")
print(f"   Date range: {crisis['month'].min()} to {crisis['month'].max()}")

# Merge datasets
print("\n3. Merging datasets on 'month'...")
combined = sentiment.merge(crisis, on='month', how='outer')
combined = combined.sort_values('month').reset_index(drop=True)

print(f"   Combined data: {combined.shape[0]} months, {combined.shape[1]} columns")
print(f"   Date range: {combined['month'].min()} to {combined['month'].max()}")

# Save full combined dataset
output_file = '../data/processed/final/district_sentiment_crisis_combined_2015_2025.csv'
combined.to_csv(output_file, index=False)
print(f"\n4. Saved combined dataset:")
print(f"   File: {output_file}")
print(f"   Size: {combined.shape}")

# Create a version with only complete data (2015-2023)
complete_data = combined[combined['sentiment_score'].notna()].copy()
output_complete = '../data/processed/final/district_sentiment_crisis_complete_2015_2023.csv'
complete_data.to_csv(output_complete, index=False)
print(f"\n5. Saved complete data (no missing sentiment):")
print(f"   File: {output_complete}")
print(f"   Size: {complete_data.shape}")


print("SAMPLE DATA (2023-01 to 2023-03)")


sample = combined[combined['month'].str.startswith('2023')].head(3)
key_cols = ['month', 'sentiment_score',
            'colombo_sentiment', 'kandy_sentiment', 'galle_sentiment', 'badulla_sentiment',
            'composite_crisis_score', 'unrest_score', 'economic_score']

print("\nSample values:")
print(sample[key_cols].to_string(index=False))

print(" MERGE COMPLETE!")

print("\nOutput files:")
print("  1. district_sentiment_crisis_combined_2015_2025.csv")
print("     → Full dataset (131 months, includes 2024-2025 with NaN sentiment)")
print("  2. district_sentiment_crisis_complete_2015_2023.csv")
print("     → Complete data only (100 months, no missing values)")
print("     → USE THIS FOR TRAINING YOUR FORECASTING MODEL")