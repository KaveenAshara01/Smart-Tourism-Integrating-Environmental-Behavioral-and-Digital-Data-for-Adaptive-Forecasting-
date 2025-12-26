import pandas as pd
import numpy as np

# Load GDELT data
print("Loading GDELT crisis data...")
gdelt = pd.read_csv('../data/crisis.csv')

print(f"Total months: {len(gdelt)}")
print(f"Date range: {gdelt['month'].min()} to {gdelt['month'].max()}")

# Crisis categories
crisis_categories = [
    'unrest', 'terror', 'economic', 'disaster', 
    'disease', 'crime', 'diplomacy'
]

# Apply weighted scoring: intensity matters more than count
# Formula: (intensity * 0.7 + events * 0.3) then normalize
for category in crisis_categories:
    event_col = f'{category}_events'
    intensity_col = f'{category}_intensity'
    
    # Weighted combination
    gdelt[f'{category}_weighted'] = (
        gdelt[intensity_col] * 0.7 + gdelt[event_col] * 0.3
    )

# Normalize each category to 0-100 scale
for category in crisis_categories:
    weighted_col = f'{category}_weighted'
    score_col = f'{category}_score'
    
    min_val = gdelt[weighted_col].min()
    max_val = gdelt[weighted_col].max()
    
    # Min-max normalization
    gdelt[score_col] = ((gdelt[weighted_col] - min_val) / (max_val - min_val)) * 100
    
    print(f"{category.capitalize()} score range: {gdelt[score_col].min():.2f} to {gdelt[score_col].max():.2f}")

# Create composite crisis index (average of all categories)
score_cols = [f'{cat}_score' for cat in crisis_categories]
gdelt['composite_crisis_score'] = gdelt[score_cols].mean(axis=1)

# Invert avg_tone (more negative = higher crisis)
# GDELT tone: -10 (very negative) to +10 (very positive)
# We want: higher score = worse situation
gdelt['tone_crisis_score'] = ((gdelt['avg_tone'] * -1) + 10) / 20 * 100

print(f"\nComposite crisis score range: {gdelt['composite_crisis_score'].min():.2f} to {gdelt['composite_crisis_score'].max():.2f}")
print(f"Tone crisis score range: {gdelt['tone_crisis_score'].min():.2f} to {gdelt['tone_crisis_score'].max():.2f}")

# Select relevant columns for final output
output_cols = ['month'] + score_cols + ['composite_crisis_score', 'tone_crisis_score', 'total_articles']
gdelt_processed = gdelt[output_cols].copy()

# Save processed crisis data
gdelt_processed.to_csv('../data/processed/crisis/monthly_crisis_scores_2015_2025.csv', index=False)
print(f"\nSaved to: monthly_crisis_scores_2015_2025.csv")

print("\n=== SAMPLE OUTPUT ===")
print(gdelt_processed.head(10))

# Statistics
print("\n=== AVERAGE CRISIS SCORES BY CATEGORY ===")
for col in score_cols:
    print(f"{col.replace('_score', '').capitalize()}: {gdelt[col].mean():.2f}")
print(f"Composite: {gdelt['composite_crisis_score'].mean():.2f}")
