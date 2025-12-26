import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Sri Lanka District Mapping (City -> District)
DISTRICT_MAPPING = {
    # Colombo District
    'Colombo': 'Colombo',
    'Dehiwala': 'Colombo',
    'Moratuwa': 'Colombo',
    'Nugegoda': 'Colombo',

    # Kandy District
    'Kandy': 'Kandy',
    'Peradeniya': 'Kandy',
    'Katugastota': 'Kandy',

    # Galle District
    'Galle': 'Galle',
    'Hikkaduwa': 'Galle',
    'Unawatuna': 'Galle',
    'Ahangama': 'Galle',

    # Badulla District (Hill Country)
    'Badulla': 'Badulla',
    'Ella': 'Badulla',
    'Haputale': 'Badulla',
    'Bandarawela': 'Badulla',
    'Pussellawa': 'Badulla',

    # Nuwara Eliya District
    'Nuwara Eliya': 'Nuwara Eliya',
    'Nanu Oya': 'Nuwara Eliya',

    # Anuradhapura District
    'Anuradhapura': 'Anuradhapura',
    'Saliyapura': 'Anuradhapura',
    'Mihintale': 'Anuradhapura',

    # Polonnaruwa District
    'Polonnaruwa': 'Polonnaruwa',
    'Habarana': 'Polonnaruwa',

    # Matale District
    'Matale': 'Matale',
    'Dambulla': 'Matale',
    'Sigiriya': 'Matale',

    # Trincomalee District
    'Trincomalee': 'Trincomalee',
    'Nilaveli': 'Trincomalee',
    'Uppuveli': 'Trincomalee',

    # Jaffna District
    'Jaffna': 'Jaffna',

    # Ampara District
    'Arugam Bay': 'Ampara',

    # Kalutara District
    'Kalutara': 'Kalutara',
    'Beruwala': 'Kalutara',
    'Bentota': 'Kalutara',

    # Gampaha District
    'Negombo': 'Gampaha',
    'Katunayake': 'Gampaha',

    # Ratnapura District
    'Ratnapura': 'Ratnapura',

    # Hambantota District
    'Hambantota': 'Hambantota',
    'Tangalle': 'Hambantota',
    'Tissamaharama': 'Hambantota',
}


def map_to_district(city):
    """Map city name to district"""
    if pd.isna(city):
        return 'Unknown'
    city_lower = city.lower()
    for key, district in DISTRICT_MAPPING.items():
        if key.lower() in city_lower:
            return district
    return 'Other'


# Load reviews
print("Loading reviews dataset...")
df = pd.read_csv('../data/Reviews.csv', encoding='latin-1', on_bad_lines='skip')

print(f"Total reviews loaded: {len(df)}")
print(f"Date range: {df['Travel_Date'].min()} to {df['Travel_Date'].max()}")

# Clean and prepare data
df['Travel_Date'] = pd.to_datetime(df['Travel_Date'], errors='coerce')
df = df.dropna(subset=['Travel_Date', 'Rating', 'Text'])

# Map cities to districts
df['District'] = df['Located_City'].apply(map_to_district)

# Filter 2015-2023 for consistency with GDELT
df = df[(df['Travel_Date'] >= '2015-01-01') & (df['Travel_Date'] <= '2023-12-31')]
print(f"Reviews 2015-2023: {len(df)}")

# Display district distribution
print("\n=== DISTRICT DISTRIBUTION ===")
district_counts = df['District'].value_counts()
print(district_counts[district_counts > 50])  # Show districts with >50 reviews


# Create 3-class sentiment labels based on rating
def classify_sentiment(rating):
    if rating >= 4:
        return 2  # Positive
    elif rating == 3:
        return 1  # Neutral
    else:
        return 0  # Negative


df['sentiment_label'] = df['Rating'].apply(classify_sentiment)
df['sentiment_score'] = df['sentiment_label'].map({2: 1, 1: 0, 0: -1})

# Extract year-month
df['year_month'] = df['Travel_Date'].dt.to_period('M').astype(str)

print("\n=== PROCESSING SENTIMENT SCORES ===")

# 1. OVERALL MONTHLY SENTIMENT (all Sri Lanka)
print("\n1. Computing overall monthly sentiment...")
monthly_sentiment = df.groupby('year_month').agg({
    'sentiment_score': 'mean',
    'sentiment_label': lambda x: (x == 2).sum(),
    'Rating': 'count'
}).reset_index()

monthly_sentiment.columns = ['month', 'sentiment_score', 'positive_count', 'review_count']

# Add neutral and negative counts
monthly_stats = df.groupby('year_month')['sentiment_label'].value_counts().unstack(fill_value=0)
if 0 in monthly_stats.columns:
    monthly_sentiment['negative_count'] = monthly_stats[0].values
else:
    monthly_sentiment['negative_count'] = 0

if 1 in monthly_stats.columns:
    monthly_sentiment['neutral_count'] = monthly_stats[1].values
else:
    monthly_sentiment['neutral_count'] = 0

# Normalize sentiment score to 0-100 scale
monthly_sentiment['sentiment_score_normalized'] = ((monthly_sentiment['sentiment_score'] + 1) / 2) * 100

print(f"   Overall monthly records: {len(monthly_sentiment)}")

# 2. DISTRICT-LEVEL MONTHLY SENTIMENT
print("\n2. Computing district-level monthly sentiment...")

# Get districts with sufficient data (>100 reviews)
major_districts = district_counts[district_counts > 100].index.tolist()
print(f"   Major districts (>100 reviews): {len(major_districts)}")
print(f"   Districts: {major_districts}")

# Filter to major districts
df_major = df[df['District'].isin(major_districts)].copy()

# Compute sentiment for each district by month
district_monthly = df_major.groupby(['year_month', 'District']).agg({
    'sentiment_score': 'mean',
    'Rating': 'count'
}).reset_index()

district_monthly.columns = ['month', 'district', 'sentiment_score', 'review_count']

# Normalize district sentiment scores
district_monthly['sentiment_score_normalized'] = ((district_monthly['sentiment_score'] + 1) / 2) * 100

# Pivot to wide format (one column per district)
district_pivot = district_monthly.pivot(
    index='month',
    columns='district',
    values='sentiment_score'
).reset_index()

# Rename columns to lowercase with underscores
district_pivot.columns = ['month'] + [f"{col.lower().replace(' ', '_')}_sentiment" for col in
                                      district_pivot.columns[1:]]

# Add review counts pivot
district_counts_pivot = district_monthly.pivot(
    index='month',
    columns='district',
    values='review_count'
).reset_index()

district_counts_pivot.columns = ['month'] + [f"{col.lower().replace(' ', '_')}_reviews" for col in
                                             district_counts_pivot.columns[1:]]

# Merge overall sentiment with district-level sentiment
combined_sentiment = monthly_sentiment.merge(district_pivot, on='month', how='left')
combined_sentiment = combined_sentiment.merge(district_counts_pivot, on='month', how='left')

# Fill NaN values (months where a district had no reviews) with overall sentiment
district_cols = [col for col in combined_sentiment.columns if col.endswith('_sentiment') and col != 'month']
for col in district_cols:
    combined_sentiment[col].fillna(combined_sentiment['sentiment_score'], inplace=True)

print(f"   Combined records: {len(combined_sentiment)}")

# 3. SAVE OUTPUTS
print("\n=== SAVING OUTPUTS ===")

# Save overall monthly sentiment
monthly_sentiment.to_csv('../data/processed/sentiment/monthly_sentiment_overall_2015_2023.csv', index=False)
print(" Saved: monthly_sentiment_overall_2015_2023.csv")

# Save district-level monthly sentiment
district_monthly.to_csv('../data/processed/sentiment/monthly_sentiment_by_district_2015_2023.csv', index=False)
print(" Saved: monthly_sentiment_by_district_2015_2023.csv")

# Save combined (overall + all districts in one file)
combined_sentiment.to_csv('../data/processed/sentiment/monthly_sentiment_combined_2015_2023.csv', index=False)
print(" Saved: monthly_sentiment_combined_2015_2023.csv")

# Save labeled dataset for training classifier
df_train = df[['Text', 'Rating', 'sentiment_label', 'sentiment_score', 'District']].copy()
df_train.to_csv('../data/processed/sentiment/reviews_labeled.csv', index=False)
print(f" Saved: reviews_labeled.csv ({len(df_train)} samples)")

# 4. DISPLAY STATISTICS
print("\n=== SUMMARY STATISTICS ===")

print(f"\nOverall Sentiment:")
print(f"  Average score: {monthly_sentiment['sentiment_score'].mean():.3f}")
print(
    f"  Score range: {monthly_sentiment['sentiment_score'].min():.3f} to {monthly_sentiment['sentiment_score'].max():.3f}")
print(f"  Average reviews/month: {monthly_sentiment['review_count'].mean():.1f}")

print(f"\nSentiment Distribution:")
print(f"  Positive: {(df['sentiment_label'] == 2).sum()} ({(df['sentiment_label'] == 2).sum() / len(df) * 100:.1f}%)")
print(f"  Neutral: {(df['sentiment_label'] == 1).sum()} ({(df['sentiment_label'] == 1).sum() / len(df) * 100:.1f}%)")
print(f"  Negative: {(df['sentiment_label'] == 0).sum()} ({(df['sentiment_label'] == 0).sum() / len(df) * 100:.1f}%)")

print(f"\nDistrict-Level Statistics:")
for district in ['Colombo', 'Kandy', 'Galle', 'Badulla']:
    if district in major_districts:
        district_data = district_monthly[district_monthly['district'] == district]
        avg_score = district_data['sentiment_score'].mean()
        total_reviews = district_data['review_count'].sum()
        print(f"  {district:15s}: avg={avg_score:+.3f}, reviews={int(total_reviews)}")

print("\n=== SAMPLE OUTPUT (first 5 months) ===")
print(combined_sentiment.head())

print("\n Processing complete!")