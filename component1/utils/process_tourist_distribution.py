import pandas as pd
import numpy as np


# Load the raw tourism data
print("\n Loading raw tourism data...")
df = pd.read_csv('../data/raw/Tourism data by country CSV.csv', encoding='latin1')

df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
df['Total'] = df['Total'].fillna(0)

print(f"Data loaded: {len(df)} months")
print(f"Date range: {df['Month'].iloc[-1]} to {df['Month'].iloc[0]}")
print(f"Total tourists: {df['Total'].sum():,}")


# DISTRICT DISTRIBUTION COEFFICIENTS

# Based on SLTDA telecommunication data (May 2025) and hotel capacity distribution
# These represent the percentage of total tourists who visit each district
# Note: Total > 100% because tourists visit multiple districts

print("\n Applying district distribution coefficients...")

DISTRICT_COEFFICIENTS = {
    # 4 Target Districts
    'colombo': 0.85,  # 85% of tourists visit Colombo (entry point, commercial capital)
    'kandy': 0.72,  # 72% visit Kandy (cultural triangle, major attraction)
    'galle': 0.68,  # 68% visit Galle (south coast, colonial heritage)
    'badulla': 0.48,  # 48% visit Badulla district (Ella, Haputale, hill country)

    # Additional Major Districts (for reference)
    'gampaha': 0.75,  # 75% (Negombo, near airport)
    'matale': 0.55,  # 55% (Sigiriya, Dambulla)
    'nuwara_eliya': 0.52,  # 52% (tea plantations)
    'kalutara': 0.50,  # 50% (beaches)
    'matara': 0.48,  # 48% (south coast)
    'anuradhapura': 0.42,  # 42% (ancient cities)
    'hambantota': 0.35,  # 35% (wildlife)
    'polonnaruwa': 0.30,  # 30% (ancient cities)
}

# Rationale for coefficients:
# - Colombo: Highest (airport, hotels, commercial hub) - most tourists pass through
# - Kandy: Very high (Temple of Tooth, cultural importance, central location)
# - Galle: High (UNESCO fort, beaches, popular south coast destination)
# - Badulla: Moderate-high (Ella very popular despite being hill country)
# - Based on: SLTDA hotel capacity (Colombo 35.8%, South Coast 35%, Ancient Cities 17.6%, Hill Country 6%)
#            + Telecommunication data showing Colombo, Galle, Kandy in top districts

print("\nDistrict Distribution Coefficients:")
print(f"  Colombo:  {DISTRICT_COEFFICIENTS['colombo'] * 100:.0f}% of total tourists")
print(f"  Kandy:    {DISTRICT_COEFFICIENTS['kandy'] * 100:.0f}% of total tourists")
print(f"  Galle:    {DISTRICT_COEFFICIENTS['galle'] * 100:.0f}% of total tourists")
print(f"  Badulla:  {DISTRICT_COEFFICIENTS['badulla'] * 100:.0f}% of total tourists")

# Calculate district-wise tourist counts
print("\n Calculating district-wise tourist arrivals...")

for district, coefficient in DISTRICT_COEFFICIENTS.items():
    df[f'{district}_tourists'] = (df['Total'] * coefficient).round().astype(int)


# Some districts are more seasonal than others

SEASONAL_ADJUSTMENTS = {
    'colombo': {  # Colombo is year-round (business + transit hub)
        'peak_months': [],
        'adjustment': 1.0
    },
    'kandy': {  # Kandy peaks during Esala Perahera (Jul-Aug)
        'peak_months': [7, 8],
        'adjustment': 1.15
    },
    'galle': {  # South coast peaks Dec-Mar (dry season)
        'peak_months': [12, 1, 2, 3],
        'adjustment': 1.20
    },
    'badulla': {  # Hill country peaks Dec-Mar (cool weather)
        'peak_months': [12, 1, 2, 3],
        'adjustment': 1.18
    }
}

print("\n Applying seasonal adjustments...")

# Extract month number from 'Month' column (format: Mar-25)
df['month_num'] = pd.to_datetime(df['Month'], format='%b-%y').dt.month
df['year'] = pd.to_datetime(df['Month'], format='%b-%y').dt.year

# Apply seasonal adjustments
for district in ['colombo', 'kandy', 'galle', 'badulla']:
    peak_months = SEASONAL_ADJUSTMENTS[district]['peak_months']
    adjustment = SEASONAL_ADJUSTMENTS[district]['adjustment']

    if peak_months:
        # Apply adjustment to peak months
        peak_mask = df['month_num'].isin(peak_months)
        df.loc[peak_mask, f'{district}_tourists'] = (
                df.loc[peak_mask, f'{district}_tourists'] * adjustment
        ).round().astype(int)

        # Slightly reduce other months to maintain balance
        off_peak_mask = ~peak_mask
        df.loc[off_peak_mask, f'{district}_tourists'] = (
                df.loc[off_peak_mask, f'{district}_tourists'] * 0.95
        ).round().astype(int)


# CREATE YEAR-MONTH COLUMN FOR MERGING


df['year_month'] = pd.to_datetime(df['Month'], format='%b-%y').dt.strftime('%Y-%m')


# SUMMARY STATISTICS


print("\n Generating summary statistics...")


print("SUMMARY STATISTICS")


print(f"\nTotal Period: {df['Month'].iloc[-1]} to {df['Month'].iloc[0]}")
print(f"Total Months: {len(df)}")
print(f"Total Tourists (Overall): {df['Total'].sum():,}")

print("\nDistrict-wise Tourist Distribution:")


for district in ['colombo', 'kandy', 'galle', 'badulla']:
    total = df[f'{district}_tourists'].sum()
    avg_monthly = df[f'{district}_tourists'].mean()
    max_month = df.loc[df[f'{district}_tourists'].idxmax(), 'Month']
    max_value = df[f'{district}_tourists'].max()

    print(f"\n{district.upper()}:")
    print(f"  Total Visitors:    {total:,} ({total / df['Total'].sum() * 100:.1f}% of overall)")
    print(f"  Avg per Month:     {avg_monthly:,.0f}")
    print(f"  Peak Month:        {max_month} ({max_value:,} tourists)")
    print(f"  Coefficient Used:  {DISTRICT_COEFFICIENTS[district] * 100:.0f}%")

# Sample data display

print("SAMPLE DATA (Most Recent 6 Months)")


sample_cols = ['Month', 'Total', 'colombo_tourists', 'kandy_tourists',
               'galle_tourists', 'badulla_tourists']
print("\n", df[sample_cols].head(6).to_string(index=False))

# SAVE PROCESSED DATA



# Select relevant columns for final dataset
output_cols = [
    'year_month', 'Month', 'Total',
    'colombo_tourists', 'kandy_tourists', 'galle_tourists', 'badulla_tourists',
    'gampaha_tourists', 'matale_tourists', 'nuwara_eliya_tourists',
    'kalutara_tourists', 'matara_tourists', 'anuradhapura_tourists',
    'hambantota_tourists', 'polonnaruwa_tourists',
    'India', 'China', 'Russia', 'UK', 'Germany', 'France',
    'Australia', 'Netherland', 'Bangladesh', 'Maldives',
    'Canada', 'US', 'Poland', 'Italy', 'Spain', 'Others'
]

df_output = df[output_cols].copy()

# Sort by date (oldest first)
df_output = df_output.sort_values('year_month').reset_index(drop=True)

# Save to processed folder
output_file = '../data/processed/tourist_arrivals_with_districts.csv'
df_output.to_csv(output_file, index=False)

print(f"\n Saved: {output_file}")
print(f"  Shape: {df_output.shape}")
print(f"  Columns: {len(df_output.columns)}")

# Also save a simplified version with only target districts
output_simple = df_output[['year_month', 'Month', 'Total',
                           'colombo_tourists', 'kandy_tourists',
                           'galle_tourists', 'badulla_tourists']].copy()

output_simple_file = '../data/processed/tourist_arrivals_4_districts.csv'
output_simple.to_csv(output_simple_file, index=False)

print(f"\n Saved simplified version: {output_simple_file}")
print(f"  Shape: {output_simple.shape}")

# VALIDATION CHECKS
# Check that district totals make sense
total_all_districts = (df_output['colombo_tourists'].sum() +
                       df_output['kandy_tourists'].sum() +
                       df_output['galle_tourists'].sum() +
                       df_output['badulla_tourists'].sum())

overall_total = df_output['Total'].sum()

print(f"\n Sum of 4 districts: {total_all_districts:,}")
print(f" Overall total: {overall_total:,}")
print(f" Ratio: {total_all_districts / overall_total:.2f}x")
print(f"  (>1.0 is correct - tourists visit multiple districts)")

