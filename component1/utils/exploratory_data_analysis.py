import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


print("EXPLORATORY DATA ANALYSIS - ALL 12 DISTRICTS")


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
print("\nLoading final training dataset...")
df = pd.read_csv('../data/processed/final_training_dataset.csv')
df['date'] = pd.to_datetime(df['year_month'] + '-01')

print(f" Data loaded: {df.shape}")
print(f" Date range: {df['year_month'].min()} to {df['year_month'].max()}")

# Create output directory
import os

os.makedirs('../metrics/eda_all_districts', exist_ok=True)

# All 12 districts
DISTRICTS = ['colombo', 'kandy', 'galle', 'badulla', 'gampaha', 'matale',
             'nuwara_eliya', 'kalutara', 'matara', 'anuradhapura',
             'hambantota', 'polonnaruwa']


# 1. TOURIST ARRIVALS TRENDS - ALL DISTRICTS


print("\n Analyzing tourist arrival trends for all 12 districts...")

fig, axes = plt.subplots(4, 3, figsize=(20, 16))
colors = plt.cm.tab20(np.linspace(0, 1, 12))

for idx, (district, color) in enumerate(zip(DISTRICTS, colors)):
    ax = axes[idx // 3, idx % 3]
    col = f'{district}_tourists'

    ax.plot(df['date'], df[col], color=color, linewidth=2, label=district.capitalize())
    ax.fill_between(df['date'], df[col], alpha=0.3, color=color)

    # Add trend line
    z = np.polyfit(range(len(df)), df[col], 1)
    p = np.poly1d(z)
    ax.plot(df['date'], p(range(len(df))), "r--", alpha=0.8, linewidth=2, label='Trend')

    ax.set_title(f'{district.capitalize()} - Tourist Arrivals', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Tourist Count', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/tourist_trends_all_districts.png', dpi=300, bbox_inches='tight')
print("   Saved: tourist_trends_all_districts.png")
plt.close()


# 2. DISTRICT COMPARISON - AVERAGE TOURISTS


print("\n Comparing average tourists across districts...")

district_means = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    district_means.append({
        'District': district.replace('_', ' ').title(),
        'Average': df[col].mean(),
        'Max': df[col].max(),
        'Min': df[col].min()
    })

district_df = pd.DataFrame(district_means).sort_values('Average', ascending=False)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(district_df))

ax.bar(x, district_df['Average'], color='steelblue', alpha=0.8, label='Average')
ax.errorbar(x, district_df['Average'],
            yerr=[district_df['Average'] - district_df['Min'],
                  district_df['Max'] - district_df['Average']],
            fmt='none', color='red', alpha=0.5, capsize=5, label='Min-Max Range')

ax.set_xlabel('District', fontsize=12, fontweight='bold')
ax.set_ylabel('Tourist Count', fontsize=12, fontweight='bold')
ax.set_title('Average Tourist Arrivals by District (2015-2023)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(district_df['District'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/district_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: district_comparison.png")
plt.close()


# 3. SEASONALITY ANALYSIS - ALL DISTRICTS


print("\n Analyzing seasonality for all districts...")

fig, axes = plt.subplots(4, 3, figsize=(20, 16))

for idx, (district, color) in enumerate(zip(DISTRICTS, colors)):
    ax = axes[idx // 3, idx % 3]
    col = f'{district}_tourists'

    # Group by month
    monthly_avg = df.groupby('month_num')[col].mean()

    ax.bar(monthly_avg.index, monthly_avg.values, color=color, alpha=0.7)
    ax.set_title(f'{district.capitalize()}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=9)
    ax.set_ylabel('Avg Tourists', fontsize=9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Seasonal Patterns - All Districts', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/seasonality_all_districts.png', dpi=300, bbox_inches='tight')
print("   Saved: seasonality_all_districts.png")
plt.close()


# 4. WEATHER IMPACT - TEMPERATURE


print("\n Analyzing weather impact (temperature)...")

fig, axes = plt.subplots(4, 3, figsize=(20, 16))

for idx, district in enumerate(DISTRICTS):
    ax = axes[idx // 3, idx % 3]

    temp_col = f'{district}_temp_mean'
    rain_col = f'{district}_rain_sum'
    tourist_col = f'{district}_tourists'

    if temp_col in df.columns:
        scatter = ax.scatter(df[temp_col], df[tourist_col],
                             alpha=0.6, s=50, c=df[rain_col], cmap='Blues')

        ax.set_title(f'{district.capitalize()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Temperature (°C)', fontsize=9)
        ax.set_ylabel('Tourists', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Rainfall (mm)', fontsize=8)

plt.suptitle('Temperature vs Tourists (Color = Rainfall)', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/weather_impact_temperature.png', dpi=300, bbox_inches='tight')
print("   Saved: weather_impact_temperature.png")
plt.close()


# 5. CORRELATION MATRIX - TOP DISTRICTS


print("\n Creating correlation matrix...")

# Select key features
key_features = []
for district in DISTRICTS:
    key_features.append(f'{district}_tourists')

key_features.extend(['sentiment_score', 'composite_crisis_score', 'Total'])

# Add weather for top 4 districts
for district in ['colombo', 'kandy', 'galle', 'badulla']:
    key_features.extend([
        f'{district}_temp_mean',
        f'{district}_rain_sum'
    ])

# Filter to existing columns
key_features = [col for col in key_features if col in df.columns]

corr_matrix = df[key_features].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap - All Districts', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/correlation_heatmap_all.png', dpi=300, bbox_inches='tight')
print("   Saved: correlation_heatmap_all.png")
plt.close()


# 6. SUMMARY STATISTICS - ALL DISTRICTS


print("\n Generating summary statistics...")

summary_stats = []

for district in DISTRICTS:
    col = f'{district}_tourists'
    stats = {
        'District': district.replace('_', ' ').title(),
        'Mean': df[col].mean(),
        'Median': df[col].median(),
        'Std': df[col].std(),
        'Min': df[col].min(),
        'Max': df[col].max(),
        'CV': df[col].std() / df[col].mean(),  # Coefficient of variation
        'Growth_Rate': ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100)
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats).sort_values('Mean', ascending=False)

# Save to CSV
summary_df.to_csv('../metrics/eda_all_districts/summary_statistics_all.csv', index=False)
print("   Saved: summary_statistics_all.csv")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Mean, Median, Min, Max
ax1 = axes[0]
x = np.arange(len(summary_df))
width = 0.2

ax1.bar(x - 1.5 * width, summary_df['Mean'], width, label='Mean', alpha=0.8, color='steelblue')
ax1.bar(x - 0.5 * width, summary_df['Median'], width, label='Median', alpha=0.8, color='orange')
ax1.bar(x + 0.5 * width, summary_df['Min'], width, label='Min', alpha=0.8, color='green')
ax1.bar(x + 1.5 * width, summary_df['Max'], width, label='Max', alpha=0.8, color='red')

ax1.set_xlabel('District', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tourist Count', fontsize=12, fontweight='bold')
ax1.set_title('Tourist Statistics by District', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df['District'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Growth Rate
ax2 = axes[1]
colors_growth = ['green' if x > 0 else 'red' for x in summary_df['Growth_Rate']]
ax2.bar(x, summary_df['Growth_Rate'], color=colors_growth, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('District', fontsize=12, fontweight='bold')
ax2.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Tourist Growth Rate (2015 to 2023)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(summary_df['District'], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/summary_statistics_all.png', dpi=300, bbox_inches='tight')
print("   Saved: summary_statistics_all.png")
plt.close()


# 7. MARKET SHARE OVER TIME


print("\n Analyzing market share over time...")

# Calculate percentage share for each district
tourist_cols = [f'{d}_tourists' for d in DISTRICTS]
df_share = df[tourist_cols].div(df[tourist_cols].sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(16, 8))

for district in DISTRICTS:
    col = f'{district}_tourists'
    ax.plot(df['date'], df_share[col], linewidth=2, label=district.capitalize(), alpha=0.7)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Market Share (%)', fontsize=12, fontweight='bold')
ax.set_title('District Market Share Over Time', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/market_share_over_time.png', dpi=300, bbox_inches='tight')
print("   Saved: market_share_over_time.png")
plt.close()


# 8. COVID IMPACT ANALYSIS


print("\n Analyzing COVID-19 impact...")

# Pre-COVID (before 2020-03), COVID (2020-03 to 2021-12), Post-COVID (2022+)
df['period'] = 'Pre-COVID'
df.loc[df['date'] >= '2020-03-01', 'period'] = 'COVID'
df.loc[df['date'] >= '2022-01-01', 'period'] = 'Post-COVID'

period_stats = []
for district in DISTRICTS:
    col = f'{district}_tourists'
    for period in ['Pre-COVID', 'COVID', 'Post-COVID']:
        period_data = df[df['period'] == period][col]
        if len(period_data) > 0:
            period_stats.append({
                'District': district.replace('_', ' ').title(),
                'Period': period,
                'Average': period_data.mean()
            })

period_df = pd.DataFrame(period_stats)

fig, ax = plt.subplots(figsize=(16, 8))

periods = ['Pre-COVID', 'COVID', 'Post-COVID']
x = np.arange(len(DISTRICTS))
width = 0.25

for i, period in enumerate(periods):
    period_data = period_df[period_df['Period'] == period].sort_values('District')
    ax.bar(x + i * width, period_data['Average'], width, label=period, alpha=0.8)

ax.set_xlabel('District', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Tourist Count', fontsize=12, fontweight='bold')
ax.set_title('COVID-19 Impact Analysis by District', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([d.replace('_', ' ').title() for d in DISTRICTS], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../metrics/eda_all_districts/covid_impact_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: covid_impact_analysis.png")
plt.close()


# FINAL SUMMARY REPORT



print(" EDA COMPLETE FOR ALL 12 DISTRICTS!")


print(f"\nAll visualizations saved to: component1/metrics/eda_all_districts/")
print("\nGenerated files:")
print("  1. tourist_trends_all_districts.png")
print("  2. district_comparison.png")
print("  3. seasonality_all_districts.png")
print("  4. weather_impact_temperature.png")
print("  5. correlation_heatmap_all.png")
print("  6. summary_statistics_all.png")
print("  7. summary_statistics_all.csv")
print("  8. market_share_over_time.png")
print("  9. covid_impact_analysis.png")


print("KEY FINDINGS:")


print("\nTop 5 Districts by Average Tourists:")
for idx, row in summary_df.head(5).iterrows():
    print(f"  {idx + 1}. {row['District']}: {row['Mean']:,.0f} tourists/month")

print("\nHighest Growth Rate:")
growth_sorted = summary_df.sort_values('Growth_Rate', ascending=False)
for idx, row in growth_sorted.head(3).iterrows():
    print(f"  {row['District']}: {row['Growth_Rate']:+.1f}%")

print("\nMost Volatile (High CV):")
cv_sorted = summary_df.sort_values('CV', ascending=False)
for idx, row in cv_sorted.head(3).iterrows():
    print(f"  {row['District']}: CV={row['CV']:.2f}")