import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# ALL DISTRICT COORDINATES (Latitude, Longitude, Elevation)
DISTRICTS = {
    # 4 Primary Target Districts
    'colombo': {'lat': 6.9271, 'lon': 79.8612, 'elevation': 7, 'region': 'Western'},
    'kandy': {'lat': 7.2906, 'lon': 80.6337, 'elevation': 500, 'region': 'Central'},
    'galle': {'lat': 6.0535, 'lon': 80.2210, 'elevation': 13, 'region': 'Southern'},
    'badulla': {'lat': 6.9934, 'lon': 81.0550, 'elevation': 680, 'region': 'Uva'},

    # Additional Major Tourism Districts
    'gampaha': {'lat': 7.0873, 'lon': 80.0142, 'elevation': 29, 'region': 'Western'},
    'matale': {'lat': 7.4675, 'lon': 80.6234, 'elevation': 364, 'region': 'Central'},
    'nuwara_eliya': {'lat': 6.9497, 'lon': 80.7891, 'elevation': 1868, 'region': 'Central'},
    'kalutara': {'lat': 6.5854, 'lon': 79.9607, 'elevation': 11, 'region': 'Western'},
    'matara': {'lat': 5.9549, 'lon': 80.5550, 'elevation': 5, 'region': 'Southern'},
    'anuradhapura': {'lat': 8.3114, 'lon': 80.4037, 'elevation': 81, 'region': 'North Central'},
    'hambantota': {'lat': 6.1429, 'lon': 81.1212, 'elevation': 18, 'region': 'Southern'},
    'polonnaruwa': {'lat': 7.9403, 'lon': 81.0188, 'elevation': 58, 'region': 'North Central'}
}


# COMPREHENSIVE WEATHER VARIABLES


WEATHER_VARIABLES = {
    # Temperature Variables
    'temperature_2m_max': 'Max Temperature (°C)',
    'temperature_2m_min': 'Min Temperature (°C)',
    'temperature_2m_mean': 'Mean Temperature (°C)',

    # Precipitation Variables
    'precipitation_sum': 'Total Precipitation (mm)',
    'rain_sum': 'Total Rainfall (mm)',
    'precipitation_hours': 'Hours of Precipitation',

    # Humidity & Pressure
    'relative_humidity_2m_mean': 'Mean Relative Humidity (%)',
    'surface_pressure_mean': 'Mean Surface Pressure (hPa)',

    # Wind Variables
    'wind_speed_10m_max': 'Max Wind Speed (km/h)',
    'wind_speed_10m_mean': 'Mean Wind Speed (km/h)',
    'wind_gusts_10m_max': 'Max Wind Gusts (km/h)',

    # Solar Radiation & Sunshine
    'shortwave_radiation_sum': 'Solar Radiation (MJ/m²)',
    'sunshine_duration': 'Sunshine Duration (seconds)',

    # Cloud Cover & Visibility
    'cloud_cover_mean': 'Mean Cloud Cover (%)',

    # Evapotranspiration
    'et0_fao_evapotranspiration_sum': 'Reference Evapotranspiration (mm)'
}


# FETCH WEATHER DATA FUNCTION


def safe_api_request(url, retries=5, cooldown=5):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = cooldown * (i + 1)
                print(f"Rate limit exceeded. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    print("Failed after multiple retries.")
    return None


def fetch_weather_for_district(district_name, lat, lon, start_date, end_date):
    """
    Fetch comprehensive historical weather data from Open-Meteo API

    Args:
        district_name (str): Name of district
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)

    Returns:
        pandas.DataFrame: Monthly aggregated weather data
    """

    print(f"\n  Fetching weather for {district_name.upper()}...")
    print(f"    Coordinates: ({lat:.4f}, {lon:.4f})")
    print(f"    Elevation: {DISTRICTS[district_name]['elevation']}m")

    # Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': list(WEATHER_VARIABLES.keys()),
        'timezone': 'Asia/Colombo'
    }

    try:
        data = safe_api_request(f"{url}?latitude={lat}&longitude={lon}"
                                f"&start_date={start_date}&end_date={end_date}"
                                f"&timezone=Asia%2FColombo&"
                                + "&".join([f"daily={v}" for v in WEATHER_VARIABLES.keys()]))

        if data is None or 'daily' not in data:
            print("     Skipping due to repeated rate limit or missing data.")
            return None

        # Convert to DataFrame
        df_data = {'date': pd.to_datetime(data['daily']['time'])}

        for var in WEATHER_VARIABLES.keys():
            if var in data['daily']:
                df_data[var] = data['daily'][var]

        df = pd.DataFrame(df_data)

        # Extract year-month
        df['year_month'] = df['date'].dt.strftime('%Y-%m')

        # Aggregate to monthly based on variable type
        agg_dict = {}

        for var in WEATHER_VARIABLES.keys():
            if var in df.columns:
                if 'mean' in var or var in ['temperature_2m_mean', 'relative_humidity_2m_mean',
                                            'surface_pressure_mean', 'wind_speed_10m_mean',
                                            'cloud_cover_mean']:
                    agg_dict[var] = 'mean'
                elif 'max' in var:
                    agg_dict[var] = 'max'
                elif 'min' in var:
                    agg_dict[var] = 'min'
                elif 'sum' in var or 'duration' in var:
                    agg_dict[var] = 'sum'
                elif 'hours' in var:
                    agg_dict[var] = 'sum'
                else:
                    agg_dict[var] = 'mean'

        monthly = df.groupby('year_month').agg(agg_dict).reset_index()

        # Add district name prefix to columns
        column_rename = {}
        for col in monthly.columns:
            if col != 'year_month':
                # Shorten column names for readability
                short_name = col.replace('temperature_2m_', 'temp_')
                short_name = short_name.replace('relative_humidity_2m_', 'humidity_')
                short_name = short_name.replace('surface_pressure_', 'pressure_')
                short_name = short_name.replace('wind_speed_10m_', 'wind_')
                short_name = short_name.replace('wind_gusts_10m_', 'wind_gust_')
                short_name = short_name.replace('shortwave_radiation_', 'solar_')
                short_name = short_name.replace('cloud_cover_', 'cloud_')
                short_name = short_name.replace('et0_fao_evapotranspiration_', 'evap_')
                short_name = short_name.replace('precipitation_', 'precip_')

                column_rename[col] = f'{district_name}_{short_name}'

        monthly = monthly.rename(columns=column_rename)

        print(f"     Fetched {len(monthly)} months ({len(column_rename)} variables)")

        return monthly

    except Exception as e:
        print(f"     Error: {e}")
        return None


import time
import requests


def expected_year_months(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months = pd.date_range(start=start, end=end, freq='MS')
    return [d.strftime('%Y-%m') for d in months]


def district_complete_for_range(existing_df, district, ym_list):
    key_col = f'{district}_temp_mean'
    if key_col not in existing_df.columns:
        return False
    if 'year_month' not in existing_df.columns:
        return False

    tmp = existing_df[['year_month', key_col]].copy()
    tmp = tmp.drop_duplicates(subset=['year_month'])
    tmp = tmp.set_index('year_month').reindex(ym_list)

    if tmp[key_col].isna().any():
        return False

    return True


# MAIN EXECUTION


print("\n Fetching comprehensive weather data from Open-Meteo API...")
print(f"Variables to fetch: {len(WEATHER_VARIABLES)}")
print("- Temperature (max, min, mean)")
print("- Precipitation & Rain")
print("- Humidity & Pressure")
print("- Wind Speed & Gusts")
print("- Solar Radiation & Sunshine")
print("- Cloud Cover")
print("- Evapotranspiration")

# Date range to match tourist data (2015-01 to 2025-03)
START_DATE = "2015-01-01"
END_DATE = "2025-03-31"

print(f"\nDate Range: {START_DATE} to {END_DATE}")
print(f"Districts: {len(DISTRICTS)}")

all_weather_data = []
failed_districts = []

output_file = '../data/processed/weather_data_all_districts_comprehensive.csv'

existing_weather = None
if os.path.exists(output_file):
    try:
        existing_weather = pd.read_csv(output_file)
        print(f"\n Existing dataset found: {output_file}")
        print(f" Existing shape: {existing_weather.shape}")
    except Exception as e:
        print(f"\n Could not read existing dataset. Will fetch from scratch. Error: {e}")
        existing_weather = None

ym_list = expected_year_months(START_DATE, END_DATE)

districts_to_fetch = []
if existing_weather is not None and 'year_month' in existing_weather.columns:
    for d in DISTRICTS.keys():
        if not district_complete_for_range(existing_weather, d, ym_list):
            districts_to_fetch.append(d)

    if len(districts_to_fetch) == 0:
        print("\n Dataset already complete for the specified time range.")
    else:
        print(f"\n Missing districts for specified time range: {', '.join(districts_to_fetch)}")
else:
    districts_to_fetch = list(DISTRICTS.keys())

for i, district_name in enumerate(districts_to_fetch, 1):
    print(f"\n[District {i}/{len(districts_to_fetch)}]")

    coords = DISTRICTS[district_name]

    weather_df = fetch_weather_for_district(
        district_name,
        coords['lat'],
        coords['lon'],
        START_DATE,
        END_DATE
    )

    if weather_df is not None:
        all_weather_data.append(weather_df)
    else:
        failed_districts.append(district_name)

    # Be nice to the API - add delay between requests
    if i < len(districts_to_fetch):
        time.sleep(10)

# MERGE ALL DISTRICTS


print(" Merging weather data from all districts...")


if existing_weather is None:
    if len(all_weather_data) > 0:
        # Start with first district
        weather_combined = all_weather_data[0]

        # Merge with remaining districts
        for i in range(1, len(all_weather_data)):
            weather_combined = weather_combined.merge(
                all_weather_data[i],
                on='year_month',
                how='outer'
            )
    else:
        print("\n Error: Could not fetch data for any districts")
        raise SystemExit(1)
else:
    weather_combined = existing_weather.copy()

    if len(all_weather_data) > 0:
        for df_new in all_weather_data:
            weather_combined = weather_combined.merge(
                df_new,
                on='year_month',
                how='left'
            )

# Sort by date
weather_combined = weather_combined.sort_values('year_month').reset_index(drop=True)

print(f"\n Successfully merged {len(all_weather_data)} districts")
print(f" Combined data shape: {weather_combined.shape}")
print(f" Date range: {weather_combined['year_month'].min()} to {weather_combined['year_month'].max()}")
print(f" Total columns: {len(weather_combined.columns)}")

if failed_districts:
    print(f"\n Warning: Failed to fetch data for: {', '.join(failed_districts)}")


# SUMMARY STATISTICS



print(" WEATHER SUMMARY STATISTICS")


for district in ['colombo', 'kandy', 'galle', 'badulla']:
    if f'{district}_temp_mean' in weather_combined.columns:
        print(f"\n{district.upper()} ({DISTRICTS[district]['region']} Province):")
        print(f"  Temperature:")
        print(f"    Average:        {weather_combined[f'{district}_temp_mean'].mean():.1f}°C")
        print(
            f"    Range:          {weather_combined[f'{district}_temp_min'].mean():.1f}°C - {weather_combined[f'{district}_temp_max'].mean():.1f}°C")

        print(f"  Precipitation:")
        print(f"    Total Rainfall: {weather_combined[f'{district}_rain_sum'].sum():.0f} mm")
        print(f"    Monthly Avg:    {weather_combined[f'{district}_rain_sum'].mean():.0f} mm")
        print(f"    Rainy Hours/mo: {weather_combined[f'{district}_precip_hours'].mean():.0f} hours")

        if f'{district}_humidity_mean' in weather_combined.columns:
            print(f"  Humidity:        {weather_combined[f'{district}_humidity_mean'].mean():.0f}%")

        if f'{district}_wind_mean' in weather_combined.columns:
            print(f"  Wind Speed:      {weather_combined[f'{district}_wind_mean'].mean():.1f} km/h")

        if f'{district}_sunshine_duration' in weather_combined.columns:
            print(
                f"  Sunshine:        {weather_combined[f'{district}_sunshine_duration'].sum() / 3600:.0f} hours total")

# District comparison

print("DISTRICT CLIMATE COMPARISON")


print(f"\n{'District':<15} {'Temp (°C)':<12} {'Rain (mm/mo)':<15} {'Humidity (%)':<12}")


for district in DISTRICTS.keys():
    if f'{district}_temp_mean' in weather_combined.columns:
        temp = weather_combined[f'{district}_temp_mean'].mean()
        rain = weather_combined[f'{district}_rain_sum'].mean()
        humid = weather_combined[
            f'{district}_humidity_mean'].mean() if f'{district}_humidity_mean' in weather_combined.columns else 0
        print(f"{district.capitalize():<15} {temp:>6.1f}        {rain:>8.0f}          {humid:>6.0f}")


# ADD DERIVED FEATURES


print(" ADDING DERIVED WEATHER FEATURES")

for district in DISTRICTS.keys():
    if f'{district}_temp_mean' in weather_combined.columns:
        # Temperature range (daily variation indicator)
        weather_combined[f'{district}_temp_range'] = (
                weather_combined[f'{district}_temp_max'] -
                weather_combined[f'{district}_temp_min']
        )

        # Rain intensity (mm per rainy hour)
        if f'{district}_precip_hours' in weather_combined.columns:
            weather_combined[f'{district}_rain_intensity'] = (
                    weather_combined[f'{district}_rain_sum'] /
                    weather_combined[f'{district}_precip_hours'].replace(0, np.nan)
            ).fillna(0)

        # Comfort index (simple: temperature - humidity/10)
        if f'{district}_humidity_mean' in weather_combined.columns:
            weather_combined[f'{district}_comfort_index'] = (
                    weather_combined[f'{district}_temp_mean'] -
                    weather_combined[f'{district}_humidity_mean'] / 10
            )

        # Dry days indicator (precipitation < 1mm)
        weather_combined[f'{district}_dry_month'] = (
                weather_combined[f'{district}_rain_sum'] < 30
        ).astype(int)

print(" Added derived features:")
print("  - Temperature Range (daily variation)")
print("  - Rain Intensity (mm per hour)")
print("  - Comfort Index (temp adjusted for humidity)")
print("  - Dry Month Indicator (<30mm rain)")

# SAVE DATA


print("SAVING WEATHER DATA")


output_file = '../data/processed/weather_data_all_districts_comprehensive.csv'
weather_combined.to_csv(output_file, index=False)

print(f"\n Saved: {output_file}")
print(f"  Shape: {weather_combined.shape}")
print(f"  Date Range: {weather_combined['year_month'].min()} to {weather_combined['year_month'].max()}")
print(f"  Districts: {len(DISTRICTS)}")
print(f"  Variables per district: ~{len(WEATHER_VARIABLES) + 4}")  # +4 for derived features

# Save column list for reference
print(f"\n✓ Total columns: {len(weather_combined.columns)}")

# Display sample

print("SAMPLE DATA (First 3 Months)")


sample_cols = ['year_month',
               'colombo_temp_mean', 'colombo_rain_sum', 'colombo_humidity_mean',
               'kandy_temp_mean', 'kandy_rain_sum', 'kandy_humidity_mean',
               'galle_temp_mean', 'galle_rain_sum', 'galle_humidity_mean',
               'badulla_temp_mean', 'badulla_rain_sum', 'badulla_humidity_mean']

# Filter columns that exist
sample_cols = [col for col in sample_cols if col in weather_combined.columns]

print("\n", weather_combined[sample_cols].head(3).to_string(index=False))

# Save metadata file
metadata = {
    'districts': list(DISTRICTS.keys()),
    'variables': list(WEATHER_VARIABLES.keys()),
    'date_range': f"{START_DATE} to {END_DATE}",
    'total_months': len(weather_combined),
    'total_columns': len(weather_combined.columns),
    'failed_districts': failed_districts
}

metadata_file = '../data/processed/weather_metadata.txt'
with open(metadata_file, 'w') as f:
    f.write("WEATHER DATA METADATA\n")
    f.write(f"Date Range: {metadata['date_range']}\n")
    f.write(f"Total Months: {metadata['total_months']}\n")
    f.write(f"Total Columns: {metadata['total_columns']}\n\n")
    f.write(f"Districts ({len(metadata['districts'])}):\n")
    for dist in metadata['districts']:
        f.write(f"  - {dist.capitalize()}\n")
    f.write(f"\nWeather Variables ({len(metadata['variables'])}):\n")
    for var, desc in WEATHER_VARIABLES.items():
        f.write(f"  - {desc}\n")
    f.write("\nDerived Features:\n")
    f.write("  - Temperature Range\n")
    f.write("  - Rain Intensity\n")
    f.write("  - Comfort Index\n")
    f.write("  - Dry Month Indicator\n")
    if metadata['failed_districts']:
        f.write(f"\nFailed Districts: {', '.join(metadata['failed_districts'])}\n")

print(f"\n Saved metadata: {metadata_file}")

print(" COMPREHENSIVE WEATHER DATA FETCHING COMPLETE!")


if existing_weather is None and len(all_weather_data) == 0:
    print("\n Error: Could not fetch data for any districts")
