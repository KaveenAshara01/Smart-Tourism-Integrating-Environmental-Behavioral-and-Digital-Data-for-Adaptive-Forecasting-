
import numpy as np
import pandas as pd
import pickle
import os
import subprocess
import shutil
from datetime import datetime
import json


class IncrementalLearner:
    """Real incremental learning with multiple strategies"""

    def __init__(self, base_model_path='../models', window_size=24):
        self.base_model_path = base_model_path
        self.window_size = window_size
        self.districts = ['colombo', 'kandy', 'galle', 'badulla', 'gampaha', 'matale',
                          'nuwara_eliya', 'kalutara', 'matara', 'anuradhapura',
                          'hambantota', 'polonnaruwa']

    def add_new_month_to_dataset(self, actuals_file, prediction_month):
        """
        Add new month of actual data to training dataset - FIXED VERSION
        """
        print(f"\n  Adding new month ({prediction_month}) to training dataset...")

        # Load actuals
        actuals = pd.read_csv(actuals_file)

        # Load existing training dataset
        training_data_path = '../data/training/final_training_dataset.csv'

        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"Training dataset not found: {training_data_path}")

        # Load with low_memory=False
        df = pd.read_csv(training_data_path, low_memory=False)

        # Check if month already exists
        if prediction_month in df['year_month'].values:
            print(f"  Warning: {prediction_month} already exists in dataset. Skipping...")
            return training_data_path

        # Parse year and month
        year, month = map(int, prediction_month.split('-'))

        # Get same month from previous years
        same_month_historical = df[df['month_num'] == month].copy()

        if len(same_month_historical) == 0:
            same_month_historical = df.tail(3).copy()

        # Create row as Series
        last_row = df.iloc[-1].copy()
        new_row_series = last_row.copy()

        # === UPDATE BASIC INFO ===
        new_row_series['year_month'] = prediction_month
        new_row_series['Month'] = f"{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]}-{str(year)[2:]}"
        new_row_series['year'] = year
        new_row_series['month_num'] = month
        new_row_series['quarter'] = (month - 1) // 3 + 1
        new_row_series['month_sin'] = np.sin(2 * np.pi * month / 12)
        new_row_series['month_cos'] = np.cos(2 * np.pi * month / 12)
        new_row_series['date'] = f'{year}-{month:02d}-01'
        new_row_series['month_index'] = len(df)

        # === UPDATE TOURIST DATA ===
        for district in self.districts:
            col = f'{district}_tourists'
            if col in actuals.columns:
                new_row_series[col] = int(actuals[col].mean())

        # Total tourists
        district_sum = sum([new_row_series.get(f'{d}_tourists', 0) for d in self.districts])
        new_row_series['Total'] = int(district_sum * 0.42)

        # === UPDATE WEATHER DATA (with safe numeric conversion) ===
        weather_suffixes = ['temp_max', 'temp_min', 'temp_mean', 'precip_sum', 'rain_sum',
                           'precip_hours', 'humidity_mean', 'pressure_mean', 'wind_max',
                           'wind_mean', 'wind_gust_max', 'solar_sum', 'sunshine_duration',
                           'cloud_mean', 'evap_sum', 'temp_range', 'rain_intensity',
                           'comfort_index', 'dry_month']

        for district in self.districts:
            for suffix in weather_suffixes:
                col = f'{district}_{suffix}'
                if col in same_month_historical.columns:
                    # Convert to numeric safely
                    numeric_vals = pd.to_numeric(same_month_historical[col], errors='coerce')
                    val = numeric_vals.mean()
                    if pd.notna(val):
                        new_row_series[col] = val

        # === UPDATE COUNTRY ARRIVALS (with safe numeric conversion) ===
        countries = ['India', 'China', 'Russia', 'UK', 'Germany', 'France', 'Australia',
                     'Netherland', 'Bangladesh', 'Maldives', 'Canada', 'US', 'Poland',
                     'Italy', 'Spain', 'Others']

        for country in countries:
            if country in same_month_historical.columns:
                # Convert to numeric safely (handles string values)
                numeric_vals = pd.to_numeric(same_month_historical[country], errors='coerce')
                val = numeric_vals.mean()
                if pd.notna(val):
                    new_row_series[country] = val

        # === SEASON INDICATORS ===
        if month in [12, 1, 2, 7, 8]:
            new_row_series['season'] = 'peak_season'
            new_row_series['season_peak_season'] = True
            new_row_series['season_inter_monsoon_1'] = False
            new_row_series['season_inter_monsoon_2'] = False
            new_row_series['season_southwest_monsoon'] = False
        elif month in [3, 4, 5]:
            new_row_series['season'] = 'inter_monsoon_1'
            new_row_series['season_peak_season'] = False
            new_row_series['season_inter_monsoon_1'] = True
            new_row_series['season_inter_monsoon_2'] = False
            new_row_series['season_southwest_monsoon'] = False
        elif month in [6]:
            new_row_series['season'] = 'southwest_monsoon'
            new_row_series['season_peak_season'] = False
            new_row_series['season_inter_monsoon_1'] = False
            new_row_series['season_inter_monsoon_2'] = False
            new_row_series['season_southwest_monsoon'] = True
        else:
            new_row_series['season'] = 'inter_monsoon_2'
            new_row_series['season_peak_season'] = False
            new_row_series['season_inter_monsoon_1'] = False
            new_row_series['season_inter_monsoon_2'] = True
            new_row_series['season_southwest_monsoon'] = False

        # === UPDATE LAG FEATURES ===
        prev_row = df.iloc[-1]

        for district in self.districts:
            tourist_col = f'{district}_tourists'

            new_row_series[f'{district}_tourists_lag1'] = prev_row.get(tourist_col, 0)

            if len(df) >= 3:
                lag3_row = df.iloc[-3]
                new_row_series[f'{district}_tourists_lag3'] = lag3_row.get(tourist_col, 0)
            else:
                new_row_series[f'{district}_tourists_lag3'] = prev_row.get(tourist_col, 0)

        # === APPEND USING loc ===
        df.loc[len(df)] = new_row_series

        # === BACKUP AND SAVE ===
        backup_path = training_data_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        shutil.copy(training_data_path, backup_path)
        print(f"  Backed up original to: {os.path.basename(backup_path)}")

        df.to_csv(training_data_path, index=False)

        print(f"  ✓ Added {prediction_month} to dataset")
        print(f"  Dataset size: {len(df)-1} → {len(df)} rows")
        print(f"  Tourist data:")
        print(f"    Total: {new_row_series['Total']:,.0f}")
        print(f"    Colombo: {new_row_series['colombo_tourists']:,.0f}")
        print(f"    Kandy: {new_row_series['kandy_tourists']:,.0f}")

        return training_data_path

    def retrain_models(self, mode='incremental', severity='medium', existing_version=None):
        """Retrain models using one of three strategies"""
        print(f"\n  Retraining mode: {mode.upper()}")

        if severity == 'high' or mode == 'full':
            training_mode = 'full'
            print("  Strategy: Full retraining on all data")
        elif mode == 'finetune' and existing_version:
            training_mode = 'finetune'
            print("  Strategy: Fine-tuning (transfer learning)")
        else:
            training_mode = 'incremental'
            print(f"  Strategy: Incremental learning (sliding window: {self.window_size} months)")

        training_script = os.path.join(os.path.dirname(__file__), 'train_adaptive_model.py')

        if not os.path.exists(training_script):
            raise FileNotFoundError(f"Training script not found: {training_script}")

        cmd = ['python', training_script, training_mode, str(self.window_size)]

        if training_mode == 'finetune' and existing_version:
            cmd.append(existing_version)

        print(f"  Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=os.path.dirname(training_script)
            )

            if result.returncode == 0:
                print("  ✓ Training completed successfully!")

                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Performance:' in line or 'MAE=' in line:
                        print(f"  {line.strip()}")

                metadata_path = '../models/temp/model_metadata.json'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    return {
                        'success': True,
                        'mode': training_mode,
                        'mae': metadata['test_performance']['mae'],
                        'mape': metadata['test_performance']['mape'],
                        'r2': metadata['test_performance']['r2'],
                        'training_samples': metadata['training_samples']
                    }
                else:
                    return {'success': True, 'mode': training_mode}
            else:
                print(f"   Training failed!")
                print(f"  Error: {result.stderr[:500]}")
                return {'success': False, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            print("   Training timeout!")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            print(f"   Error: {e}")
            return {'success': False, 'error': str(e)}

    def save_trained_models(self, version):
        """Move models from temp to versioned directory"""
        temp_dir = '../models/temp'
        version_dir = os.path.join(self.base_model_path, version)

        if not os.path.exists(temp_dir):
            raise FileNotFoundError(f"Temp models not found: {temp_dir}")

        os.makedirs(version_dir, exist_ok=True)

        files_moved = 0
        for filename in os.listdir(temp_dir):
            src = os.path.join(temp_dir, filename)
            dst = os.path.join(version_dir, filename)
            shutil.move(src, dst)
            files_moved += 1

        print(f"  ✓ Moved {files_moved} files to {version}")

        version_info = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'files': os.listdir(version_dir)
        }

        with open(os.path.join(version_dir, 'version_info.json'), 'w') as f:
            json.dump(version_info, f, indent=2)

        return version_dir