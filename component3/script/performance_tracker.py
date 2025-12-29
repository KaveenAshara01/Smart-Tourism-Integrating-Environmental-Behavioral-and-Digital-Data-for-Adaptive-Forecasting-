
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List
import sqlite3
from datetime import datetime
import json


class PerformanceTracker:
    """Track and store model performance metrics"""

    def __init__(self, db_path='../data/monitoring.db'):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                district TEXT NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                mape REAL NOT NULL,
                r2 REAL NOT NULL,
                model_version TEXT NOT NULL
            )
        ''')

        # Drift detection results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                drift_detected INTEGER NOT NULL,
                severity TEXT NOT NULL,
                votes INTEGER NOT NULL,
                ks_test_result TEXT,
                cusum_result TEXT,
                page_hinkley_result TEXT,
                action_taken TEXT
            )
        ''')

        # Retraining history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                old_version TEXT NOT NULL,
                new_version TEXT NOT NULL,
                training_samples INTEGER NOT NULL,
                training_duration REAL NOT NULL,
                improvement_mae REAL,
                improvement_r2 REAL
            )
        ''')

        # Ensemble weights history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                xgboost_weight REAL NOT NULL,
                lstm_weight REAL NOT NULL,
                optimization_method TEXT NOT NULL,
                xgboost_mape REAL,
                lstm_mape REAL
            )
        ''')

        conn.commit()
        conn.close()

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate all performance metrics"""
        epsilon = 1e-10

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }

    def log_performance(self, prediction_month: str, district: str,
                        actual: np.ndarray, predicted: np.ndarray,
                        model_version: str):
        """Log performance metrics to database"""
        metrics = self.calculate_metrics(actual, predicted)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, prediction_month, district, mae, rmse, mape, r2, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction_month,
            district,
            metrics['mae'],
            metrics['rmse'],
            metrics['mape'],
            metrics['r2'],
            model_version
        ))

        conn.commit()
        conn.close()

        return metrics

    def log_drift_event(self, prediction_month: str, drift_result: Dict, action_taken: str):
        """Log drift detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO drift_events 
            (timestamp, prediction_month, drift_detected, severity, votes,
             ks_test_result, cusum_result, page_hinkley_result, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction_month,
            1 if drift_result['drift_detected'] else 0,
            drift_result['severity'],
            drift_result['votes'],
            json.dumps(drift_result['methods']['ks_test']),
            json.dumps(drift_result['methods']['cusum']),
            json.dumps(drift_result['methods']['page_hinkley']),
            action_taken
        ))

        conn.commit()
        conn.close()

    def log_retraining(self, trigger_reason: str, old_version: str, new_version: str,
                       training_samples: int, training_duration: float,
                       old_mae: float = None, new_mae: float = None,
                       old_r2: float = None, new_r2: float = None):
        """Log retraining event"""
        improvement_mae = None
        improvement_r2 = None

        if old_mae and new_mae:
            improvement_mae = ((old_mae - new_mae) / old_mae) * 100

        if old_r2 and new_r2:
            improvement_r2 = ((new_r2 - old_r2) / old_r2) * 100

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO retraining_history 
            (timestamp, trigger_reason, old_version, new_version, 
             training_samples, training_duration, improvement_mae, improvement_r2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            trigger_reason,
            old_version,
            new_version,
            training_samples,
            training_duration,
            improvement_mae,
            improvement_r2
        ))

        conn.commit()
        conn.close()

    def log_ensemble_weights(self, prediction_month: str, weights: Dict,
                             xgb_mape: float = None, lstm_mape: float = None):
        """Log ensemble weight optimization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ensemble_weights 
            (timestamp, prediction_month, xgboost_weight, lstm_weight, 
             optimization_method, xgboost_mape, lstm_mape)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction_month,
            weights['xgboost_weight'],
            weights['lstm_weight'],
            weights['method'],
            xgb_mape,
            lstm_mape
        ))

        conn.commit()
        conn.close()

    def get_performance_trend(self, district: str = None, limit: int = 12) -> pd.DataFrame:
        """Get performance trend over time"""
        conn = sqlite3.connect(self.db_path)

        if district:
            query = '''
                SELECT prediction_month, mae, rmse, mape, r2, model_version
                FROM performance_metrics
                WHERE district = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(district, limit))
        else:
            query = '''
                SELECT prediction_month, AVG(mae) as mae, AVG(rmse) as rmse, 
                       AVG(mape) as mape, AVG(r2) as r2
                FROM performance_metrics
                GROUP BY prediction_month
                ORDER BY prediction_month DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))

        conn.close()
        return df

    def get_drift_history(self, limit: int = 10) -> pd.DataFrame:
        """Get drift detection history"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT timestamp, prediction_month, drift_detected, severity, 
                   votes, action_taken
            FROM drift_events
            ORDER BY timestamp DESC
            LIMIT ?
        '''

        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        return df