import numpy as np
from scipy import stats
from typing import Dict, Tuple
import json


class DriftDetector:
    """Robust drift detection using multiple statistical methods"""

    def __init__(self, ks_alpha=0.05, cusum_threshold=3.0, ph_delta=0.005, ph_lambda=50):
        self.ks_alpha = ks_alpha
        self.cusum_threshold = cusum_threshold
        self.ph_delta = ph_delta
        self.ph_lambda = ph_lambda
        self.ph_sum = 0
        self.ph_min = 0

    def kolmogorov_smirnov_test(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """KS test for distribution drift"""
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = p_value < self.ks_alpha

        return {
            'method': 'KS-Test',
            'drift_detected': bool(drift_detected),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'severity': 'high' if p_value < 0.01 else 'medium' if drift_detected else 'none'
        }

    def cusum_test(self, errors: np.ndarray) -> Dict:
        """CUSUM control chart"""
        mean_error = np.mean(errors)
        cumsum = np.cumsum(errors - mean_error)
        max_cumsum = np.abs(cumsum).max()
        drift_detected = max_cumsum > self.cusum_threshold

        return {
            'method': 'CUSUM',
            'drift_detected': bool(drift_detected),
            'max_cumsum': float(max_cumsum),
            'threshold': self.cusum_threshold,
            'severity': 'high' if max_cumsum > self.cusum_threshold * 1.5 else 'medium' if drift_detected else 'none'
        }

    def page_hinkley_test(self, error: float) -> Dict:
        """Page-Hinkley sequential test"""
        self.ph_sum += error - self.ph_delta

        if self.ph_sum < self.ph_min:
            self.ph_min = self.ph_sum

        ph_statistic = self.ph_sum - self.ph_min
        drift_detected = ph_statistic > self.ph_lambda

        if drift_detected:
            self.ph_sum = 0
            self.ph_min = 0

        return {
            'method': 'Page-Hinkley',
            'drift_detected': bool(drift_detected),
            'statistic': float(ph_statistic),
            'severity': 'high' if ph_statistic > self.ph_lambda * 1.5 else 'medium' if drift_detected else 'none'
        }

    def detect_drift(self, reference_errors: np.ndarray, current_errors: np.ndarray) -> Dict:
        """
        Multi-method drift detection with consensus
        Returns drift decision based on majority vote
        """
        ks_result = self.kolmogorov_smirnov_test(reference_errors, current_errors)
        cusum_result = self.cusum_test(current_errors)
        ph_result = self.page_hinkley_test(np.mean(current_errors))

        # Consensus: at least 2 methods agree
        votes = sum([
            ks_result['drift_detected'],
            cusum_result['drift_detected'],
            ph_result['drift_detected']
        ])

        consensus_drift = votes >= 2

        # Determine overall severity
        severities = [ks_result['severity'], cusum_result['severity'], ph_result['severity']]
        if 'high' in severities:
            severity = 'high'
        elif 'medium' in severities:
            severity = 'medium'
        else:
            severity = 'none'

        return {
            'drift_detected': consensus_drift,
            'votes': int(votes),
            'severity': severity,
            'methods': {
                'ks_test': ks_result,
                'cusum': cusum_result,
                'page_hinkley': ph_result
            }
        }