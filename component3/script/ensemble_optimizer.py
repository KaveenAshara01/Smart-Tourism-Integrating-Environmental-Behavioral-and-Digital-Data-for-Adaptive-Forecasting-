import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple
import json


class EnsembleOptimizer:
    """Adaptive ensemble weight optimization"""

    def __init__(self, method='inverse_error'):
        """
        method: 'inverse_error', 'bayesian', or 'gradient'
        """
        self.method = method
        self.history = []

    def inverse_error_weighting(self, xgb_mape: float, lstm_mape: float) -> Tuple[float, float]:
        """
        Simple inverse error weighting
        Better performing model gets higher weight
        """
        epsilon = 1e-10

        inverse_xgb = 1 / (xgb_mape + epsilon)
        inverse_lstm = 1 / (lstm_mape + epsilon)

        total = inverse_xgb + inverse_lstm

        xgb_weight = inverse_xgb / total
        lstm_weight = inverse_lstm / total

        return xgb_weight, lstm_weight

    def bayesian_optimization(self,
                              xgb_predictions: np.ndarray,
                              lstm_predictions: np.ndarray,
                              actuals: np.ndarray) -> Tuple[float, float]:
        """
        Bayesian optimization to find optimal weights
        """

        def objective(weights):
            ensemble_pred = weights[0] * xgb_predictions + weights[1] * lstm_predictions
            mse = np.mean((actuals - ensemble_pred) ** 2)
            return mse

        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1), (0, 1)]

        result = minimize(
            objective,
            x0=[0.5, 0.5],
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        if result.success:
            return float(result.x[0]), float(result.x[1])
        else:
            # Fallback to equal weights
            return 0.5, 0.5

    def gradient_based_optimization(self,
                                    xgb_predictions: np.ndarray,
                                    lstm_predictions: np.ndarray,
                                    actuals: np.ndarray,
                                    learning_rate=0.01,
                                    iterations=100) -> Tuple[float, float]:
        """
        Gradient descent to optimize weights
        """
        # Initialize weights
        w_xgb = 0.5
        w_lstm = 0.5

        for _ in range(iterations):
            # Current predictions
            ensemble_pred = w_xgb * xgb_predictions + w_lstm * lstm_predictions

            # Calculate gradients
            error = ensemble_pred - actuals
            grad_xgb = np.mean(error * xgb_predictions)
            grad_lstm = np.mean(error * lstm_predictions)

            # Update weights
            w_xgb -= learning_rate * grad_xgb
            w_lstm -= learning_rate * grad_lstm

            # Normalize to sum to 1
            total = w_xgb + w_lstm
            w_xgb /= total
            w_lstm /= total

            # Clip to valid range
            w_xgb = np.clip(w_xgb, 0, 1)
            w_lstm = 1 - w_xgb

        return float(w_xgb), float(w_lstm)

    def optimize_weights(self,
                         xgb_predictions: np.ndarray = None,
                         lstm_predictions: np.ndarray = None,
                         actuals: np.ndarray = None,
                         xgb_mape: float = None,
                         lstm_mape: float = None) -> Dict:
        """
        Main optimization method
        """
        if self.method == 'inverse_error' and xgb_mape is not None and lstm_mape is not None:
            xgb_weight, lstm_weight = self.inverse_error_weighting(xgb_mape, lstm_mape)

        elif self.method == 'bayesian' and xgb_predictions is not None:
            xgb_weight, lstm_weight = self.bayesian_optimization(xgb_predictions, lstm_predictions, actuals)

        elif self.method == 'gradient' and xgb_predictions is not None:
            xgb_weight, lstm_weight = self.gradient_based_optimization(xgb_predictions, lstm_predictions, actuals)

        else:
            # Default equal weights
            xgb_weight, lstm_weight = 0.5, 0.5

        result = {
            'xgboost_weight': xgb_weight,
            'lstm_weight': lstm_weight,
            'method': self.method
        }

        self.history.append(result)

        return result

    def get_weight_history(self) -> list:
        """Return optimization history"""
        return self.history