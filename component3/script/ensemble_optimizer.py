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

    def evaluate_weights(self,
                         weights: Tuple[float, float],
                         xgb_predictions: np.ndarray,
                         lstm_predictions: np.ndarray,
                         actuals: np.ndarray,
                         metric: str = 'mse') -> float:
        """
        Evaluate ensemble performance for given weights
        """
        w_xgb, w_lstm = weights
        ensemble_pred = w_xgb * xgb_predictions + w_lstm * lstm_predictions

        if metric == 'mse':
            return float(np.mean((actuals - ensemble_pred) ** 2))

        elif metric == 'mae':
            return float(np.mean(np.abs(actuals - ensemble_pred)))

        else:
            raise ValueError(f"Unsupported metric: {metric}")

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
            total = w_xgb + w_lstm + 1e-10
            w_xgb /= total
            w_lstm /= total

            # Clip to valid range
            w_xgb = np.clip(w_xgb, 0, 1)
            w_lstm = 1 - w_xgb

        return float(w_xgb), float(w_lstm)

    def optimize_weights_auto(self,
                              xgb_predictions: np.ndarray = None,
                              lstm_predictions: np.ndarray = None,
                              actuals: np.ndarray = None,
                              xgb_mape: float = None,
                              lstm_mape: float = None,
                              eval_metric: str = 'mse') -> Dict:
        """
        Automatically choose the best ensemble weighting strategy
        """

        candidates = []

        # ---- Case 1: Only MAPE available → inverse-error only ----
        if (xgb_predictions is None or lstm_predictions is None or actuals is None):
            if xgb_mape is None or lstm_mape is None:
                # Absolute fallback
                return {
                    'method': 'equal',
                    'xgboost_weight': 0.5,
                    'lstm_weight': 0.5
                }

            w_inv = self.inverse_error_weighting(xgb_mape, lstm_mape)
            return {
                'method': 'inverse_error',
                'xgboost_weight': w_inv[0],
                'lstm_weight': w_inv[1]
            }

        # ---- Case 2: Full data available → evaluate multiple strategies ----

        # Inverse-error baseline (optional but recommended for research)
        if xgb_mape is not None and lstm_mape is not None:
            w_inv = self.inverse_error_weighting(xgb_mape, lstm_mape)
            err_inv = self.evaluate_weights(
                w_inv, xgb_predictions, lstm_predictions, actuals, eval_metric
            )
            candidates.append(('inverse_error', w_inv, err_inv))

        # Bayesian optimization
        w_bayes = self.bayesian_optimization(
            xgb_predictions, lstm_predictions, actuals
        )
        err_bayes = self.evaluate_weights(
            w_bayes, xgb_predictions, lstm_predictions, actuals, eval_metric
        )
        candidates.append(('bayesian', w_bayes, err_bayes))

        # Gradient-based optimization
        w_grad = self.gradient_based_optimization(
            xgb_predictions, lstm_predictions, actuals
        )
        err_grad = self.evaluate_weights(
            w_grad, xgb_predictions, lstm_predictions, actuals, eval_metric
        )
        candidates.append(('gradient', w_grad, err_grad))

        # ---- Select best method ----
        best_method, best_weights, best_error = min(
            candidates, key=lambda x: x[2]
        )

        result = {
            'method': best_method,
            'xgboost_weight': float(best_weights[0]),
            'lstm_weight': float(best_weights[1]),
            'evaluation_metric': eval_metric,
            'score': float(best_error),
            'all_candidates': [
                {
                    'method': m,
                    'xgboost_weight': float(w[0]),
                    'lstm_weight': float(w[1]),
                    'score': float(e)
                }
                for m, w, e in candidates
            ]
        }

        self.history.append(result)
        return result

    def get_weight_history(self) -> list:
        """Return optimization history"""
        return self.history