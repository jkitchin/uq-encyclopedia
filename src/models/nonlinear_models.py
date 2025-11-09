"""Nonlinear regression models."""

import numpy as np
from typing import Callable, List
from pycse import nlinfit


class NonlinearModel:
    """Nonlinear regression using pycse.nlinfit."""

    def __init__(self, model_func: Callable, param_names: List[str], initial_guess: List[float]):
        """
        Initialize nonlinear model.

        Parameters
        ----------
        model_func : callable
            Model function with signature: func(x, *params) -> y  (pycse.nlinfit format)
        param_names : list of str
            Names of parameters
        initial_guess : list of float
            Initial parameter guess for optimization
        """
        self.model_func = model_func
        self.param_names = param_names
        self.initial_guess = np.array(initial_guess)
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the nonlinear model using nlinfit."""
        X_flat = X.flatten()
        y_flat = y.flatten() if y.ndim > 1 else y

        # Use nlinfit to fit the model
        # nlinfit expects func(x, *params)
        # NOTE: The third return value from nlinfit is NOT data residuals!
        self.params, self.params_ci, _ = nlinfit(
            self.model_func, X_flat, y_flat, self.initial_guess, alpha=0.05
        )

        # Compute actual data residuals
        y_pred = self.model_func(X_flat, *self.params)
        self.residuals = y_flat - y_pred

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted parameters."""
        if self.params is None:
            raise ValueError("Model must be fit before prediction")

        X_flat = X.flatten()
        # Call function with signature func(x, *params)
        y_pred = self.model_func(X_flat, *self.params)

        return y_pred


# Model function definitions for each dataset type
# NOTE: pycse.nlinfit expects signature: func(x, *params)
def exponential_decay_func(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def logistic_growth_func(x, L, k, x0):
    """Logistic growth: y = L / (1 + exp(-k*(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def michaelis_menten_func(x, Vmax, Km):
    """Michaelis-Menten: y = (Vmax * x) / (Km + x)"""
    return (Vmax * x) / (Km + x)


def gaussian_func(x, a, mu, sigma):
    """Gaussian: y = a * exp(-((x - mu)^2) / (2 * sigma^2))"""
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))
