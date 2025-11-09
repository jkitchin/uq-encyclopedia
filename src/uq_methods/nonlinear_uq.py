"""Uncertainty quantification methods for nonlinear models."""

import numpy as np
from scipy import stats
from typing import Optional
from pycse import nlinfit
from src.uq_methods.base import UQMethod, UncertaintyResult
from src.benchmark.registry import register_uq_method


@register_uq_method(family='nonlinear', method='nlinfit')
class NlinfitUQ(UQMethod):
    """
    Nonlinear fit prediction intervals using pycse.nlinfit.

    Uses pycse.nlinfit to compute prediction intervals for nonlinear regression
    based on the Jacobian and residual standard error.

    Reference: https://github.com/jkitchin/pycse
    """

    def __init__(self, confidence_level: float = 0.95, **kwargs):
        """
        Initialize nlinfit UQ method.

        Parameters
        ----------
        confidence_level : float
            Confidence level (default 0.95)
        **kwargs
            Additional method parameters
        """
        super().__init__(confidence_level=confidence_level, **kwargs)

    def compute_intervals(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> UncertaintyResult:
        """
        Compute prediction intervals using pycse.nlinfit.

        Parameters
        ----------
        model : NonlinearModel
            Fitted nonlinear regression model
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        y_test : np.ndarray, optional
            Test targets (not used)

        Returns
        -------
        result : UncertaintyResult
            Uncertainty quantification results
        """
        # Flatten inputs
        X_train_flat = X_train.flatten()
        X_test_flat = X_test.flatten()
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train

        alpha = 1 - self.confidence_level

        # Use nlinfit to get parameter estimates and confidence intervals
        params, params_ci, _ = nlinfit(
            model.model_func, X_train_flat, y_train_flat, model.initial_guess, alpha=alpha
        )

        # Compute predictions (func signature: func(x, *params))
        y_pred_train = model.model_func(X_train_flat, *params)
        y_pred = model.model_func(X_test_flat, *params)

        # Compute residual standard error from actual data residuals
        # NOTE: nlinfit returns parameter-related residuals, not data residuals!
        data_residuals = y_train_flat - y_pred_train
        n = len(y_train_flat)
        p = len(params)
        df = n - p
        sigma = np.sqrt(np.sum(data_residuals**2) / df)

        # Compute Jacobian for prediction intervals
        def jacobian_func(params_val, x_val):
            """Numerical Jacobian of model function."""
            eps = 1e-8
            J = np.zeros((len(x_val), len(params_val)))
            f0 = model.model_func(x_val, *params_val)

            for i in range(len(params_val)):
                params_perturb = params_val.copy()
                params_perturb[i] += eps
                f_perturb = model.model_func(x_val, *params_perturb)
                J[:, i] = (f_perturb - f0) / eps

            return J

        # Compute covariance matrix of parameters
        J_train = jacobian_func(params, X_train_flat)
        try:
            cov_params = sigma**2 * np.linalg.inv(J_train.T @ J_train)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            cov_params = sigma**2 * np.linalg.pinv(J_train.T @ J_train)

        # Compute prediction variance
        J_test = jacobian_func(params, X_test_flat)

        # Prediction variance: sigma^2 * (1 + J * cov_params * J^T)
        pred_var = sigma**2 * np.ones(len(X_test_flat))
        for i in range(len(X_test_flat)):
            pred_var[i] += J_test[i:i+1, :] @ cov_params @ J_test[i:i+1, :].T

        pred_std = np.sqrt(pred_var)

        # Compute prediction intervals using t-distribution
        t_stat = stats.t.ppf(1 - alpha/2, df)
        margin = t_stat * pred_std

        y_lower = y_pred - margin
        y_upper = y_pred + margin

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=self.confidence_level,
            std=pred_std,
            metadata={
                'method': 'nlinfit',
                'parameters': params,
                'parameter_ci': params_ci,
                't_statistic': t_stat,
                'degrees_of_freedom': df,
                'sigma': sigma,
            }
        )


@register_uq_method(family='nonlinear', method='conformal')
class ConformalPredictionNonlinear(UQMethod):
    """
    Conformal Prediction for nonlinear regression.

    Distribution-free method that provides prediction intervals with guaranteed
    coverage under exchangeability assumptions.
    """

    def __init__(self, confidence_level: float = 0.95, calibration_fraction: float = 0.2, **kwargs):
        """
        Initialize conformal prediction UQ method.

        Parameters
        ----------
        confidence_level : float
            Confidence level (default 0.95)
        calibration_fraction : float
            Fraction of training data to use for calibration (default 0.2)
        **kwargs
            Additional method parameters
        """
        super().__init__(confidence_level=confidence_level, **kwargs)
        self.calibration_fraction = calibration_fraction

    def compute_intervals(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> UncertaintyResult:
        """
        Compute prediction intervals using conformal prediction.

        Parameters
        ----------
        model : NonlinearModel
            Fitted nonlinear regression model
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        y_test : np.ndarray, optional
            Test targets (not used)

        Returns
        -------
        result : UncertaintyResult
            Uncertainty quantification results
        """
        # Flatten inputs
        X_train_flat = X_train.flatten()
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        X_test_flat = X_test.flatten()

        # Split training data into proper training and calibration sets
        n_train = len(X_train_flat)
        n_cal = int(n_train * self.calibration_fraction)
        n_proper_train = n_train - n_cal

        # Use last portion for calibration
        X_proper_train = X_train_flat[:n_proper_train]
        y_proper_train = y_train_flat[:n_proper_train]
        X_cal = X_train_flat[n_proper_train:]
        y_cal = y_train_flat[n_proper_train:]

        # Re-fit model on proper training set
        model.fit(X_proper_train.reshape(-1, 1), y_proper_train)

        # Compute nonconformity scores on calibration set
        y_cal_pred = model.predict(X_cal.reshape(-1, 1))
        nonconformity_scores = np.abs(y_cal - y_cal_pred)

        # Compute quantile for prediction intervals
        alpha = 1 - self.confidence_level
        # Add 1 to numerator for finite-sample correction
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)  # Cap at 1.0
        quantile = np.quantile(nonconformity_scores, q_level)

        # Compute predictions and intervals on test set
        y_pred = model.predict(X_test_flat.reshape(-1, 1))
        y_lower = y_pred - quantile
        y_upper = y_pred + quantile

        # Estimate std as quantile / z-score
        z_score = stats.norm.ppf(1 - alpha/2)
        estimated_std = np.full_like(y_pred, quantile / z_score)

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=self.confidence_level,
            std=estimated_std,
            metadata={
                'method': 'conformal',
                'quantile': quantile,
                'n_calibration': n_cal,
                'calibration_fraction': self.calibration_fraction,
            }
        )
