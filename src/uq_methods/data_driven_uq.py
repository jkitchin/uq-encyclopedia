"""Uncertainty quantification methods for data-driven models."""

import numpy as np
from scipy import stats
from typing import Optional
from src.uq_methods.base import UQMethod, UncertaintyResult
from src.benchmark.registry import register_uq_method


@register_uq_method(family='data_driven', method='ensemble')
class EnsembleUQ(UQMethod):
    """
    Ensemble-based uncertainty quantification.

    Uses the ensemble of predictions to compute mean and prediction intervals
    based on the empirical distribution of ensemble members.
    """

    def __init__(self, confidence_level: float = 0.95, **kwargs):
        """
        Initialize ensemble UQ method.

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
        Compute prediction intervals using ensemble predictions.

        Parameters
        ----------
        model : DPOSEModel
            Fitted data-driven model with ensemble predictions
        X_train : np.ndarray
            Training features (not used, but kept for API consistency)
        y_train : np.ndarray
            Training targets (not used, but kept for API consistency)
        X_test : np.ndarray
            Test features
        y_test : np.ndarray, optional
            Test targets (not used)

        Returns
        -------
        result : UncertaintyResult
            Uncertainty quantification results
        """
        # Get ensemble predictions
        y_ensemble = model.predict_ensemble(X_test)

        # Compute mean and std across ensemble
        if y_ensemble.ndim == 1:
            # Single prediction per sample
            y_pred = y_ensemble
            y_std = np.zeros_like(y_pred)
            y_lower = y_pred
            y_upper = y_pred
        else:
            # Ensemble predictions: shape (n_samples, n_ensemble)
            y_pred = y_ensemble.mean(axis=1)
            y_std = y_ensemble.std(axis=1)

            # Compute prediction intervals using quantiles of ensemble
            alpha = 1 - self.confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2

            y_lower = np.quantile(y_ensemble, lower_quantile, axis=1)
            y_upper = np.quantile(y_ensemble, upper_quantile, axis=1)

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=self.confidence_level,
            std=y_std,
            metadata={
                'method': 'ensemble',
                'n_ensemble': y_ensemble.shape[1] if y_ensemble.ndim > 1 else 1,
                'ensemble_predictions': y_ensemble,
            }
        )


@register_uq_method(family='data_driven', method='ensemble_calibrated')
class CalibratedEnsembleUQ(UQMethod):
    """
    Calibrated ensemble-based uncertainty quantification.

    Uses the ensemble predictions but scales the intervals based on
    validation set calibration to achieve better coverage.
    """

    def __init__(self, confidence_level: float = 0.95, calibration_fraction: float = 0.2, **kwargs):
        """
        Initialize calibrated ensemble UQ method.

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
        self.calibration_factor = 1.0

    def compute_intervals(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> UncertaintyResult:
        """
        Compute calibrated prediction intervals.

        Parameters
        ----------
        model : DPOSEModel
            Fitted data-driven model
        X_train : np.ndarray
            Training features (used for calibration)
        y_train : np.ndarray
            Training targets (used for calibration)
        X_test : np.ndarray
            Test features
        y_test : np.ndarray, optional
            Test targets (not used)

        Returns
        -------
        result : UncertaintyResult
            Uncertainty quantification results
        """
        # First get uncalibrated ensemble predictions
        y_ensemble_test = model.predict_ensemble(X_test)

        if y_ensemble_test.ndim == 1:
            # Single prediction - no ensemble
            return UncertaintyResult(
                y_pred=y_ensemble_test,
                y_lower=y_ensemble_test,
                y_upper=y_ensemble_test,
                confidence_level=self.confidence_level,
                std=np.zeros_like(y_ensemble_test),
                metadata={'method': 'ensemble_calibrated', 'calibration_factor': 1.0}
            )

        # Calibrate on training data
        y_ensemble_train = model.predict_ensemble(X_train)
        y_pred_train = y_ensemble_train.mean(axis=1)
        y_std_train = y_ensemble_train.std(axis=1)

        # Flatten y_train if needed
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train

        # Compute z-scores for training data
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs((y_train_flat - y_pred_train) / y_std_train)
            z_scores = z_scores[np.isfinite(z_scores)]

        # Calibration factor: empirical quantile of z-scores
        if len(z_scores) > 0:
            self.calibration_factor = np.quantile(z_scores, self.confidence_level)
        else:
            self.calibration_factor = 1.96  # Default for 95% confidence

        # Ensure minimum calibration factor
        self.calibration_factor = max(self.calibration_factor, 1.0)

        # Compute calibrated intervals for test set
        y_pred = y_ensemble_test.mean(axis=1)
        y_std = y_ensemble_test.std(axis=1)

        y_lower = y_pred - self.calibration_factor * y_std
        y_upper = y_pred + self.calibration_factor * y_std

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=self.confidence_level,
            std=y_std,
            metadata={
                'method': 'ensemble_calibrated',
                'calibration_factor': self.calibration_factor,
                'n_ensemble': y_ensemble_test.shape[1],
            }
        )
