"""Base classes and interfaces for uncertainty quantification methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class UncertaintyResult:
    """
    Container for uncertainty quantification results.

    Attributes
    ----------
    y_pred : np.ndarray
        Point predictions, shape (n_samples,)
    y_lower : np.ndarray
        Lower bounds of prediction intervals, shape (n_samples,)
    y_upper : np.ndarray
        Upper bounds of prediction intervals, shape (n_samples,)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% intervals)
    std : np.ndarray, optional
        Standard deviations if available, shape (n_samples,)
    metadata : dict, optional
        Additional method-specific information
    """
    y_pred: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    confidence_level: float
    std: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def interval_width(self) -> np.ndarray:
        """Compute prediction interval widths."""
        return self.y_upper - self.y_lower

    @property
    def mean_interval_width(self) -> float:
        """Compute mean prediction interval width."""
        return np.mean(self.interval_width)

    @property
    def median_interval_width(self) -> float:
        """Compute median prediction interval width."""
        return np.median(self.interval_width)


class UQMethod(ABC):
    """
    Abstract base class for uncertainty quantification methods.

    UQ methods are responsible for:
    - Computing prediction intervals
    - Estimating uncertainty in predictions
    - Providing calibrated confidence levels
    """

    def __init__(self, confidence_level: float = 0.95, **kwargs):
        """
        Initialize UQ method.

        Parameters
        ----------
        confidence_level : float
            Desired confidence level (default 0.95 for 95% intervals)
        **kwargs
            Method-specific parameters
        """
        self.confidence_level = confidence_level
        self.method_params = kwargs

    @abstractmethod
    def compute_intervals(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> UncertaintyResult:
        """
        Compute prediction intervals for test data.

        Parameters
        ----------
        model : Model
            Fitted model instance
        X_train : np.ndarray
            Training input features, shape (n_train, n_features)
        y_train : np.ndarray
            Training target values, shape (n_train,)
        X_test : np.ndarray
            Test input features, shape (n_test, n_features)
        y_test : np.ndarray, optional
            Test target values for methods that need them (e.g., conformal)

        Returns
        -------
        result : UncertaintyResult
            Uncertainty quantification results
        """
        pass

    def get_name(self) -> str:
        """
        Get UQ method name.

        Returns
        -------
        name : str
            Method name
        """
        return self.__class__.__name__

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get UQ method metadata.

        Returns
        -------
        metadata : dict
            Method metadata
        """
        return {
            'name': self.get_name(),
            'confidence_level': self.confidence_level,
            'method_params': self.method_params,
        }


class BootstrapUQMixin:
    """
    Mixin class for bootstrap-based uncertainty quantification.

    Can be combined with any UQ method to add bootstrap capabilities.
    """

    def bootstrap_intervals(
        self,
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        **model_kwargs,
    ) -> UncertaintyResult:
        """
        Compute prediction intervals via bootstrapping.

        Parameters
        ----------
        model_class : type
            Model class to instantiate
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
        **model_kwargs
            Additional arguments for model initialization

        Returns
        -------
        result : UncertaintyResult
            Bootstrap-based uncertainty results
        """
        n_train = len(y_train)
        n_test = len(X_test)
        predictions = np.zeros((n_bootstrap, n_test))

        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            # Fit model and predict
            model = model_class(**model_kwargs)
            model.fit(X_boot, y_boot)
            predictions[i] = model.predict(X_test)

        # Compute statistics
        y_pred = np.mean(predictions, axis=0)
        y_std = np.std(predictions, axis=0)

        # Compute percentile-based intervals
        alpha = 1 - confidence_level
        y_lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
        y_upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=confidence_level,
            std=y_std,
            metadata={'n_bootstrap': n_bootstrap},
        )
