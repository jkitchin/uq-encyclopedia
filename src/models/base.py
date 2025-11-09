"""Base classes and interfaces for models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class Model(ABC):
    """
    Abstract base class for all models.

    Models are responsible for:
    - Fitting to training data
    - Making predictions
    - Providing metadata about the model
    """

    def __init__(self, **kwargs):
        """
        Initialize model with hyperparameters.

        Parameters
        ----------
        **kwargs
            Model-specific hyperparameters
        """
        self.hyperparameters = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        """
        Fit the model to training data.

        Parameters
        ----------
        X : np.ndarray
            Training input features, shape (n_samples, n_features)
        y : np.ndarray
            Training target values, shape (n_samples,)

        Returns
        -------
        self : Model
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
            Predicted values, shape (n_samples,)
        """
        pass

    def get_name(self) -> str:
        """
        Get model name.

        Returns
        -------
        name : str
            Model name
        """
        return self.__class__.__name__

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.

        Returns
        -------
        hyperparameters : dict
            Dictionary of hyperparameters
        """
        return self.hyperparameters

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns
        -------
        metadata : dict
            Model metadata including name, hyperparameters, and fit status
        """
        return {
            'name': self.get_name(),
            'hyperparameters': self.get_hyperparameters(),
            'is_fitted': self.is_fitted,
        }


class LinearModel(Model):
    """
    Base class for linear models (models linear in parameters).

    Linear models have the form: y = X @ beta
    where beta are the model parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coef_ = None  # Coefficients (beta)
        self.intercept_ = None  # Intercept (if applicable)

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get fitted coefficients.

        Returns
        -------
        coef : np.ndarray or None
            Model coefficients if fitted, None otherwise
        """
        return self.coef_


class NonlinearModel(Model):
    """
    Base class for nonlinear models (nonlinear in parameters).

    Nonlinear models require iterative optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params_ = None  # Fitted parameters
        self.optimization_result_ = None  # Store optimization info

    def get_parameters(self) -> Optional[np.ndarray]:
        """
        Get fitted parameters.

        Returns
        -------
        params : np.ndarray or None
            Model parameters if fitted, None otherwise
        """
        return self.params_
