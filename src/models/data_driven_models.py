"""Data-driven models for regression."""

import numpy as np
from typing import Optional
from pycse.sklearn.dpose import DPOSE
import jax.numpy as jnp
from jax import nn as jnn


class DPOSEModel:
    """
    DPOSE: Direct Propagation of Shallow Ensembles.

    A shallow ensemble neural network providing calibrated uncertainty estimates.
    """

    def __init__(
        self,
        n_hidden: int = 20,
        n_ensemble: int = 32,
        activation: str = 'tanh',
        seed: int = 42,
        loss_type: str = 'crps',
        optimizer: str = 'bfgs',
        min_sigma: float = 0.001,
    ):
        """
        Initialize DPOSE model.

        Parameters
        ----------
        n_hidden : int
            Number of neurons in hidden layer
        n_ensemble : int
            Number of ensemble members (output layer size)
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid')
        seed : int
            Random seed for reproducibility
        loss_type : str
            Loss function type ('crps' or 'nll')
        optimizer : str
            Optimizer to use ('bfgs', 'adam', 'sgd')
        min_sigma : float
            Minimum standard deviation for predictions
        """
        self.n_hidden = n_hidden
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.min_sigma = min_sigma

        # Map activation name to JAX function
        self.activation_map = {
            'tanh': jnn.tanh,
            'relu': jnn.relu,
            'sigmoid': jnn.sigmoid,
        }
        if activation not in self.activation_map:
            raise ValueError(f"Activation '{activation}' not supported. Use: {list(self.activation_map.keys())}")
        self.activation = self.activation_map[activation]
        self.activation_name = activation

        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the DPOSE model."""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get input dimension
        n_features = X.shape[1]

        # Create DPOSE model with architecture: (n_features, n_hidden, n_ensemble)
        self.model = DPOSE(
            layers=(n_features, self.n_hidden, self.n_ensemble),
            activation=self.activation,
            seed=self.seed,
            loss_type=self.loss_type,
            optimizer=self.optimizer,
            min_sigma=self.min_sigma,
        )

        # Flatten y if needed
        y_flat = y.flatten() if y.ndim > 1 else y

        # Fit model
        self.model.fit(X, y_flat)

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Predict using the DPOSE model.

        Parameters
        ----------
        X : np.ndarray
            Input features
        return_std : bool
            If True, return (mean, std) tuple

        Returns
        -------
        y_pred : np.ndarray or tuple
            Predictions (mean only or (mean, std))
        """
        if self.model is None:
            raise ValueError("Model must be fit before prediction")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get predictions from all ensemble members
        y_pred = self.model.predict(X)

        if return_std:
            # DPOSE predict returns ensemble predictions
            # We need to compute mean and std across ensemble
            if y_pred.ndim > 1:
                mean = y_pred.mean(axis=1)
                std = y_pred.std(axis=1)
            else:
                mean = y_pred
                std = np.zeros_like(y_pred)
            return mean, std
        else:
            # Return mean prediction
            if y_pred.ndim > 1:
                return y_pred.mean(axis=1)
            else:
                return y_pred

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        y_ensemble : np.ndarray
            Predictions from all ensemble members, shape (n_samples, n_ensemble)
        """
        if self.model is None:
            raise ValueError("Model must be fit before prediction")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict_ensemble(X)
