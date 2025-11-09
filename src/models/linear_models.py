"""Linear models for regression with uncertainty quantification."""

import numpy as np
from typing import Optional, Callable, List
from src.models.base import LinearModel
from src.benchmark.registry import register_model


@register_model(family='linear')
class OLSRegression(LinearModel):
    """
    Ordinary Least Squares (OLS) linear regression.

    Supports polynomial and custom basis functions.
    Stores information needed for uncertainty quantification (hat matrix, residuals, etc.).
    """

    def __init__(
        self,
        degree: int = 1,
        fit_intercept: bool = True,
        basis_functions: Optional[List[Callable]] = None,
        **kwargs
    ):
        """
        Initialize OLS regression model.

        Parameters
        ----------
        degree : int
            Polynomial degree (only used if basis_functions is None)
        fit_intercept : bool
            Whether to fit an intercept term
        basis_functions : list of callable, optional
            Custom basis functions. Each should take X and return transformed features.
            If None, uses polynomial basis up to specified degree.
        **kwargs
            Additional arguments passed to LinearModel
        """
        super().__init__(**kwargs)
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.basis_functions = basis_functions

        # These will be set during fit
        self.X_train_ = None  # Store training data for UQ
        self.y_train_ = None
        self.residuals_ = None
        self.sigma_squared_ = None  # Residual variance
        self.XtX_inv_ = None  # (X^T X)^{-1} for UQ calculations

    def _create_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Create design matrix with basis functions.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        X_design : np.ndarray
            Design matrix, shape (n_samples, n_basis)
        """
        X_flat = X.flatten()

        if self.basis_functions is not None:
            # Use custom basis functions
            features = [f(X_flat) for f in self.basis_functions]
            X_design = np.column_stack(features)
        else:
            # Use polynomial basis
            if self.degree == 1:
                X_design = X_flat.reshape(-1, 1)
            else:
                # Create polynomial features [x, x^2, x^3, ..., x^degree]
                X_design = np.column_stack([X_flat**i for i in range(1, self.degree + 1)])

        # Add intercept column if requested
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(X_flat)), X_design])

        return X_design

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLSRegression':
        """
        Fit OLS regression model.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        self : OLSRegression
            Fitted model
        """
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Create design matrix
        X_design = self._create_design_matrix(X)

        # Solve normal equations: beta = (X^T X)^{-1} X^T y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y

        # Store (X^T X)^{-1} for uncertainty quantification
        self.XtX_inv_ = np.linalg.inv(XtX)
        self.coef_ = self.XtX_inv_ @ Xty

        # Compute residuals and variance
        y_pred = X_design @ self.coef_
        self.residuals_ = y - y_pred
        n, p = X_design.shape
        self.sigma_squared_ = np.sum(self.residuals_**2) / (n - p)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
            Predictions, shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_design = self._create_design_matrix(X)
        return X_design @ self.coef_

    def get_hat_matrix_diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Get diagonal of hat matrix H = X(X^T X)^{-1}X^T.

        Used for computing prediction intervals.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        h : np.ndarray
            Diagonal of hat matrix, shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_design = self._create_design_matrix(X)
        # h_i = x_i^T (X^T X)^{-1} x_i
        h = np.sum(X_design @ self.XtX_inv_ * X_design, axis=1)
        return h

    def get_prediction_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Get variance of predictions.

        Var[y_pred] = sigma^2 * x^T (X^T X)^{-1} x

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        variance : np.ndarray
            Prediction variance, shape (n_samples,)
        """
        h = self.get_hat_matrix_diagonal(X)
        return self.sigma_squared_ * h

    def get_prediction_std(self, X: np.ndarray) -> np.ndarray:
        """
        Get standard deviation of predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features)

        Returns
        -------
        std : np.ndarray
            Prediction standard deviation, shape (n_samples,)
        """
        return np.sqrt(self.get_prediction_variance(X))


# Convenience classes for specific polynomial degrees
@register_model(family='linear', degree=1)
class LinearRegression(OLSRegression):
    """Linear regression (degree 1 polynomial)."""
    def __init__(self, **kwargs):
        super().__init__(degree=1, **kwargs)


@register_model(family='linear', degree=2)
class QuadraticRegression(OLSRegression):
    """Quadratic regression (degree 2 polynomial)."""
    def __init__(self, **kwargs):
        super().__init__(degree=2, **kwargs)


@register_model(family='linear', degree=3)
class CubicRegression(OLSRegression):
    """Cubic regression (degree 3 polynomial)."""
    def __init__(self, **kwargs):
        super().__init__(degree=3, **kwargs)


# Specialized models for specific datasets
@register_model(family='linear', basis='lennard_jones')
class LennardJonesRegression(OLSRegression):
    """
    Regression for Lennard-Jones potential.

    Uses basis functions [r^{-12}, r^{-6}] to linearize the potential.
    """
    def __init__(self, r_min: float = 0.9, r_max: float = 3.0, **kwargs):
        """
        Initialize Lennard-Jones regression.

        Parameters
        ----------
        r_min : float
            Minimum r value (for scaling)
        r_max : float
            Maximum r value (for scaling)
        **kwargs
            Additional arguments
        """
        self.r_min = r_min
        self.r_max = r_max

        # Define basis functions for LJ potential
        def basis_r_inv_12(x):
            r = r_min + x * (r_max - r_min)
            return r**(-12)

        def basis_r_inv_6(x):
            r = r_min + x * (r_max - r_min)
            return r**(-6)

        basis_functions = [basis_r_inv_12, basis_r_inv_6]
        super().__init__(basis_functions=basis_functions, fit_intercept=True, **kwargs)


@register_model(family='linear', basis='shomate')
class ShomateRegression(OLSRegression):
    """
    Regression for Shomate equation.

    Uses basis functions [1, t, t², t³, 1/t²] where t = T/1000.
    """
    def __init__(self, T_min: float = 298.0, T_max: float = 1000.0, **kwargs):
        """
        Initialize Shomate regression.

        Parameters
        ----------
        T_min : float
            Minimum temperature (K)
        T_max : float
            Maximum temperature (K)
        **kwargs
            Additional arguments
        """
        self.T_min = T_min
        self.T_max = T_max

        # Define basis functions for Shomate equation
        def basis_1(x):
            return np.ones_like(x)

        def basis_t(x):
            T = T_min + x * (T_max - T_min)
            return T / 1000.0

        def basis_t2(x):
            T = T_min + x * (T_max - T_min)
            t = T / 1000.0
            return t**2

        def basis_t3(x):
            T = T_min + x * (T_max - T_min)
            t = T / 1000.0
            return t**3

        def basis_inv_t2(x):
            T = T_min + x * (T_max - T_min)
            t = T / 1000.0
            return 1.0 / (t**2)

        basis_functions = [basis_1, basis_t, basis_t2, basis_t3, basis_inv_t2]
        super().__init__(basis_functions=basis_functions, fit_intercept=False, **kwargs)
