"""Utilities for dataset generation, noise, and splitting."""

import numpy as np
from typing import Tuple, Callable, Optional


def generate_noise(
    y: np.ndarray,
    noise_model: str,
    noise_level: float,
    X: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate noise for data.

    Parameters
    ----------
    y : np.ndarray
        Clean target values
    noise_model : str
        'homoskedastic' or 'heteroskedastic'
    noise_level : float
        Noise level as fraction (e.g., 0.02 for 2%)
    X : np.ndarray, optional
        Input features (needed for some heteroskedastic models)
    seed : int, optional
        Random seed

    Returns
    -------
    noise : np.ndarray
        Noise array
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_model == 'homoskedastic':
        # Constant noise: σ = noise_level * std(y)
        sigma = noise_level * np.std(y)
        noise = np.random.normal(0, sigma, size=y.shape)

    elif noise_model == 'heteroskedastic':
        # X-dependent noise: σ(x) = noise_level * |y(x)|
        # Use small constant to avoid zero noise
        sigma = noise_level * (np.abs(y) + 1e-6)
        noise = np.random.normal(0, 1, size=y.shape) * sigma

    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    return noise


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    gap_fraction: float = 0.25,
    extrapolation_fraction: Tuple[float, float] = (0.125, 0.125),
    n_test: int = 500,
    domain: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Create train/test/gap/extrapolation splits from data.

    Parameters
    ----------
    X : np.ndarray
        Full input array
    y : np.ndarray
        Full target array
    gap_fraction : float
        Fraction of domain to reserve as gap
    extrapolation_fraction : tuple
        (low, high) fractions for extrapolation
    n_test : int
        Number of test samples
    domain : tuple, optional
        (min, max) of domain. If None, inferred from X
    seed : int, optional
        Random seed

    Returns
    -------
    splits : dict
        Dictionary containing all data splits
    """
    if seed is not None:
        np.random.seed(seed)

    if domain is None:
        x_min, x_max = X.min(), X.max()
    else:
        x_min, x_max = domain

    domain_width = x_max - x_min

    # Define region boundaries
    extrap_low_fraction, extrap_high_fraction = extrapolation_fraction
    extrap_low_end = x_min + extrap_low_fraction * domain_width
    extrap_high_start = x_max - extrap_high_fraction * domain_width

    # Gap region (centered)
    gap_width = gap_fraction * domain_width
    gap_center = (extrap_low_end + extrap_high_start) / 2
    gap_start = gap_center - gap_width / 2
    gap_end = gap_center + gap_width / 2

    # Flatten X for comparison
    X_flat = X.flatten()

    # Create masks for each region
    mask_extrap_low = X_flat < extrap_low_end
    mask_extrap_high = X_flat > extrap_high_start
    mask_gap = (X_flat >= gap_start) & (X_flat <= gap_end)
    mask_train = (
        (X_flat >= extrap_low_end) &
        (X_flat <= extrap_high_start) &
        ~mask_gap
    )

    # Extract each region
    splits = {
        'X_train': X[mask_train],
        'y_train': y[mask_train],
        'X_gap': X[mask_gap],
        'y_gap': y[mask_gap],
        'X_extrap_low': X[mask_extrap_low],
        'y_extrap_low': y[mask_extrap_low],
        'X_extrap_high': X[mask_extrap_high],
        'y_extrap_high': y[mask_extrap_high],
    }

    # Generate test set (uniform over domain)
    X_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)
    splits['X_test'] = X_test

    return splits


def apply_function_to_grid(
    func: Callable,
    domain: Tuple[float, float],
    n_points: int,
    add_noise: bool = False,
    noise_model: str = 'homoskedastic',
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to a regular grid.

    Parameters
    ----------
    func : callable
        Function to apply, signature: func(x) -> y
    domain : tuple
        (min, max) of domain
    n_points : int
        Number of points in grid
    add_noise : bool
        Whether to add noise
    noise_model : str
        Noise model if add_noise=True
    noise_level : float
        Noise level if add_noise=True
    seed : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Input grid, shape (n_points, 1)
    y : np.ndarray
        Function values (possibly with noise), shape (n_points,)
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = domain
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    y = func(X.flatten())

    if add_noise:
        noise = generate_noise(y, noise_model, noise_level, X)
        y = y + noise

    return X, y


def sample_function(
    func: Callable,
    domain: Tuple[float, float],
    n_points: int,
    add_noise: bool = False,
    noise_model: str = 'homoskedastic',
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a function at random points.

    Parameters
    ----------
    func : callable
        Function to sample, signature: func(x) -> y
    domain : tuple
        (min, max) of domain
    n_points : int
        Number of samples
    add_noise : bool
        Whether to add noise
    noise_model : str
        Noise model if add_noise=True
    noise_level : float
        Noise level if add_noise=True
    seed : int, optional
        Random seed

    Returns
    -------
    X : np.ndarray
        Random input samples, shape (n_points, 1)
    y : np.ndarray
        Function values (possibly with noise), shape (n_points,)
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = domain
    X = np.random.uniform(x_min, x_max, n_points).reshape(-1, 1)
    y = func(X.flatten())

    if add_noise:
        noise = generate_noise(y, noise_model, noise_level, X)
        y = y + noise

    return X, y
