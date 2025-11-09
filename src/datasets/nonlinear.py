"""Nonlinear datasets for UQ benchmarking."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from src.datasets.base import Dataset


class ExponentialDecayDataset(Dataset):
    """Exponential decay: y = a * exp(-b * x) + c"""

    def __init__(
        self,
        n_samples: int = 100,
        noise_model: str = 'homoskedastic',
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        a: float = 2.0,
        b: float = 3.0,
        c: float = 0.5
    ):
        super().__init__(
            n_samples=n_samples,
            domain=(0.0, 1.0),
            gap_fraction=0.25,
            extrapolation_fraction=(0.125, 0.125),
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed if seed is not None else 42
        )
        self.a = a
        self.b = b
        self.c = c

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean exponential decay data."""
        return self.a * np.exp(-self.b * X) + self.c


    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"y = {self.a} * exp(-{self.b} * x) + {self.c}"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {'a': self.a, 'b': self.b, 'c': self.c}

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'nonlinear'


class LogisticGrowthDataset(Dataset):
    """Logistic growth: y = L / (1 + exp(-k*(x - x0)))"""

    def __init__(
        self,
        n_samples: int = 100,
        noise_model: str = 'homoskedastic',
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        L: float = 1.0,
        k: float = 10.0,
        x0: float = 0.5
    ):
        super().__init__(
            n_samples=n_samples,
            domain=(0.0, 1.0),
            gap_fraction=0.25,
            extrapolation_fraction=(0.125, 0.125),
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed if seed is not None else 42
        )
        self.L = L
        self.k = k
        self.x0 = x0

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean logistic growth data."""
        return self.L / (1 + np.exp(-self.k * (X - self.x0)))

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"y = {self.L} / (1 + exp(-{self.k} * (x - {self.x0})))"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {'L': self.L, 'k': self.k, 'x0': self.x0}

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'nonlinear'


class MichaelisMentenDataset(Dataset):
    """Michaelis-Menten kinetics: y = (Vmax * x) / (Km + x)"""

    def __init__(
        self,
        n_samples: int = 100,
        noise_model: str = 'homoskedastic',
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        Vmax: float = 1.0,
        Km: float = 0.3
    ):
        super().__init__(
            n_samples=n_samples,
            domain=(0.0, 1.0),
            gap_fraction=0.25,
            extrapolation_fraction=(0.125, 0.125),
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed if seed is not None else 42
        )
        self.Vmax = Vmax
        self.Km = Km

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean Michaelis-Menten data."""
        return (self.Vmax * X) / (self.Km + X)

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"y = ({self.Vmax} * x) / ({self.Km} + x)"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {'Vmax': self.Vmax, 'Km': self.Km}

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'nonlinear'


class GaussianDataset(Dataset):
    """Gaussian/Bell curve: y = a * exp(-((x - mu)^2) / (2 * sigma^2))"""

    def __init__(
        self,
        n_samples: int = 100,
        noise_model: str = 'homoskedastic',
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        a: float = 1.0,
        mu: float = 0.5,
        sigma: float = 0.15
    ):
        super().__init__(
            n_samples=n_samples,
            domain=(0.0, 1.0),
            gap_fraction=0.25,
            extrapolation_fraction=(0.125, 0.125),
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed if seed is not None else 42
        )
        self.a = a
        self.mu = mu
        self.sigma = sigma

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean Gaussian data."""
        return self.a * np.exp(-((X - self.mu)**2) / (2 * self.sigma**2))

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"y = {self.a} * exp(-((x - {self.mu})^2) / (2 * {self.sigma}^2))"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {'a': self.a, 'mu': self.mu, 'sigma': self.sigma}

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'nonlinear'
