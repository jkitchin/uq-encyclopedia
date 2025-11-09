"""Linear datasets for UQ benchmarking.

These datasets are 'linear' in the sense that they are linear in their parameters,
meaning they can be fit using linear regression after appropriate basis transformation.
"""

import numpy as np
from typing import Dict, Any
from src.datasets.base import Dataset
from src.benchmark.registry import register_dataset


@register_dataset(family='linear')
class LineDataset(Dataset):
    """
    Simple linear dataset: y = mx + b

    Parameters are chosen to give y values in [0, 1] range when x is in [0, 1].
    """

    def __init__(self, slope: float = 0.8, intercept: float = 0.1, **kwargs):
        """
        Initialize line dataset.

        Parameters
        ----------
        slope : float
            Slope of the line
        intercept : float
            Y-intercept
        **kwargs
            Additional arguments passed to Dataset
        """
        super().__init__(**kwargs)
        self.slope = slope
        self.intercept = intercept

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean y values."""
        return self.slope * X + self.intercept

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"y = {self.slope:.2f}x + {self.intercept:.2f}"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {
            'slope': self.slope,
            'intercept': self.intercept,
        }

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'linear'


@register_dataset(family='linear')
class PolynomialDataset(Dataset):
    """
    Polynomial dataset: y = a₀ + a₁x + a₂x² + a₃x³

    Coefficients are chosen to give interesting behavior in [0, 1] range.
    """

    def __init__(
        self,
        coefficients: np.ndarray = None,
        degree: int = 3,
        **kwargs
    ):
        """
        Initialize polynomial dataset.

        Parameters
        ----------
        coefficients : np.ndarray, optional
            Polynomial coefficients [a₀, a₁, a₂, ..., aₙ]
            If None, uses default coefficients for interesting behavior
        degree : int
            Polynomial degree (only used if coefficients is None)
        **kwargs
            Additional arguments passed to Dataset
        """
        super().__init__(**kwargs)
        self.degree = degree

        if coefficients is None:
            # Default coefficients for degree 3: creates a curve with inflection
            # Normalized to give values roughly in [0, 1] for x in [0, 1]
            if degree == 3:
                self.coefficients = np.array([0.1, -0.5, 1.5, -0.3])
            elif degree == 2:
                self.coefficients = np.array([0.2, 0.3, 0.5])
            else:
                # Random coefficients normalized
                np.random.seed(42)
                self.coefficients = np.random.randn(degree + 1)
                # Normalize to [0, 1] range
                x_test = np.linspace(0, 1, 100)
                y_test = np.polyval(self.coefficients[::-1], x_test)
                y_min, y_max = y_test.min(), y_test.max()
                self.coefficients = (self.coefficients - y_min) / (y_max - y_min)
        else:
            self.coefficients = np.array(coefficients)
            self.degree = len(coefficients) - 1

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean y values."""
        # polyval expects coefficients in reverse order (highest degree first)
        return np.polyval(self.coefficients[::-1], X)

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        terms = []
        for i, coef in enumerate(self.coefficients):
            if i == 0:
                terms.append(f"{coef:.3f}")
            elif i == 1:
                terms.append(f"{coef:.3f}x")
            else:
                terms.append(f"{coef:.3f}x^{i}")
        return "y = " + " + ".join(terms)

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {
            'coefficients': self.coefficients.tolist(),
            'degree': self.degree,
        }

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'linear'


@register_dataset(family='linear')
class LennardJonesDataset(Dataset):
    """
    Normalized Lennard-Jones potential: E(r) = 4ε[(σ/r)¹² - (σ/r)⁶]

    This is linearizable via transformation: E = 4ε[r⁻¹² - r⁻⁶]
    We use normalized form with ε=1, σ=1, scaled to [0, 1] domain.

    The potential has a minimum at r = 2^(1/6) ≈ 1.122.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        r_min: float = 0.9,
        r_max: float = 3.0,
        **kwargs
    ):
        """
        Initialize Lennard-Jones dataset.

        Parameters
        ----------
        epsilon : float
            Depth of potential well
        sigma : float
            Distance at which potential is zero
        r_min : float
            Minimum r value (in original units)
        r_max : float
            Maximum r value (in original units)
        **kwargs
            Additional arguments passed to Dataset
        """
        # Override domain to be [0, 1] for consistency
        kwargs['domain'] = (0.0, 1.0)
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.sigma = sigma
        self.r_min = r_min
        self.r_max = r_max

    def _scale_x_to_r(self, X: np.ndarray) -> np.ndarray:
        """Convert scaled X in [0, 1] to r values."""
        return self.r_min + X * (self.r_max - self.r_min)

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean y values."""
        # Convert X to r
        r = self._scale_x_to_r(X)

        # Lennard-Jones potential
        sr = self.sigma / r
        energy = 4 * self.epsilon * (sr**12 - sr**6)

        # Normalize to approximately [0, 1] range
        # Shift so minimum is at 0
        e_min = -self.epsilon  # Minimum of LJ potential
        energy_shifted = energy - e_min

        # Scale to [0, 1] based on range
        r_test = np.linspace(self.r_min, self.r_max, 1000)
        sr_test = self.sigma / r_test
        e_test = 4 * self.epsilon * (sr_test**12 - sr_test**6) - e_min
        e_max = np.max(e_test)

        return energy_shifted / e_max

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        return f"E(r) = 4ε[(σ/r)¹² - (σ/r)⁶], ε={self.epsilon:.2f}, σ={self.sigma:.2f}"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {
            'epsilon': self.epsilon,
            'sigma': self.sigma,
            'r_min': self.r_min,
            'r_max': self.r_max,
        }

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'linear'


@register_dataset(family='linear')
class ShomateDataset(Dataset):
    """
    Shomate equation for heat capacity:
    Cp/R = A + B*t + C*t² + D*t³ + E/t²

    where t = T/1000 (temperature in K divided by 1000)

    This is linear in parameters [A, B, C, D, E] with basis functions
    [1, t, t², t³, 1/t²].

    We use coefficients for a realistic species (e.g., O₂) and normalize.
    """

    def __init__(
        self,
        coefficients: np.ndarray = None,
        T_min: float = 298.0,
        T_max: float = 1000.0,
        **kwargs
    ):
        """
        Initialize Shomate dataset.

        Parameters
        ----------
        coefficients : np.ndarray, optional
            Shomate coefficients [A, B, C, D, E]
            If None, uses coefficients for O₂
        T_min : float
            Minimum temperature (K)
        T_max : float
            Maximum temperature (K)
        **kwargs
            Additional arguments passed to Dataset
        """
        # Override domain to be [0, 1] for consistency
        kwargs['domain'] = (0.0, 1.0)
        super().__init__(**kwargs)

        self.T_min = T_min
        self.T_max = T_max

        if coefficients is None:
            # Shomate coefficients for O₂ (298-1000K)
            # Source: NIST-JANAF thermochemical tables
            self.coefficients = np.array([
                29.659,      # A
                6.137261,    # B
                -1.186521,   # C
                0.095780,    # D
                -0.219663,   # E
            ])
        else:
            self.coefficients = np.array(coefficients)

    def _scale_x_to_T(self, X: np.ndarray) -> np.ndarray:
        """Convert scaled X in [0, 1] to temperature."""
        return self.T_min + X * (self.T_max - self.T_min)

    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean y values."""
        # Convert X to temperature
        T = self._scale_x_to_T(X)
        t = T / 1000.0  # Shomate uses T/1000

        # Compute Cp/R using Shomate equation
        A, B, C, D, E = self.coefficients
        Cp = A + B*t + C*t**2 + D*t**3 + E/(t**2)

        # Normalize to [0, 1] range
        T_test = np.linspace(self.T_min, self.T_max, 1000)
        t_test = T_test / 1000.0
        Cp_test = (A + B*t_test + C*t_test**2 + D*t_test**3 + E/(t_test**2))
        Cp_min, Cp_max = Cp_test.min(), Cp_test.max()

        return (Cp - Cp_min) / (Cp_max - Cp_min)

    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        A, B, C, D, E = self.coefficients
        return f"Cp/R = {A:.2f} + {B:.2f}t + {C:.2f}t² + {D:.2f}t³ + {E:.2f}/t²"

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        return {
            'coefficients': self.coefficients.tolist(),
            'T_min': self.T_min,
            'T_max': self.T_max,
        }

    def _get_family(self) -> str:
        """Return dataset family."""
        return 'linear'
