"""Base classes and interfaces for dataset generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class DataSplit:
    """Container for dataset splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_gap: Optional[np.ndarray] = None  # Interpolation region
    y_gap: Optional[np.ndarray] = None  # True values in gap (for evaluation)
    X_extrap_low: Optional[np.ndarray] = None  # Lower extrapolation region
    y_extrap_low: Optional[np.ndarray] = None
    X_extrap_high: Optional[np.ndarray] = None  # Upper extrapolation region
    y_extrap_high: Optional[np.ndarray] = None


@dataclass
class DatasetMetadata:
    """Metadata describing a dataset."""
    name: str
    family: str  # 'linear', 'nonlinear', 'data_driven'
    function_form: str  # Mathematical form or description
    n_samples: int
    domain: Tuple[float, float]
    gap_fraction: float
    extrapolation_fraction: Tuple[float, float]  # (low, high)
    noise_model: str  # 'homoskedastic' or 'heteroskedastic'
    noise_level: float  # Percentage (1, 2, 5, 10)
    seed: int
    parameters: Dict[str, Any]  # Function-specific parameters


class Dataset(ABC):
    """
    Abstract base class for all datasets.

    Datasets are responsible for:
    - Generating clean (noiseless) data
    - Adding noise (homo/heteroskedastic)
    - Creating train/test/gap/extrapolation splits
    - Providing metadata
    """

    def __init__(
        self,
        n_samples: int = 1500,
        domain: Tuple[float, float] = (0.0, 1.0),
        gap_fraction: float = 0.25,
        extrapolation_fraction: Tuple[float, float] = (0.125, 0.125),
        noise_model: str = 'homoskedastic',
        noise_level: float = 0.02,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate
        domain : tuple
            (min, max) of the domain
        gap_fraction : float
            Fraction of domain to reserve as interpolation gap
        extrapolation_fraction : tuple
            (low, high) fractions for extrapolation regions
        noise_model : str
            'homoskedastic' or 'heteroskedastic'
        noise_level : float
            Noise level as fraction (e.g., 0.02 for 2%)
        seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.domain = domain
        self.gap_fraction = gap_fraction
        self.extrapolation_fraction = extrapolation_fraction
        self.noise_model = noise_model
        self.noise_level = noise_level
        self.seed = seed

        # Set random seed
        np.random.seed(seed)

    @abstractmethod
    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """
        Generate clean (noiseless) y values for given X.

        Parameters
        ----------
        X : np.ndarray
            Input values

        Returns
        -------
        y : np.ndarray
            Clean output values
        """
        pass

    @abstractmethod
    def get_function_form(self) -> str:
        """Return string describing the mathematical form."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of function-specific parameters."""
        pass

    def _add_noise(self, X: np.ndarray, y_clean: np.ndarray) -> np.ndarray:
        """
        Add noise to clean data.

        Parameters
        ----------
        X : np.ndarray
            Input values
        y_clean : np.ndarray
            Clean output values

        Returns
        -------
        y_noisy : np.ndarray
            Noisy output values
        """
        if self.noise_model == 'homoskedastic':
            # Constant noise: σ = noise_level * std(y_clean)
            sigma = self.noise_level * np.std(y_clean)
            noise = np.random.normal(0, sigma, size=y_clean.shape)
        elif self.noise_model == 'heteroskedastic':
            # X-dependent noise: σ(x) = noise_level * |y_clean(x)|
            sigma = self.noise_level * np.abs(y_clean)
            noise = np.random.normal(0, 1, size=y_clean.shape) * sigma
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

        return y_clean + noise

    def generate(self) -> DataSplit:
        """
        Generate complete dataset with train/test/gap/extrapolation splits.

        Returns
        -------
        data_split : DataSplit
            Container with all data splits
        """
        # Define regions
        x_min, x_max = self.domain
        domain_width = x_max - x_min

        # Extrapolation regions
        extrap_low_fraction, extrap_high_fraction = self.extrapolation_fraction
        extrap_low_end = x_min + extrap_low_fraction * domain_width
        extrap_high_start = x_max - extrap_high_fraction * domain_width

        # Gap region (centered in main domain)
        gap_width = self.gap_fraction * domain_width
        gap_center = (extrap_low_end + extrap_high_start) / 2
        gap_start = gap_center - gap_width / 2
        gap_end = gap_center + gap_width / 2

        # Generate samples
        n_extrap_low = int(self.n_samples * extrap_low_fraction)
        n_extrap_high = int(self.n_samples * extrap_high_fraction)
        n_gap = int(self.n_samples * self.gap_fraction)
        n_train = self.n_samples - n_extrap_low - n_extrap_high - n_gap

        # Create X values for each region
        X_extrap_low = np.random.uniform(x_min, extrap_low_end, n_extrap_low)
        X_train_low = np.random.uniform(extrap_low_end, gap_start, n_train // 2)
        X_gap = np.random.uniform(gap_start, gap_end, n_gap)
        X_train_high = np.random.uniform(gap_end, extrap_high_start, n_train - n_train // 2)
        X_extrap_high = np.random.uniform(extrap_high_start, x_max, n_extrap_high)

        # Combine training data
        X_train = np.concatenate([X_train_low, X_train_high])
        X_train = np.sort(X_train)

        # Test set: uniform over entire domain
        n_test = max(500, self.n_samples // 3)
        X_test = np.linspace(x_min, x_max, n_test)

        # Generate clean y values
        y_train_clean = self._generate_clean(X_train)
        y_test_clean = self._generate_clean(X_test)
        y_gap_clean = self._generate_clean(X_gap)
        y_extrap_low_clean = self._generate_clean(X_extrap_low)
        y_extrap_high_clean = self._generate_clean(X_extrap_high)

        # Add noise
        y_train = self._add_noise(X_train, y_train_clean)
        y_test = self._add_noise(X_test, y_test_clean)
        y_gap = y_gap_clean  # Keep gap clean for evaluation
        y_extrap_low = y_extrap_low_clean  # Keep extrapolation clean for evaluation
        y_extrap_high = y_extrap_high_clean

        return DataSplit(
            X_train=X_train.reshape(-1, 1),
            y_train=y_train,
            X_test=X_test.reshape(-1, 1),
            y_test=y_test,
            X_gap=X_gap.reshape(-1, 1),
            y_gap=y_gap,
            X_extrap_low=X_extrap_low.reshape(-1, 1),
            y_extrap_low=y_extrap_low,
            X_extrap_high=X_extrap_high.reshape(-1, 1),
            y_extrap_high=y_extrap_high,
        )

    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata describing this dataset.

        Returns
        -------
        metadata : DatasetMetadata
            Dataset metadata
        """
        return DatasetMetadata(
            name=self.__class__.__name__,
            family=self._get_family(),
            function_form=self.get_function_form(),
            n_samples=self.n_samples,
            domain=self.domain,
            gap_fraction=self.gap_fraction,
            extrapolation_fraction=self.extrapolation_fraction,
            noise_model=self.noise_model,
            noise_level=self.noise_level,
            seed=self.seed,
            parameters=self.get_parameters(),
        )

    @abstractmethod
    def _get_family(self) -> str:
        """Return dataset family: 'linear', 'nonlinear', or 'data_driven'."""
        pass
