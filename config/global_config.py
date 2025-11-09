"""Global configuration for the UQ Encyclopedia project."""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DATA_DIR = DATA_DIR / "generated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV_DIR = RESULTS_DIR / "csv"
RESULTS_JSON_DIR = RESULTS_DIR / "json"
RESULTS_HDF5_DIR = RESULTS_DIR / "hdf5"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DIR = REPORTS_DIR / "dashboard"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# Random Seed Configuration
# ============================================================================

# Master seed for all experiments
MASTER_SEED = 42

# ============================================================================
# Dataset Configuration
# ============================================================================

# Default dataset parameters
DATASET_DEFAULTS = {
    'n_samples': 1500,
    'domain': (0.0, 1.0),
    'gap_fraction': 0.25,
    'extrapolation_fraction': (0.125, 0.125),
    'n_test_samples': 500,
}

# Noise configurations to test
NOISE_MODELS = ['homoskedastic', 'heteroskedastic']
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%

# ============================================================================
# Learning Curve Configuration
# ============================================================================

# Sample sizes for learning curves (log-spaced up to 100)
LEARNING_CURVE_SIZES = [10, 20, 30, 50, 75, 100]

# Bootstrap replicates (adaptive based on model type)
BOOTSTRAP_REPLICATES = {
    'linear': 500,
    'nonlinear': 250,
    'gp': 100,
    'expensive': 100,
}

# ============================================================================
# UQ Configuration
# ============================================================================

# Default confidence level for prediction intervals
CONFIDENCE_LEVEL = 0.95

# Alternative confidence levels for sensitivity analysis
CONFIDENCE_LEVELS = [0.68, 0.90, 0.95, 0.99]  # 1σ, 90%, 95%, 99%

# ============================================================================
# Computational Configuration
# ============================================================================

# Number of parallel workers for multiprocessing
# Set to -1 to use all available cores, or None to disable parallelization
N_WORKERS = None  # Will use single-threaded for laptop execution

# Maximum number of workers if N_WORKERS is -1
MAX_WORKERS = os.cpu_count() or 4

# Use actual number of workers
if N_WORKERS == -1:
    N_WORKERS = MAX_WORKERS

# Progress bar configuration
SHOW_PROGRESS = True

# ============================================================================
# Visualization Configuration
# ============================================================================

# Default figure size
FIGURE_SIZE = (10, 6)

# Figure DPI
FIGURE_DPI = 150

# Style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # matplotlib style

# Color palette
COLOR_PALETTE = 'tab10'

# Export formats
EXPORT_FORMATS = ['png', 'svg']

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Whether to save intermediate results
SAVE_INTERMEDIATE_RESULTS = True

# Checkpoint frequency (save after every N experiments)
CHECKPOINT_FREQUENCY = 10

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        GENERATED_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        RESULTS_CSV_DIR,
        RESULTS_JSON_DIR,
        RESULTS_HDF5_DIR,
        RESULTS_FIGURES_DIR,
        REPORTS_DIR,
        DASHBOARD_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_dataset_path(dataset_name: str, noise_model: str, noise_level: float) -> Path:
    """
    Get path for a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    noise_model : str
        Noise model ('homoskedastic' or 'heteroskedastic')
    noise_level : float
        Noise level as fraction

    Returns
    -------
    path : Path
        Path to dataset file
    """
    filename = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}.npz"
    return GENERATED_DATA_DIR / filename


def get_results_path(experiment_id: str, format: str = 'csv') -> Path:
    """
    Get path for experiment results.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier
    format : str
        File format ('csv', 'json', or 'hdf5')

    Returns
    -------
    path : Path
        Path to results file
    """
    if format == 'csv':
        return RESULTS_CSV_DIR / f"{experiment_id}.csv"
    elif format == 'json':
        return RESULTS_JSON_DIR / f"{experiment_id}.json"
    elif format == 'hdf5':
        return RESULTS_HDF5_DIR / f"{experiment_id}.h5"
    else:
        raise ValueError(f"Unknown format: {format}")


# Initialize directories when module is imported
ensure_directories()
