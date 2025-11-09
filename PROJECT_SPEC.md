# UQ Encyclopedia - Comprehensive Project Specification

## Project Overview
A comprehensive benchmark framework for evaluating uncertainty quantification (UQ) methods across linear, nonlinear, and data-driven models. The project will systematically compare UQ approaches in interpolation and extrapolation scenarios with varying data availability and noise characteristics.

---

## 1. Project Structure (Hybrid Approach)

```
uq-encyclopedia/
├── config/
│   ├── global_config.py          # Master seed, paths, global settings
│   ├── dataset_specs.json         # Dataset parameters
│   └── experiment_configs/        # Experiment-specific configs
├── src/
│   ├── datasets/                  # Dataset generation modules
│   │   ├── base.py               # Abstract base classes
│   │   ├── linear.py             # Linear models
│   │   ├── nonlinear.py          # Nonlinear models
│   │   └── generators.py         # Noise generation, train/test splits
│   ├── models/                    # Model implementations
│   │   ├── base.py               # Model interface
│   │   ├── linear_models.py
│   │   ├── nonlinear_models.py
│   │   └── data_driven_models.py
│   ├── uq_methods/                # UQ method implementations
│   │   ├── base.py               # UQ method interface
│   │   ├── conformal.py
│   │   ├── bayesian.py
│   │   └── bootstrap.py
│   ├── metrics/                   # Evaluation metrics
│   │   ├── coverage.py           # PICP, calibration
│   │   ├── efficiency.py         # Interval widths
│   │   └── accuracy.py           # RMSE, MAE, R²
│   ├── visualization/             # Standardized plotting
│   │   ├── plots.py              # Core plotting functions
│   │   ├── calibration.py        # Calibration plots
│   │   ├── learning_curves.py
│   │   └── comparison_plots.py
│   ├── benchmark/                 # Benchmark framework
│   │   ├── runner.py             # Experiment orchestration
│   │   ├── registry.py           # Plugin registration system
│   │   └── results.py            # Results aggregation
│   └── utils/
│       ├── seeds.py              # Seed management
│       ├── io.py                 # File I/O utilities
│       └── parallel.py           # Parallelization helpers
├── data/
│   ├── generated/                # Pre-generated datasets
│   └── processed/                # Processed results
├── notebooks/
│   ├── 01_linear_models/
│   ├── 02_nonlinear_models/
│   ├── 03_data_driven_models/
│   └── 04_analysis_and_comparison/
├── results/
│   ├── csv/                      # Summary tables
│   ├── json/                     # Metadata and configs
│   ├── hdf5/                     # Raw predictions and intervals
│   └── figures/
├── reports/
│   └── dashboard/                # HTML dashboard
└── tests/
    └── ...
```

---

## 2. Dataset Specifications

### Common Parameters
- Total samples: ~1500 per dataset
- Domain: Scaled to [0, 1]
- Gap: 20-30% of domain (interpolation region)
- Extrapolation: Both ends of domain (~10-15% on each side)
- Storage: Pre-generated and saved

### Noise Models
- **Homoskedastic**: σ = constant
- **Heteroskedastic**: σ = σ(x)
- **Noise levels**: 1%, 2%, 5%, 10%

### 2.1 Linear Models
1. **Line**: y = mx + b
2. **Polynomial**: y = a₀ + a₁x + a₂x² + a₃x³
3. **Lennard-Jones** (normalized): E(r) = 4[(1/r)¹² - (1/r)⁶]
4. **Shomate**: Cₚ/R = A + Bt + Ct² + Dt³ + E/t² (T: 298-1000K, scaled)

### 2.2 Nonlinear Models
1. **Arrhenius**: k = A exp(-Ea/RT)
2. **Morse**: V(r) = De[1 - exp(-a(r-re))]²
3. **Gaussian**: y = A exp(-(x-μ)²/(2σ²))

### 2.3 Data-Driven Models
1. **Sigmoid/Discontinuity**: Sharp transition or step function

---

## 3. Learning Curve Configuration

### Sample Sizes
Log-spaced up to 100 samples:
- Suggested: [10, 20, 30, 50, 75, 100]
- Each sample size: Train model, evaluate on test set

### Bootstrap Replicates (Adaptive)
- Linear models: 500 replicates
- Nonlinear models: 250 replicates
- GP/expensive models: 100 replicates

---

## 4. Models and UQ Methods

### 4.1 Linear Models
**Models**: Ordinary Least Squares (OLS)

**UQ Methods**:
- Hat matrix method (pycse.regress)
- Bayesian Linear Regression (sklearn)
- Conformal Prediction (MAPIE)

### 4.2 Nonlinear Models
**Models**: Nonlinear Least Squares (scipy.optimize)

**UQ Methods**:
- Delta method
- Conformal Prediction
- Bayesian (if feasible via MCMC)

### 4.3 Data-Driven Models
**Models**:
1. Gaussian Process Regression
   - Kernels: Linear (DotProduct), RBF, WhiteKernel (noise)
   - Combinations: RBF + WhiteKernel, etc.
2. DPOSE (Deep Polynomial Structure Embedding)
3. NNBR (Neural Network Basis Regression)
4. Random Forest Regressor

**UQ Methods**:
- sklearn return_std functionality
- Conformal Prediction (MAPIE)

---

## 5. Benchmark Framework Architecture

### Plugin System Components

```python
# Example interface
class Dataset(ABC):
    @abstractmethod
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class UQMethod(ABC):
    @abstractmethod
    def compute_intervals(self, model, X, y, X_test, alpha=0.05):
        pass
```

### Registry System
- Decorator-based registration
- Automatic discovery of new plugins
- Configuration-driven experiment specification

---

## 6. Evaluation Metrics

### Coverage & Calibration
- Prediction Interval Coverage Probability (PICP)
- Calibration plots (expected vs observed coverage)
- Coverage by region (interpolation vs extrapolation)

### Efficiency
- Mean interval width
- Median interval width
- Interval width vs coverage trade-off

### Accuracy
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² coefficient of determination

---

## 7. Computational Design

### Target Platform
Laptop execution with the following optimizations:
- Use multiprocessing for embarrassingly parallel tasks (bootstrap, learning curves)
- Optimize memory usage (clear models between experiments)
- Progress bars and checkpointing for long-running experiments
- Estimated total runtime: 6-24 hours for full suite

---

## 8. Reproducibility

### Seed Management
- Global master seed in `config/global_config.py`
- Deterministic seed derivation: `seed_i = hash(master_seed + experiment_id) % 2^32`
- Log all seeds used in metadata

### Version Control
- All code in git
- Generated datasets tracked with DVC or similar
- Environment captured (requirements.txt / environment.yml)

---

## 9. Outputs and Reporting

### Data Formats
- **CSV**: Summary statistics, metric tables (easy viewing)
- **JSON**: Experiment metadata, configurations
- **HDF5**: Raw predictions, uncertainty intervals, bootstrap samples

### Visualization
- Comprehensive plotting module with consistent styling
- Plot types: learning curves, calibration curves, prediction intervals, comparison plots
- Export formats: PNG (high-res), SVG (vector)

### Dashboard
- HTML interactive dashboard (plotly/bokeh)
- Supporting Jupyter notebooks for detailed analysis
- Auto-generated from experiment results

---

## 10. Development Workflow

### Phase 1: Core Infrastructure
- Base classes and interfaces
- Dataset generators
- Seed management

### Phase 2: Linear Models
- Implement all linear datasets
- Three UQ methods
- Metrics and visualization

### Phase 3: Nonlinear Models
- Nonlinear datasets and fitting
- Delta method, conformal prediction

### Phase 4: Data-Driven Models
- GP, Random Forest, NNBR, DPOSE
- Advanced UQ methods

### Phase 5: Benchmarking & Analysis
- Run full suite of experiments
- Generate comprehensive comparison
- Create dashboard
