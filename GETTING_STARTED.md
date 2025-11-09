# Getting Started with UQ Encyclopedia

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd uq-encyclopedia
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package
```bash
# Basic installation
pip install -e .

# With notebook support
pip install -e ".[notebooks]"

# With development tools
pip install -e ".[dev]"

# All extras
pip install -e ".[notebooks,dev,docs]"
```

## Project Structure

The project is organized as follows:

- `src/` - Source code
  - `datasets/` - Dataset generation modules
  - `models/` - Model implementations
  - `uq_methods/` - UQ method implementations
  - `metrics/` - Evaluation metrics
  - `visualization/` - Plotting utilities
  - `benchmark/` - Benchmark framework
  - `utils/` - Utility functions
- `config/` - Configuration files
- `data/` - Generated and processed datasets
- `notebooks/` - Jupyter notebooks for analysis
- `results/` - Experimental results
- `reports/` - Generated reports and dashboards
- `tests/` - Unit tests

## Quick Start

### Phase 1: Core Infrastructure (Completed)

The core infrastructure includes:

1. **Base Classes**: Abstract base classes for datasets, models, and UQ methods
2. **Seed Management**: Reproducible random seed handling
3. **Registry System**: Plugin architecture for easy extension
4. **I/O Utilities**: Saving/loading datasets and results
5. **Configuration**: Global configuration management

### Creating a Custom Dataset

To create a new dataset, inherit from the `Dataset` base class:

```python
from src.datasets.base import Dataset
from src.benchmark.registry import register_dataset
import numpy as np

@register_dataset()
class MyCustomDataset(Dataset):
    def _generate_clean(self, X: np.ndarray) -> np.ndarray:
        """Generate clean y values."""
        return np.sin(2 * np.pi * X)

    def get_function_form(self) -> str:
        """Return mathematical form."""
        return "y = sin(2πx)"

    def get_parameters(self) -> dict:
        """Return parameters."""
        return {"frequency": 1.0}

    def _get_family(self) -> str:
        """Return dataset family."""
        return "linear"  # or 'nonlinear', 'data_driven'
```

### Creating a Custom Model

To create a new model, inherit from the `Model` base class:

```python
from src.models.base import Model
from src.benchmark.registry import register_model
import numpy as np
from sklearn.linear_model import Ridge

@register_model()
class MyCustomModel(Model):
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.model_ = Ridge()
        self.model_.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model_.predict(X)
```

### Creating a Custom UQ Method

To create a new UQ method, inherit from the `UQMethod` base class:

```python
from src.uq_methods.base import UQMethod, UncertaintyResult
from src.benchmark.registry import register_uq_method
import numpy as np

@register_uq_method()
class MyCustomUQMethod(UQMethod):
    def compute_intervals(self, model, X_train, y_train, X_test, y_test=None):
        """Compute prediction intervals."""
        y_pred = model.predict(X_test)

        # Your UQ logic here
        # ...

        return UncertaintyResult(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            confidence_level=self.confidence_level,
        )
```

## Configuration

The global configuration is in `config/global_config.py`. Key settings:

- `MASTER_SEED`: Master random seed (default: 42)
- `DATASET_DEFAULTS`: Default dataset parameters
- `NOISE_MODELS`: Noise types to test
- `NOISE_LEVELS`: Noise levels to test (1%, 2%, 5%, 10%)
- `LEARNING_CURVE_SIZES`: Sample sizes for learning curves
- `CONFIDENCE_LEVEL`: Default confidence level (0.95)

## Seed Management

The project uses a centralized seed management system for reproducibility:

```python
from src.utils.seeds import get_experiment_seed, set_global_seed

# Get a deterministic seed for an experiment
seed = get_experiment_seed("my_experiment")

# Set global random seeds
set_global_seed(42)
```

## Registry System

The registry system allows you to register and discover datasets, models, and UQ methods:

```python
from src.benchmark.registry import (
    list_datasets, list_models, list_uq_methods,
    get_dataset, get_model, get_uq_method
)

# List available components
print("Datasets:", list_datasets())
print("Models:", list_models())
print("UQ Methods:", list_uq_methods())

# Get a registered component
DatasetClass = get_dataset("MyCustomDataset")
dataset = DatasetClass()
```

## Next Steps

### Phase 2: Linear Models (In Progress)

The next phase involves:
1. Implementing linear datasets (line, polynomial, Lennard-Jones, Shomate)
2. Implementing linear models (OLS)
3. Implementing UQ methods (hat matrix, Bayesian, conformal)
4. Creating metrics and visualization

Stay tuned for updates!

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

When adding new components:
1. Inherit from appropriate base classes
2. Register using the decorator system
3. Follow existing naming conventions
4. Add tests for new functionality
5. Update documentation

## License

[Add your license here]
