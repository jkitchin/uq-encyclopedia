# Reproducing Linear Models Results

This guide shows how to reproduce all results presented in the interactive dashboard.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete benchmark (all 64 experiments)
python run_linear_benchmark.py

# 3. Generate visualizations
python generate_linear_visualizations.py
python generate_individual_fits.py

# 4. Create interactive dashboard
python create_interactive_dashboard.py
python add_guide_to_dashboard.py
python add_dataset_plots_to_guide.py
python add_implementation_details.py
python add_reproducibility_section.py

# 5. Open dashboard
open reports/dashboard/linear_models_interactive.html
```

## Step-by-Step Instructions

### 1. Run Complete Benchmark

Runs all 64 experiments (4 datasets × 2 noise models × 4 noise levels × 2 UQ methods):

```bash
python run_linear_benchmark.py
```

**Runtime**: ~2-3 minutes
**Outputs**:
- `results/csv/linear_models_comprehensive.csv` - Detailed results for all 64 experiments
- `results/csv/linear_models_summary.csv` - Summary statistics by UQ method

### 2. Generate Static Visualizations

Creates 11 publication-quality figures:

```bash
python generate_linear_visualizations.py
```

**Outputs** in `results/figures/`:
- `coverage_by_dataset.png` - Coverage comparison across datasets
- `coverage_vs_noise_level.png` - How coverage changes with noise
- `coverage_by_noise_model.png` - Homoskedastic vs heteroskedastic
- `heatmap_coverage_hatmatrix.png` - Coverage heatmap for Hat Matrix
- `heatmap_coverage_bayesian.png` - Coverage heatmap for Bayesian
- `regional_coverage.png` - Interpolation vs extrapolation performance
- `summary_grid.png` - 4-panel comprehensive summary
- And 4 more...

### 3. Generate Individual Fit Plots

Creates 64 individual plots showing model fits with uncertainty bands:

```bash
python generate_individual_fits.py
```

**Outputs**: `results/figures/individual_fits/` (64 PNG files)
Each shows: training data, true function, predictions, 95% intervals, gap/extrapolation regions

### 4. Create Interactive Dashboard

```bash
# Base dashboard with interactive Plotly charts
python create_interactive_dashboard.py

# Add educational guide (5 tabs with explanations)
python add_guide_to_dashboard.py

# Add dataset visualizations to Datasets tab
python add_dataset_plots_to_guide.py

# Add implementation details (libraries, formulas, code links)
python add_implementation_details.py

# Add reproducibility instructions
python add_reproducibility_section.py
```

**Output**: `reports/dashboard/linear_models_interactive.html` (self-contained, works offline)

### 5. Compute Calibration-Sharpness Analysis

Analyzes the coverage-efficiency trade-off:

```bash
python compute_calibration_sharpness.py
```

**Key finding**: Bayesian achieves 100% coverage but with intervals 37x wider than Hat Matrix!

**Output**: `results/csv/calibration_sharpness_scores.csv`

### 6. Hyperparameter Tuning (Optional)

Grid search for optimal Bayesian hyperparameters:

```bash
# Run grid search (81 combinations, ~5-10 min)
python tune_bayesian_hyperparameters.py

# Analyze results
python analyze_tuning_results.py
```

**Finding**: All parameter combinations achieve 100% coverage with similar widths.
**Optimal params**: α₁=1e-7, α₂=1e-5, λ₁=1e-5, λ₂=1e-7 (marginal improvement)

## Single Experiment Example

To run a single experiment (quick test):

```bash
python test_linear_models.py
```

This runs one configuration and creates `results/figures/linear_models_comparison.png`

## Source Code Reference

| Component | File Path | Description |
|-----------|-----------|-------------|
| **Datasets** | `src/datasets/linear.py` | Line, Polynomial, Lennard-Jones, Shomate |
| **Models** | `src/models/linear_models.py` | OLS regression with polynomial/custom basis |
| **UQ Methods** | `src/uq_methods/linear_uq.py` | Hat Matrix, Bayesian, Conformal |
| **Metrics** | `src/metrics/` | Coverage, efficiency, accuracy, calibration |
| **Seeding** | `src/utils/seeds.py` | Deterministic experiment seeds |
| **Config** | `config/global_config.py` | Global parameters and paths |

## Key Implementation Details

### Hat Matrix Method
- **File**: `src/uq_methods/linear_uq.py` (lines 11-127)
- **Formula**: ŷ ± t(α/2, df) × σ × √(1 + h_i)
- **Library**: `pycse.predict` (https://github.com/jkitchin/pycse)
- **Reference**: Classical OLS prediction intervals with t-distribution

### Bayesian Linear Regression
- **File**: `src/uq_methods/linear_uq.py` (lines 115-229)
- **Library**: `sklearn.linear_model.BayesianRidge`
- **Priors**: Gamma priors on noise and weight precision
- **Key fix**: Now uses design matrix transformation (bug fixed!)

### Design Matrix Transformation
- **File**: `src/models/linear_models.py` (lines 52-84)
- **Critical**: Transforms raw features to polynomial/basis functions
- **Example**: For degree-3 polynomial: X → [1, x, x², x³]

### Coverage Metric
- **File**: `src/metrics/coverage.py`
- **Formula**: PICP = (# intervals containing true value) / (# total points)

### Calibration-Sharpness Score (NEW!)
- **File**: `src/metrics/efficiency.py` (lines 259-328)
- **Formula**: 10 × |coverage - 0.95| + normalized_width
- **Purpose**: Quantifies coverage-efficiency trade-off

## Expected Results

After running the complete benchmark, you should see:

**Hat Matrix**:
- Coverage: 88.2% ± 12.3%
- Normalized width: 0.074 ± 0.082
- Cal-Sharp score: 0.80 (better)

**Bayesian**:
- Coverage: 100.0% ± 0.0% (perfect calibration!)
- Normalized width: 2.74 ± 1.22 (37x wider than Hat Matrix)
- Cal-Sharp score: 3.24 (worse due to overly wide intervals)

**Conclusion**: Hat Matrix provides better balance between coverage and sharpness.

## Requirements

```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
plotly>=5.14.0
tqdm>=4.65.0
h5py>=3.8.0
pycse>=2.1.0  # For Hat Matrix UQ (regress, predict)
```

Install with:
```bash
pip install -r requirements.txt
```

## Reproducibility

All experiments use deterministic seeding:
- Master seed: 42
- Experiment seeds: SHA256(master_seed + experiment_id) % 2³²
- No randomness in results - running twice gives identical outputs

## Questions?

See the interactive dashboard for:
- **What is Coverage?** tab: Definitions and examples
- **Study Design** tab: Factorial design and data splitting
- **UQ Methods** tab: Implementation details with formulas
- **How to Interpret** tab: Reading charts and understanding metrics

Or check the source code files listed above!
