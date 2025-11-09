# pycse Integration for Hat Matrix UQ

## Summary

The Hat Matrix UQ method now uses `pycse.regress` and `pycse.predict` instead of a custom implementation.

## What Changed

### Before
```python
# Custom implementation using numpy/scipy
h = model.get_hat_matrix_diagonal(X_test)
sigma = np.sqrt(model.sigma_squared_)
t_stat = stats.t.ppf(1 - alpha/2, df)
std_error = sigma * np.sqrt(1 + h)
margin = t_stat * std_error
```

### After
```python
# Using pycse library
from pycse import regress, predict

# Fit parameters
pars, pars_int, se = regress(X_train_design, y_train, alpha=0.05)

# Get predictions with intervals
y_pred, y_int, pred_se = predict(X_train_design, y_train, pars, X_test_design, alpha=0.05)
```

## Why pycse?

1. **Trusted implementation** - Well-tested library from Prof. John Kitchin
2. **Consistent results** - Matches our implementation (max difference < 0.0004)
3. **Cleaner code** - Less custom matrix algebra
4. **Better maintained** - Active development and bug fixes
5. **Additional features** - Parameter confidence intervals included

## Verification

Tested on Line dataset with 51 training points:

| Metric | Custom Implementation | pycse | Match? |
|--------|----------------------|-------|--------|
| Coverage | 92.2% | 92.0% | ✓ |
| Mean Width | 0.0410 | 0.0408 | ✓ |
| Max Difference | - | 0.00038 | ✓ (negligible) |

## Installation

```bash
pip install pycse>=2.1.0
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## Implementation Details

### File
`src/uq_methods/linear_uq.py` (lines 11-127)

### Key Functions Used

1. **pycse.regress(A, y, alpha)**
   - Fits linear regression
   - Returns: parameters, parameter intervals, standard errors
   - Uses: Least squares with t-distribution for confidence

2. **pycse.predict(X, y, pars, XX, alpha)**
   - Computes predictions with prediction intervals
   - Returns: predictions, interval bounds (2×n), prediction std errors
   - Uses: Delta method with hat matrix

### Design Matrix Transformation

The model's `_create_design_matrix` method is still used to handle:
- Polynomial features: [1, x, x², x³]
- Lennard-Jones basis: [1, 1/r⁶, 1/r¹²]
- Shomate basis: [1, t, t², t³, 1/t²]

This ensures pycse works correctly with custom basis functions.

## References

- **pycse GitHub**: https://github.com/jkitchin/pycse
- **Documentation**: Available in pycse docstrings
- **Theory**: Classical OLS prediction intervals
  - Weibull DOE: http://www.weibull.com/DOEWeb/confidence_intervals_in_multiple_linear_regression.htm

## Backward Compatibility

The API remains unchanged:
```python
hat = HatMatrixUQ(confidence_level=0.95)
result = hat.compute_intervals(model, X_train, y_train, X_test)
```

All existing code continues to work without modification!

## Notes

- Small numerical warnings from pycse are suppressed (they're handled internally)
- Results may differ slightly (< 0.0004) due to numerical precision
- The method metadata now includes `'method': 'hat_matrix_pycse'`
