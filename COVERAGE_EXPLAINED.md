# Understanding Coverage Metrics in This Benchmark

## What Does "Coverage" Mean?

**Coverage** measures the percentage of test points where the true value falls within the predicted 95% confidence interval.

**Target:** 95% coverage means that ideally, 95 out of 100 test points should have their true values within the predicted uncertainty bounds.

## Critical: What Test Points Are Included?

**The coverage metric in this benchmark includes BOTH interpolation AND extrapolation regions.**

### Data Split Structure

Each experiment uses the following spatial regions (for domain [0, 1]):

```
[0.0 --------- 0.125 -------- 0.375 -- 0.5 -- 0.625 --------- 0.875 --------- 1.0]
 └─ extrap_low ─┘ train_low ┘       gap      └─ train_high ─┘ └─ extrap_high ─┘
     (12.5%)        (25%)        (25%)             (25%)             (12.5%)
```

**Breakdown:**

1. **Extrapolation Low (0.0 - 0.125)**:
   - OUTSIDE the training data range
   - Model has never seen data here
   - Tests model behavior beyond training bounds

2. **Training Low (0.125 - 0.375)**:
   - Training points are sampled from this region
   - Model learned from data here

3. **Gap Region (0.375 - 0.625)**:
   - INSIDE the training bounds but NO training points
   - Tests interpolation with sparse coverage
   - This is a deliberately challenging region

4. **Training High (0.625 - 0.875)**:
   - Training points are sampled from this region
   - Model learned from data here

5. **Extrapolation High (0.875 - 1.0)**:
   - OUTSIDE the training data range
   - Model has never seen data here
   - Tests model behavior beyond training bounds

### What's Evaluated for Coverage?

**Test set:** 500 uniformly spaced points from 0.0 to 1.0 (ENTIRE domain)

This means coverage is computed over:
- ✅ **~12.5% extrapolation** (low region, outside training)
- ✅ **~25% near-interpolation** (training low region)
- ✅ **~25% gap interpolation** (hardest - no nearby training data)
- ✅ **~25% near-interpolation** (training high region)
- ✅ **~12.5% extrapolation** (high region, outside training)

### Why This Matters

**Different challenges in each region:**

| Region | Difficulty | Why Coverage May Drop |
|--------|-----------|----------------------|
| **Training regions** | Easy | Lots of nearby data, should have high coverage (~98-100%) |
| **Gap region** | Hard | No nearby training points, uncertainty estimates must be well-calibrated (~85-95%) |
| **Extrapolation** | Very Hard | Beyond training range, many methods overconfident (~70-90%) |

**The reported coverage is a weighted average across all these regions.**

### What This Means for Interpretation

A method with 88% coverage might have:
- 100% coverage in training regions (excellent)
- 90% coverage in gap region (good interpolation)
- 70% coverage in extrapolation (poor, overconfident)

This overall number reflects the method's ability to:
1. **Interpolate** in regions without training data (gap)
2. **Extrapolate** beyond training bounds
3. **Maintain calibration** across the entire domain

### Comparison to Traditional Benchmarks

**Most machine learning benchmarks only test:**
- Random train/test split from same distribution
- Often just interpolation

**This benchmark is more challenging because:**
- Gap region: Tests interpolation with sparse data
- Extrapolation regions: Tests behavior outside training domain
- Heteroskedastic noise: Tests with varying uncertainty levels

This makes the benchmark more realistic for scientific/engineering applications where:
- We often need to predict in regions without training data
- We must extrapolate beyond measured conditions
- Uncertainty varies across the domain

## Bottom Line

When you see a coverage value in this benchmark (e.g., 88.8% for GP):

**It includes:**
- ✅ Points with lots of nearby training data (easy)
- ✅ Points in the gap with no nearby training data (hard)
- ✅ Points outside the training range (very hard)

**It is NOT just:**
- ❌ Random points from the same distribution as training
- ❌ Only interpolation within training bounds

This makes lower coverage values (e.g., 85-90%) still represent good performance, especially if coverage is maintained across all regions including extrapolation.

## What About Other Metrics (RMSE, R²)?

**All metrics are computed on the same evaluation set as coverage.**

This means:
- **RMSE** (Root Mean Squared Error): Includes prediction errors in extrapolation regions
- **R²** (Coefficient of Determination): Measures variance explained across entire domain including extrapolation
- **Mean Width**: Average uncertainty interval width across all regions

### Interpreting R²

Traditional R² interpretation:
- R² = 0.99: Excellent fit (99% of variance explained)
- R² = 0.95: Good fit
- R² = 0.90: Acceptable fit

**In this benchmark**, these values are MORE impressive because they include extrapolation:
- R² = 0.95 here means the model maintains accuracy even in extrapolation
- Lower R² values (e.g., 0.85-0.90) are still good if the model predicts well in interpolation but struggles with extrapolation

### Design Note

Computing R² over extrapolation is somewhat non-standard:
- **Pro**: Consistent with coverage, tests full model capability
- **Con**: Extrapolation errors can dominate the metric

For scientific applications where extrapolation is common, this approach provides a realistic assessment of model utility.

## Recommendations for Interpretation

**Good Coverage (90-97%)**: Method maintains calibration across interpolation and extrapolation

**Acceptable Coverage (85-90%)**: Method has good interpolation but may be overconfident in extrapolation

**Poor Coverage (<85%)**: Method significantly underestimates uncertainty, especially in challenging regions

**Too High Coverage (>98%)**: Method may be too conservative, producing unnecessarily wide intervals
