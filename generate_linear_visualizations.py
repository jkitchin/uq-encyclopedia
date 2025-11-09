#!/usr/bin/env python
"""Generate comprehensive visualizations from linear models benchmark results."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, '.')

from config.global_config import RESULTS_CSV_DIR, RESULTS_FIGURES_DIR

print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Load results
results_path = RESULTS_CSV_DIR / 'linear_models_comprehensive.csv'
print(f"\nLoading results from: {results_path}")
df = pd.read_csv(results_path)

print(f"Loaded {len(df)} experiment results")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Create figure directory if needed
RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# 1. Coverage by Dataset and UQ Method
print("\n1. Coverage by Dataset and UQ Method...")
fig, ax = plt.subplots(figsize=(12, 6))

datasets = df['dataset'].unique()
uq_methods = df['uq_method'].unique()
x = np.arange(len(datasets))
width = 0.35

for i, method in enumerate(uq_methods):
    method_data = df[df['uq_method'] == method].groupby('dataset')['coverage'].mean()
    offset = width * (i - len(uq_methods)/2 + 0.5)
    ax.bar(x + offset, method_data, width, label=method, alpha=0.8)

ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Target (95%)')
ax.axhspan(0.90, 1.00, alpha=0.1, color='green', label='±5% tolerance')
ax.set_xlabel('Dataset')
ax.set_ylabel('Coverage')
ax.set_title('Coverage by Dataset and UQ Method')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'coverage_by_dataset.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: coverage_by_dataset.png")
plt.close()

# 2. Coverage by Noise Level
print("2. Coverage by Noise Level...")
fig, ax = plt.subplots(figsize=(10, 6))

for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('noise_level')['coverage'].agg(['mean', 'std'])
    ax.errorbar(method_data.index * 100, method_data['mean'], yerr=method_data['std'],
                marker='o', linewidth=2, markersize=8, label=method, capsize=5)

ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Target')
ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('Coverage')
ax.set_title('Coverage vs Noise Level')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'coverage_vs_noise_level.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: coverage_vs_noise_level.png")
plt.close()

# 3. Coverage: Homoskedastic vs Heteroskedastic
print("3. Coverage: Homoskedastic vs Heteroskedastic...")
fig, ax = plt.subplots(figsize=(10, 6))

noise_models = df['noise_model'].unique()
x = np.arange(len(uq_methods))
width = 0.35

for i, noise_model in enumerate(noise_models):
    model_data = df[df['noise_model'] == noise_model].groupby('uq_method')['coverage'].mean()
    offset = width * (i - len(noise_models)/2 + 0.5)
    ax.bar(x + offset, model_data, width, label=noise_model.capitalize(), alpha=0.8)

ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Target')
ax.set_xlabel('UQ Method')
ax.set_ylabel('Coverage')
ax.set_title('Coverage: Homoskedastic vs Heteroskedastic Noise')
ax.set_xticks(x)
ax.set_xticklabels(uq_methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'coverage_by_noise_model.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: coverage_by_noise_model.png")
plt.close()

# 4. RMSE by Dataset
print("4. RMSE by Dataset...")
fig, ax = plt.subplots(figsize=(12, 6))

for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('dataset')['rmse'].mean()
    ax.plot(method_data.index, method_data.values, marker='o',
            linewidth=2, markersize=8, label=method)

ax.set_xlabel('Dataset')
ax.set_ylabel('RMSE')
ax.set_title('RMSE by Dataset and UQ Method')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'rmse_by_dataset.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: rmse_by_dataset.png")
plt.close()

# 5. Interval Width by Dataset
print("5. Interval Width by Dataset...")
fig, ax = plt.subplots(figsize=(12, 6))

for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('dataset')['mean_width'].mean()
    ax.plot(method_data.index, method_data.values, marker='o',
            linewidth=2, markersize=8, label=method)

ax.set_xlabel('Dataset')
ax.set_ylabel('Mean Interval Width')
ax.set_title('Prediction Interval Width by Dataset')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'width_by_dataset.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: width_by_dataset.png")
plt.close()

# 6. Coverage vs Width Trade-off
print("6. Coverage vs Width Trade-off...")
fig, ax = plt.subplots(figsize=(10, 8))

for dataset in datasets:
    for method in uq_methods:
        subset = df[(df['dataset'] == dataset) & (df['uq_method'] == method)]
        marker = 'o' if method == uq_methods[0] else 's'
        ax.scatter(subset['coverage'], subset['mean_width'],
                  s=100, alpha=0.6, marker=marker,
                  label=f'{dataset} - {method}')

ax.axvline(x=0.95, color='r', linestyle='--', linewidth=2, label='Target Coverage')
ax.set_xlabel('Coverage')
ax.set_ylabel('Mean Interval Width')
ax.set_title('Coverage vs Width Trade-off')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'coverage_vs_width.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: coverage_vs_width.png")
plt.close()

# 7. Heatmap of Coverage by Dataset and Noise Level
print("7. Heatmap: Coverage by Dataset and Noise Level...")
for method in uq_methods:
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot_data = df[df['uq_method'] == method].pivot_table(
        values='coverage',
        index='dataset',
        columns='noise_level',
        aggfunc='mean'
    )

    # Convert noise levels to percentages for better labels
    pivot_data.columns = [f'{int(x*100)}%' for x in pivot_data.columns]

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.85, vmax=1.0, center=0.95, ax=ax, cbar_kws={'label': 'Coverage'})

    ax.set_title(f'Coverage Heatmap: {method}')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Dataset')

    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / f'heatmap_coverage_{method.lower()}.png',
                dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: heatmap_coverage_{method.lower()}.png")
    plt.close()

# 8. Regional Coverage Comparison
print("8. Regional Coverage (Interpolation vs Extrapolation)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

regions = ['extrap_low', 'interpolation', 'extrap_high']
region_labels = ['Extrapolation\n(Low)', 'Interpolation', 'Extrapolation\n(High)']

for idx, method in enumerate(uq_methods):
    ax = axes[idx]

    region_data = []
    for region in regions:
        col_name = f'coverage_{region}'
        if col_name in df.columns:
            region_data.append(df[df['uq_method'] == method][col_name].mean())
        else:
            region_data.append(np.nan)

    bars = ax.bar(region_labels, region_data, alpha=0.7)
    ax.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Target')
    ax.set_ylabel('Coverage')
    ax.set_title(f'{method}: Coverage by Region')
    ax.set_ylim([0.85, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Color bars based on target
    for bar, coverage in zip(bars, region_data):
        if not np.isnan(coverage):
            if abs(coverage - 0.95) < 0.05:
                bar.set_color('green')
            elif abs(coverage - 0.95) < 0.10:
                bar.set_color('orange')
            else:
                bar.set_color('red')

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'regional_coverage.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: regional_coverage.png")
plt.close()

# 9. Comprehensive Summary Grid
print("9. Comprehensive Summary Grid...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Coverage by dataset
ax = axes[0, 0]
for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('dataset')['coverage'].mean()
    ax.plot(range(len(method_data)), method_data.values, marker='o', label=method)
ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_ylabel('Coverage')
ax.set_title('Coverage by Dataset')
ax.legend()
ax.grid(True, alpha=0.3)

# RMSE by noise level
ax = axes[0, 1]
for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('noise_level')['rmse'].mean()
    ax.plot(method_data.index * 100, method_data.values, marker='o', label=method)
ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('RMSE')
ax.set_title('RMSE vs Noise Level')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Width by noise level
ax = axes[1, 0]
for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('noise_level')['mean_width'].mean()
    ax.plot(method_data.index * 100, method_data.values, marker='o', label=method)
ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('Mean Width')
ax.set_title('Interval Width vs Noise Level')
ax.legend()
ax.grid(True, alpha=0.3)

# Miscalibration by dataset
ax = axes[1, 1]
for method in uq_methods:
    method_data = df[df['uq_method'] == method].groupby('dataset')['miscalibration_area'].mean()
    ax.plot(range(len(method_data)), method_data.values, marker='o', label=method)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_ylabel('Miscalibration Area')
ax.set_title('Miscalibration by Dataset (lower is better)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_FIGURES_DIR / 'summary_grid.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: summary_grid.png")
plt.close()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\nAll figures saved to: {RESULTS_FIGURES_DIR}")
print(f"\nGenerated {9 + len(uq_methods)} figures:")
print("  - coverage_by_dataset.png")
print("  - coverage_vs_noise_level.png")
print("  - coverage_by_noise_model.png")
print("  - rmse_by_dataset.png")
print("  - width_by_dataset.png")
print("  - coverage_vs_width.png")
for method in uq_methods:
    print(f"  - heatmap_coverage_{method.lower()}.png")
print("  - regional_coverage.png")
print("  - summary_grid.png")
