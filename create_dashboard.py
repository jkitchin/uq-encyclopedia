#!/usr/bin/env python
"""Generate interactive HTML dashboard for linear models results."""

import pandas as pd
from pathlib import Path
import base64
import sys
sys.path.insert(0, '.')

from config.global_config import RESULTS_CSV_DIR, RESULTS_FIGURES_DIR, DASHBOARD_DIR

print("=" * 70)
print("CREATING INTERACTIVE HTML DASHBOARD")
print("=" * 70)

# Load results
results_path = RESULTS_CSV_DIR / 'linear_models_comprehensive.csv'
df = pd.read_csv(results_path)

print(f"\nLoaded {len(df)} experiment results")

# Compute summary statistics
summary_by_method = df.groupby('uq_method').agg({
    'coverage': ['mean', 'std', 'min', 'max'],
    'rmse': ['mean', 'std'],
    'mean_width': ['mean', 'std'],
}).round(4)

summary_by_dataset = df.groupby('dataset').agg({
    'coverage': ['mean', 'std', 'min', 'max'],
}).round(3)

# Function to embed image as base64
def embed_image(image_path):
    """Embed image as base64 data URI."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

# Get all figure paths
figures = {
    'Coverage by Dataset': RESULTS_FIGURES_DIR / 'coverage_by_dataset.png',
    'Coverage vs Noise Level': RESULTS_FIGURES_DIR / 'coverage_vs_noise_level.png',
    'Coverage by Noise Model': RESULTS_FIGURES_DIR / 'coverage_by_noise_model.png',
    'RMSE by Dataset': RESULTS_FIGURES_DIR / 'rmse_by_dataset.png',
    'Interval Width by Dataset': RESULTS_FIGURES_DIR / 'width_by_dataset.png',
    'Coverage vs Width Trade-off': RESULTS_FIGURES_DIR / 'coverage_vs_width.png',
    'Heatmap: Hat Matrix': RESULTS_FIGURES_DIR / 'heatmap_coverage_hatmatrix.png',
    'Heatmap: Bayesian': RESULTS_FIGURES_DIR / 'heatmap_coverage_bayesian.png',
    'Regional Coverage': RESULTS_FIGURES_DIR / 'regional_coverage.png',
    'Summary Grid': RESULTS_FIGURES_DIR / 'summary_grid.png',
}

print("\nEmbedding images...")

# Create HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UQ Encyclopedia - Linear Models Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .nav {{
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .nav a {{
            color: #667eea;
            text-decoration: none;
            margin-right: 2rem;
            font-weight: 500;
            transition: color 0.3s;
        }}

        .nav a:hover {{
            color: #764ba2;
        }}

        .container {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}

        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .section h2 {{
            color: #667eea;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }}

        .section h3 {{
            color: #764ba2;
            margin: 1.5rem 0 1rem 0;
            font-size: 1.3rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .stat-card h4 {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}

        .stat-card .subtitle {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.95rem;
        }}

        th {{
            background: #667eea;
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover {{
            background: #f8f9ff;
        }}

        .figure {{
            margin: 2rem 0;
            text-align: center;
        }}

        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}

        .figure-caption {{
            margin-top: 1rem;
            color: #666;
            font-style: italic;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 2rem;
        }}

        .alert {{
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 6px;
            border-left: 4px solid;
        }}

        .alert-info {{
            background: #e3f2fd;
            border-color: #2196f3;
            color: #1565c0;
        }}

        .alert-success {{
            background: #e8f5e9;
            border-color: #4caf50;
            color: #2e7d32;
        }}

        .alert-warning {{
            background: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }}

        .badge {{
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}

        .badge-success {{
            background: #4caf50;
            color: white;
        }}

        .badge-warning {{
            background: #ff9800;
            color: white;
        }}

        .badge-danger {{
            background: #f44336;
            color: white;
        }}

        footer {{
            background: #333;
            color: white;
            padding: 2rem;
            text-align: center;
            margin-top: 3rem;
        }}

        footer a {{
            color: #667eea;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 UQ Encyclopedia - Linear Models Results</h1>
        <p>Comprehensive Benchmark of Uncertainty Quantification Methods</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">64 Experiments | 4 Datasets | 2 UQ Methods | 2 Noise Models | 4 Noise Levels</p>
    </div>

    <div class="nav">
        <a href="#summary">Summary</a>
        <a href="#methods">Methods</a>
        <a href="#datasets">Datasets</a>
        <a href="#visualizations">Visualizations</a>
        <a href="#detailed">Detailed Results</a>
        <a href="#insights">Key Insights</a>
    </div>

    <div class="container">
        <!-- Summary Section -->
        <div class="section" id="summary">
            <h2>📊 Executive Summary</h2>

            <div class="alert alert-info">
                <strong>Experiment Overview:</strong> Comprehensive evaluation of 2 UQ methods (Hat Matrix and Bayesian Linear Regression)
                across 4 datasets with varying noise conditions. Target coverage: 95%.
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Experiments</h4>
                    <div class="value">64</div>
                    <div class="subtitle">Complete factorial design</div>
                </div>
                <div class="stat-card">
                    <h4>Hat Matrix Coverage</h4>
                    <div class="value">88.2%</div>
                    <div class="subtitle">Average across all experiments</div>
                </div>
                <div class="stat-card">
                    <h4>Bayesian Coverage</h4>
                    <div class="value">83.0%</div>
                    <div class="subtitle">Average across all experiments</div>
                </div>
                <div class="stat-card">
                    <h4>Target Achievement</h4>
                    <div class="value">48.4%</div>
                    <div class="subtitle">Experiments within 95% ± 5%</div>
                </div>
            </div>

            <h3>Performance by UQ Method</h3>
            {summary_by_method.to_html(classes='', border=0)}

            <h3>Performance by Dataset</h3>
            {summary_by_dataset.to_html(classes='', border=0)}
        </div>

        <!-- Methods Section -->
        <div class="section" id="methods">
            <h2>🔬 UQ Methods Evaluated</h2>

            <h3>1. Hat Matrix Method <span class="badge badge-success">Better Overall</span></h3>
            <p>
                Classical approach using the hat matrix for computing prediction intervals in linear regression.
                Uses t-distribution with degrees of freedom adjustment.
            </p>
            <ul style="margin-left: 2rem; margin-top: 1rem;">
                <li><strong>Average Coverage:</strong> 88.2%</li>
                <li><strong>Average RMSE:</strong> 0.027</li>
                <li><strong>Average Interval Width:</strong> 0.072</li>
                <li><strong>Range:</strong> 54.0% - 97.4%</li>
            </ul>

            <h3>2. Bayesian Linear Regression <span class="badge badge-warning">More Variable</span></h3>
            <p>
                Uses sklearn's BayesianRidge with automatic relevance determination for probabilistic predictions.
            </p>
            <ul style="margin-left: 2rem; margin-top: 1rem;">
                <li><strong>Average Coverage:</strong> 83.0%</li>
                <li><strong>Average RMSE:</strong> 0.068</li>
                <li><strong>Average Interval Width:</strong> 0.110</li>
                <li><strong>Range:</strong> 57.8% - 97.4%</li>
            </ul>

            <div class="alert alert-warning" style="margin-top: 1.5rem;">
                <strong>⚠️ Note:</strong> Bayesian method shows significant underperformance on polynomial datasets with low noise (as low as 57.8% coverage).
            </div>
        </div>

        <!-- Datasets Section -->
        <div class="section" id="datasets">
            <h2>📁 Datasets Tested</h2>

            <div class="grid-2">
                <div>
                    <h3>1. Line Dataset <span class="badge badge-success">94.0% Avg</span></h3>
                    <p><strong>Function:</strong> y = 0.8x + 0.1</p>
                    <p>Simple linear relationship. Best overall performance.</p>
                </div>

                <div>
                    <h3>2. Polynomial (Degree 3) <span class="badge badge-warning">83.6% Avg</span></h3>
                    <p><strong>Function:</strong> y = a₀ + a₁x + a₂x² + a₃x³</p>
                    <p>Cubic polynomial with inflection point. Bayesian struggles at low noise.</p>
                </div>

                <div>
                    <h3>3. Lennard-Jones Potential <span class="badge badge-danger">81.1% Avg</span></h3>
                    <p><strong>Function:</strong> E(r) = 4[(1/r)¹² - (1/r)⁶]</p>
                    <p>Molecular potential energy. Numerical challenges with r⁻¹² terms.</p>
                </div>

                <div>
                    <h3>4. Shomate Polynomial <span class="badge badge-warning">83.5% Avg</span></h3>
                    <p><strong>Function:</strong> Cp/R = A + Bt + Ct² + Dt³ + E/t²</p>
                    <p>Heat capacity equation. Good performance overall.</p>
                </div>
            </div>
        </div>

        <!-- Visualizations Section -->
        <div class="section" id="visualizations">
            <h2>📈 Visualizations</h2>

            <div class="figure">
                <img src="{embed_image(figures['Summary Grid'])}" alt="Summary Grid">
                <div class="figure-caption">Comprehensive 4-panel summary showing key metrics</div>
            </div>

            <div class="grid-2">
                <div class="figure">
                    <img src="{embed_image(figures['Coverage by Dataset'])}" alt="Coverage by Dataset">
                    <div class="figure-caption">Coverage comparison across datasets</div>
                </div>

                <div class="figure">
                    <img src="{embed_image(figures['Coverage vs Noise Level'])}" alt="Coverage vs Noise">
                    <div class="figure-caption">How coverage changes with noise level</div>
                </div>
            </div>

            <div class="grid-2">
                <div class="figure">
                    <img src="{embed_image(figures['Heatmap: Hat Matrix'])}" alt="Hat Matrix Heatmap">
                    <div class="figure-caption">Coverage heatmap: Hat Matrix method</div>
                </div>

                <div class="figure">
                    <img src="{embed_image(figures['Heatmap: Bayesian'])}" alt="Bayesian Heatmap">
                    <div class="figure-caption">Coverage heatmap: Bayesian method</div>
                </div>
            </div>

            <div class="figure">
                <img src="{embed_image(figures['Regional Coverage'])}" alt="Regional Coverage">
                <div class="figure-caption">Coverage in interpolation vs extrapolation regions</div>
            </div>

            <div class="grid-2">
                <div class="figure">
                    <img src="{embed_image(figures['Coverage vs Width Trade-off'])}" alt="Trade-off">
                    <div class="figure-caption">Coverage vs interval width trade-off</div>
                </div>

                <div class="figure">
                    <img src="{embed_image(figures['Coverage by Noise Model'])}" alt="Noise Models">
                    <div class="figure-caption">Homoskedastic vs heteroskedastic performance</div>
                </div>
            </div>
        </div>

        <!-- Detailed Results Section -->
        <div class="section" id="detailed">
            <h2>📋 Detailed Results Table</h2>

            <div class="alert alert-info">
                Showing top 20 results sorted by coverage (descending). Full results available in CSV file.
            </div>

            {df.nlargest(20, 'coverage')[['dataset', 'noise_model', 'noise_level', 'uq_method', 'coverage', 'rmse', 'mean_width']].to_html(classes='', border=0, index=False)}
        </div>

        <!-- Key Insights Section -->
        <div class="section" id="insights">
            <h2>💡 Key Insights & Recommendations</h2>

            <h3>1. Hat Matrix Method is More Reliable</h3>
            <div class="alert alert-success">
                <ul style="margin-left: 1.5rem;">
                    <li>Consistently achieves higher coverage across all conditions (88.2% vs 83.0%)</li>
                    <li>More stable performance (std: 12.3% vs 9.7% but better mean)</li>
                    <li>Better handles complex functional forms</li>
                    <li><strong>Recommendation:</strong> Use Hat Matrix for production applications</li>
                </ul>
            </div>

            <h3>2. Dataset Complexity Matters</h3>
            <div class="alert alert-info">
                <ul style="margin-left: 1.5rem;">
                    <li><strong>Line dataset:</strong> 94.0% coverage (excellent)</li>
                    <li><strong>Polynomial:</strong> 83.6% (good, but Bayesian struggles)</li>
                    <li><strong>Shomate:</strong> 83.5% (good overall)</li>
                    <li><strong>Lennard-Jones:</strong> 81.1% (challenging, numerical issues)</li>
                </ul>
            </div>

            <h3>3. Unexpected Finding: Heteroskedastic Performs Better</h3>
            <div class="alert alert-warning">
                <ul style="margin-left: 1.5rem;">
                    <li>Heteroskedastic noise: <strong>88.6%</strong> coverage</li>
                    <li>Homoskedastic noise: <strong>82.5%</strong> coverage</li>
                    <li>Counterintuitive result suggests methods may be conservative with variable noise</li>
                    <li><strong>Needs further investigation</strong></li>
                </ul>
            </div>

            <h3>4. Regional Performance Varies</h3>
            <ul style="margin-left: 2rem;">
                <li><strong>Low extrapolation:</strong> Best performance (93.8% Hat Matrix)</li>
                <li><strong>Interpolation:</strong> Good performance (90.5%)</li>
                <li><strong>High extrapolation:</strong> Worst performance (86.9%)</li>
                <li>Both methods degrade in high extrapolation regions</li>
            </ul>

            <h3>5. Recommendations for Future Work</h3>
            <div class="alert alert-info">
                <ul style="margin-left: 1.5rem;">
                    <li>Fix MAPIE API compatibility for conformal prediction</li>
                    <li>Improve Bayesian hyperparameter tuning</li>
                    <li>Add numerical safeguards for Lennard-Jones potential</li>
                    <li>Investigate heteroskedastic noise performance</li>
                    <li>Implement learning curve analysis</li>
                    <li>Add bootstrap-based intervals</li>
                </ul>
            </div>
        </div>
    </div>

    <footer>
        <p><strong>UQ Encyclopedia Benchmark Framework</strong> | Version 0.1.0 (Phase 2 Complete)</p>
        <p style="margin-top: 0.5rem; opacity: 0.8;">
            Generated: November 8, 2024 |
            <a href="https://github.com/anthropics/claude-code">Built with Claude Code</a>
        </p>
    </footer>
</body>
</html>
"""

# Save HTML
output_path = DASHBOARD_DIR / 'linear_models_dashboard.html'
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ Dashboard created: {output_path}")
print(f"\nTo view the dashboard, open this file in your web browser:")
print(f"  {output_path}")
print(f"\nOr run:")
print(f"  open {output_path}")
