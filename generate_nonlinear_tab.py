#!/usr/bin/env python3
"""
Generate the nonlinear models tab content for the dashboard.
"""

import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('results/csv/nonlinear_models_comprehensive.csv')

# Calculate summary statistics
total_experiments = len(df)
num_datasets = df['dataset'].nunique()
num_methods = df['uq_method'].nunique()
num_noise_models = df['noise_model'].nunique()
num_noise_levels = df['noise_level'].nunique()

# Calculate mean coverage by method
coverage_by_method = df.groupby('uq_method')['coverage'].mean()
nlinfit_coverage = coverage_by_method['Nlinfit']
conformal_coverage = coverage_by_method['Conformal']

# Calculate target achievement (within 95% ± 5%, i.e., 90% - 100%)
target_achievement = ((df['coverage'] >= 0.90) & (df['coverage'] <= 1.0)).sum() / total_experiments

# Performance table by UQ method
perf_table = df.groupby('uq_method').agg({
    'coverage': ['mean', 'std', 'min', 'max'],
    'rmse': ['mean', 'std'],
    'mean_width': ['mean', 'std']
}).round(4)

# Generate the HTML
html = f'''<div class="header">
        <h1>🎯 UQ Encyclopedia - Nonlinear Models Results</h1>
        <p>Comprehensive Benchmark of Uncertainty Quantification Methods</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">{total_experiments} Experiments | {num_datasets} Datasets | {num_methods} UQ Methods | {num_noise_models} Noise Models | {num_noise_levels} Noise Levels</p>
    </div>

    <div class="nav">
        <a href="#summary-nl">Summary</a>
        <a href="#methods-nl">Methods</a>
        <a href="#datasets-nl">Datasets</a>
        <a href="#visualizations-nl">Visualizations</a>
        <a href="#detailed-nl">Detailed Results</a>
        <a href="#insights-nl">Key Insights</a>
    </div>

    <div class="container">
        <!-- Summary Section -->
        <div class="section" id="summary-nl">
            <h2>📊 Executive Summary</h2>

            <div class="alert alert-info">
                <strong>Experiment Overview:</strong> Comprehensive evaluation of 2 UQ methods (Nlinfit and Conformal)
                across 4 nonlinear datasets with varying noise conditions. Target coverage: 95%.
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Experiments</h4>
                    <div class="value">{total_experiments}</div>
                    <div class="subtitle">Complete factorial design</div>
                </div>
                <div class="stat-card">
                    <h4>Nlinfit Coverage</h4>
                    <div class="value">{nlinfit_coverage*100:.1f}%</div>
                    <div class="subtitle">Average across all experiments</div>
                </div>
                <div class="stat-card">
                    <h4>Conformal Coverage</h4>
                    <div class="value">{conformal_coverage*100:.1f}%</div>
                    <div class="subtitle">Average across all experiments</div>
                </div>
                <div class="stat-card">
                    <h4>Target Achievement</h4>
                    <div class="value">{target_achievement*100:.1f}%</div>
                    <div class="subtitle">Experiments within 95% ± 5%</div>
                </div>
            </div>

            <h3>Performance by UQ Method</h3>
            {perf_table.to_html()}
        </div>

        <!-- Methods Section -->
        <div class="section" id="methods-nl">
            <h2>🔧 UQ Methods</h2>

            <div class="method-card">
                <h3>Nlinfit (Delta Method)</h3>
                <p><strong>Approach:</strong> Uses the delta method to propagate parameter uncertainties from nonlinear least squares regression.</p>
                <p><strong>Key Features:</strong></p>
                <ul>
                    <li>Analytical uncertainty propagation based on parameter covariance matrix</li>
                    <li>Assumes asymptotic normality of parameter estimates</li>
                    <li>Fast computation using first-order Taylor approximation</li>
                    <li>Standard approach in nonlinear regression packages (e.g., MATLAB, R)</li>
                </ul>
                <p><strong>Average Coverage:</strong> {nlinfit_coverage*100:.1f}%</p>
            </div>

            <div class="method-card">
                <h3>Conformal Prediction</h3>
                <p><strong>Approach:</strong> Distribution-free uncertainty quantification using calibration set.</p>
                <p><strong>Key Features:</strong></p>
                <ul>
                    <li>Model-agnostic, distribution-free method</li>
                    <li>Provides finite-sample coverage guarantees</li>
                    <li>Requires splitting data into training and calibration sets</li>
                    <li>Adapts to local prediction difficulty</li>
                </ul>
                <p><strong>Average Coverage:</strong> {conformal_coverage*100:.1f}%</p>
            </div>
        </div>

        <!-- Datasets Section -->
        <div class="section" id="datasets-nl">
            <h2>📁 Datasets</h2>

            <div class="dataset-card">
                <h3>Exponential Decay</h3>
                <p><strong>Model:</strong> y = A * exp(-k * x) + C</p>
                <p><strong>Description:</strong> Classic exponential decay model commonly used in kinetics, radioactive decay, and relaxation processes.</p>
                <p><strong>Parameters:</strong> Amplitude (A), decay constant (k), offset (C)</p>
                <p><strong>Experiments:</strong> {df[df['dataset']=='ExponentialDecay'].shape[0]} configurations</p>
            </div>

            <div class="dataset-card">
                <h3>Logistic Growth</h3>
                <p><strong>Model:</strong> y = K / (1 + exp(-r * (x - x₀)))</p>
                <p><strong>Description:</strong> Sigmoid function modeling population growth, learning curves, and dose-response relationships.</p>
                <p><strong>Parameters:</strong> Carrying capacity (K), growth rate (r), midpoint (x₀)</p>
                <p><strong>Experiments:</strong> {df[df['dataset']=='LogisticGrowth'].shape[0]} configurations</p>
            </div>

            <div class="dataset-card">
                <h3>Michaelis-Menten</h3>
                <p><strong>Model:</strong> y = (Vmax * x) / (Km + x)</p>
                <p><strong>Description:</strong> Enzyme kinetics model describing reaction rate as a function of substrate concentration.</p>
                <p><strong>Parameters:</strong> Maximum velocity (Vmax), Michaelis constant (Km)</p>
                <p><strong>Experiments:</strong> {df[df['dataset']=='MichaelisMenten'].shape[0]} configurations</p>
            </div>

            <div class="dataset-card">
                <h3>Gaussian</h3>
                <p><strong>Model:</strong> y = A * exp(-(x - μ)² / (2σ²))</p>
                <p><strong>Description:</strong> Gaussian/normal distribution used for peak fitting in spectroscopy and chromatography.</p>
                <p><strong>Parameters:</strong> Amplitude (A), mean (μ), standard deviation (σ)</p>
                <p><strong>Experiments:</strong> {df[df['dataset']=='Gaussian'].shape[0]} configurations</p>
            </div>
        </div>

        <!-- Visualizations Section -->
        <div class="section" id="visualizations-nl">
            <h2>📈 Visualizations</h2>
            <p>Click on any row in the Detailed Results table below to view interactive visualizations showing:</p>
            <ul>
                <li>Data points with noise</li>
                <li>Fitted nonlinear model</li>
                <li>Uncertainty intervals from each UQ method</li>
                <li>Coverage statistics and error metrics</li>
            </ul>
        </div>

        <!-- Detailed Results Section -->
        <div class="section" id="detailed-nl">
            <h2>📋 Detailed Results</h2>
            <p>Click on any row to view the detailed fit visualization.</p>
            <div id="nonlinear-results">
'''

# Generate the table
html += '''                <table id="resultsTableNL">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Noise Model</th>
                            <th>Noise Level</th>
                            <th>UQ Method</th>
                            <th>Coverage</th>
                            <th>RMSE</th>
                            <th>Mean Width</th>
                            <th>R²</th>
                        </tr>
                    </thead>
                    <tbody>
'''

# Generate table rows
for _, row in df.iterrows():
    exp_id = row['exp_id']
    dataset = row['dataset']
    noise_model = row['noise_model']
    noise_level = row['noise_level']
    uq_method = row['uq_method']
    coverage = row['coverage']
    rmse = row['rmse']
    mean_width = row['mean_width']
    r2 = row['r2']

    # Determine coverage class
    if coverage >= 0.90 and coverage <= 1.0:
        coverage_class = 'coverage-good'
    elif coverage >= 0.85:
        coverage_class = 'coverage-acceptable'
    else:
        coverage_class = 'coverage-poor'

    # Build file path
    file_path = f'../../results/figures/nonlinear_fits_html/{exp_id}_{uq_method}.html'

    html += f'''                        <tr class="clickable-row" onclick="showFit('{file_path}', '{dataset}', '{noise_model}', {noise_level}, '{uq_method}', {coverage})">
                            <td>{dataset}</td>
                            <td>{noise_model}</td>
                            <td>{noise_level:.2f}</td>
                            <td>{uq_method}</td>
                            <td class="{coverage_class}">{coverage:.3f}</td>
                            <td>{rmse:.4f}</td>
                            <td>{mean_width:.4f}</td>
                            <td>{r2:.4f}</td>
                        </tr>
'''

html += '''                    </tbody>
                </table>
            </div>
        </div>

        <!-- Key Insights Section -->
        <div class="section" id="insights-nl">
            <h2>💡 Key Insights</h2>

            <div class="insight-card">
                <h3>🎯 Overall Performance</h3>
                <ul>
                    <li><strong>Nlinfit (Delta Method)</strong> achieved {:.1f}% average coverage across all experiments</li>
                    <li><strong>Conformal Prediction</strong> achieved {:.1f}% average coverage across all experiments</li>
                    <li>{:.1f}% of experiments achieved coverage within the target range (90-100%)</li>
                    <li>Both methods show excellent predictive accuracy with high R² values (>0.99 for low noise)</li>
                </ul>
            </div>

            <div class="insight-card">
                <h3>📊 Method Comparison</h3>
                <ul>
                    <li><strong>Nlinfit</strong> tends to provide more consistent coverage across different datasets, especially for homoskedastic noise</li>
                    <li><strong>Conformal</strong> shows more variable coverage but adapts better to heteroskedastic noise in some cases</li>
                    <li>Coverage degrades for both methods as noise level increases, particularly for heteroskedastic noise</li>
                    <li>Interval widths scale appropriately with noise level for both methods</li>
                </ul>
            </div>

            <div class="insight-card">
                <h3>🔬 Dataset-Specific Findings</h3>
                <ul>
                    <li><strong>Exponential Decay:</strong> Both methods perform well for homoskedastic noise; Nlinfit shows better coverage for heteroskedastic noise</li>
                    <li><strong>Logistic Growth:</strong> Excellent performance for both methods, with Conformal showing adaptive interval widths</li>
                    <li><strong>Michaelis-Menten:</strong> Conformal achieves perfect coverage (100%) at high heteroskedastic noise, though with wider intervals</li>
                    <li><strong>Gaussian:</strong> Most challenging dataset; both methods struggle with coverage, especially for heteroskedastic noise</li>
                </ul>
            </div>

            <div class="insight-card">
                <h3>⚠️ Challenges and Limitations</h3>
                <ul>
                    <li>Gaussian peak fitting is particularly challenging for UQ methods due to parameter correlation</li>
                    <li>Heteroskedastic noise creates additional challenges, especially at higher noise levels</li>
                    <li>Delta method assumes asymptotic normality, which may not hold well for small samples or highly nonlinear models</li>
                    <li>Conformal prediction requires data splitting, reducing effective training set size</li>
                </ul>
            </div>

            <div class="insight-card">
                <h3>✅ Recommendations</h3>
                <ul>
                    <li>For <strong>homoskedastic noise</strong>: Nlinfit (delta method) provides reliable, fast uncertainty estimates</li>
                    <li>For <strong>heteroskedastic noise</strong>: Consider both methods; Conformal can provide adaptive intervals</li>
                    <li>For <strong>critical applications</strong>: Use both methods and compare results for robustness</li>
                    <li>For <strong>peak fitting (Gaussian)</strong>: Consider alternative parameterizations or Bayesian approaches for better UQ</li>
                </ul>
            </div>
        </div>
    </div>'''.format(
    nlinfit_coverage*100,
    conformal_coverage*100,
    target_achievement*100
)

print(html)
