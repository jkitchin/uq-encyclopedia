#!/usr/bin/env python
"""Add educational guide section to the interactive dashboard."""

import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.global_config import DASHBOARD_DIR

print("Adding educational guide to dashboard...")

# Read the current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the guide section HTML
guide_section = """
        <!-- Educational Guide Section -->
        <div class="section" id="guide">
            <h2>📚 Understanding This Study</h2>

            <div class="guide-nav">
                <button class="guide-btn active" onclick="showGuideTab('coverage')">What is Coverage?</button>
                <button class="guide-btn" onclick="showGuideTab('design')">Study Design</button>
                <button class="guide-btn" onclick="showGuideTab('datasets')">Datasets</button>
                <button class="guide-btn" onclick="showGuideTab('methods')">UQ Methods</button>
                <button class="guide-btn" onclick="showGuideTab('interpret')">How to Interpret</button>
            </div>

            <!-- Coverage Tab -->
            <div id="guide-coverage" class="guide-content active">
                <h3>🎯 What is Coverage?</h3>

                <div class="info-box">
                    <h4>Definition</h4>
                    <p><strong>Coverage</strong> measures how often the prediction intervals contain the true values.</p>
                    <p>If we create 95% prediction intervals, we expect them to contain the true value 95% of the time.</p>
                </div>

                <h4>Example:</h4>
                <ul>
                    <li>We predict 100 test points with 95% intervals</li>
                    <li>If 95 of those intervals contain the true values → <span class="good">Coverage = 95%</span> ✓</li>
                    <li>If only 85 contain the true values → <span class="warning">Coverage = 85%</span> (under-coverage)</li>
                    <li>If 99 contain the true values → <span class="info">Coverage = 99%</span> (over-coverage, too conservative)</li>
                </ul>

                <div class="info-box info">
                    <h4>Why is this important?</h4>
                    <p>Good UQ methods should be <strong>well-calibrated</strong>:</p>
                    <ul>
                        <li><strong>95% coverage ≈ 95%</strong>: The method is reliable and trustworthy</li>
                        <li><strong>&lt; 90% coverage</strong>: Intervals are too narrow (overconfident, dangerous!)</li>
                        <li><strong>&gt; 99% coverage</strong>: Intervals are too wide (underconfident, not useful)</li>
                    </ul>
                </div>

                <h4>Prediction Interval vs Confidence Interval</h4>
                <div class="comparison-grid">
                    <div>
                        <strong>Prediction Interval (this study)</strong>
                        <p>Captures uncertainty about a <em>future observation</em></p>
                        <p>Includes both:</p>
                        <ul>
                            <li>Model uncertainty (how well we know the function)</li>
                            <li>Data noise (irreducible randomness)</li>
                        </ul>
                        <p>Wider than confidence intervals</p>
                    </div>
                    <div>
                        <strong>Confidence Interval</strong>
                        <p>Captures uncertainty about the <em>mean prediction</em></p>
                        <p>Only includes model uncertainty</p>
                        <p>Narrower, shrinks with more data</p>
                    </div>
                </div>

                <div class="example-box">
                    <h4>Visual Example</h4>
                    <p>Click on any row in the results table to see prediction intervals in action:</p>
                    <ul>
                        <li><span style="color: blue;">Blue shaded region</span> = 95% prediction interval</li>
                        <li><span style="color: black;">Black line</span> = True function</li>
                        <li><span style="color: blue; font-weight: bold;">Blue dashed line</span> = Model prediction</li>
                        <li><span style="color: gray;">Gray dots</span> = Training data (noisy observations)</li>
                    </ul>
                    <p>Count how many black line points fall inside the blue shaded region → that's the coverage!</p>
                </div>
            </div>

            <!-- Study Design Tab -->
            <div id="guide-design" class="guide-content">
                <h3>🔬 Study Design</h3>

                <div class="info-box">
                    <h4>Objective</h4>
                    <p>Comprehensively evaluate uncertainty quantification (UQ) methods for linear regression models under various conditions.</p>
                </div>

                <h4>Factorial Experimental Design</h4>
                <p>We tested <strong>all combinations</strong> of:</p>

                <div class="factor-grid">
                    <div class="factor-card">
                        <h5>📁 Datasets (4)</h5>
                        <ul>
                            <li>Line</li>
                            <li>Polynomial (degree 3)</li>
                            <li>Lennard-Jones potential</li>
                            <li>Shomate polynomial</li>
                        </ul>
                    </div>

                    <div class="factor-card">
                        <h5>🔊 Noise Models (2)</h5>
                        <ul>
                            <li><strong>Homoskedastic:</strong> Constant noise (σ = constant)</li>
                            <li><strong>Heteroskedastic:</strong> Variable noise (σ = σ(x))</li>
                        </ul>
                    </div>

                    <div class="factor-card">
                        <h5>📊 Noise Levels (4)</h5>
                        <ul>
                            <li>1% (very clean data)</li>
                            <li>2% (clean)</li>
                            <li>5% (moderate)</li>
                            <li>10% (noisy)</li>
                        </ul>
                    </div>

                    <div class="factor-card">
                        <h5>🎯 UQ Methods (2)</h5>
                        <ul>
                            <li>Hat Matrix (classical)</li>
                            <li>Bayesian Linear Regression</li>
                        </ul>
                    </div>
                </div>

                <div class="calculation-box">
                    <p><strong>Total experiments:</strong> 4 × 2 × 4 × 2 = <strong>64 experiments</strong></p>
                </div>

                <h4>Data Splitting Strategy</h4>
                <div class="split-diagram">
                    <div class="split-section train">
                        <strong>Training Data</strong>
                        <p>~51 samples</p>
                        <p>Used to fit the model</p>
                    </div>
                    <div class="split-section gap">
                        <strong>Gap (25%)</strong>
                        <p>~25 samples</p>
                        <p>Interpolation test</p>
                    </div>
                    <div class="split-section extrap">
                        <strong>Extrapolation</strong>
                        <p>Low & High regions</p>
                        <p>Beyond training data</p>
                    </div>
                    <div class="split-section test">
                        <strong>Test Data</strong>
                        <p>500 samples</p>
                        <p>Dense grid evaluation</p>
                    </div>
                </div>

                <p><strong>Why this split?</strong></p>
                <ul>
                    <li><strong>Training:</strong> Data available for fitting the model</li>
                    <li><strong>Gap:</strong> Tests interpolation (predicting within the domain but where we have no training data)</li>
                    <li><strong>Extrapolation:</strong> Tests prediction beyond the training domain (more challenging)</li>
                    <li><strong>Test:</strong> Dense evaluation grid for comprehensive coverage assessment</li>
                </ul>

                <h4>Reproducibility</h4>
                <div class="info-box success">
                    <ul>
                        <li>✓ Fixed random seed (42) for all experiments</li>
                        <li>✓ Deterministic seed derivation for each configuration</li>
                        <li>✓ All code and configurations version controlled</li>
                        <li>✓ Complete results saved to CSV for re-analysis</li>
                    </ul>
                </div>
            </div>

            <!-- Datasets Tab -->
            <div id="guide-datasets" class="guide-content">
                <h3>📁 Datasets Explained</h3>

                <div class="dataset-card">
                    <h4>1. Line Dataset</h4>
                    <p><strong>Function:</strong> y = 0.8x + 0.1</p>
                    <p><strong>Type:</strong> Simplest linear relationship</p>
                    <p><strong>Complexity:</strong> ⭐ (Very Easy)</p>
                    <p><strong>Why test this?</strong> Baseline - all methods should excel here. If coverage is poor on this, something is fundamentally wrong.</p>
                    <p><strong>Expected:</strong> ~94% coverage ✓</p>
                </div>

                <div class="dataset-card">
                    <h4>2. Polynomial Dataset (Degree 3)</h4>
                    <p><strong>Function:</strong> y = a₀ + a₁x + a₂x² + a₃x³</p>
                    <p><strong>Type:</strong> Cubic polynomial with inflection point</p>
                    <p><strong>Complexity:</strong> ⭐⭐ (Moderate)</p>
                    <p><strong>Why test this?</strong> Tests if methods handle non-constant curvature. More parameters = more uncertainty.</p>
                    <p><strong>Challenge:</strong> Bayesian method struggles at low noise (57.8% coverage!)</p>
                    <p><strong>Expected:</strong> ~83.6% coverage</p>
                </div>

                <div class="dataset-card">
                    <h4>3. Lennard-Jones Potential</h4>
                    <p><strong>Function:</strong> E(r) = 4ε[(σ/r)¹² - (σ/r)⁶]</p>
                    <p><strong>Type:</strong> Molecular potential energy (chemistry/physics)</p>
                    <p><strong>Complexity:</strong> ⭐⭐⭐⭐ (Challenging)</p>
                    <p><strong>Why test this?</strong> Real-world function with extreme behavior (r⁻¹² terms → numerical instability)</p>
                    <p><strong>Challenge:</strong> Numerical issues when r is small (divide-by-zero warnings)</p>
                    <p><strong>Expected:</strong> ~81.1% coverage (lowest overall)</p>
                </div>

                <div class="dataset-card">
                    <h4>4. Shomate Polynomial</h4>
                    <p><strong>Function:</strong> Cp/R = A + Bt + Ct² + Dt³ + E/t²</p>
                    <p><strong>Type:</strong> Heat capacity equation (thermodynamics)</p>
                    <p><strong>Complexity:</strong> ⭐⭐⭐ (Moderate-High)</p>
                    <p><strong>Why test this?</strong> Real scientific application with 1/t² term (different from standard polynomials)</p>
                    <p><strong>Domain:</strong> Temperature 298-1000K (scaled to [0,1])</p>
                    <p><strong>Expected:</strong> ~83.5% coverage</p>
                </div>

                <div class="info-box">
                    <h4>Common Features</h4>
                    <ul>
                        <li>All scaled to domain [0, 1] for fair comparison</li>
                        <li>All normalized to roughly [0, 1] output range</li>
                        <li>Same train/test split strategy</li>
                        <li>100 total samples per dataset</li>
                    </ul>
                </div>
            </div>

            <!-- Methods Tab -->
            <div id="guide-methods" class="guide-content">
                <h3>🎯 UQ Methods Explained</h3>

                <div class="method-card">
                    <h4>1. Hat Matrix Method (Classical)</h4>
                    <p><strong>Type:</strong> Frequentist / Classical Statistics</p>
                    <p><strong>Average Coverage:</strong> 88.2% (Better overall)</p>

                    <h5>How it works:</h5>
                    <ol>
                        <li>Fit ordinary least squares (OLS) regression</li>
                        <li>Compute the "hat matrix" H = X(X^T X)^(-1)X^T</li>
                        <li>Prediction variance at point x: σ²(1 + h_i) where h_i is diagonal element</li>
                        <li>Use t-distribution for intervals: ŷ ± t_{α/2,df} × σ√(1 + h_i)</li>
                    </ol>

                    <h5>Assumptions:</h5>
                    <ul>
                        <li>Errors are normally distributed</li>
                        <li>Constant variance (homoskedastic) - but tested on heteroskedastic too</li>
                        <li>Linear relationship</li>
                    </ul>

                    <h5>Strengths:</h5>
                    <ul class="pros">
                        <li>Fast computation</li>
                        <li>Well-established theory</li>
                        <li>Reliable across datasets (88.2% avg)</li>
                        <li>Exact solution (no iterations)</li>
                    </ul>

                    <h5>Weaknesses:</h5>
                    <ul class="cons">
                        <li>Assumes normality</li>
                        <li>Can underperform on complex datasets</li>
                        <li>Less flexible than Bayesian</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>2. Bayesian Linear Regression</h4>
                    <p><strong>Type:</strong> Bayesian / Probabilistic</p>
                    <p><strong>Average Coverage:</strong> 83.0% (More variable)</p>

                    <h5>How it works:</h5>
                    <ol>
                        <li>Place prior distributions on model parameters</li>
                        <li>Update priors with data → posterior distribution</li>
                        <li>Predictions are distributions (not point estimates)</li>
                        <li>Use BayesianRidge (sklearn) with automatic relevance determination</li>
                    </ol>

                    <h5>What makes it "Bayesian":</h5>
                    <ul>
                        <li>Treats parameters as random variables (not fixed unknowns)</li>
                        <li>Incorporates prior knowledge/beliefs</li>
                        <li>Produces full probability distributions for predictions</li>
                    </ul>

                    <h5>Strengths:</h5>
                    <ul class="pros">
                        <li>Naturally provides uncertainty estimates</li>
                        <li>Automatic feature selection (ARD)</li>
                        <li>Theoretically principled</li>
                        <li>Can incorporate prior knowledge</li>
                    </ul>

                    <h5>Weaknesses:</h5>
                    <ul class="cons">
                        <li>Lower average coverage (83.0%)</li>
                        <li>Highly variable (57.8% - 97.4%)</li>
                        <li>Struggles on polynomial + low noise</li>
                        <li>Sensitive to hyperparameters</li>
                        <li>Slower than Hat Matrix</li>
                    </ul>
                </div>

                <div class="info-box warning">
                    <h4>⚠️ Key Finding</h4>
                    <p>Hat Matrix method is more reliable across conditions. Bayesian method needs hyperparameter tuning to improve performance, especially for polynomial datasets with low noise.</p>
                </div>
            </div>

            <!-- Interpretation Tab -->
            <div id="guide-interpret" class="guide-content">
                <h3>📖 How to Interpret Results</h3>

                <h4>Understanding the Table</h4>
                <div class="interpretation-grid">
                    <div>
                        <h5>Coverage Column</h5>
                        <ul>
                            <li><span class="good">Green (≥0.90):</span> Good coverage</li>
                            <li><span class="warning">Orange (0.80-0.90):</span> Acceptable</li>
                            <li><span class="poor">Red (&lt;0.80):</span> Poor (intervals too narrow)</li>
                        </ul>
                        <p><strong>Target:</strong> 0.95 ± 0.05 (i.e., 0.90 to 1.00)</p>
                    </div>

                    <div>
                        <h5>RMSE (Root Mean Squared Error)</h5>
                        <p>Measures prediction accuracy (lower is better)</p>
                        <p>This is about the point predictions, not the intervals</p>
                        <p><strong>Typical values:</strong> 0.002 - 0.08</p>
                    </div>

                    <div>
                        <h5>Mean Width</h5>
                        <p>Average width of prediction intervals</p>
                        <p><strong>Tradeoff:</strong> Wider = more coverage, but less useful</p>
                        <p>Ideal: Narrow intervals that still achieve 95% coverage</p>
                    </div>
                </div>

                <h4>Using the Interactive Charts</h4>

                <h5>1. Coverage by Dataset</h5>
                <p>Shows which datasets are easier/harder for UQ</p>
                <ul>
                    <li>Line dataset: easiest (94% avg)</li>
                    <li>Lennard-Jones: hardest (81% avg)</li>
                    <li>Compare Hat Matrix vs Bayesian side-by-side</li>
                </ul>

                <h5>2. Coverage vs Noise Level</h5>
                <p>Shows how noise affects coverage</p>
                <ul>
                    <li>Counter-intuitive: Coverage slightly improves with more noise!</li>
                    <li>Suggests methods are slightly underconfident at low noise</li>
                </ul>

                <h5>3. Heatmaps</h5>
                <p>Quick visual comparison across all conditions</p>
                <ul>
                    <li>Green = good coverage (near 0.95)</li>
                    <li>Red = poor coverage</li>
                    <li>Look for patterns: which combinations work well?</li>
                </ul>

                <h5>4. RMSE vs Coverage Scatter</h5>
                <p>Shows the accuracy-coverage tradeoff</p>
                <ul>
                    <li>Ideal: High coverage (x-axis near 0.95) AND low RMSE (y-axis near 0)</li>
                    <li>Bubble size = interval width</li>
                    <li>Look for points in bottom-right quadrant</li>
                </ul>

                <h5>5. Regional Coverage</h5>
                <p>Tests interpolation vs extrapolation</p>
                <ul>
                    <li>Interpolation: predicting in gaps within training data</li>
                    <li>Extrapolation: predicting beyond training domain</li>
                    <li>Both methods degrade in high extrapolation region</li>
                </ul>

                <h4>What Makes a "Good" Result?</h4>
                <div class="info-box success">
                    <h5>Ideal UQ Method:</h5>
                    <ol>
                        <li><strong>Coverage ≈ 95%:</strong> Well-calibrated</li>
                        <li><strong>Low RMSE:</strong> Accurate point predictions</li>
                        <li><strong>Narrow intervals:</strong> Precise (but not too narrow!)</li>
                        <li><strong>Consistent across conditions:</strong> Robust</li>
                        <li><strong>Similar coverage in all regions:</strong> Reliable everywhere</li>
                    </ol>
                </div>

                <h4>Common Patterns to Look For</h4>
                <ul>
                    <li><strong>Undercoverage (&lt;90%):</strong> Intervals too narrow → dangerous for decision-making</li>
                    <li><strong>Overcoverage (&gt;98%):</strong> Intervals too wide → not useful</li>
                    <li><strong>Method struggles on specific dataset:</strong> May indicate poor model assumptions</li>
                    <li><strong>Heteroskedastic performs better:</strong> Surprising finding, needs investigation</li>
                </ul>

                <div class="example-box">
                    <h4>Try This Exercise:</h4>
                    <ol>
                        <li>Filter table to "Polynomial" dataset</li>
                        <li>Filter to "Homoskedastic" noise</li>
                        <li>Filter to "1%" noise level</li>
                        <li>Compare HatMatrix (90.6%) vs Bayesian (57.8%)</li>
                        <li>Click both rows to see the fits</li>
                        <li>Notice Bayesian has way too narrow intervals!</li>
                    </ol>
                </div>
            </div>
        </div>
"""

# Add CSS for the guide section
guide_css = """
        .guide-nav {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .guide-btn {
            background: #f0f0f0;
            border: 2px solid #ddd;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.3s;
        }

        .guide-btn:hover {
            background: #e0e0f0;
            border-color: #667eea;
        }

        .guide-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .guide-content {
            display: none;
        }

        .guide-content.active {
            display: block;
        }

        .info-box {
            background: #f0f7ff;
            border-left: 4px solid #2196f3;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 4px;
        }

        .info-box.success {
            background: #f0fff4;
            border-color: #4caf50;
        }

        .info-box.warning {
            background: #fff8e1;
            border-color: #ff9800;
        }

        .info-box.info {
            background: #e3f2fd;
            border-color: #2196f3;
        }

        .comparison-grid, .factor-grid, .interpretation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .dataset-card, .method-card, .factor-card {
            border: 2px solid #e0e0e0;
            padding: 1.5rem;
            border-radius: 8px;
            background: #fafafa;
        }

        .dataset-card h4, .method-card h4 {
            color: #667eea;
            margin-bottom: 1rem;
        }

        .split-diagram {
            display: flex;
            gap: 0.5rem;
            margin: 1.5rem 0;
        }

        .split-section {
            flex: 1;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }

        .split-section.train {
            background: #e3f2fd;
        }

        .split-section.gap {
            background: #fff8e1;
        }

        .split-section.extrap {
            background: #ffebee;
        }

        .split-section.test {
            background: #f3e5f5;
        }

        .calculation-box {
            background: #667eea;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
            font-size: 1.1rem;
            margin: 1rem 0;
        }

        .example-box {
            background: #f5f5f5;
            border: 2px dashed #999;
            padding: 1.5rem;
            border-radius: 6px;
            margin: 1.5rem 0;
        }

        .pros, .cons {
            margin: 0.5rem 0;
        }

        .pros li {
            color: #2e7d32;
        }

        .cons li {
            color: #c62828;
        }

        .good {
            color: #4caf50;
            font-weight: bold;
        }

        .warning {
            color: #ff9800;
            font-weight: bold;
        }

        .poor {
            color: #f44336;
            font-weight: bold;
        }
"""

# Add JavaScript for tab switching
guide_js = """
        function showGuideTab(tabName) {
            // Hide all tabs
            var contents = document.getElementsByClassName('guide-content');
            for (var i = 0; i < contents.length; i++) {
                contents[i].classList.remove('active');
            }

            // Remove active from all buttons
            var buttons = document.getElementsByClassName('guide-btn');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }

            // Show selected tab
            document.getElementById('guide-' + tabName).classList.add('active');
            event.target.classList.add('active');
        }
"""

# Insert guide section after summary stats
html_content = html_content.replace(
    '        <!-- Interactive Results Table -->',
    guide_section + '\n        <!-- Interactive Results Table -->'
)

# Add CSS
html_content = html_content.replace(
    '        .stat-card .value {',
    guide_css + '\n        .stat-card .value {'
)

# Add JavaScript
html_content = html_content.replace(
    '    <script>',
    '    <script>\n' + guide_js
)

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Guide section added to: {dashboard_path}")
print("\nThe dashboard now includes:")
print("  - What is Coverage? (with examples)")
print("  - Study Design explanation")
print("  - Dataset descriptions")
print("  - UQ Methods explained")
print("  - How to Interpret results")
print("\nRefresh your browser to see the new guide!")
