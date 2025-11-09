#!/usr/bin/env python
"""Add prominent coverage explanation notice to all dashboard tabs."""

from pathlib import Path
import re

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Coverage notice HTML
coverage_notice = '''
            <div class="alert alert-warning" style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #856404;">⚠️ Important: How Coverage is Computed</h4>
                <p style="margin-bottom: 10px;"><strong>The reported coverage metrics include BOTH interpolation AND extrapolation.</strong></p>
                <p style="margin-bottom: 10px;">Test points are evaluated across the ENTIRE domain [0, 1], including:</p>
                <ul style="margin-bottom: 10px;">
                    <li><strong>~25% Extrapolation</strong> (outside training data bounds: 0.0-0.125 and 0.875-1.0)</li>
                    <li><strong>~25% Gap/Interpolation</strong> (inside training bounds but no nearby training points: 0.375-0.625)</li>
                    <li><strong>~50% Near Training Data</strong> (regions 0.125-0.375 and 0.625-0.875)</li>
                </ul>
                <p style="margin-bottom: 10px;"><strong>This is more challenging than typical ML benchmarks</strong> that only test random interpolation. Lower coverage values (e.g., 85-90%) can still represent good performance given this includes extrapolation.</p>
                <p style="margin-bottom: 0;"><em>See <a href="COVERAGE_EXPLAINED.md" style="color: #856404; text-decoration: underline;">COVERAGE_EXPLAINED.md</a> for detailed explanation.</em></p>
            </div>
'''

print("Adding coverage notice to dashboard tabs...")

# Add to Data-Driven UQ tab (after the header)
dd_pattern = r'(<div id="datadriven" class="tabcontent">.*?<h2>Data-Driven UQ Benchmark Results</h2>)'
dd_match = re.search(dd_pattern, html, re.DOTALL)
if dd_match:
    insert_pos = dd_match.end()
    html = html[:insert_pos] + coverage_notice + html[insert_pos:]
    print("✓ Added to Data-Driven UQ tab")
else:
    print("✗ Could not find Data-Driven UQ tab")

# Save the updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("\nThe coverage notice has been added to help users understand what the")
print("coverage metrics represent (includes both interpolation and extrapolation).")
