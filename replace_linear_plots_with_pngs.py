#!/usr/bin/env python
"""Replace inline Plotly scripts with PNG images in the linear models tab."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_png_conversion')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Dataset names
datasets = ['Line', 'Polynomial', 'LennardJones', 'Shomate']
noise_models = ['homoskedastic', 'heteroskedastic']
noise_levels = [1, 2, 5, 10]

count = 0

for dataset in datasets:
    for noise_model in noise_models:
        for noise_level in noise_levels:
            # Pattern to match: <div id="plot_XXX"...></div><script>...Plotly.newPlot...  </script>
            plot_id = f"plot_{dataset}_{noise_model}_{noise_level}"

            # This regex matches the entire Plotly div + script block
            # The pattern is: <div id="plot_XXX"...></div> followed by <script>...Plotly.newPlot...</script>
            pattern = (
                rf'<div id="{plot_id}"[^>]*></div>\s*'
                r'<script type="text/javascript">.*?'
                r'Plotly\.newPlot\(.*?\).*?'
                r'</script>'
            )

            # Replacement: simple IMG tag
            png_file = f"linear_plots_png/{dataset}_{noise_model}_{noise_level}_HatMatrix.png"
            replacement = f'<img src="{png_file}" style="width:100%; height:auto; border: 1px solid #ddd; border-radius: 4px;">'

            # Perform replacement
            html_new = re.sub(pattern, replacement, html, flags=re.DOTALL)

            if html_new != html:
                count += 1
                html = html_new
                print(f"  ✓ Replaced {plot_id}")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print(f"✓ Replaced {count} Plotly plots with PNG images")
print("\nRefresh your browser to see the PNG plots!")
