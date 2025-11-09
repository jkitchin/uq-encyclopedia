#!/usr/bin/env python
"""Fix Plotly plots not rendering in linear tab by adding resize on tab switch."""

from pathlib import Path
import re

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_plotly_fix')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Find and replace the showMainTab function
old_function = '''        function showMainTab(tabName) {
            // Hide all tab contents
            var contents = document.getElementsByClassName('main-tab-content');
            for (var i = 0; i < contents.length; i++) {
                contents[i].classList.remove('active');
            }

            // Remove active class from all tab buttons
            var buttons = document.getElementsByClassName('main-tab-btn');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }

            // Show selected tab content
            document.getElementById('main-tab-' + tabName).classList.add('active');

            // Activate clicked button
            event.target.classList.add('active');
        }'''

new_function = '''        function showMainTab(tabName) {
            // Hide all tab contents
            var contents = document.getElementsByClassName('main-tab-content');
            for (var i = 0; i < contents.length; i++) {
                contents[i].classList.remove('active');
            }

            // Remove active class from all tab buttons
            var buttons = document.getElementsByClassName('main-tab-btn');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }

            // Show selected tab content
            document.getElementById('main-tab-' + tabName).classList.add('active');

            // Activate clicked button
            event.target.classList.add('active');

            // Resize Plotly plots in the newly shown tab (fixes hidden plot issue)
            setTimeout(function() {
                var activeTab = document.getElementById('main-tab-' + tabName);
                var plotlyDivs = activeTab.getElementsByClassName('plotly-graph-div');
                for (var i = 0; i < plotlyDivs.length; i++) {
                    if (plotlyDivs[i].id && window.Plotly) {
                        window.Plotly.Plots.resize(plotlyDivs[i]);
                    }
                }
            }, 100);
        }'''

if old_function in html:
    html = html.replace(old_function, new_function)
    print("✓ Updated showMainTab function to resize Plotly plots on tab switch")
else:
    print("⚠️ Could not find showMainTab function pattern")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("✓ Plotly plots should now render correctly when switching to the linear tab")
print("\nRefresh your browser and click the Linear Models tab to see the plots!")
