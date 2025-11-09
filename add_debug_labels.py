#!/usr/bin/env python
"""Add visible debug labels to help identify what sections are being displayed."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read the dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_debug')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Add conspicuous labels to each Detailed Results section
# This will help identify which one the user is seeing

# Nonlinear tab
html = html.replace(
    '<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\n\n            <div class="filter-section">\n                <select id="filterDatasetNL"',
    '<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\n            <p style="background: yellow; padding: 10px; font-weight: bold; border: 3px solid red;">⚠️ DEBUG: This is the ONLY Detailed Results table in the Nonlinear tab (line ~2898)</p>\n\n            <div class="filter-section">\n                <select id="filterDatasetNL"'
)

# Data-driven tab
html = html.replace(
    '<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\n\n            <div class="filter-section">\n                <select id="filterDatasetDD"',
    '<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\n            <p style="background: yellow; padding: 10px; font-weight: bold; border: 3px solid red;">⚠️ DEBUG: This is the ONLY Detailed Results table in the Data-Driven tab (line ~3339)</p>\n\n            <div class="filter-section">\n                <select id="filterDatasetDD"'
)

# Also add labels to other major sections
html = html.replace(
    '<h3>📁 Datasets Explained</h3>',
    '<h3>📁 Datasets Explained</h3>\n                <p style="background: lightblue; padding: 5px; font-size: 0.9em;">ℹ️ DEBUG: This is the Datasets Explained section (informational cards, NOT a results table)</p>'
)

html = html.replace(
    '<h3>🎯 UQ Methods Explained</h3>',
    '<h3>🎯 UQ Methods Explained</h3>\n                <p style="background: lightgreen; padding: 5px; font-size: 0.9em;">ℹ️ DEBUG: This is the UQ Methods section (informational cards, NOT a results table)</p>'
)

html = html.replace(
    '<h3>Training Data Visualizations</h3>',
    '<h3>Training Data Visualizations</h3>\n                <p style="background: lightcoral; padding: 5px; font-size: 0.9em;">ℹ️ DEBUG: This is the Training Data Visualizations section (plots, NOT a results table)</p>'
)

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated with debug labels: {dashboard_path}")
print("\nAdded conspicuous colored labels to:")
print("  • Detailed Results sections (YELLOW with RED border)")
print("  • Datasets Explained sections (LIGHT BLUE)")
print("  • UQ Methods sections (LIGHT GREEN)")
print("  • Training Data Visualizations (LIGHT CORAL)")
print("\nPlease open the dashboard now and:")
print("  1. Click on the 'Nonlinear Models' tab")
print("  2. Scroll through and note which sections you see")
print("  3. Take a screenshot showing both 'duplicate tables' you're seeing")
print("  4. Describe what content/color labels appear on each one")
print("\nThis will help identify exactly what's being duplicated!")
