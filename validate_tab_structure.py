#!/usr/bin/env python
"""Validate that tabs are properly closed and not nested."""

from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Find tab boundaries
linear_start = html.find('<div id="main-tab-linear"')
nonlinear_start = html.find('<div id="main-tab-nonlinear"')
datadriven_start = html.find('<div id="main-tab-datadriven"')

print("Tab boundaries:")
print(f"  Linear starts at: {linear_start}")
print(f"  Nonlinear starts at: {nonlinear_start}")
print(f"  Data-driven starts at: {datadriven_start}")

# Extract each tab section
linear_section = html[linear_start:nonlinear_start]
nonlinear_section = html[nonlinear_start:datadriven_start]
datadriven_section = html[datadriven_start:]

# Count divs in each section
def count_divs(section, name):
    opening = section.count('<div')
    closing = section.count('</div>')
    print(f"\n{name}:")
    print(f"  Opening <div> tags: {opening}")
    print(f"  Closing </div> tags: {closing}")
    print(f"  Balance: {opening - closing}")
    return opening - closing

linear_balance = count_divs(linear_section, "Linear tab")
nonlinear_balance = count_divs(nonlinear_section, "Nonlinear tab")

# Check if linear tab is properly closed
if linear_balance != 1:  # Should be 1 because it starts with <div id="main-tab-linear"
    print(f"\n⚠️  WARNING: Linear tab div balance is {linear_balance}, expected 1")
    print("   This means the linear tab may not be properly closed!")

    # Find where the linear tab should close
    lines = linear_section.split('\n')
    last_20_lines = lines[-20:]
    print("\n   Last 20 lines of linear tab section:")
    for i, line in enumerate(last_20_lines, start=len(lines)-20):
        print(f"     {i}: {line[:100]}")
else:
    print("\n✓ Linear tab appears to be properly structured")

if nonlinear_balance != 1:
    print(f"\n⚠️  WARNING: Nonlinear tab div balance is {nonlinear_balance}, expected 1")
else:
    print("✓ Nonlinear tab appears to be properly structured")

# Check what's immediately after the linear tab
after_linear = html[nonlinear_start-200:nonlinear_start]
print("\n200 characters before nonlinear tab starts:")
print(after_linear)
