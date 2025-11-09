.PHONY: help dashboard clean clean-results clean-all
.PHONY: benchmark-linear benchmark-nonlinear benchmark-datadriven benchmark-all
.PHONY: viz-linear viz-nonlinear viz-datadriven viz-all
.PHONY: test install

# Default target
help:
	@echo "UQ Encyclopedia - Available Make Targets"
	@echo "=========================================="
	@echo ""
	@echo "Dashboard & Viewing:"
	@echo "  make dashboard          - Launch the interactive dashboard in browser"
	@echo ""
	@echo "Running Benchmarks:"
	@echo "  make benchmark-all      - Run all benchmarks (linear + nonlinear + data-driven)"
	@echo "  make benchmark-linear   - Run linear models benchmark"
	@echo "  make benchmark-nonlinear - Run nonlinear models benchmark"
	@echo "  make benchmark-datadriven - Run data-driven models benchmark"
	@echo ""
	@echo "Generating Visualizations:"
	@echo "  make viz-all            - Generate all visualizations"
	@echo "  make viz-linear         - Generate linear model visualizations"
	@echo "  make viz-nonlinear      - Generate nonlinear model visualizations"
	@echo "  make viz-datadriven     - Generate data-driven model visualizations"
	@echo ""
	@echo "Setup & Maintenance:"
	@echo "  make install            - Install Python dependencies"
	@echo "  make test               - Run test suite"
	@echo "  make clean              - Clean generated visualizations"
	@echo "  make clean-results      - Clean benchmark results (keep dashboard)"
	@echo "  make clean-all          - Clean everything (results + visualizations)"
	@echo ""
	@echo "Project Status:"
	@echo "  make status             - Show project status and disk usage"

# Launch dashboard in default browser
dashboard:
	@echo "Launching UQ Encyclopedia Dashboard..."
	@if [ -f reports/dashboard/dashboard.html ]; then \
		open reports/dashboard/dashboard.html; \
		echo "✓ Dashboard opened in browser"; \
	else \
		echo "✗ Dashboard not found. Generate results first with 'make benchmark-all'"; \
		exit 1; \
	fi

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Run benchmarks
benchmark-linear:
	@echo "Running linear models benchmark..."
	python run_linear_benchmark.py
	@echo "✓ Linear benchmark complete"

benchmark-nonlinear:
	@echo "Running nonlinear models benchmark..."
	python run_nonlinear_benchmark.py
	@echo "✓ Nonlinear benchmark complete"

benchmark-datadriven:
	@echo "Running data-driven models benchmark..."
	@echo "Training GP models..."
	python train_gp_models.py
	@echo "Training NNGMM models..."
	python train_nngmm_models.py
	@echo "Training NNBR models..."
	python train_nnbr_models.py
	@echo "✓ Data-driven benchmark complete"

benchmark-all: benchmark-linear benchmark-nonlinear benchmark-datadriven
	@echo ""
	@echo "=========================================="
	@echo "✓ All benchmarks complete!"
	@echo "=========================================="
	@echo "Results saved to:"
	@echo "  - results/linear_fits/"
	@echo "  - results/nonlinear_fits/"
	@echo "  - results/gp_fits/"
	@echo "  - results/nngmm_fits/"
	@echo "  - results/nnbr_fits/"
	@echo ""
	@echo "Next: Run 'make viz-all' to generate visualizations"

# Generate visualizations
viz-linear:
	@echo "Generating linear model visualizations..."
	python generate_linear_visualizations.py
	python generate_linear_plot_pngs.py
	@echo "✓ Linear visualizations complete"

viz-nonlinear:
	@echo "Generating nonlinear model visualizations..."
	python generate_nonlinear_fits_plotly.py
	python generate_nonlinear_tab.py
	@echo "✓ Nonlinear visualizations complete"

viz-datadriven:
	@echo "Generating data-driven model visualizations..."
	python generate_gp_fits_plotly.py
	python generate_nngmm_fits_plotly.py
	python generate_nnbr_fits_plotly.py
	python rebuild_datadriven_table.py
	@echo "✓ Data-driven visualizations complete"

viz-all: viz-linear viz-nonlinear viz-datadriven
	@echo ""
	@echo "=========================================="
	@echo "✓ All visualizations generated!"
	@echo "=========================================="
	@echo "Visualizations saved to:"
	@echo "  - results/figures/linear_fits_html/"
	@echo "  - results/figures/nonlinear_fits_html/"
	@echo "  - results/figures/gp_fits_html/"
	@echo "  - results/figures/nngmm_fits_html/"
	@echo "  - results/figures/nnbr_fits_html/"
	@echo "  - reports/dashboard/"
	@echo ""
	@echo "Launch dashboard: make dashboard"

# Complete workflow
all: benchmark-all viz-all
	@echo ""
	@echo "=========================================="
	@echo "✓ Complete workflow finished!"
	@echo "=========================================="
	@echo ""
	@echo "Launch dashboard with: make dashboard"

# Test suite
test:
	@echo "Running test suite..."
	@if [ -d tests ]; then \
		python -m pytest tests/ -v; \
	else \
		echo "No tests directory found"; \
	fi

# Cleaning targets
clean:
	@echo "Cleaning generated visualizations..."
	@rm -rf results/figures/
	@rm -rf reports/dashboard/*.html
	@echo "✓ Visualizations cleaned"

clean-results:
	@echo "Cleaning benchmark results..."
	@rm -rf results/linear_fits/
	@rm -rf results/nonlinear_fits/
	@rm -rf results/gp_fits/
	@rm -rf results/nngmm_fits/
	@rm -rf results/nnbr_fits/
	@echo "✓ Results cleaned"

clean-all: clean clean-results
	@echo "Cleaning Python cache..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ All generated files cleaned"

# Project status
status:
	@echo "UQ Encyclopedia - Project Status"
	@echo "================================="
	@echo ""
	@echo "Git Status:"
	@git status --short || echo "Not a git repository"
	@echo ""
	@echo "Disk Usage:"
	@du -sh results/ 2>/dev/null || echo "  results/: Not found"
	@du -sh reports/ 2>/dev/null || echo "  reports/: Not found"
	@du -sh data/ 2>/dev/null || echo "  data/: Not found"
	@echo ""
	@echo "Result Files:"
	@echo -n "  Linear results: "
	@find results/linear_fits -name "*.json" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo -n "  Nonlinear results: "
	@find results/nonlinear_fits -name "*.json" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo -n "  GP results: "
	@find results/gp_fits -name "*.json" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo -n "  NNGMM results: "
	@find results/nngmm_fits -name "*.json" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo -n "  NNBR results: "
	@find results/nnbr_fits -name "*.json" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo ""
	@echo "Visualizations:"
	@echo -n "  HTML files: "
	@find results/figures -name "*.html" 2>/dev/null | wc -l | xargs echo || echo "0"
	@echo -n "  Dashboard: "
	@if [ -f reports/dashboard/dashboard.html ]; then \
		ls -lh reports/dashboard/dashboard.html | awk '{print $$5}'; \
	else \
		echo "Not found"; \
	fi

# Quick start guide
quickstart:
	@echo "UQ Encyclopedia - Quick Start"
	@echo "=============================="
	@echo ""
	@echo "1. Install dependencies:"
	@echo "   make install"
	@echo ""
	@echo "2. Run all benchmarks (takes ~2-4 hours):"
	@echo "   make benchmark-all"
	@echo ""
	@echo "3. Generate all visualizations:"
	@echo "   make viz-all"
	@echo ""
	@echo "4. Launch dashboard:"
	@echo "   make dashboard"
	@echo ""
	@echo "Or run everything at once:"
	@echo "   make all"
	@echo ""
	@echo "For help: make help"
