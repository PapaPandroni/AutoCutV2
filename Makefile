# AutoCut V2 - Development Makefile
# Provides convenient commands for testing, development, and CI/CD

# Variables
PYTHON = python3
VENV_ACTIVATE = . env/bin/activate
PYTEST = pytest
PYTEST_ARGS = -v --tb=short

# Use bash as shell
SHELL := /bin/bash

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
BOLD = \033[1m
NC = \033[0m # No Color

.PHONY: help setup test test-unit test-integration test-performance clean lint format install deps demo quality type-check security validate pre-commit setup-hooks

# Default target
help: ## Show this help message
	@echo "$(BOLD)AutoCut V2 - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Set up complete development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	python3 -m venv env
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install -e .
	$(VENV_ACTIVATE) && pip install -e ".[dev]"
	@$(MAKE) setup-hooks
	@echo "$(GREEN)$(BOLD)Development environment ready!$(NC)"
	@echo "$(YELLOW)Activate with: source env/bin/activate$(NC)"
	@echo "$(BLUE)Run 'make validate' to check everything is working$(NC)"


deps: ## Install production dependencies only
	@echo "$(YELLOW)Installing production dependencies...$(NC)"
	$(VENV_ACTIVATE) && pip install -e .
	@echo "$(GREEN)Production dependencies installed!$(NC)"



upgrade-deps: ## Upgrade all development dependencies
	@echo "$(YELLOW)Upgrading development dependencies...$(NC)"
	$(VENV_ACTIVATE) && pip install --upgrade -e ".[dev]"
	@echo "$(GREEN)Development dependencies upgraded!$(NC)"

# Testing commands
test: ## Run all tests
	@echo "$(YELLOW)Running all tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS)

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) tests/unit/ -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) tests/integration/ -m "not slow"

test-performance: ## Run performance tests only
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) tests/performance/

test-quick: ## Run quick tests (no slow tests)
	@echo "$(YELLOW)Running quick tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) -m "not slow and not integration"

test-slow: ## Run all tests including slow ones
	@echo "$(YELLOW)Running all tests including slow ones...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) --runslow

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) --cov=src --cov-report=term-missing --cov-report=html

test-hardware: ## Run hardware-specific tests  
	@echo "$(YELLOW)Running hardware tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) -k "hardware or gpu or benchmark" --ignore=tests/integration/ --ignore=tests/cli/

test-media: ## Run tests that require media files
	@echo "$(YELLOW)Running media-dependent tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) tests/integration/test_full_pipeline.py tests/reliability/ -k "not test_hardware"

# Application commands - NEW CLI Interface
demo: ## Run AutoCut demo using new CLI (main entry point)
	@echo "$(YELLOW)Running AutoCut demo with new CLI...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) autocut.py demo --pattern balanced

demo-quick: ## Run quick AutoCut demo with new CLI
	@echo "$(YELLOW)Running quick AutoCut demo with new CLI...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) autocut.py demo --quick --pattern balanced



validate-video: ## Validate video compatibility using new CLI
	@echo "$(YELLOW)Testing video validation with new CLI...$(NC)"
	@if [ -n "$(VIDEO)" ]; then \
		$(VENV_ACTIVATE) && $(PYTHON) autocut.py validate "$(VIDEO)" --detailed; \
	else \
		echo "$(BLUE)Usage: make validate-video VIDEO=path/to/video.mp4$(NC)"; \
		echo "$(BLUE)Or use: python autocut.py validate path/to/video.mp4$(NC)"; \
	fi

benchmark: ## Run system performance benchmark using new CLI
	@echo "$(YELLOW)Running system benchmark with new CLI...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) autocut.py benchmark --detailed


# CLI help and information
cli-help: ## Show AutoCut CLI help
	@echo "$(YELLOW)AutoCut CLI Help:$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) autocut.py --help

cli-process-help: ## Show help for process command
	@echo "$(YELLOW)AutoCut Process Command Help:$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) autocut.py process --help

# Code Quality Commands
lint: ## Run Ruff linter with auto-fixes
	@echo "$(YELLOW)Running Ruff linter...$(NC)"
	$(VENV_ACTIVATE) && ruff check --fix .
	@echo "$(GREEN)Linting completed!$(NC)"

lint-check: ## Run Ruff linter without fixes (CI mode)
	@echo "$(YELLOW)Running Ruff linter (check only)...$(NC)"
	$(VENV_ACTIVATE) && ruff check .

format: ## Run Ruff formatter
	@echo "$(YELLOW)Running Ruff formatter...$(NC)"
	$(VENV_ACTIVATE) && ruff format .
	@echo "$(GREEN)Code formatting completed!$(NC)"

type-check: ## Run MyPy static type checking
	@echo "$(YELLOW)Running MyPy type checking...$(NC)"
	$(VENV_ACTIVATE) && mypy src/ autocut.py
	@echo "$(GREEN)Type checking completed!$(NC)"

security: ## Run Bandit security scanner
	@echo "$(YELLOW)Running Bandit security scanner...$(NC)"
	$(VENV_ACTIVATE) && bandit -c pyproject.toml -r .
	@echo "$(GREEN)Security scan completed!$(NC)"

quality: lint format type-check security ## Run all code quality checks
	@echo "$(GREEN)$(BOLD)✓ All code quality checks passed!$(NC)"

# Pre-commit and Git Integration
setup-hooks: ## Set up pre-commit hooks (requires pre-commit package)
	@echo "$(YELLOW)Setting up pre-commit hooks...$(NC)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		$(VENV_ACTIVATE) && pre-commit install && pre-commit install --hook-type commit-msg; \
		echo "$(GREEN)Pre-commit hooks installed!$(NC)"; \
	else \
		echo "$(RED)pre-commit not found. Install with: pip install pre-commit$(NC)"; \
	fi

pre-commit: ## Run pre-commit hooks manually (requires pre-commit package)
	@echo "$(YELLOW)Running pre-commit hooks...$(NC)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		$(VENV_ACTIVATE) && pre-commit run --all-files; \
	else \
		echo "$(RED)pre-commit not found. Running basic checks instead...$(NC)"; \
		$(MAKE) lint format type-check security; \
	fi

validate: ## Validate all code (lint, type-check, security, test)
	@echo "$(BOLD)$(YELLOW)Running comprehensive validation...$(NC)"
	@echo "$(BLUE)1/4 Linting...$(NC)"
	@$(MAKE) lint-check
	@echo "$(BLUE)2/4 Type checking...$(NC)"
	@$(MAKE) type-check  
	@echo "$(BLUE)3/4 Security scanning...$(NC)"
	@$(MAKE) security
	@echo "$(BLUE)4/4 Testing...$(NC)"
	@$(MAKE) test-quick
	@echo "$(GREEN)$(BOLD)✅ All validation checks passed!$(NC)"

# Cleanup commands
clean: ## Clean up generated files and caches
	@echo "$(YELLOW)Cleaning up generated files and caches...$(NC)"
	find . -path ./env -prune -o -type f -name "*.pyc" -print0 | xargs -0 rm -f
	find . -path ./env -prune -o -type d -name "__pycache__" -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf coverage.lcov
	rm -rf coverage.json
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -f output/*.mp4 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(YELLOW)Cleaning everything including virtual environment...$(NC)"
	rm -rf env/
	@echo "$(GREEN)Full cleanup completed!$(NC)"

# Git helpers - simplified  
commit-test: test-quick ## Run quick tests before commit
	@echo "$(GREEN)Quick tests passed - ready to commit!$(NC)"

# Information commands
info: ## Show project information
	@echo "$(BOLD)AutoCut V2 - Project Information$(NC)"
	@echo "$(BLUE)Status:$(NC) Production-Ready Core Pipeline (January 2025)"
	@echo "$(BLUE)Python:$(NC) $(shell python3 --version 2>/dev/null || echo 'Not found')"
	@echo "$(BLUE)Virtual Env:$(NC) $(shell [ -d env ] && echo 'Present' || echo 'Missing - run make setup')"
	@echo "$(BLUE)CLI Interface:$(NC) autocut.py (Complete)"
	@echo "$(BLUE)Test Files:$(NC) $(shell find test_media -name '*.mov' -o -name '*.mp4' 2>/dev/null | wc -l) video(s), $(shell find test_media -name '*.mp3' -o -name '*.wav' 2>/dev/null | wc -l) audio"
	@echo "$(BLUE)Architecture:$(NC) Modular, Memory-Safe, Cross-Platform"
	@echo "$(BLUE)Testing:$(NC) pytest with $(shell find tests -name 'test_*.py' 2>/dev/null | wc -l) test files"
	@echo "$(BLUE)Next Phase:$(NC) GUI Development (Step 6)"


# CI/CD simulation
ci-test: ## Simulate CI/CD testing pipeline
	@echo "$(BOLD)$(YELLOW)Simulating CI/CD Pipeline...$(NC)"
	@echo "$(BLUE)Step 1:$(NC) Setup check"
	@[ -d env ] || (echo "$(RED)Virtual environment missing!$(NC)" && exit 1)
	@echo "$(BLUE)Step 2:$(NC) Dependency check"
	$(VENV_ACTIVATE) && python -c "import pytest; import src" || (echo "$(RED)Dependencies missing!$(NC)" && exit 1)
	@echo "$(BLUE)Step 3:$(NC) Unit tests"
	$(VENV_ACTIVATE) && $(PYTEST) tests/unit/ -v --tb=short -x
	@echo "$(BLUE)Step 4:$(NC) Integration tests"
	$(VENV_ACTIVATE) && $(PYTEST) tests/integration/ -v --tb=short -x -m "not slow"
	@echo "$(GREEN)$(BOLD)✓ CI/CD Pipeline simulation completed successfully!$(NC)"

# Development workflow helpers
dev-test: ## Quick development test cycle
	@echo "$(YELLOW)Development test cycle...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) tests/unit/ -v --tb=short --maxfail=3
	@echo "$(GREEN)Development tests completed!$(NC)"


# Quick reference

# Version and release helpers  
version: ## Show current version info
	@echo "AutoCut V2 - Production Pipeline Complete"
	@echo "Core Features: Audio/Video Analysis, Beat Sync, Hardware Acceleration" 
	@echo "Architecture: Modular (src/video/, src/hardware/, src/core/)"
	@echo "Status: Ready for GUI Development Phase"