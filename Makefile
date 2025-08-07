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

.PHONY: help setup test test-unit test-integration test-performance clean lint format install deps demo

# Default target
help: ## Show this help message
	@echo "$(BOLD)AutoCut V2 - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Set up development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	python3 -m venv env
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install -r requirements.txt
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Activate with: source env/bin/activate$(NC)"

install: deps ## Install dependencies (alias for deps)

deps: ## Install/update dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(VENV_ACTIVATE) && pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed!$(NC)"

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
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) -m "hardware or gpu"

test-media: ## Run tests that require media files
	@echo "$(YELLOW)Running media-dependent tests...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) $(PYTEST_ARGS) -m "media_required"

# Application commands  
demo: ## Run AutoCut demo (main entry point)
	@echo "$(YELLOW)Running AutoCut demo...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) test_autocut_demo.py

demo-quick: ## Run quick AutoCut demo with limited files
	@echo "$(YELLOW)Running quick AutoCut demo...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) test_autocut_demo.py --videos 3

validate: ## Validate iPhone H.265 compatibility for test files
	@echo "$(YELLOW)Validating iPhone H.265 compatibility...$(NC)"
	@echo "$(BLUE)iPhone H.265 validation temporarily disabled - use pytest for validation tests$(NC)"

benchmark: ## Run system performance benchmark
	@echo "$(YELLOW)Running system benchmark...$(NC)"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.hardware.detection import HardwareDetector; detector = HardwareDetector(); settings = detector.detect_optimal_settings('fast'); print('Encoder:', settings.get('encoder_type', 'Unknown')); print('Settings detected successfully')"

# Development commands
lint: ## Run linting (when available)
	@echo "$(YELLOW)Running code linting...$(NC)"
	@echo "$(BLUE)Linting setup not yet implemented$(NC)"

format: ## Format code (when available) 
	@echo "$(YELLOW)Formatting code...$(NC)"
	@echo "$(BLUE)Code formatting setup not yet implemented$(NC)"

# Cleanup commands
clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f output/*.mp4 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(YELLOW)Cleaning everything including virtual environment...$(NC)"
	rm -rf env/
	@echo "$(GREEN)Full cleanup completed!$(NC)"

# Git helpers
commit-test: test-quick ## Run quick tests before commit
	@echo "$(GREEN)Quick tests passed - ready to commit!$(NC)"

commit-full: test ## Run all tests before commit
	@echo "$(GREEN)All tests passed - ready to commit!$(NC)"

# Information commands
info: ## Show project information
	@echo "$(BOLD)AutoCut V2 - Project Information$(NC)"
	@echo "$(BLUE)Status:$(NC) Week 2 Testing Framework - IMPLEMENTED"
	@echo "$(BLUE)Python:$(NC) $(shell python3 --version 2>/dev/null || echo 'Not found')"
	@echo "$(BLUE)Virtual Env:$(NC) $(shell [ -d env ] && echo 'Present' || echo 'Missing - run make setup')"
	@echo "$(BLUE)Test Files:$(NC) $(shell find test_media -name '*.mov' -o -name '*.mp4' 2>/dev/null | wc -l) video(s), $(shell find test_media -name '*.mp3' -o -name '*.wav' 2>/dev/null | wc -l) audio"
	@echo "$(BLUE)Architecture:$(NC) Modular (Week 1 refactoring complete)"
	@echo "$(BLUE)Testing:$(NC) pytest with $(shell find tests -name 'test_*.py' 2>/dev/null | wc -l) test files"

status: info ## Show project status (alias for info)

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

dev-integration: ## Development integration test
	@echo "$(YELLOW)Development integration test...$(NC)"
	$(VENV_ACTIVATE) && $(PYTEST) tests/integration/test_full_pipeline.py::TestFullPipeline::test_complete_autocut_pipeline -v -s
	@echo "$(GREEN)Integration test completed!$(NC)"

# Quick reference
usage: help ## Show usage (alias for help)

# Version and release helpers  
version: ## Show current version info
	@echo "AutoCut V2 - Refactoring Phase"
	@echo "Week 2: Testing Framework - COMPLETE"
	@echo "Architecture: Modular (src/video/, src/hardware/, src/core/)"
	@echo "Testing: pytest framework with comprehensive test coverage"