# AutoCut V2 - Development Environment Setup Commands

This document provides the exact commands and configurations for setting up a modern Python development environment optimized for large refactoring projects.

## Quick Start

### Automated Setup (Recommended)

```bash
# Make setup script executable and run it
chmod +x setup-dev-env.sh
./setup-dev-env.sh
```

### Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# 2. Install development dependencies
pip install -e ".[dev]"

# 3. Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# 4. Verify setup
make validate
```

## Installation Commands

### Core Development Tools

```bash
# Install individual tools (if not using pyproject.toml)
pip install ruff>=0.8.0              # Fast linter and formatter
pip install mypy>=1.17.0             # Static type checking  
pip install bandit[toml]>=1.8.0      # Security vulnerability scanning
pip install pytest>=7.0.0            # Testing framework
pip install pytest-cov>=4.0.0        # Coverage plugin
pip install pre-commit>=3.0.0        # Git hooks management
```

### Additional Development Dependencies

```bash
# Testing plugins
pip install pytest-xdist>=3.0.0      # Parallel test execution
pip install pytest-benchmark>=4.0.0   # Benchmarking
pip install pytest-timeout>=2.1.0     # Test timeouts
pip install pytest-mock>=3.10.0      # Mock fixtures
pip install pytest-randomly>=3.12.0   # Randomize test order

# Type stubs
pip install types-psutil>=5.9.0      # psutil type stubs
pip install types-requests>=2.28.0    # requests type stubs
pip install types-setuptools>=65.0.0  # setuptools type stubs

# Additional tools  
pip install tox>=4.0.0               # Multi-environment testing
pip install interrogate>=1.5.0        # Docstring coverage
pip install coverage-conditional-plugin>=0.9.0  # Conditional coverage
```

## Tool Usage Commands

### Ruff (Linting and Formatting)

```bash
# Lint with auto-fixes
ruff check --fix .

# Format code
ruff format .

# Check only (no fixes) - for CI
ruff check .

# Show statistics
ruff check . --statistics

# Check specific files
ruff check src/video_analyzer.py

# Show rule explanations
ruff rule E501
```

### MyPy (Type Checking)

```bash
# Type check main modules
mypy src/ autocut.py

# Type check with configuration file
mypy --config-file=pyproject.toml src/

# Type check specific file
mypy src/video_analyzer.py

# Show error codes
mypy --show-error-codes src/

# Install missing stubs
mypy --install-types src/

# Generate MyPy cache statistics
mypy --cache-dir=.mypy_cache --show-cache-stats src/
```

### Bandit (Security Scanning)

```bash
# Security scan with TOML config
bandit -c pyproject.toml -r .

# Generate JSON report
bandit -c pyproject.toml -r . -f json -o security-report.json

# High confidence issues only
bandit -c pyproject.toml -r . -i

# Scan specific directories
bandit -c pyproject.toml -r src/

# Show available tests
bandit --list-tests

# Explain specific test
bandit -t B101 --help
```

### Pytest (Testing)

```bash
# Run all tests with coverage
pytest

# Run fast tests only (excluding slow markers)
pytest -m "not slow"

# Run with parallel execution
pytest -n auto

# Run specific test file
pytest tests/unit/test_video_analyzer.py

# Run with detailed coverage
pytest --cov=src --cov-report=term-missing --cov-report=html

# Run with multiple output formats
pytest --cov-report=html --cov-report=xml --cov-report=lcov

# Run with test selection
pytest -k "test_audio"

# Run with maximum failures
pytest --maxfail=5

# Show test durations
pytest --durations=10

# Run tests with markers
pytest -m integration
pytest -m "unit and not slow"

# Debug mode (no capture)
pytest -s

# Verbose output
pytest -v

# Very verbose with coverage
pytest -vv --cov=src --cov-report=term-missing
```

### Coverage Reporting

```bash
# Generate HTML coverage report
pytest --cov-report=html
open htmlcov/index.html  # View in browser

# Terminal coverage report
pytest --cov-report=term-missing

# XML coverage (for CI/CD)
pytest --cov-report=xml

# LCOV format (for VS Code)
pytest --cov-report=lcov

# JSON format
pytest --cov-report=json

# Multiple formats simultaneously
pytest --cov-report=html --cov-report=xml --cov-report=term-missing

# Coverage with branch analysis
pytest --cov-branch

# Fail if coverage below threshold
pytest --cov-fail-under=80
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg  
pre-commit install --hook-type pre-push

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff
pre-commit run mypy

# Update hook versions
pre-commit autoupdate

# Clean hook cache
pre-commit clean

# Uninstall hooks
pre-commit uninstall
```

## Make Commands (Recommended)

### Environment Setup

```bash
make setup          # Complete development environment setup
make deps-dev       # Install development dependencies
make deps-test      # Install test dependencies only
make setup-hooks    # Set up pre-commit hooks
make upgrade-deps   # Upgrade development dependencies
```

### Code Quality

```bash
make lint           # Ruff linting with auto-fixes
make lint-check     # Ruff linting without fixes (CI mode)
make format         # Ruff code formatting
make type-check     # MyPy static type checking  
make security       # Bandit security scanning
make quality        # All code quality checks
```

### Testing

```bash
make test           # All tests with coverage
make test-fast      # Fast tests only
make test-unit      # Unit tests
make test-integration # Integration tests  
make test-cov       # Tests with detailed coverage report
make test-slow      # All tests including slow ones
```

### Validation and CI

```bash
make validate       # Complete validation pipeline
make pre-commit     # Run pre-commit hooks manually
make ci-test        # Simulate CI/CD pipeline
```

### Maintenance

```bash
make clean          # Clean caches and generated files
make clean-all      # Clean everything including venv
make info           # Show project information
make help           # Show all available commands
```

## Environment Verification

### Check Installation

```bash
# Check tool versions
ruff --version
mypy --version  
bandit --version
pytest --version
pre-commit --version

# Verify Python version
python --version

# Check virtual environment
echo $VIRTUAL_ENV

# List installed packages
pip list | grep -E "(ruff|mypy|bandit|pytest)"
```

### Health Check

```bash
# Quick health check script
./check-dev-env.sh

# Manual health checks
ruff check . --statistics
mypy src/ autocut.py --no-error-summary
bandit -c pyproject.toml -r . -q
pytest -m "not slow" --maxfail=1 -q
```

### Configuration Validation

```bash
# Validate pyproject.toml syntax
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Check pre-commit config
pre-commit validate-config

# Test MyPy configuration
mypy --show-config

# Test Ruff configuration  
ruff check --show-settings .
```

## Troubleshooting Commands

### Clear Caches

```bash
# Clear all tool caches
rm -rf .mypy_cache/
rm -rf .pytest_cache/  
rm -rf .ruff_cache/
rm -rf __pycache__/

# Or use make command
make clean
```

### Reset Environment

```bash
# Complete environment reset
make clean-all
./setup-dev-env.sh
```

### Debug Tool Issues

```bash
# MyPy debugging
mypy --show-traceback src/
mypy --pdb src/

# Pytest debugging
pytest --pdb
pytest --lf  # Last failed tests
pytest --ff  # Failed first

# Ruff debugging
ruff check --show-files .
ruff check --explain E501
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Quality Checks
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run validation
      run: make validate
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit CI

```yaml
# .github/workflows/pre-commit.yml
name: pre-commit
on:
  pull_request:
  push:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
    - uses: pre-commit/action@v3.0.1
```

This comprehensive setup provides a modern, production-ready Python development environment optimized for large refactoring projects with aggressive code quality standards.