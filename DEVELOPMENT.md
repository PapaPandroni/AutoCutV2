# AutoCut V2 - Modern Python Development Environment

This document describes the complete development environment setup for AutoCut V2, optimized for large refactoring projects with aggressive code quality standards.

## Overview

The development environment includes:

- **Ruff**: Fast Python linter and formatter (replaces Black, isort, flake8, and more)
- **MyPy**: Static type checking with strict mode
- **Bandit**: Security vulnerability scanning
- **Pytest**: Testing framework with comprehensive coverage reporting
- **Pre-commit**: Automated code quality checks on git commits

## Quick Setup

### Automated Setup (Recommended)

```bash
# Clone and navigate to project
cd /path/to/AutoCutV2

# Run automated setup script
./setup-dev-env.sh

# Activate virtual environment
source env/bin/activate

# Verify setup
make validate
```

### Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

## Tool Configurations

All tools are configured in `pyproject.toml` following modern Python best practices:

### Ruff Configuration

Ruff is configured with aggressive settings for large refactoring projects:

- **Target**: Python 3.8+ compatibility
- **Line length**: 88 characters (Black-compatible)
- **Rules**: Comprehensive rule set including pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, and more
- **Auto-fixing**: Enabled for most rules (except unused imports/variables during refactoring)

```bash
# Lint with auto-fix
ruff check --fix .

# Format code
ruff format .

# Check without fixing (CI mode)  
ruff check .
```

### MyPy Configuration

MyPy is configured in strict mode with gradual typing support:

- **Strict mode**: Enabled for maximum type safety
- **Python version**: 3.8
- **Gradual typing**: Legacy modules have relaxed settings during refactoring
- **Third-party stubs**: Configured for external libraries

```bash
# Type check main modules
mypy src/ autocut.py

# Type check specific file
mypy src/video_analyzer.py
```

### Bandit Configuration

Bandit scans for security vulnerabilities:

- **Format**: TOML configuration in `pyproject.toml`
- **Exclusions**: Test directories, media files, and known false positives
- **Focus**: Critical security tests for production code

```bash
# Security scan
bandit -c pyproject.toml -r .

# Generate report
bandit -c pyproject.toml -r . -f json -o security-report.json
```

### Pytest Configuration

Pytest is configured for comprehensive testing:

- **Coverage**: Branch coverage with 80% minimum threshold
- **Parallel execution**: Support for pytest-xdist
- **Multiple report formats**: Terminal, HTML, XML, LCOV
- **Test markers**: Organized by speed, type, and requirements

```bash
# Run all tests with coverage
pytest

# Run fast tests only
pytest -m "not slow"

# Run with parallel execution
pytest -n auto

# Generate HTML coverage report
pytest --cov-report=html
```

## Development Workflow

### Daily Development

```bash
# Activate environment
source env/bin/activate

# Before starting work
make validate

# During development (quick checks)
make quality

# Before committing
make pre-commit
```

### Pre-commit Hooks

The pre-commit hooks automatically run:

1. **Ruff linter** with auto-fixes
2. **Ruff formatter**  
3. **MyPy type checking**
4. **Bandit security scan**
5. **General file hygiene** (trailing whitespace, file endings, etc.)
6. **Fast tests** (non-slow tests only)

### Make Commands

All development tasks are available via Make:

```bash
# Setup and environment
make setup          # Complete development environment setup
make deps-dev        # Install development dependencies  
make setup-hooks     # Set up pre-commit hooks

# Code quality
make lint            # Ruff linting with auto-fixes
make format          # Code formatting
make type-check      # MyPy type checking
make security        # Bandit security scan
make quality         # All quality checks

# Testing
make test            # All tests with coverage
make test-fast       # Fast tests only
make test-unit       # Unit tests
make test-integration # Integration tests
make test-cov        # Tests with detailed coverage

# Validation and CI
make validate        # Complete validation pipeline
make pre-commit      # Run pre-commit hooks manually

# Maintenance
make clean           # Clean caches and generated files
make upgrade-deps    # Upgrade development dependencies
```

## Configuration Files

### pyproject.toml

The main configuration file contains settings for all tools:

```toml
[tool.ruff]
target-version = "py38" 
line-length = 88
# ... comprehensive rule configuration

[tool.mypy]
strict = true
python_version = "3.8"
# ... strict mode with gradual typing

[tool.bandit]
exclude_dirs = [".venv", "tests", "test_media"]
# ... security-focused configuration

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--cov=src", "--cov-report=html"]
# ... comprehensive test configuration

[tool.coverage.run]
source = ["src", "."]
branch = true
fail_under = 80
# ... detailed coverage settings
```

### .pre-commit-config.yaml

Pre-commit hooks configuration:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking  
- **Bandit**: Security scanning
- **Built-in hooks**: File hygiene, syntax checks
- **Testing**: Fast test execution

### Makefile

Development commands and workflow automation with colored output and error handling.

## IDE Integration

### VS Code

Install extensions:
- **Ruff**: Official Ruff extension
- **Python**: Microsoft Python extension  
- **MyPy**: MyPy type checker
- **Coverage Gutters**: Coverage visualization

Configuration in `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./env/bin/python",
  "ruff.configuration": "./pyproject.toml",
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "coverage-gutters.coverageReportFileName": "coverage.xml"
}
```

### Other IDEs

- **PyCharm**: Configure external tools for Ruff, MyPy, and Bandit
- **Vim/Neovim**: Use coc-ruff, coc-pyright, and ale plugins
- **Emacs**: Use flycheck with ruff-format and mypy checkers

## Continuous Integration

The configuration is optimized for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.8'

- name: Install dependencies
  run: |
    pip install -e ".[dev]"

- name: Run validation
  run: make validate
```

## Refactoring Best Practices

This environment is specifically optimized for large refactoring projects:

### Gradual Typing Strategy

1. **Global strict mode**: New code must be fully typed
2. **Legacy module overrides**: Temporarily disable strict checking for legacy modules
3. **Progressive enhancement**: Gradually enable strict typing as modules are refactored

### Code Quality Strategy

1. **Aggressive linting**: Comprehensive rule set catches issues early
2. **Auto-fixing**: Automatically fix formatting and simple issues
3. **Security focus**: Bandit catches security vulnerabilities during refactoring
4. **Test coverage**: Maintain 80% coverage throughout refactoring

### Development Process

1. **Before starting**: Run `make validate` to ensure clean baseline
2. **During development**: Use `make quality` for quick quality checks
3. **Before commits**: Pre-commit hooks automatically validate changes
4. **CI/CD integration**: Automated validation in continuous integration

## Troubleshooting

### Common Issues

**MyPy errors on legacy code:**
```bash
# Temporarily disable strict checking for specific modules
# Add to pyproject.toml:
[[tool.mypy.overrides]]
module = "src.legacy_module"
disallow_untyped_defs = false
```

**Ruff false positives:**
```bash
# Disable specific rules per file
# Add to pyproject.toml:
[tool.ruff.lint.per-file-ignores]
"src/legacy.py" = ["E501", "F401"]
```

**Bandit false positives:**
```bash
# Skip specific tests
# Add to pyproject.toml:
[tool.bandit]
skips = ["B101", "B603"]
```

**Coverage failing:**
```bash
# Check coverage report
pytest --cov-report=term-missing

# Exclude lines from coverage
# Add comments: # pragma: no cover
```

### Performance Optimization

- **Parallel testing**: Use `pytest -n auto` for faster test execution
- **MyPy incremental**: Uses sqlite cache for faster subsequent runs
- **Ruff speed**: Significantly faster than traditional tools (Black, flake8, isort)

## Advanced Usage

### Custom Rule Sets

Create project-specific rule configurations:

```toml
[tool.ruff.lint]
select = ["ALL"]  # Start with all rules
ignore = [
  # Project-specific ignores
  "D100",  # Missing docstring (enable gradually)
  "ANN",   # Type annotations (enable gradually)
]
```

### Coverage Analysis

Generate detailed coverage reports:

```bash
# HTML report with branch coverage
pytest --cov-report=html --cov-branch

# Terminal report showing missing lines
pytest --cov-report=term-missing

# JSON report for programmatic analysis
pytest --cov-report=json
```

### Security Scanning

Advanced Bandit usage:

```bash
# Generate detailed JSON report
bandit -c pyproject.toml -r . -f json -o security-report.json

# Scan specific directories only
bandit -c pyproject.toml -r src/

# High confidence issues only
bandit -c pyproject.toml -r . -i
```

### Performance Profiling

Integration with performance tools:

```bash
# Memory profiling with pytest
pytest --profile

# Code profiling
python -m cProfile -o profile.stats autocut.py demo
```

## Contributing

When contributing to the project:

1. **Setup**: Use `./setup-dev-env.sh` for consistent environment
2. **Validation**: Run `make validate` before submitting PRs
3. **Testing**: Ensure tests pass and coverage is maintained
4. **Code quality**: All quality checks must pass
5. **Security**: No security vulnerabilities in Bandit scan

## Maintenance

### Dependency Updates

```bash
# Update development tools
make upgrade-deps

# Update pre-commit hooks
pre-commit autoupdate

# Verify updates work
make validate
```

### Environment Refresh

```bash
# Clean everything
make clean-all

# Rebuild environment
./setup-dev-env.sh
```

This development environment provides a solid foundation for maintaining high code quality during large refactoring projects while ensuring security and comprehensive testing coverage.