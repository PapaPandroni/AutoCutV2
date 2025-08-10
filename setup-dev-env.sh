#!/bin/bash
# AutoCut V2 - Development Environment Setup Script
# Sets up the complete modern Python development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BOLD}${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..50})${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

print_header "AutoCut V2 - Development Environment Setup"
echo "This script will set up a complete modern Python development environment with:"
echo "  • Ruff (fast linting and formatting)"
echo "  • MyPy (static type checking)"
echo "  • Bandit (security vulnerability scanning)"
echo "  • Pytest (testing framework with coverage)"
echo "  • Pre-commit hooks"
echo ""

# Check Python version
print_header "Checking Python Version"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
    print_error "Python 3.8 or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected (compatible)"

# Create virtual environment
print_header "Setting Up Virtual Environment"
if [ -d "env" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf env
fi

python3 -m venv env
print_success "Virtual environment created"

# Activate virtual environment
source env/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_header "Upgrading Pip"
pip install --upgrade pip
print_success "Pip upgraded to latest version"

# Install development dependencies
print_header "Installing Development Dependencies"
print_status "Installing AutoCut with development extras..."
pip install -e ".[dev]"
print_success "Development dependencies installed"

# Verify installations
print_header "Verifying Tool Installations"

check_tool() {
    local tool=$1
    local cmd=$2
    
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -1)
        print_success "$tool: $version"
        return 0
    else
        print_error "$tool: Not found"
        return 1
    fi
}

TOOLS_OK=true
check_tool "Ruff" "ruff" || TOOLS_OK=false
check_tool "MyPy" "mypy" || TOOLS_OK=false
check_tool "Bandit" "bandit" || TOOLS_OK=false  
check_tool "Pytest" "pytest" || TOOLS_OK=false
check_tool "Pre-commit" "pre-commit" || TOOLS_OK=false

if [ "$TOOLS_OK" = false ]; then
    print_error "Some tools are missing. Please check the installation."
    exit 1
fi

# Set up pre-commit hooks
print_header "Setting Up Pre-commit Hooks"
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    print_success "Pre-commit hooks installed"
else
    print_warning ".pre-commit-config.yaml not found. Skipping pre-commit setup."
fi

# Run initial validation
print_header "Running Initial Validation"
print_status "Running Ruff linter..."
ruff check . --fix || print_warning "Ruff found issues (fixed automatically)"

print_status "Running Ruff formatter..."
ruff format . || print_warning "Ruff formatting had issues"

print_status "Running MyPy type check on main modules..."
mypy --config-file=pyproject.toml src/ autocut.py || print_warning "MyPy found type issues"

print_status "Running Bandit security scan..."
bandit -c pyproject.toml -r . -f json -o bandit-report.json || print_warning "Bandit found security issues"

print_status "Running fast tests..."
pytest -m "not slow" --maxfail=3 -q || print_warning "Some tests failed"

# Display usage information
print_header "Setup Complete!"
echo -e "${GREEN}✅ Development environment is ready!${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source env/bin/activate${NC}"
echo ""
echo "2. Available development commands:"
echo -e "   ${BLUE}make help${NC}                 - Show all available commands"
echo -e "   ${BLUE}make validate${NC}             - Run all quality checks"
echo -e "   ${BLUE}make quality${NC}              - Run linting, formatting, type-check, security"
echo -e "   ${BLUE}make test${NC}                 - Run tests with coverage"
echo -e "   ${BLUE}make pre-commit${NC}           - Run pre-commit hooks manually"
echo ""
echo "3. Tool-specific commands:"
echo -e "   ${BLUE}ruff check --fix .${NC}        - Lint and auto-fix issues"  
echo -e "   ${BLUE}ruff format .${NC}             - Format all Python files"
echo -e "   ${BLUE}mypy src/ autocut.py${NC}      - Type checking"
echo -e "   ${BLUE}bandit -c pyproject.toml -r .${NC} - Security scan"
echo -e "   ${BLUE}pytest${NC}                    - Run tests"
echo ""
echo -e "${BOLD}Configuration files:${NC}"
echo -e "   ${BLUE}pyproject.toml${NC}            - Main configuration (all tools)"
echo -e "   ${BLUE}.pre-commit-config.yaml${NC}   - Pre-commit hooks"
echo -e "   ${BLUE}Makefile${NC}                  - Development commands"
echo ""
echo -e "${BOLD}For large refactoring projects:${NC}"
echo "• Use 'make validate' before commits"
echo "• Run 'make quality' regularly during development"  
echo "• The tools are configured for aggressive code quality checking"
echo "• MyPy strict mode helps catch type issues early"
echo "• Bandit scans for security vulnerabilities"
echo "• Coverage reporting ensures comprehensive testing"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"

# Create a simple status script
cat > check-dev-env.sh << 'EOF'
#!/bin/bash
# Quick development environment status check

echo "AutoCut V2 - Development Environment Status"
echo "==========================================="
echo "Virtual Environment: $([ -n "$VIRTUAL_ENV" ] && echo "Active ($VIRTUAL_ENV)" || echo "Not active")"
echo "Python: $(python --version 2>&1)"
echo "Ruff: $(ruff --version 2>&1 | head -1)"
echo "MyPy: $(mypy --version 2>&1)"
echo "Bandit: $(bandit --version 2>&1 | head -1)"
echo "Pytest: $(pytest --version 2>&1 | head -1)"
echo "Pre-commit: $(pre-commit --version 2>&1)"
echo ""
echo "Quick health check:"
echo "==================="
ruff check . --statistics || echo "❌ Ruff check failed"
mypy --config-file=pyproject.toml src/ autocut.py > /dev/null 2>&1 && echo "✅ MyPy check passed" || echo "❌ MyPy check failed"
bandit -c pyproject.toml -r . -q > /dev/null 2>&1 && echo "✅ Bandit check passed" || echo "❌ Bandit check failed"
pytest -m "not slow" --maxfail=1 -q > /dev/null 2>&1 && echo "✅ Fast tests passed" || echo "❌ Fast tests failed"
EOF

chmod +x check-dev-env.sh
print_success "Created check-dev-env.sh for quick status checks"

# Clean up any temporary files
rm -f bandit-report.json

print_success "Setup completed successfully!"