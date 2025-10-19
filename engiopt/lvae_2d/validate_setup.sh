#!/bin/bash
# Validate LVAE sweep setup before launching on Euler
# Usage: ./validate_setup.sh

set -e

echo "=========================================="
echo "LVAE Sweep Setup Validation"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Check if we're on Euler
if [[ $(hostname) == *"euler"* ]]; then
    ON_EULER=true
    echo "✓ Running on Euler HPC"
else
    ON_EULER=false
    echo "ℹ Running on local machine (some checks will be skipped)"
fi
echo ""

#-------------------------------------------------------------------------------
# Check files exist
#-------------------------------------------------------------------------------
echo "Checking required files..."

FILES=(
    "engiopt/lvae_2d/lvae_2d.py"
    "engiopt/lvae_2d/aes.py"
    "engiopt/lvae_2d/utils.py"
    "engiopt/lvae_2d/sweep_lvae_2d.yaml"
    "engiopt/lvae_2d/sweep_lvae_2d_bayes.yaml"
    "engiopt/lvae_2d/sweep_lvae_2d_spectral_norm.yaml"
    "engiopt/lvae_2d/euler_sweep.slurm"
    "engiopt/lvae_2d/euler_launch_sweep.sh"
    "engiopt/lvae_2d/euler_sweep_monitor.sh"
    "engiopt/lvae_2d/analyze_sweeps.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
        ((ERRORS++))
    fi
done
echo ""

#-------------------------------------------------------------------------------
# Check scripts are executable
#-------------------------------------------------------------------------------
echo "Checking script permissions..."

EXEC_FILES=(
    "engiopt/lvae_2d/euler_launch_sweep.sh"
    "engiopt/lvae_2d/euler_sweep_monitor.sh"
)

for file in "${EXEC_FILES[@]}"; do
    if [ -x "$file" ]; then
        echo "  ✓ $file is executable"
    else
        echo "  ⚠ $file not executable (run: chmod +x $file)"
        ((WARNINGS++))
    fi
done
echo ""

#-------------------------------------------------------------------------------
# Check Python environment
#-------------------------------------------------------------------------------
echo "Checking Python environment..."

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  ✓ Python found: $PYTHON_VERSION"

    # Check for required packages
    PACKAGES=("torch" "wandb" "tyro" "numpy" "matplotlib")
    for pkg in "${PACKAGES[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            echo "  ✓ $pkg installed"
        else
            echo "  ✗ $pkg NOT installed"
            ((ERRORS++))
        fi
    done
else
    echo "  ✗ Python not found"
    ((ERRORS++))
fi
echo ""

#-------------------------------------------------------------------------------
# Check WandB configuration
#-------------------------------------------------------------------------------
echo "Checking WandB configuration..."

if command -v wandb &> /dev/null; then
    echo "  ✓ wandb CLI found"

    # Check if logged in
    if wandb login --relogin --timeout 1 &>/dev/null; then
        echo "  ✓ WandB logged in"
    else
        echo "  ⚠ WandB not logged in (run: wandb login)"
        ((WARNINGS++))
    fi
else
    echo "  ✗ wandb CLI not found"
    ((ERRORS++))
fi
echo ""

#-------------------------------------------------------------------------------
# Euler-specific checks
#-------------------------------------------------------------------------------
if [ "$ON_EULER" = true ]; then
    echo "Checking Euler HPC configuration..."

    # Check modules
    if command -v module &> /dev/null; then
        echo "  ✓ Module system available"

        # Check if required modules are loaded
        if module list 2>&1 | grep -q "python"; then
            echo "  ✓ Python module loaded"
        else
            echo "  ⚠ Python module not loaded (run: module load python_cuda/3.11.6)"
            ((WARNINGS++))
        fi

        if module list 2>&1 | grep -q "cuda"; then
            echo "  ✓ CUDA module loaded"
        else
            echo "  ⚠ CUDA module not loaded (run: module load cuda/12.4.1)"
            ((WARNINGS++))
        fi
    else
        echo "  ✗ Module system not available"
        ((ERRORS++))
    fi

    # Check SLURM
    if command -v sbatch &> /dev/null; then
        echo "  ✓ SLURM (sbatch) available"
    else
        echo "  ✗ SLURM not available"
        ((ERRORS++))
    fi

    # Check directories
    if [ -d "$SCRATCH" ]; then
        echo "  ✓ \$SCRATCH directory exists: $SCRATCH"
    else
        echo "  ⚠ \$SCRATCH not set or doesn't exist"
        ((WARNINGS++))
    fi

    # Check for logs directory
    if [ ! -d "logs" ]; then
        echo "  ⚠ logs/ directory doesn't exist (will be created automatically)"
        mkdir -p logs
        echo "    Created logs/ directory"
    else
        echo "  ✓ logs/ directory exists"
    fi
else
    echo "Skipping Euler-specific checks (not on Euler)"
fi
echo ""

#-------------------------------------------------------------------------------
# Validate YAML files
#-------------------------------------------------------------------------------
echo "Validating sweep configuration files..."

if command -v python &> /dev/null; then
    for yaml_file in engiopt/lvae_2d/sweep_*.yaml; do
        if python -c "import yaml; yaml.safe_load(open('$yaml_file'))" 2>/dev/null; then
            echo "  ✓ $yaml_file is valid YAML"
        else
            echo "  ✗ $yaml_file has YAML syntax errors"
            ((ERRORS++))
        fi
    done
else
    echo "  ⚠ Cannot validate YAML (Python not available)"
    ((WARNINGS++))
fi
echo ""

#-------------------------------------------------------------------------------
# Check training script
#-------------------------------------------------------------------------------
echo "Testing training script syntax..."

if python -m py_compile engiopt/lvae_2d/lvae_2d.py 2>/dev/null; then
    echo "  ✓ lvae_2d.py compiles successfully"
else
    echo "  ✗ lvae_2d.py has syntax errors"
    ((ERRORS++))
fi

if python -m py_compile engiopt/lvae_2d/aes.py 2>/dev/null; then
    echo "  ✓ aes.py compiles successfully"
else
    echo "  ✗ aes.py has syntax errors"
    ((ERRORS++))
fi
echo ""

#-------------------------------------------------------------------------------
# Check for common issues
#-------------------------------------------------------------------------------
echo "Checking for common configuration issues..."

# Check WANDB_ENTITY in scripts
ENTITY_IN_LAUNCH=$(grep "WANDB_ENTITY=" engiopt/lvae_2d/euler_launch_sweep.sh | head -1 | cut -d'"' -f2)
ENTITY_IN_SLURM=$(grep "WANDB_ENTITY=" engiopt/lvae_2d/euler_sweep.slurm | grep "export" | head -1 | cut -d'{' -f2 | cut -d':' -f1)

if [ -n "$ENTITY_IN_LAUNCH" ]; then
    echo "  ✓ WandB entity set in launch script: $ENTITY_IN_LAUNCH"
else
    echo "  ⚠ WandB entity not found in launch script"
    ((WARNINGS++))
fi

# Check PROJECT_DIR in SLURM script
PROJECT_DIR=$(grep 'PROJECT_DIR=' engiopt/lvae_2d/euler_sweep.slurm | head -1 | cut -d'"' -f2)
if [ -n "$PROJECT_DIR" ]; then
    echo "  ✓ Project directory set in SLURM script: $PROJECT_DIR"
else
    echo "  ⚠ Project directory not found in SLURM script"
    ((WARNINGS++))
fi

# Check if sweep files have reasonable parameter ranges
for sweep_file in engiopt/lvae_2d/sweep_lvae_2d.yaml; do
    if grep -q "plummet_threshold" "$sweep_file" && grep -q "alpha" "$sweep_file"; then
        echo "  ✓ Sweep file contains both plummet and lognorm parameters"
    else
        echo "  ⚠ Sweep file might be missing parameters"
        ((WARNINGS++))
    fi
done
echo ""

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo "=========================================="
echo "Validation Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ All checks passed! Setup is ready."
    echo ""
    echo "Next steps:"
    if [ "$ON_EULER" = true ]; then
        echo "  ./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200"
    else
        echo "  1. Copy code to Euler:"
        echo "     rsync -avz EngiOpt/ euler:~/projects/EngiOpt/"
        echo "  2. SSH to Euler and run:"
        echo "     ./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200"
    fi
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ $WARNINGS warning(s) found. Setup should work but review warnings above."
    echo ""
    echo "You can proceed, but address warnings for best results."
    exit 0
else
    echo "✗ $ERRORS error(s) and $WARNINGS warning(s) found."
    echo ""
    echo "Please fix errors before running sweeps."
    exit 1
fi
