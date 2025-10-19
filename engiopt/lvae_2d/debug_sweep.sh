#!/bin/bash
# Debug script to check sweep status and find log files
# Usage: ./debug_sweep.sh [job_id]

echo "=========================================="
echo "LVAE Sweep Debugging"
echo "=========================================="
echo ""

# Check for logs directory
echo "1. Checking logs directory..."
if [ -d "logs" ]; then
    echo "   ✓ logs/ directory exists"
    LOG_COUNT=$(ls -1 logs/sweep_*.out 2>/dev/null | wc -l)
    ERR_COUNT=$(ls -1 logs/sweep_*.err 2>/dev/null | wc -l)
    echo "   Found $LOG_COUNT .out files and $ERR_COUNT .err files"

    if [ $LOG_COUNT -gt 0 ]; then
        echo ""
        echo "   Recent log files:"
        ls -lht logs/sweep_*.out | head -5
        echo ""
        echo "   Sample from most recent log:"
        LATEST_LOG=$(ls -t logs/sweep_*.out | head -1)
        echo "   File: $LATEST_LOG"
        echo "   ---"
        head -50 "$LATEST_LOG"
        echo "   ---"
        tail -50 "$LATEST_LOG"
    fi
else
    echo "   ✗ logs/ directory MISSING - this is likely the problem!"
    echo "   Creating logs/ directory now..."
    mkdir -p logs
    echo "   ✓ Created logs/ directory"
fi

echo ""
echo "2. Checking SLURM job status..."
if command -v squeue &> /dev/null; then
    echo ""
    squeue -u $USER
else
    echo "   (squeue not available - not on Euler?)"
fi

echo ""
echo "3. Checking recent SLURM jobs..."
if command -v sacct &> /dev/null; then
    echo ""
    sacct -u $USER --starttime=$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed
else
    echo "   (sacct not available - not on Euler?)"
fi

echo ""
echo "4. Checking for sweep info files..."
ls -lht engiopt/lvae_2d/sweep_info_*.txt 2>/dev/null | head -5

echo ""
echo "5. Checking WandB sweep status..."
if [ -f "engiopt/lvae_2d/sweep_info_heatconduction2d_grid.txt" ]; then
    echo "   Latest sweep info:"
    cat engiopt/lvae_2d/sweep_info_heatconduction2d_grid.txt
fi

echo ""
echo "=========================================="
echo "Debugging Complete"
echo "=========================================="
echo ""
echo "If logs/ directory was missing, try resubmitting:"
echo "  sbatch --export=SWEEP_ID=<your-sweep-id> --array=1-200%50 engiopt/lvae_2d/euler_sweep.slurm"
echo ""
