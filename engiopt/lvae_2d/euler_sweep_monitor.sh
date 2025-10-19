#!/bin/bash
# Monitor and manage sweep jobs on Euler
# Usage: ./euler_sweep_monitor.sh [command] [job_id]

set -e

COMMAND=${1:-status}
JOB_ID=${2:-}

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------
show_usage() {
    cat <<EOF
LVAE Sweep Monitor for Euler HPC

Usage: $0 [command] [job_id]

Commands:
  status [job_id]   - Show status of jobs (all or specific job)
  logs [job_id]     - Show recent logs (tail -f)
  errors [job_id]   - Show error logs
  cancel [job_id]   - Cancel sweep jobs
  requeue [job_id]  - Requeue failed jobs
  summary           - Show summary statistics
  sweeps            - List all sweep info files

Examples:
  $0 status                 # Show all your jobs
  $0 status 12345          # Show specific job array
  $0 logs 12345            # Tail logs for job 12345
  $0 errors 12345          # Show errors for job 12345
  $0 cancel 12345          # Cancel job array 12345
  $0 summary               # Show completion statistics
  $0 sweeps                # List all saved sweep info
EOF
}

get_user_jobs() {
    squeue -u "$USER" -h -o "%i %j %t %M %D %C %R" 2>/dev/null || echo ""
}

#-------------------------------------------------------------------------------
# Commands
#-------------------------------------------------------------------------------
case $COMMAND in
    status)
        echo "=========================================="
        echo "SLURM Job Status"
        echo "=========================================="
        if [ -n "$JOB_ID" ]; then
            echo "Job ID: $JOB_ID"
            echo ""
            squeue -j "$JOB_ID" -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %.6C %R" 2>/dev/null || echo "No jobs found with ID $JOB_ID"
            echo ""
            echo "Array task breakdown:"
            squeue -j "$JOB_ID" -t ALL -o "%.10i %.2t" 2>/dev/null | tail -n +2 | awk '{print $2}' | sort | uniq -c | awk '{
                state=$2
                count=$1
                if (state=="PD") label="PENDING"
                else if (state=="R") label="RUNNING"
                else if (state=="CG") label="COMPLETING"
                else if (state=="CD") label="COMPLETED"
                else if (state=="F") label="FAILED"
                else if (state=="CA") label="CANCELLED"
                else label=state
                printf "  %-12s %5d\n", label, count
            }'
        else
            squeue -u "$USER" -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %.6C %R" 2>/dev/null || echo "No jobs running"
        fi
        echo "=========================================="
        ;;

    logs)
        if [ -z "$JOB_ID" ]; then
            echo "ERROR: Job ID required for logs command"
            echo "Usage: $0 logs <job_id>"
            exit 1
        fi

        LOG_FILES=$(ls -t logs/sweep_${JOB_ID}_*.out 2>/dev/null | head -5)
        if [ -z "$LOG_FILES" ]; then
            echo "No log files found for job $JOB_ID"
            echo "Looking in: logs/sweep_${JOB_ID}_*.out"
            exit 1
        fi

        echo "=========================================="
        echo "Tailing logs for job $JOB_ID"
        echo "=========================================="
        echo "Showing 5 most recent log files:"
        echo "$LOG_FILES"
        echo "=========================================="
        echo ""
        tail -f $LOG_FILES
        ;;

    errors)
        if [ -z "$JOB_ID" ]; then
            echo "ERROR: Job ID required for errors command"
            echo "Usage: $0 errors <job_id>"
            exit 1
        fi

        ERROR_FILES=$(ls logs/sweep_${JOB_ID}_*.err 2>/dev/null)
        if [ -z "$ERROR_FILES" ]; then
            echo "No error files found for job $JOB_ID"
            exit 1
        fi

        echo "=========================================="
        echo "Errors for job $JOB_ID"
        echo "=========================================="
        for err_file in $ERROR_FILES; do
            if [ -s "$err_file" ]; then  # Only show non-empty files
                echo ""
                echo "=== $err_file ==="
                tail -20 "$err_file"
            fi
        done
        ;;

    cancel)
        if [ -z "$JOB_ID" ]; then
            echo "ERROR: Job ID required for cancel command"
            echo "Usage: $0 cancel <job_id>"
            exit 1
        fi

        echo "Cancelling job $JOB_ID..."
        scancel "$JOB_ID"
        echo "Done. Verify with: squeue -j $JOB_ID"
        ;;

    requeue)
        if [ -z "$JOB_ID" ]; then
            echo "ERROR: Job ID required for requeue command"
            echo "Usage: $0 requeue <job_id>"
            exit 1
        fi

        echo "Requeuing failed tasks for job $JOB_ID..."
        scontrol requeue "$JOB_ID"
        echo "Done."
        ;;

    summary)
        echo "=========================================="
        echo "Sweep Summary Statistics"
        echo "=========================================="

        # Find all sweep info files
        INFO_FILES=$(ls engiopt/lvae_2d/sweep_info_*.txt 2>/dev/null)
        if [ -z "$INFO_FILES" ]; then
            echo "No sweep info files found"
            exit 0
        fi

        for info_file in $INFO_FILES; do
            echo ""
            echo "--- $(basename "$info_file") ---"
            grep -E "(Created|Problem|Sweep ID|SLURM Job)" "$info_file" || true

            # Try to get job statistics if job ID exists
            SLURM_JOB=$(grep "SLURM Job:" "$info_file" | awk '{print $3}')
            if [ -n "$SLURM_JOB" ]; then
                echo "  Job Status:"
                squeue -j "$SLURM_JOB" -t ALL -o "%.2t" 2>/dev/null | tail -n +2 | sort | uniq -c | awk '{
                    state=$2
                    count=$1
                    if (state=="PD") label="PENDING"
                    else if (state=="R") label="RUNNING"
                    else if (state=="CG") label="COMPLETING"
                    else if (state=="CD") label="COMPLETED"
                    else if (state=="F") label="FAILED"
                    else if (state=="CA") label="CANCELLED"
                    else label=state
                    printf "    %-12s %5d\n", label, count
                }' || echo "    Job completed or not found"
            fi
        done
        echo "=========================================="
        ;;

    sweeps)
        echo "=========================================="
        echo "Saved Sweep Information"
        echo "=========================================="
        INFO_FILES=$(ls -t engiopt/lvae_2d/sweep_info_*.txt 2>/dev/null)
        if [ -z "$INFO_FILES" ]; then
            echo "No sweep info files found"
        else
            for info_file in $INFO_FILES; do
                echo ""
                echo "=== $(basename "$info_file") ==="
                cat "$info_file"
            done
        fi
        echo "=========================================="
        ;;

    help|--help|-h)
        show_usage
        ;;

    *)
        echo "ERROR: Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
