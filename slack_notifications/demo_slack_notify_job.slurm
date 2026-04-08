#!/usr/bin/env bash
#SBATCH -J slack_notify_demo
#SBATCH -o slack_notify_demo.%j.out
#SBATCH -e slack_notify_demo.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
##SBATCH -p amd-rtx


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SEND_SLACK="$SCRIPT_DIR/send_slack_message.sh"

msg="Slurm job allocated: job_id=${SLURM_JOB_ID:-?} node=$(hostname) user=${USER:-?} partition=${SLURM_JOB_PARTITION:-?}"
"$SEND_SLACK" "$msg"

echo "Slack notification sent; holding allocation (top batch loop). Cancel with: scancel ${SLURM_JOB_ID:-<jobid>}"
# Batch mode, no -n: runs until the job ends (wall limit or scancel). Low churn.
exec top -b -d 60
