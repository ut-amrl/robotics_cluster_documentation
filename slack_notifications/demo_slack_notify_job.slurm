#!/usr/bin/env bash
#SBATCH -J slack_notify_demo
#SBATCH -o slack_notify_demo.%j.out
#SBATCH -e slack_notify_demo.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -p pvc


set -euo pipefail

SCRIPT_DIR="/work2/09340/joydeepb/stampede3/projects/robotics_cluster_documentation/slack_notifications"
SEND_SLACK="$SCRIPT_DIR/send_slack_message.sh"
if [[ ! -f "$SEND_SLACK" ]]; then
  echo "Error: could not find send_slack_message.sh at: $SEND_SLACK" >&2
fi

msg="Slurm job allocated: job_id=${SLURM_JOB_ID:-?} node=$(hostname) user=${USER:-?} partition=${SLURM_JOB_PARTITION:-?}"
bash "$SEND_SLACK" "$msg"

echo "Slack notification sent; holding allocation (top batch loop). Cancel with: scancel ${SLURM_JOB_ID:-<jobid>}"
# Batch mode, no -n: runs until the job ends (wall limit or scancel). Low churn.
exec top -b -d 60
