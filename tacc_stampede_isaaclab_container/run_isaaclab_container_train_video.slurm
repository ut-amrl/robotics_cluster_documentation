#!/bin/bash
#SBATCH -J isaaclab_container
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -o isaaclab_container.%j.out
#SBATCH -e isaaclab_container.%j.err

set -euo pipefail

# Some modulefiles/source'd completion scripts are not nounset-clean.
set +u
module reset
module load nvidia
module load tacc-apptainer/1.4.1
set -u

export ISAACLAB_IMG_DIR="$SCRATCH/containers/isaaclab"
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer/tmp"
mkdir -p "$ISAACLAB_IMG_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

ISAACLAB_SIF="$ISAACLAB_IMG_DIR/isaac-lab_2.3.2.sif"
if [[ ! -f "$ISAACLAB_SIF" ]]; then
  apptainer pull "$ISAACLAB_SIF" docker://nvcr.io/nvidia/isaac-lab:2.3.2
fi

RUN_DIR="$SCRATCH/isaaclab_runs/ant_video_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/kit-cache" "$RUN_DIR/kit-data" "$RUN_DIR/isaaclab-tmp"

TERM=xterm apptainer exec --cleanenv --fakeroot --nv \
  --bind "$RUN_DIR/results:/results,$RUN_DIR/kit-cache:/isaac-sim/kit/cache,$RUN_DIR/kit-data:/isaac-sim/kit/data,$RUN_DIR/isaaclab-tmp:/tmp/isaaclab" \
  --pwd /results \
  "$ISAACLAB_SIF" \
  bash --noprofile --norc -lc '
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Ant-v0 \
  --headless \
  --video \
  --video_length 200 \
  --video_interval 2000 \
  --max_iterations 50
'

echo "============================================================"
echo "Isaac Lab training SLURM job completed successfully."
echo "Run artifacts copied to: $RUN_DIR"
echo "============================================================"
