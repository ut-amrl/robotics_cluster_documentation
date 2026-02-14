#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./enter_isaaclab_container.sh [--output-dir DIR]
  ./enter_isaaclab_container.sh -o DIR
  ./enter_isaaclab_container.sh DIR

Description:
  Launch an interactive Isaac Lab container shell on Stampede3 with:
  - clean environment isolation (--cleanenv)
  - GPU passthrough (--nv)
  - root-mapped namespace (--fakeroot)
  - writable bind mounts for Isaac Sim cache/data and run outputs

Behavior:
  - If no output directory is provided, creates:
      $SCRATCH/isaaclab_runs/interactive_YYYYmmdd_HHMMSS
  - Uses image:
      $SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif
    unless ISAACLAB_SIF is already set in the environment.
EOF
}

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; usage; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$1"
        shift
      else
        echo "Error: unexpected extra argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${SCRATCH:-}" ]]; then
  echo "Error: SCRATCH is not set. Run this on a Stampede3 login/compute shell." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SCRATCH/isaaclab_runs/interactive_$(date +%Y%m%d_%H%M%S)"
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

# Some modulefiles are not nounset-clean. Relax nounset only for module ops.
set +u
module reset
module load nvidia
module load tacc-apptainer/1.4.1
set -u

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/.apptainer/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

ISAACLAB_SIF="${ISAACLAB_SIF:-$SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif}"
if [[ ! -f "$ISAACLAB_SIF" ]]; then
  echo "Error: Isaac Lab image not found at: $ISAACLAB_SIF" >&2
  echo "Pull it first with:" >&2
  echo "  apptainer pull \"$ISAACLAB_SIF\" docker://nvcr.io/nvidia/isaac-lab:2.3.2" >&2
  exit 1
fi

RESULTS_DIR="$OUTPUT_DIR/results"
KIT_CACHE_DIR="$OUTPUT_DIR/kit-cache"
KIT_DATA_DIR="$OUTPUT_DIR/kit-data"
mkdir -p "$RESULTS_DIR" "$KIT_CACHE_DIR" "$KIT_DATA_DIR"

cat <<EOF
Launching Isaac Lab interactive shell
  image:      $ISAACLAB_SIF
  output dir: $OUTPUT_DIR
  results:    $RESULTS_DIR
  kit cache:  $KIT_CACHE_DIR
  kit data:   $KIT_DATA_DIR

Inside container, run:
  export TERM=xterm
  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y
  cd /results
  /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless --video --video_length 200 --video_interval 2000 --max_iterations 10
EOF

TERM=xterm apptainer exec --cleanenv --fakeroot --nv \
  --bind "$RESULTS_DIR:/results,$KIT_CACHE_DIR:/isaac-sim/kit/cache,$KIT_DATA_DIR:/isaac-sim/kit/data" \
  --pwd /results \
  "$ISAACLAB_SIF" \
  bash --noprofile --norc -i
