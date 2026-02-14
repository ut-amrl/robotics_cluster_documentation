# Isaac Lab NGC Container on Stampede3 (Apptainer)

Run NVIDIA Isaac Lab container `nvcr.io/nvidia/isaac-lab:2.3.2` on Stampede3 `amd-rtx` with Apptainer.

## 1) Start an interactive GPU node

```bash
idev -p amd-rtx -N 1 -n 1 -t 02:00:00
```

## 2) Pull image once

```bash
module reset
module load nvidia
module load tacc-apptainer/1.4.1

export ISAACLAB_SIF="$SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif"
mkdir -p "$(dirname "$ISAACLAB_SIF")"
apptainer pull "$ISAACLAB_SIF" docker://nvcr.io/nvidia/isaac-lab:2.3.2
```

If pull is denied, authenticate to NGC and retry:

```bash
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='<your-ngc-api-key>'
apptainer pull "$ISAACLAB_SIF" docker://nvcr.io/nvidia/isaac-lab:2.3.2
```

## 3) Enter interactive container

### Option A: helper script (recommended)

Use the helper in this folder:

```bash
./enter_isaaclab_container.sh
./enter_isaaclab_container.sh --output-dir "$SCRATCH/isaaclab_runs/my_interactive_run"
```

This creates writable run directories and launches Apptainer with the right flags:
`--cleanenv --fakeroot --nv` plus bind mounts for `/results`, `/isaac-sim/kit/cache`, and `/isaac-sim/kit/data`.

### Option B: manual launch commands

```bash
export RUN_DIR="$SCRATCH/isaaclab_runs/interactive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/kit-cache" "$RUN_DIR/kit-data"

TERM=xterm apptainer exec --cleanenv --fakeroot --nv \
  --bind "$RUN_DIR/results:/results,$RUN_DIR/kit-cache:/isaac-sim/kit/cache,$RUN_DIR/kit-data:/isaac-sim/kit/data" \
  --pwd /results \
  "$ISAACLAB_SIF" \
  bash --noprofile --norc -i
```

## 4) Run RL training with video

Inside the container shell:

```bash
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
cd /results

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Ant-v0 \
  --headless \
  --video \
  --video_length 200 \
  --video_interval 2000 \
  --max_iterations 10
```

## 5) Locate outputs

After training, logs and videos are under the output directory printed by
`enter_isaaclab_container.sh` (the `output dir:` line):

```bash
ls -lah <output_dir>/results/logs/rsl_rl/ant/*/videos/train
```

Or from inside container:

```bash
ls -lah /results/logs/rsl_rl/ant/*/videos/train
```

## Optional: batch mode

The provided batch example `run_isaaclab_container_train_video.slurm` runs
headless reinforcement learning training for the Isaac Lab Ant locomotion task
(`Isaac-Ant-v0`) using the `rsl_rl` training script with video capture enabled.

Submit with:

```bash
sbatch run_isaaclab_container_train_video.slurm
```

By default, each job writes results to:

```bash
$SCRATCH/isaaclab_runs/ant_video_<jobid>/results
```

Inside that directory, Isaac Lab creates run artifacts under
`logs/rsl_rl/ant/<timestamp>/`, including:

- TensorBoard/event logs and training summaries
- Model checkpoints
- Captured training videos (under `videos/train/`)
