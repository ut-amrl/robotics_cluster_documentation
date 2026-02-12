# IsaacLab Installation Guide (Stampede3 / Micromamba)

Instructions for installing [IsaacLab v2.1.0](https://isaac-sim.github.io/IsaacLab/release/2.1.0/) with Isaac Sim 4.5.0 in a micromamba environment on Stampede3 GPU nodes.

**Automated install:** You can skip the manual steps entirely by submitting the provided sbatch script:

```bash
sbatch install_isaaclab.slurm
```

See [`install_isaaclab.slurm`](install_isaaclab.slurm) for the full script. It creates the environment, installs all dependencies, patches version pins, and runs a verification training loop — all in a single batch job.

For more context on Isaac Lab's installation options, see the [official IsaacLab installation documentation](https://isaac-sim.github.io/IsaacLab/release/2.1.0/source/setup/installation/index.html).

## System Details

- **OS:** Rocky Linux 9.7 (Blue Onyx), Kernel 5.14.0
- **CPU:** 2x AMD EPYC 9555 64-Core (128 cores total)
- **RAM:** ~1.5 TB
- **GPUs:** 8x NVIDIA RTX PRO 6000 Blackwell Server Edition
- **GPU Driver:** 590.48.01
- **CUDA (system):** 13.1
- **GLIBC:** 2.34

## Step 1: Create the Micromamba Environment

```bash
# Create environment with Python 3.10 (required by Isaac Sim)
micromamba create -n isaaclab_install python=3.10 -c conda-forge -y
micromamba activate isaaclab_install
```

## Step 2: Install PyTorch Nightly

The Blackwell GPUs on this cluster require PyTorch nightly with cu128 (the default PyTorch 2.5.1 that ships with Isaac Sim does not support Blackwell).

```bash
pip install --upgrade pip
pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Step 3: Install Isaac Sim 4.5.0

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

This takes several minutes as it downloads ~2 GB of packages.

## Step 4: Clone and Checkout IsaacLab v2.1.0

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0
```

## Step 5: Patch Torch Version Pins

IsaacLab's source pins `torch==2.5.1`, which conflicts with the nightly PyTorch required for Blackwell. Relax the constraint before installing:

```bash
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab/setup.py
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab_rl/setup.py
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab_tasks/setup.py
```

## Step 6: Install IsaacLab Extensions

Two environment variables are needed to work around build issues on this cluster:

```bash
export TERM=xterm                          # fixes "tabs" error in non-interactive shells
export CMAKE_POLICY_VERSION_MINIMUM=3.5    # fixes egl_probe build with CMake 4.x

./isaaclab.sh --install
```

This takes several minutes.

## Step 7: Reinstall PyTorch Nightly

The IsaacLab installer downgrades PyTorch to 2.5.1. Reinstall the nightly build:

```bash
pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --upgrade --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 8: Accept the EULA (Non-Interactive)

The first import of `isaacsim` triggers an interactive EULA prompt. Accept it non-interactively:

```bash
echo "Yes" | python -c "import isaacsim"
```

This persists the acceptance for all future imports.

## Step 9: Verify with RL Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --max_iterations 10
```

You should see 10 learning iterations with reward/loss statistics. Training runs at ~440K steps/s on these nodes.

For a full training run or a different task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

### Training with RGB Video Recording

To enable RGB camera rendering and record training videos, add the `--video` flag:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --video \
    --video_length 200 \
    --video_interval 2000 \
    --max_iterations 50
```

- `--video` enables RGB rendering and records mp4 clips during training.
- `--video_length N` sets how many steps each clip records (default: 200).
- `--video_interval N` records a clip every N steps (default: 2000).

Videos are saved to `logs/rsl_rl/<task>/<timestamp>/videos/train/`.

## Known Issues and Workarounds

### 1. `isaaclab.sh` fails: "terminal type 'dumb' cannot reset tabs"

The script runs `tabs 4` which requires a real terminal type. Non-interactive shells default to `TERM=dumb`.

**Fix:** `export TERM=xterm` before running `isaaclab.sh`.

### 2. `egl_probe` fails to build with CMake 4.x

The `egl_probe` package's `CMakeLists.txt` uses a `cmake_minimum_required` version below 3.5, which CMake 4.x no longer supports.

**Fix:** `export CMAKE_POLICY_VERSION_MINIMUM=3.5` before running the installer.

### 3. PyTorch downgrade during IsaacLab install

IsaacLab's `setup.py` files pin `torch==2.5.1`. The installer overwrites the nightly build with this older version, which does not work on Blackwell GPUs.

**Fix:** Patch the `setup.py` files (Step 5) before installing, then reinstall PyTorch nightly after (Step 7).

### 4. EULA blocks non-interactive scripts

The first `import isaacsim` triggers an interactive EULA prompt that hangs in scripts.

**Fix:** `echo "Yes" | python -c "import isaacsim"` to accept once, non-interactively.

### 5. CUDA peer-to-peer warning on multi-GPU nodes

You may see: `Cuda failure: 'peer access is already enabled'`. This is a harmless warning on 8-GPU nodes. Training completes successfully despite it.

## Automated Installation via sbatch

Instead of running the steps above manually, submit the provided SLURM batch script which performs the entire installation end-to-end on a GPU node:

```bash
sbatch install_isaaclab.slurm
```

The script ([`install_isaaclab.slurm`](install_isaaclab.slurm)) performs all 9 steps above, including environment creation, patching, and a verification training run. Monitor progress with:

```bash
tail -f isaaclab_install.<job_id>.out
```

For the official upstream installation instructions and alternative methods, refer to the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/release/2.1.0/source/setup/installation/index.html).
