#!/usr/bin/env bash
#SBATCH -J isaaclab_install        # Job name
#SBATCH -o isaaclab_install.%j.out # Output file (%j = job ID)
#SBATCH -e isaaclab_install.%j.err # Error file
#SBATCH -p amd-rtx                 # Queue (partition)
#SBATCH -N 1                       # 1 node
#SBATCH -n 1                       # 1 task
#SBATCH -t 01:00:00                # Wall time (1 hour, generous for install)
#SBATCH -A IRI26004		   # Allocation

set -e

# --------------------------------------------------------------------------
# IsaacLab v2.1.0 full installation script for Stampede3 amd-rtx nodes
# System: 8x NVIDIA RTX PRO 6000 Blackwell, AMD EPYC 9555, Rocky Linux 9.7
# --------------------------------------------------------------------------

echo "=== IsaacLab Installation Starting ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load the nvidia module (brings in cuda, openmpi, etc.)
module load nvidia
module list

# Initialize micromamba in this non-login shell
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

# --------------------------------------------------------------------------
# Step 1: Create micromamba environment
# --------------------------------------------------------------------------
echo ""
echo "=== Step 1: Creating micromamba environment ==="
micromamba create -n isaaclab python=3.10 -c conda-forge -y
micromamba activate isaaclab
echo "Python: $(python --version)"

# --------------------------------------------------------------------------
# Step 2: Install PyTorch nightly (required for Blackwell GPUs)
# --------------------------------------------------------------------------
echo ""
echo "=== Step 2: Installing PyTorch nightly (cu128) ==="
pip install --upgrade pip
pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# --------------------------------------------------------------------------
# Step 3: Install Isaac Sim 4.5.0
# --------------------------------------------------------------------------
echo ""
echo "=== Step 3: Installing Isaac Sim 4.5.0 ==="
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# --------------------------------------------------------------------------
# Step 4: Clone and checkout IsaacLab v2.1.0
# --------------------------------------------------------------------------
echo ""
echo "=== Step 4: Cloning IsaacLab v2.1.0 ==="
cd $WORK

if [ -d "IsaacLab" ]; then
    echo "IsaacLab directory already exists, updating..."
    cd IsaacLab
    git fetch --tags
    git checkout v2.1.0
else
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    git checkout v2.1.0
fi

# --------------------------------------------------------------------------
# Step 5: Patch torch version pins (Blackwell compatibility)
# --------------------------------------------------------------------------
echo ""
echo "=== Step 5: Patching torch version pins ==="
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab/setup.py
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab_rl/setup.py
sed -i 's/"torch==2.5.1"/"torch>=2.5.1"/g' source/isaaclab_tasks/setup.py
echo "Patched setup.py files"

# --------------------------------------------------------------------------
# Step 6: Install IsaacLab extensions
# --------------------------------------------------------------------------
echo ""
echo "=== Step 6: Installing IsaacLab extensions ==="
export TERM=xterm                        # Fix for "tabs" error
export CMAKE_POLICY_VERSION_MINIMUM=3.5  # Fix for egl_probe CMake 4.x build
./isaaclab.sh --install

# --------------------------------------------------------------------------
# Step 7: Reinstall PyTorch nightly (installer downgrades to 2.5.1)
# --------------------------------------------------------------------------
echo ""
echo "=== Step 7: Reinstalling PyTorch nightly ==="
pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --upgrade --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# --------------------------------------------------------------------------
# Step 8: Accept EULA non-interactively
# --------------------------------------------------------------------------
echo ""
echo "=== Step 8: Accepting EULA ==="
echo "Yes" | python -c "import isaacsim"
echo "EULA accepted"

# --------------------------------------------------------------------------
# Step 9: Verify with RL training (10 iterations)
# --------------------------------------------------------------------------
echo ""
echo "=== Step 9: Verification - RL training test ==="
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --max_iterations 10

echo ""
echo "=== Installation complete! ==="
echo "Date: $(date)"
