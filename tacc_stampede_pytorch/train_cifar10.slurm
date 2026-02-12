#!/usr/bin/env bash
#SBATCH -J pytorch_cifar10         # Job name
#SBATCH -o pytorch_cifar10.%j.out  # Output file (%j = job ID)
#SBATCH -e pytorch_cifar10.%j.err  # Error file
#SBATCH -p amd-rtx                 # Queue (partition)
#SBATCH -N 1                       # 1 node
#SBATCH -n 1                       # 1 task
#SBATCH -t 00:30:00                # Wall time
#SBATCH -A IRI26004                # Allocation

set -e

# --------------------------------------------------------------------------
# PyTorch CIFAR-10 training on Stampede3 amd-rtx nodes
# Creates a micromamba environment, installs PyTorch nightly (cu128),
# and trains a ResNet-18 on CIFAR-10 with mixed-precision training.
# --------------------------------------------------------------------------

echo "=== PyTorch CIFAR-10 Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Initialize micromamba
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

# --------------------------------------------------------------------------
# Step 1: Create environment (skip if it already exists)
# --------------------------------------------------------------------------
ENV_NAME="pytorch"
if micromamba env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists, activating..."
else
    echo "Creating environment '$ENV_NAME'..."
    micromamba create -n "$ENV_NAME" python=3.11 -c conda-forge -y
fi
micromamba activate "$ENV_NAME"
echo "Python: $(python --version)"

# --------------------------------------------------------------------------
# Step 2: Install PyTorch nightly with CUDA 12.8 (if not already installed)
# --------------------------------------------------------------------------
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "Installing PyTorch nightly (cu128)..."
    pip install --upgrade pip
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
fi

# --------------------------------------------------------------------------
# Step 3: Verify GPU access
# --------------------------------------------------------------------------
echo ""
echo "=== GPU Verification ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

# --------------------------------------------------------------------------
# Step 4: Train CIFAR-10
# --------------------------------------------------------------------------
echo ""
echo "=== Training ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/train_cifar10.py" \
    --epochs 5 \
    --batch-size 256 \
    --gpu 0 \
    --data-dir "$SCRATCH/datasets/cifar10" \
    --output "$SCRATCH/pytorch_cifar10_$SLURM_JOB_ID"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
