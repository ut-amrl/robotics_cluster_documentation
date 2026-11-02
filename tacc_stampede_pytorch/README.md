# PyTorch GPU Environment Guide (Stampede3 / Micromamba)

Set up PyTorch on Stampede3, check that it can use the GPUs, and run a training job. These examples use the two-GPU `rtx-small` queue. For larger jobs, see the [queue guide](../README.md#our-gpu-nodes-amd-rtx-and-rtx-small).

## Quick Start

If you just want to run the included CIFAR-10 training example end-to-end:

```bash
sbatch -p rtx-small train_cifar10.slurm
```

This creates the environment, installs PyTorch, and trains a ResNet-18 on CIFAR-10 for 5 epochs. Monitor with:

```bash
tail -f pytorch_cifar10.<job_id>.out
```

## Why PyTorch Nightly?

Both queues have **NVIDIA RTX PRO 6000 Blackwell** GPUs (96 GB VRAM each). These require PyTorch built with CUDA 12.8+ support. As of early 2025, the stable PyTorch release does not yet include Blackwell support — you must use either a **nightly** build or a stable release >= 2.9.

## Step 1: Create the Micromamba Environment

From a login node or compute node:

```bash
micromamba create -n pytorch python=3.11 -c conda-forge -y
micromamba activate pytorch
```

Install into `$WORK` (the default micromamba root) so the environment persists across sessions.

## Step 2: Install PyTorch with CUDA 12.8

```bash
pip install --upgrade pip
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

This installs PyTorch nightly with CUDA 12.8 support (~3 GB download). If a stable release supporting Blackwell is available (PyTorch >= 2.9), you can drop `--pre` and use the stable index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Step 3: Verify GPU Access (Interactive)

Start an interactive session on a two-GPU `rtx-small` node:

```bash
idev -p rtx-small -N 1 -n 1 -t 0:30:00
```

Once allocated, activate the environment and test:

```bash
micromamba activate pytorch

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version:   {torch.version.cuda}')
print(f'GPU count:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.0f} GB)')
x = torch.randn(1000, 1000, device='cuda')
print(f'Tensor on GPU: {x.device}, sum = {x.sum().item():.4f}')
"
```

On `rtx-small`, check for these lines in the output:

```
CUDA available: True
GPU count:      2
```

### Quick Matmul Benchmark

```bash
python -c "
import torch, time
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    c = torch.mm(a, b)
torch.cuda.synchronize()
tflops = 100 * 2 * 4096**3 / (time.time() - t0) / 1e12
print(f'FP32 matmul: {tflops:.1f} TFLOPS')
"
```

We measured **~54 TFLOPS** FP32 on an `amd-rtx` node.

## Step 4: Run a Training Job via sbatch

The included [`train_cifar10.py`](train_cifar10.py) trains a ResNet-18 on CIFAR-10 with mixed-precision (AMP), cosine LR scheduling, and model checkpointing. Submit it with the provided SLURM script:

```bash
sbatch -p rtx-small train_cifar10.slurm
```

Or run it interactively:

```bash
python train_cifar10.py --epochs 20 --gpu 0 --output $SCRATCH/cifar10_ckpt
```

### Training Script Options

```
--epochs N        Number of epochs (default: 5)
--batch-size N    Batch size (default: 256)
--lr F            Learning rate (default: 0.1)
--gpu N           GPU index (default: 0)
--workers N       Data loader workers (default: 4)
--output DIR      Checkpoint directory (default: ./checkpoints)
--data-dir DIR    Dataset download directory (default: ./data)
```

### Example Output

```
Training on: cuda:0 (NVIDIA RTX PRO 6000 Blackwell Server Edition)
PyTorch: 2.11.0.dev20260211+cu128, CUDA: 12.8
Config: epochs=3, batch_size=256, lr=0.1
Epoch 1/3 | Train Loss: 2.0414, Acc: 28.62% | Test Loss: 1.6193, Acc: 40.25% | Time: 2.5s | LR: 0.075000
Epoch 2/3 | Train Loss: 1.4468, Acc: 46.52% | Test Loss: 1.3146, Acc: 51.47% | Time: 2.1s | LR: 0.025000
Epoch 3/3 | Train Loss: 1.1313, Acc: 58.85% | Test Loss: 1.0920, Acc: 60.73% | Time: 2.1s | LR: 0.000000

Training complete. Best test accuracy: 60.73%
Checkpoint saved to: ./checkpoints/best_model.pt
```

With the full 200 epochs, this setup reaches ~93-94% test accuracy on CIFAR-10.

## Adapting for Your Own Project

Use this environment as a starting point. Install additional packages as needed:

```bash
micromamba activate pytorch
pip install transformers datasets accelerate   # Hugging Face
pip install lightning                           # PyTorch Lightning
pip install wandb                               # Experiment tracking
```

### Multi-GPU Training

For training on multiple GPUs with `torchrun`, start one process per GPU:

```bash
# On rtx-small (2 GPUs)
torchrun --nproc_per_node=2 your_training_script.py

# On amd-rtx (8 GPUs)
torchrun --nproc_per_node=8 your_training_script.py
```

Use these commands within a job on the matching queue. Set `--nproc_per_node` to the number of GPUs your training script will use on that node.

### Checkpointing to $SCRATCH

For long training runs, save checkpoints to `$SCRATCH` for fast I/O:

```bash
python train_cifar10.py --output $SCRATCH/my_experiment --data-dir $SCRATCH/datasets
```

Remember that `$SCRATCH` files are **purged after 10 days of inactivity**. Copy final results to `$WORK` for long-term storage.

## Files in This Directory

| File | Description |
|------|-------------|
| [`train_cifar10.py`](train_cifar10.py) | CIFAR-10 ResNet-18 training script (single GPU, mixed precision) |
| [`train_cifar10.slurm`](train_cifar10.slurm) | SLURM batch script: creates environment, installs PyTorch, runs training |
