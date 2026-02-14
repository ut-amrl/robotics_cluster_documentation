# Texas Robotics Cluster Notes

Notes and setup guides for [Texas Robotics](https://robotics.utexas.edu/) use of [TACC Stampede3](https://docs.tacc.utexas.edu/hpc/stampede3/).

## Our Nodes: `amd-rtx` Queue

Texas Robotics contributed 8 nodes to Stampede3 and has priority access to them via the `amd-rtx` queue.

| Spec | Value |
|------|-------|
| **Nodes** | 8 |
| **CPU** | 2x AMD EPYC 9555 64-Core (128 cores per node) |
| **RAM** | ~1.5 TB per node |
| **GPUs** | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition per node |
| **GPU Driver** | 590.48.01 |
| **CUDA** | 13.1 (via `module load nvidia`) |
| **OS** | Rocky Linux 9.7, Kernel 5.14.0 |

## Other Stampede3 Queues

Texas Robotics allocations also have access to the rest of Stampede3. See the [full queue documentation](https://docs.tacc.utexas.edu/hpc/stampede3/#queues) for details.

| Queue | Nodes | Cores/Node | RAM/Node | GPUs | Description |
|-------|------:|----------:|---------:|------|-------------|
| `skx-dev` | 72 | 48 | 192 GB | -- | Dev/debug queue (2 hr limit) |
| `skx` | 1,160 | 48 | 192 GB | -- | Skylake CPU nodes |
| `icx` | 224 | 80 | 256 GB | -- | Ice Lake CPU nodes |
| `spr` | 616 | 112 | 128 GB HBM | -- | Sapphire Rapids HBM nodes |
| `nvdimm` | 3 | 80 | 4 TB | -- | Large-memory Ice Lake nodes |
| `h100` | 24 | 96 | 1 TB | 4x NVIDIA H100 | GPU nodes |
| `pvc` | 20 | 96 | 1 TB | 4x Intel PVC | GPU nodes |
| **`amd-rtx`** | **8** | **128** | **~1.5 TB** | **8x RTX PRO 6000** | **TR priority** |

## Software Status (`amd-rtx`)

| Component | Status | Notes |
|-----------|--------|-------|
| Isaac Lab NGC container v2.3.2 | Working (Apptainer, recommended) | Best path for latest Isaac Lab on Stampede3: run `nvcr.io/nvidia/isaac-lab:2.3.2` via Apptainer; see [container guide](tacc_stampede_isaaclab_container/) |
| IsaacLab v2.1.0 | Working (legacy path) | Older pip/micromamba install path; use container above for latest Isaac Lab version. See [setup guide](tacc_stampede_isaaclab/) |
| Isaac Sim 5.1.x (source build) | Working (with caveats) | Build from source on GLIBC 2.34 nodes; see [Isaac Sim source-build guide](tacc_stampede_isaacsim_source_build/) |
| PyTorch (nightly, cu128) | Working | Default PyTorch 2.5.1 does **not** support Blackwell; see [PyTorch GPU guide](tacc_stampede_pytorch/) |
| vLLM 0.15 | Working | LLM inference server with TP/DP support; see [vLLM guide](tacc_stampede_vllm/) |
| Isaac Sim 4.5.0 | Working | pip install |

## Connecting

See the [Stampede3 access documentation](https://docs.tacc.utexas.edu/hpc/stampede3/#access). You will need [TACC Multi-Factor Authentication](https://docs.tacc.utexas.edu/basics/mfa) -- you will be prompted for a 2FA code at login.

```bash
ssh <username>@stampede3.tacc.utexas.edu
```

## Interactive Development Sessions

The `idev` command is the easiest way to get an interactive shell on a compute node:

```bash
idev -p amd-rtx                      # default time (usually 30 min)
idev -p amd-rtx -t 2:30:00           # 2.5 hours
```

Once allocated, you can see your assigned node and SSH directly to it:

```bash
squeue -u $USER                      # find the node name (NODELIST column)
ssh <node-name>                       # e.g. ssh c571-002
```

This is useful when you need multiple terminals on the same compute node.

## Checking Node Availability

From the [TACC docs on monitoring with sinfo](https://docs.tacc.utexas.edu/hpc/stampede3/#jobs-monitoring-sinfo), this command gives a compact overview of all queues:

```bash
$ sinfo -S+P -o "%18P %8a %20F"
PARTITION          AVAIL    NODES(A/I/O/T)
amd-rtx            up       1/7/0/8
h100               up       14/4/6/24
icx                up       206/8/10/224
nvdimm             up       2/1/0/3
pvc                up       6/10/4/20
skx                up       1091/5/64/1160
skx-dev*           up       10/35/27/72
spr                up       255/237/124/616
```

The `NODES(A/I/O/T)` column shows **A**llocated / **I**dle / **O**ther (down, drained, etc.) / **T**otal. In the example above, the `amd-rtx` queue has 1 node in use, 7 idle, and 8 total.

For per-node detail on our queue, use `sinfo -Nel -p amd-rtx`.

## Common SLURM Commands

```bash
sbatch my_job.slurm                   # submit a batch job
squeue -u $USER                       # your running/pending jobs
scancel <job_id>                      # cancel a job
```

See the [TACC job submission docs](https://docs.tacc.utexas.edu/hpc/stampede3/#running) for full details.

## Module System

Stampede3 uses [Lmod](https://docs.tacc.utexas.edu/hpc/stampede3/#admin-modules) for software management:

```bash
module list                           # currently loaded modules
module spider <keyword>               # search available modules
module load nvidia                    # load NVIDIA stack (CUDA, OpenMPI, etc.)
module reset                          # reset to system defaults
```

## Storage and Quotas

See the [Stampede3 file systems documentation](https://docs.tacc.utexas.edu/hpc/stampede3/#system-filesystems) for complete details.

| Path | Quota | Backed Up | Purge Policy |
|------|-------|-----------|-------------|
| `$HOME` | 15 GB, 300K files | Yes | -- |
| `$WORK` | 1 TB, 3M files (shared across all TACC systems) | No | -- |
| `$SCRATCH` | No quota (~10 PB total) | No | Files not accessed in 10 days may be purged |

Check your usage:

```bash
/usr/local/etc/taccinfo
```
Example output: 

```bash
--------------------- Project balances for user joydeepb ----------------------
| Name           Avail SUs     Expires |                                      |
| IRI26004           99508  2027-02-04 |                                      |
------------------------ Disk quotas for user joydeepb ------------------------
| Disk         Usage (GB)     Limit    %Used   File Usage       Limit   %Used |
| /scratch           24.0       0.0     0.00         8383           0    0.00 |
| /home1              0.2      14.0     1.75          398      500000    0.08 |
| /work             241.1    1024.0    23.54       849645     3000000   28.32 |
-------------------------------------------------------------------------------
```

Handy navigation aliases (built-in on TACC systems):

```bash
cdh                                   # cd $HOME
cdw                                   # cd $WORK
cds                                   # cd $SCRATCH
```

### Best Practices

- **`$HOME`**: Small config files and dotfiles only. Do not install software here.
- **`$WORK`**: Persistent software installs, conda/micromamba environments, cloned repos, and datasets you need long-term. Shared across TACC systems via [Stockyard](https://tacc.utexas.edu/systems/stockyard/).
- **`$SCRATCH`**: Large training outputs, checkpoints, and temporary data. Fast I/O, but **files are purged after 10 days of inactivity**. Do not use as long-term storage.
- Avoid many small file operations on `$HOME` and `$WORK` -- they are not designed for high-throughput I/O.
- If you need more than 1 TB of persistent storage, see [TACC Corral](https://docs.tacc.utexas.edu/hpc/corral/).

## Python Environment Management

We recommend [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for managing Python environments. It is a standalone C++ binary that resolves and installs conda packages significantly faster than conda or mamba, with no base environment overhead.

```bash
# Install micromamba (one-time)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create an environment
micromamba create -n myenv python=3.10 -c conda-forge -y

# Activate / deactivate
micromamba activate myenv
micromamba deactivate
```

Install environments into `$WORK` so they persist and are available across login and compute nodes.

## Setup Guides

| Guide | Description |
|-------|-------------|
| [PyTorch GPU Environment](tacc_stampede_pytorch/) | Set up micromamba + PyTorch with GPU support on Blackwell nodes. Includes an interactive verification walkthrough and an [sbatch training script](tacc_stampede_pytorch/train_cifar10.slurm). |
| [vLLM LLM Serving](tacc_stampede_vllm/) | Install vLLM and serve LLMs with single-GPU, data-parallel (8 GPU), and tensor-parallel configurations. Includes sbatch scripts for [14B single-GPU](tacc_stampede_vllm/serve_single_gpu.slurm), [14B data-parallel](tacc_stampede_vllm/serve_data_parallel.slurm), and [32B tensor-parallel](tacc_stampede_vllm/serve_tensor_parallel.slurm). |
| [Isaac Lab NGC Container](tacc_stampede_isaaclab_container/) | **Recommended for latest Isaac Lab.** Run NVIDIA's `nvcr.io/nvidia/isaac-lab:2.3.2` image on Stampede3 using Apptainer. Includes interactive launch helpers and an [sbatch video-training example](tacc_stampede_isaaclab_container/run_isaaclab_container_train_video.slurm). |
| [IsaacLab on Stampede3](tacc_stampede_isaaclab/) | Legacy install guide for IsaacLab v2.1.0 + Isaac Sim 4.5.0 via pip/micromamba. Includes an [sbatch script](tacc_stampede_isaaclab/install_isaaclab.slurm). |
| [Isaac Sim Source Build (GLIBC 2.34)](tacc_stampede_isaacsim_source_build/) | Build Isaac Sim from source on Blackwell nodes where pip wheels are incompatible. Includes a full automation script ([build_isaacsim_stampede.sh](tacc_stampede_isaacsim_source_build/build_isaacsim_stampede.sh)) and a warehouse SDG smoke test. |
