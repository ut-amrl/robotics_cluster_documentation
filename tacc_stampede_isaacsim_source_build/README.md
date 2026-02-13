# IsaacSim Source Build on TACC Stampede (Blackwell Node, GLIBC 2.34)

This README documents a working source-build flow for `IsaacSim` on this cluster node.

Why source build here:
- Host GLIBC is `2.34` (`ldd --version`), and pip-distributed IsaacSim binaries are not compatible for this setup.

Validated on this node:
- GPU: NVIDIA RTX PRO 6000 Blackwell
- Driver: 590.48.01
- Host default GCC: 13.2.0
- Host GLIBC: 2.34

## Quick-start: use the automation script

If you are on the same cluster family, use the script first:
- `build_isaacsim_stampede.sh`

This script automates clone + LFS + micromamba GCC11 + wrappers + build + validation.

### 1) Review/edit variables

Open the script and adjust the variables at the top:
- `CLONE_PARENT_DIR`
- `REPO_URL`
- `REPO_NAME`
- `REPO_BRANCH`
- `UPDATE_EXISTING_REPO`
- `MAMBA_ENV_NAME`
- `BUILD_CONFIG`
- `BUILD_JOBS`
- `ACCEPT_EULA`

### 2) Run it

```bash
cd /path/to/script/location
chmod +x build_isaacsim_stampede.sh
ACCEPT_EULA=yes ./build_isaacsim_stampede.sh
```

Notes:
- Use `ACCEPT_EULA=yes` for non-interactive first-time setup.
- If the repo already exists and you want it refreshed, set `UPDATE_EXISTING_REPO=true`.

### 3) Example with overrides

```bash
CLONE_PARENT_DIR="$HOME/work" \
REPO_BRANCH="main" \
BUILD_JOBS=32 \
ACCEPT_EULA=yes \
./build_isaacsim_stampede.sh
```

---

## What actually failed first

1) **Compiler version gate in IsaacSim build tooling**
- Repo requires GCC 11 by default (`repo.toml` check).
- Default host compiler is GCC 13.

2) **Cluster oneAPI environment polluted include paths**
- `CPATH` had oneAPI/TBB include directories.
- This caused header/API mismatches in early C++ compilation.

3) **CUDA 11.8 + system GCC 13 incompatibility for nvcc host compile**
- CUDA C++ compilation failed with `_Float32`/`_Float64` errors.
- Root cause: `nvcc` picked host `g++` 13 from PATH.

## Working solution

- Use micromamba to install GCC 11 toolchain.
- Put wrapper `gcc`/`g++` scripts in front of PATH so:
  - IsaacSim compiler check sees GCC 11.
  - `nvcc` host compiler resolution also uses GCC 11.
- Unset oneAPI include env vars before build invocation.

---

## Step-by-step

Define these once for your environment:

```bash
export REPO_ROOT="/path/to/IsaacSim"
export MAMBA_ENV_PREFIX="$(micromamba env list | awk '/isaacsim-gcc11/{print $NF}')"
```

If `isaacsim-gcc11` does not exist yet, create it first (next section), then re-run the
`MAMBA_ENV_PREFIX` command.

### 1) Create GCC11 environment

```bash
micromamba create -y -n isaacsim-gcc11 -c conda-forge \
  gcc_linux-64=11 gxx_linux-64=11 binutils make cmake ninja
```

### 2) Create wrapper compilers in repo

```bash
mkdir -p "$REPO_ROOT/.toolchain/bin"
```

Create `gcc` wrapper:

```bash
cat > "$REPO_ROOT/.toolchain/bin/gcc" <<EOF
#!/usr/bin/env bash
exec "$MAMBA_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" "\$@"
EOF
```

Create `g++` wrapper:

```bash
cat > "$REPO_ROOT/.toolchain/bin/g++" <<EOF
#!/usr/bin/env bash
exec "$MAMBA_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" "\$@"
EOF
```

Make wrappers executable:

```bash
chmod +x "$REPO_ROOT/.toolchain/bin/gcc" \
         "$REPO_ROOT/.toolchain/bin/g++"
```

Sanity check:

```bash
PATH="$REPO_ROOT/.toolchain/bin:$PATH" gcc --version
PATH="$REPO_ROOT/.toolchain/bin:$PATH" g++ --version
```

Expected: GCC/G++ `11.4.0` (conda-forge build).

### 3) Build IsaacSim (release)

```bash
cd "$REPO_ROOT"

env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
  PATH="$REPO_ROOT/.toolchain/bin:$PATH" \
  ./build.sh --config release -j 16
```

Notes:
- First run downloads many large dependencies.
- You may be prompted to accept the Omniverse EULA if not yet accepted.
- This command ran successfully end-to-end here (fetch + generate + build + post-build).

### 4) Quick validation

```bash
test -x "$REPO_ROOT/_build/linux-x86_64/release/isaac-sim.sh" && echo "launcher exists"
"$REPO_ROOT/_build/linux-x86_64/release/python.sh" -c "import sys; print(sys.version)"
```

### 5) Warehouse SDG Python smoke test

I also validated IsaacSim by running a headless synthetic-data generation script against a warehouse stage.

Script used:
- `$REPO_ROOT/tmp_warehouse_sdg_smoke.py`

Run command:

```bash
cd "$REPO_ROOT"
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
  _build/linux-x86_64/release/python.sh tmp_warehouse_sdg_smoke.py
```

What this script does:
- Opens `/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd`
- Runs Replicator `BasicWriter` for 3 frames
- Writes semantic segmentation + 2D tight boxes under `_out_warehouse_sdg_smoke`

Expected output artifacts:

```bash
ls "$REPO_ROOT/_out_warehouse_sdg_smoke"
```

You should see files like:
- `semantic_segmentation_labels_0000.json`
- `bounding_box_2d_tight_labels_0000.json`
- `bounding_box_2d_tight_prim_paths_0000.json`
- `metadata.txt`

Observed result here:
- Script exited successfully (`exit code 0`) and annotation JSON files were generated.

---

## Issues encountered and fixes

- **Issue:** `BuildError: GCC version check failed: expected 11.*.* but found 13.2.0`
  - **Fix:** Provide GCC11 wrappers in PATH (above), so `gcc`/`g++` resolve to GCC11.

- **Issue:** oneAPI/TBB include contamination caused compile errors
  - **Fix:** Build with `env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT ...`

- **Issue:** CUDA compile errors like `_Float32` undefined in `/usr/include/stdlib.h`
  - **Fix:** Ensure `nvcc` sees GCC11 by placing wrapper `g++` first in PATH.

---

## Optional: split phases

If you want staged execution:

```bash
# fetch only
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
  PATH="$REPO_ROOT/.toolchain/bin:$PATH" \
  ./build.sh --fetch-only --config release

# generate only
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
  PATH="$REPO_ROOT/.toolchain/bin:$PATH" \
  ./build.sh --generate --config release

# build only
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
  PATH="$REPO_ROOT/.toolchain/bin:$PATH" \
  ./build.sh --build-only --config release -j 16
```
