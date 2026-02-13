#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Configurable variables
################################################################################

# Where to clone the IsaacSim repository
CLONE_PARENT_DIR="${CLONE_PARENT_DIR:-$HOME/projects}"

# IsaacSim repo information
REPO_URL="${REPO_URL:-https://github.com/isaac-sim/IsaacSim.git}"
REPO_NAME="${REPO_NAME:-IsaacSim}"
REPO_BRANCH="${REPO_BRANCH:-main}"

# If repo already exists, update it (fetch + checkout branch + pull)
UPDATE_EXISTING_REPO="${UPDATE_EXISTING_REPO:-false}"  # true|false

# Micromamba environment for GCC 11
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-isaacsim-gcc11}"

# Build options
BUILD_CONFIG="${BUILD_CONFIG:-release}"  # release|debug
BUILD_JOBS="${BUILD_JOBS:-16}"

# EULA handling
# Set to "yes" to auto-accept the EULA prompt during bootstrap.
ACCEPT_EULA="${ACCEPT_EULA:-no}"  # yes|no

################################################################################
# Derived paths
################################################################################

REPO_DIR="${CLONE_PARENT_DIR}/${REPO_NAME}"
TOOLCHAIN_DIR="${REPO_DIR}/.toolchain/bin"

################################################################################
# Helpers
################################################################################

log() {
  echo
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

ensure_parent_dir() {
  mkdir -p "${CLONE_PARENT_DIR}"
}

clone_or_update_repo() {
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning ${REPO_URL} into ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  else
    log "Repository already exists at ${REPO_DIR}"
    if [[ "${UPDATE_EXISTING_REPO}" == "true" ]]; then
      log "Updating existing repository"
      git -C "${REPO_DIR}" fetch --all --tags
      git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
      git -C "${REPO_DIR}" pull --ff-only
    fi
  fi

  # Ensure desired branch is checked out.
  git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
}

setup_git_lfs() {
  log "Configuring Git LFS and pulling LFS objects"
  git lfs install
  git -C "${REPO_DIR}" lfs pull
}

setup_micromamba_env() {
  eval "$(micromamba shell hook -s bash)"

  if ! micromamba env list | awk '{print $1}' | grep -qx "${MAMBA_ENV_NAME}"; then
    log "Creating micromamba environment ${MAMBA_ENV_NAME} with GCC 11 toolchain"
    micromamba create -y -n "${MAMBA_ENV_NAME}" -c conda-forge \
      gcc_linux-64=11 gxx_linux-64=11 binutils make cmake ninja
  else
    log "Micromamba environment ${MAMBA_ENV_NAME} already exists"
  fi
}

resolve_mamba_env_prefix() {
  eval "$(micromamba shell hook -s bash)"
  local prefix
  prefix="$(micromamba env list | awk -v env="${MAMBA_ENV_NAME}" '$1==env{print $NF}')"
  [[ -n "${prefix}" ]] || die "Failed to resolve prefix for micromamba env ${MAMBA_ENV_NAME}"
  echo "${prefix}"
}

create_toolchain_wrappers() {
  local env_prefix="$1"
  log "Creating GCC/G++ wrappers under ${TOOLCHAIN_DIR}"
  mkdir -p "${TOOLCHAIN_DIR}"

  cat > "${TOOLCHAIN_DIR}/gcc" <<EOF
#!/usr/bin/env bash
exec "${env_prefix}/bin/x86_64-conda-linux-gnu-gcc" "\$@"
EOF

  cat > "${TOOLCHAIN_DIR}/g++" <<EOF
#!/usr/bin/env bash
exec "${env_prefix}/bin/x86_64-conda-linux-gnu-g++" "\$@"
EOF

  chmod +x "${TOOLCHAIN_DIR}/gcc" "${TOOLCHAIN_DIR}/g++"

  PATH="${TOOLCHAIN_DIR}:$PATH" gcc --version | head -n 1
  PATH="${TOOLCHAIN_DIR}:$PATH" g++ --version | head -n 1
}

bootstrap_eula_if_needed() {
  cd "${REPO_DIR}"

  if [[ -f ".eula_accepted" ]]; then
    log "EULA already accepted"
    return
  fi

  if [[ "${ACCEPT_EULA}" != "yes" ]]; then
    die "EULA not accepted yet. Re-run with ACCEPT_EULA=yes to continue non-interactively."
  fi

  log "Accepting EULA non-interactively via build bootstrap"
  printf 'yes\n' | ./build.sh --help >/dev/null
}

build_isaacsim() {
  cd "${REPO_DIR}"
  log "Starting IsaacSim build (${BUILD_CONFIG}, jobs=${BUILD_JOBS})"

  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
    PATH="${TOOLCHAIN_DIR}:$PATH" \
    ./build.sh --config "${BUILD_CONFIG}" -j "${BUILD_JOBS}"
}

validate_build() {
  local launcher="${REPO_DIR}/_build/linux-x86_64/${BUILD_CONFIG}/isaac-sim.sh"
  local pysh="${REPO_DIR}/_build/linux-x86_64/${BUILD_CONFIG}/python.sh"

  [[ -x "${launcher}" ]] || die "Build finished but launcher not found: ${launcher}"
  [[ -x "${pysh}" ]] || die "Build finished but python.sh not found: ${pysh}"

  log "Launcher verified: ${launcher}"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u INCLUDE -u TBBROOT \
    "${pysh}" -c "import sys; print('IsaacSim Python OK:', sys.version.split()[0])"
}

################################################################################
# Main
################################################################################

require_cmd git
require_cmd git-lfs
require_cmd micromamba
require_cmd awk

ensure_parent_dir
clone_or_update_repo
setup_git_lfs
setup_micromamba_env

MAMBA_ENV_PREFIX="$(resolve_mamba_env_prefix)"
create_toolchain_wrappers "${MAMBA_ENV_PREFIX}"
bootstrap_eula_if_needed
build_isaacsim
validate_build

log "Done. IsaacSim repository: ${REPO_DIR}"
