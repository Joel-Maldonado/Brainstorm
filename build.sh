#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

PROFILE="release"
RUN_CHECK=1
COPY_BINARY=1
INSTALL_DEPS=0

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --release        Build release profile (default)
  --debug          Build debug profile
  --no-check       Skip `cargo check`
  --no-copy        Do not copy built binary to ./brainstorm
  --install-deps   Install Python deps with uv (torch==2.4.0, pygame, python-chess)
  -h, --help       Show this help text
EOF
}

log() {
  printf '[build] %s\n' "$*"
}

for arg in "$@"; do
  case "$arg" in
    --release) PROFILE="release" ;;
    --debug) PROFILE="debug" ;;
    --no-check) RUN_CHECK=0 ;;
    --no-copy) COPY_BINARY=0 ;;
    --install-deps) INSTALL_DEPS=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

cd "$REPO_ROOT"

UV_PY312="$(uv python find 3.12 2>/dev/null || true)"
if [[ -z "$UV_PY312" && "$INSTALL_DEPS" -eq 1 ]]; then
  log "Installing Python 3.12 with uv"
  uv python install 3.12
  UV_PY312="$(uv python find 3.12 2>/dev/null || true)"
fi

if [[ -z "$UV_PY312" ]]; then
  echo "Python 3.12 not found via uv. Run: uv python install 3.12" >&2
  exit 1
fi

export PATH="$(dirname "$UV_PY312"):$PATH"
export LIBTORCH_USE_PYTORCH=1

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  log "Installing Python dependencies into uv-managed Python 3.12"
  uv pip install --system --break-system-packages --python "$UV_PY312" \
    torch==2.4.0 pygame python-chess
fi

TORCH_VERSION="$(uv run --python "$UV_PY312" python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
try:
    import torch
except Exception:
    print("")
else:
    print(torch.__version__.split("+")[0])
PY
)"

if [[ -z "$TORCH_VERSION" ]]; then
  echo "PyTorch is not installed for $UV_PY312." >&2
  echo "Run: ./build.sh --install-deps" >&2
  exit 1
fi

if [[ "$TORCH_VERSION" != "2.4.0" ]]; then
  echo "Installed torch version is ${TORCH_VERSION}, but this repo expects 2.4.0 (tch 0.17.0)." >&2
  echo "Run: ./build.sh --install-deps" >&2
  exit 1
fi

TORCH_LIB="$(uv run --python "$UV_PY312" python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "lib")
PY
)"
export DYLD_LIBRARY_PATH="${TORCH_LIB}:${DYLD_LIBRARY_PATH:-}"

log "Using Python: $UV_PY312"
log "Using torch: $TORCH_VERSION"
log "Using torch lib dir: $TORCH_LIB"

if [[ "$RUN_CHECK" -eq 1 ]]; then
  log "Running cargo check"
  cargo check
fi

if [[ "$PROFILE" == "release" ]]; then
  log "Building release binary"
  cargo build --release
  BUILT_BIN="${REPO_ROOT}/target/release/brainstorm"
else
  log "Building debug binary"
  cargo build
  BUILT_BIN="${REPO_ROOT}/target/debug/brainstorm"
fi

if [[ "$COPY_BINARY" -eq 1 ]]; then
  cp "$BUILT_BIN" "${REPO_ROOT}/brainstorm"
  log "Copied binary to ${REPO_ROOT}/brainstorm"
fi

log "Build complete: $BUILT_BIN"
