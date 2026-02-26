#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

cd "$REPO_ROOT"

UV_PY312="$(uv python find 3.12 2>/dev/null || true)"
if [[ -z "$UV_PY312" ]]; then
  echo "Python 3.12 not found via uv. Run: uv python install 3.12" >&2
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

export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="${TORCH_LIB}:${DYLD_LIBRARY_PATH:-}"

if [[ ! -x "${REPO_ROOT}/brainstorm" ]]; then
  if [[ -x "${REPO_ROOT}/target/release/brainstorm" ]]; then
    cp "${REPO_ROOT}/target/release/brainstorm" "${REPO_ROOT}/brainstorm"
  else
    echo "Engine binary not found. Run: ./build.sh" >&2
    exit 1
  fi
fi

exec uv run --python "$UV_PY312" gui.py
