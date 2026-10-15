#!/usr/bin/env bash
# CPU development environment only. GPU allocation is an explicit separate job.
set -euo pipefail
lafa_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lafa_venv_dir="${LAFA_VENV_DIR:-/workspace/lafa-venv}"
python3 -m venv --system-site-packages "$lafa_venv_dir"
"$lafa_venv_dir/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
"$lafa_venv_dir/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  -e "$lafa_repo_dir[test,notebooks]" build
"$lafa_venv_dir/bin/python" -m pip check
"$lafa_venv_dir/bin/python" -c 'import lafa, torch; print("LaFA", lafa.__version__, "PyTorch", torch.__version__, "CUDA available:", torch.cuda.is_available())'
