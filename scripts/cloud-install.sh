#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Meridian-Rank
export PIP_CACHE_DIR=/tmp/meridian-pip-cache
export MPLCONFIGDIR=/tmp/meridian-mpl
export XDG_CACHE_HOME=/tmp/meridian-cache
python -m venv .venv
if .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  .venv/bin/python -m pip install -c constraints/python312-cloud.txt -e '.[dev]'
else
  .venv/bin/python -m pip install -e '.[dev]'
fi
.venv/bin/python -m pip check
