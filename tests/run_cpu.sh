#!/usr/bin/env bash
# The BudgetPix tests that need no GPU, no weights and no upstream checkout; stops at the first failure.
#   bash tests/run_cpu.sh
# Sampling from the checkpoints and the training steps of every family are tests/run_gpu.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
step() { echo; echo "=== $1"; }
step "1/1 token layouts of every family: tiling, budgets, packing"; $PY tests/test_layouts.py
echo; echo "all BudgetPix CPU tests passed"
