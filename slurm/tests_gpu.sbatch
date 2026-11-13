#!/usr/bin/env bash
# The full test suite (tests/run_gpu.sh) on one GPU, about 30 minutes; its header lists the variables it reads.
#   MINIT2I_ROOT=... PIXELDIT_ROOT=... PMF_ROOT=... MINIT2I_WEIGHTS=... PIXELDIT_WEIGHTS=... sbatch slurm/tests_gpu.sbatch
#SBATCH --job-name=budgetpix-tests
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/budgetpix-tests-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash tests/run_gpu.sh
