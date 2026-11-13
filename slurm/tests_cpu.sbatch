#!/usr/bin/env bash
# The CPU tests (tests/run_cpu.sh): no GPU, no weights.
#   sbatch slurm/tests_cpu.sbatch
#SBATCH --job-name=budgetpix-tests-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/budgetpix-tests-cpu-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
bash tests/run_cpu.sh
