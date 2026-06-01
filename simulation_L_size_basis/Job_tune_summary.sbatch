#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=KEDRL_SIM3_SUM
#SBATCH --output=logs/sim3_tune_summary.%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=64
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env -- delta ai
module purge
module load cuda/12.6.1
module load default gcc/14.2.0 craype-accel-nvidia90
module load python/miniforge3_pytorch/2.7.0
##load module/env -- delta
#module load slurm-env/0.1
#module load cudatoolkit/25.3_12.8
#module load pytorch-conda/2.8
#eval "$(conda shell.bash hook)"   # IMPORTANT
#conda activate delta_env
echo $SLURM_JOB_PARTITION
echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

set -euo pipefail

BASE=$PWD
unset KEDRL_SRC
KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
KEDRL_PIP_TARGET="$BASE/runs/.kedrl_site_summary_${SLURM_JOB_ID:-manual}"
rm -rf "$KEDRL_PIP_TARGET"
mkdir -p "$KEDRL_PIP_TARGET"
echo "Installing ke_drl from git: ${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
  --target "$KEDRL_PIP_TARGET" "git+${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
export PYTHONPATH="$KEDRL_PIP_TARGET:${PYTHONPATH:-}"
python3 - <<'PY'
import ke_drl
print(f"Installed ke_drl source: {ke_drl.__file__}")
PY

shopt -s nullglob
for RUN in "$BASE"/runs/sim3_global_*; do
  [ -d "$RUN" ] || continue
  [ -f "$RUN/main_tune_global.py" ] || continue
  grep -q -- "--aggregate-only" "$RUN/main_tune_global.py" || { echo "Skipping $RUN: old tuning driver."; continue; }
  if ! compgen -G "$RUN/mu/mu_hat_*.csv" >/dev/null; then
    echo "Skipping $RUN: no mu_hat files yet."
    continue
  fi
  COMBO_ID=$(python3 - "$RUN" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
meta = run / "metrics" / "tuning_combo_metadata.json"
if meta.exists():
    print(json.loads(meta.read_text(encoding="utf-8"))["combo_id"])
else:
    m = re.search(r"_([0-9]+)$", run.name)
    if not m:
        raise SystemExit(f"cannot infer combo id from {run}")
    print(int(m.group(1)))
PY
)
  cp "$BASE/main_tune_global.py" "$RUN/"
  cp "$BASE/mu_plot.py" "$RUN/"
  cp "$BASE/sim_eval.py" "$RUN/"
  cp "$BASE/sim_utils.py" "$RUN/"
  cp "$BASE/params_tune.yaml" "$RUN/"
  cp "$BASE/tuning_grid.yaml" "$RUN/"
  python3 - "$BASE/params_tune.yaml" "$RUN/params.yaml" <<'PY'
import sys
import yaml

base_params = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
run_params = yaml.safe_load(open(sys.argv[2], "r", encoding="utf-8"))
if "reporting_filter" in base_params:
    run_params["reporting_filter"] = base_params["reporting_filter"]
elif "diagnostic_filter" in base_params:
    run_params["diagnostic_filter"] = base_params["diagnostic_filter"]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    yaml.safe_dump(run_params, f, sort_keys=False)
PY
  echo "Aggregating $RUN (combo ${COMBO_ID})"
  (cd "$RUN" && python3 main_tune_global.py --combo-id "$COMBO_ID" --aggregate-only)
done

python3 summarize_tuning.py
