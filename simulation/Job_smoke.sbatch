#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=KEDRL_SMOKE
#SBATCH --output=logs/smoke.%j.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
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

set -euo pipefail

BASE=$PWD
RUN="$BASE/runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$BASE/logs" "$RUN" "$RUN/data" "$RUN/logs" "$RUN/mu" "$RUN/metrics" "$RUN/plots"

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../src" ]; then
  KEDRL_SRC="$BASE/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../kedrl_git/src" ]; then
  KEDRL_SRC="$BASE/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ] && [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
  echo "KEDRL_SRC: $KEDRL_SRC"
else
  echo "KEDRL_SRC=${KEDRL_SRC:-unset} not found; using installed ke_drl package." >&2
  if [ "${KEDRL_REINSTALL_FROM_GIT:-0}" = "1" ]; then
    python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"
  fi
  python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("ke_drl")
if spec is None:
    raise SystemExit('ke_drl is not installed. Run: python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"')
print(f"Installed ke_drl source: {spec.origin}")
PY
fi

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

cp "$BASE/main_offlinedata.py" "$RUN/"
cp "$BASE/main_MonteCarloZ.py" "$RUN/"
cp "$BASE/main_est.py" "$RUN/"
cp "$BASE/mu_plot.py" "$RUN/"
cp "$BASE/sim_utils.py" "$RUN/"
cp "$BASE/sim_eval.py" "$RUN/"
cp "$BASE/validate_sim_config.py" "$RUN/"
cp "$BASE/parallel_offlinedata.py" "$RUN/"
cp "$BASE/params_smoke.yaml" "$RUN/params.yaml"

cd "$RUN"
echo "Smoke run folder: $(pwd)"
echo "Using params:"
python3 - <<'PY'
import yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
print({
    "n_ids": p["n_ids"],
    "Z_sim": p["Z_sim"],
    "experiment": p["experiment"],
    "benchmark": p["benchmark"],
    "target_set": p["target_set"],
    "num_grid_points": p["num_grid_points"],
    "num_steps": p["optimization"]["num_steps"],
})
PY

echo "===== SMOKE 0/4: config validation ====="
python3 validate_sim_config.py --params params.yaml

N_REP=$(python3 - <<'PY'
import yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
print(int((p.get("experiment") or {}).get("num_replicates", 1)))
PY
)

echo "===== SMOKE 1/4: offline data for ${N_REP} replicates ====="
python3 parallel_offlinedata.py --params params.yaml

echo "===== SMOKE 2/4: one fixed Monte Carlo benchmark true Z ====="
python3 main_MonteCarloZ.py

echo "===== SMOKE 3/4: one global B fit per replicate ====="
for ((r=0; r<N_REP; r++)); do
  export SLURM_ARRAY_TASK_ID="$r"
  export OFFLINE_DATA_ID="$r"
  python3 main_est.py
done
unset SLURM_ARRAY_TASK_ID
unset OFFLINE_DATA_ID

echo "===== SMOKE 4/4: aggregate plots/metrics ====="
python3 mu_plot.py

echo "Smoke outputs:"
ls -lh data/fit_0.pt data/Zeval_0.pt metrics/per_run_metrics.csv plots/mu_summary_UG.png
echo "SMOKE TEST DONE"
