#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=RUN_GLOBAL
#SBATCH --output=logs/master-launch.%j.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=END
#SBATCH --mail-user=mehrdad3@illinois.edu

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env -- delta ai
#module purge
#module load cuda/12.6.1
#module load craype-accel-nvidia90
#module load python/miniforge3_pytorch/2.7.0

##load module/env -- delta
module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"   # IMPORTANT
conda activate delta_env

echo $SLURM_JOB_PARTITION
echo $CUDA_VISIBLE_DEVICES

set -euo pipefail

# ============================================================ #
# Setup
# ============================================================ #
BASE=$PWD
OUTBASE=$BASE/runs
RUN="$OUTBASE/global_eval_${SLURM_JOB_ID}"
mkdir -p "$OUTBASE" logs

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../src" ]; then
  KEDRL_SRC="$BASE/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../kedrl_git/src" ]; then
  KEDRL_SRC="$BASE/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ] && [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
else
  echo "Warning: KEDRL_SRC=${KEDRL_SRC:-unset} not found; relying on installed ke_drl package." >&2
fi

read -r OFFLINE_DATA_ID N_EVAL < <(python3 - <<'PY'
import yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
cfg = p.get("evaluation") or {}
target = p.get("target_set") or {}
print(int(cfg.get("offline_data_id", 0)), int(cfg.get("num_points", target.get("num_points", 30))))
PY
)

DATA_DIR="$BASE/data"
if [ ! -f "$DATA_DIR/offline_data_${OFFLINE_DATA_ID}.pt" ]; then
  echo "Missing $DATA_DIR/offline_data_${OFFLINE_DATA_ID}.pt. Run Job_data.sbatch first." >&2
  exit 1
fi

for ((j=0; j<N_EVAL; j++)); do
  if [ ! -f "$DATA_DIR/Z_true_${j}.pt" ]; then
    echo "Missing $DATA_DIR/Z_true_${j}.pt. Run Job_Z.sbatch and wait for all ${N_EVAL} array tasks." >&2
    exit 1
  fi
done

mkdir -p "$RUN" "$RUN/logs"

# ------------------------------------------------------------ #
# Copy required files and data snapshot for the global fit
# ------------------------------------------------------------ #
cp "$BASE/main_est.py"       "$RUN/"
cp "$BASE/mu_plot.py"        "$RUN/"
cp "$BASE/sim_utils.py"      "$RUN/"
cp "$BASE/sim_eval.py"       "$RUN/"
cp "$BASE/Job_est.sbatch"    "$RUN/"
cp "$BASE/Job_plot.sbatch"   "$RUN/"
cp "$BASE/params.yaml"       "$RUN/"
if [ -d "$BASE/utils" ]; then cp -r "$BASE/utils" "$RUN/"; fi
cp -r "$BASE/data"           "$RUN/"

# ------------------------------------------------------------ #
# Submit dependent global-estimation and plotting jobs
# ------------------------------------------------------------ #
cd "$RUN"
outdir="$RUN"

jid_est=$(sbatch --parsable \
    --output=${outdir}/logs/Job_est.%j.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_est.sbatch)

jid_plot=$(sbatch --parsable \
    --dependency=afterok:${jid_est} \
    --output=${outdir}/logs/Job_plot.%j.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_plot.sbatch)

echo "Submitted global simulation: Job_est=${jid_est}, Job_plot=${jid_plot}"
echo "Run folder: ${RUN}"
