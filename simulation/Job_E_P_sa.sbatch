#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=RUN_GLOBAL
#SBATCH --output=logs/master-launch.%j.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA40x4
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
RUN="$OUTBASE/main_${SLURM_JOB_ID}"
mkdir -p "$OUTBASE" logs

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

N_REP=$(python3 - <<'PY'
import yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
print(int((p.get("experiment") or {}).get("num_replicates", 1)))
PY
)

DATA_DIR="$BASE/data"
for ((j=0; j<N_REP; j++)); do
  if [ ! -f "$DATA_DIR/offline_data_${j}.pt" ]; then
    echo "Missing $DATA_DIR/offline_data_${j}.pt. Run Job_data.sbatch and wait for all ${N_REP} array tasks." >&2
    exit 1
  fi
  if [ ! -f "$DATA_DIR/Z_true_${j}.pt" ]; then
    echo "Missing $DATA_DIR/Z_true_${j}.pt. Run Job_Z.sbatch and wait for all ${N_REP} array tasks." >&2
    exit 1
  fi
done
python3 validate_sim_config.py --params "$BASE/params.yaml" --data "$DATA_DIR/offline_data_0.pt"
N_REP_ENV="$N_REP" DATA_DIR_ENV="$DATA_DIR" python3 - <<'PY'
import os
from pathlib import Path

import torch

n_rep = int(os.environ["N_REP_ENV"])
data_dir = Path(os.environ["DATA_DIR_ENV"])
for j in range(n_rep):
    z_path = data_dir / f"Z_true_{j}.pt"
    blob = torch.load(z_path, map_location="cpu")
    meta = blob.get("metadata", {})
    offline_id = int(meta.get("offline_data_id", -999999))
    if offline_id != j:
        raise SystemExit(
            f"{z_path} has metadata offline_data_id={offline_id}, expected {j}. "
            "This usually means stale true-Z files are mixed with the current run."
        )
    if "s_star" not in meta or "a_star" not in meta:
        raise SystemExit(f"{z_path} is missing benchmark s_star/a_star metadata.")
print(f"Benchmark true-Z metadata OK for {n_rep} replicates.")
PY
echo "Found ${N_REP} offline datasets and benchmark true-Z files."

mkdir -p "$RUN" "$RUN/logs"

# ------------------------------------------------------------ #
# Copy required files and data snapshot for the global fit
# ------------------------------------------------------------ #
cp "$BASE/main_est.py"       "$RUN/"
cp "$BASE/mu_plot.py"        "$RUN/"
cp "$BASE/sim_utils.py"      "$RUN/"
cp "$BASE/sim_eval.py"       "$RUN/"
cp "$BASE/validate_sim_config.py" "$RUN/"
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
    --array=0-$((N_REP-1)) \
    --output=${outdir}/logs/Job_est.%A_%a.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_est.sbatch)

jid_plot=$(sbatch --parsable \
    --dependency=afterok:${jid_est} \
    --output=${outdir}/logs/Job_plot.%j.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_plot.sbatch)

echo "Submitted global simulation: Job_est=${jid_est}, Job_plot=${jid_plot}"
echo "Run folder: ${RUN}"
