#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=avg_metrics
#SBATCH --output=logs/avg_metrics_%A_%a.log
#SBATCH --time=00:3:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=NONE
#SBATCH --mail-user=mehrdad3@illinois.edu

cd ${OUTDIR:-${BASE:-$PWD}}
mkdir -p logs mu plots metrics

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

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../src" ]; then
  export KEDRL_SRC="$PWD/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../kedrl_git/src" ]; then
  export KEDRL_SRC="$PWD/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ] && [ -d "$KEDRL_SRC" ]; then
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

## load modules/env -- delta
#module load cuda/12.3.0
#source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
#conda activate /u/mehrdad3/.conda/envs/delta_env
#echo "Using Python from: $(which python)"
#python -m pip show pot

python3 mu_plot.py
