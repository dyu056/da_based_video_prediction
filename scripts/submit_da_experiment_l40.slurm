#!/bin/bash
#SBATCH --job-name=da_video_experiment
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_da_%j.out
#SBATCH --error=slurm_da_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/ssd/da_final_project}"
ENV_NAME="${ENV_NAME:-da-video-l40}"
ENV_TYPE="${ENV_TYPE:-conda}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-l40}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_NAME="${RUN_NAME:-da_experiment_${SLURM_JOB_ID}}"
CHECKPOINT="${CHECKPOINT:-outputs/da_host_openloop_v1/best_model.pt}"
SPLIT="${SPLIT:-val}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OBSERVATION_KIND="${OBSERVATION_KIND:-full}"
OBSERVE_EVERY="${OBSERVE_EVERY:-2}"
MASK_FRACTION="${MASK_FRACTION:-0.25}"
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-16}"
OBSERVATION_NOISE="${OBSERVATION_NOISE:-0.02}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-8}"
LATENT_NOISE="${LATENT_NOISE:-0.03}"
HIDDEN_NOISE="${HIDDEN_NOISE:-0.01}"
CELL_NOISE="${CELL_NOISE:-0.01}"
SEED="${SEED:-42}"

echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project directory: ${PROJECT_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Observation kind: ${OBSERVATION_KIND}"
echo "Observe every: ${OBSERVE_EVERY}"
echo "Environment type: ${ENV_TYPE}"

activate_conda_env() {
  find_conda() {
    if command -v conda >/dev/null 2>&1; then
      command -v conda
      return 0
    fi

    local candidate
    for candidate in \
      "$HOME/miniconda3/bin/conda" \
      "$HOME/miniconda/bin/conda" \
      "/opt/miniconda3/bin/conda" \
      "/usr/local/miniconda3/bin/conda"; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done

    return 1
  }

  CONDA_BIN="$(find_conda || true)"
  if [[ -z "${CONDA_BIN}" ]]; then
    echo "conda not found."
    exit 1
  fi

  CONDA_BASE="$("${CONDA_BIN}" info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

activate_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv activate script not found at ${VENV_DIR}/bin/activate"
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
}

activate_user_site() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter ${PYTHON_BIN} was not found."
    exit 1
  fi
}

cd "${PROJECT_DIR}"
mkdir -p "outputs/${RUN_NAME}"

case "${ENV_TYPE}" in
  conda)
    activate_conda_env
    ;;
  venv)
    activate_venv
    ;;
  user)
    activate_user_site
    ;;
  *)
    echo "Unsupported ENV_TYPE=${ENV_TYPE}. Use conda, venv, or user."
    exit 1
    ;;
esac

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

nvidia-smi || true
which "${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

srun "${PYTHON_BIN}" -u scripts/run_da_experiment.py \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --eval-sequences "${EVAL_SEQUENCES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --observation-kind "${OBSERVATION_KIND}" \
  --observe-every "${OBSERVE_EVERY}" \
  --mask-fraction "${MASK_FRACTION}" \
  --downsample-size "${DOWNSAMPLE_SIZE}" \
  --observation-noise "${OBSERVATION_NOISE}" \
  --ensemble-size "${ENSEMBLE_SIZE}" \
  --latent-noise "${LATENT_NOISE}" \
  --hidden-noise "${HIDDEN_NOISE}" \
  --cell-noise "${CELL_NOISE}" \
  --seed "${SEED}" \
  --device cuda \
  --output-dir "outputs/${RUN_NAME}"
