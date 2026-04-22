#!/bin/bash
#SBATCH --job-name=da_video_baseline
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_baseline_%j.out
#SBATCH --error=slurm_baseline_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/ssd/da_final_project}"
ENV_NAME="${ENV_NAME:-da-video-l40}"
ENV_TYPE="${ENV_TYPE:-conda}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-l40}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL="${MODEL:-simvp}"
RUN_NAME="${RUN_NAME:-${MODEL}_baseline_${SLURM_JOB_ID}}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-4000}"
VAL_SEQUENCES="${VAL_SEQUENCES:-800}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-10}"
PRED_FRAMES="${PRED_FRAMES:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
NUM_DIGITS="${NUM_DIGITS:-2}"
LATENT_DIM="${LATENT_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-0}"
PRED_L1_WEIGHT="${PRED_L1_WEIGHT:-1.0}"
PRED_BCE_WEIGHT="${PRED_BCE_WEIGHT:-0.4}"
PRED_MSE_WEIGHT="${PRED_MSE_WEIGHT:-0.2}"
TEACHER_FORCE_START="${TEACHER_FORCE_START:-0.6}"
TEACHER_FORCE_END="${TEACHER_FORCE_END:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project directory: ${PROJECT_DIR}"
echo "Model: ${MODEL}"
echo "Run name: ${RUN_NAME}"
echo "Latent dim: ${LATENT_DIM}"
echo "Hidden dim: ${HIDDEN_DIM}"
echo "Num workers: ${NUM_WORKERS}"
echo "Save every epoch: ${SAVE_EVERY_EPOCH}"
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

nvidia-smi || true
which "${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

EXTRA_ARGS=()
if [[ "${SAVE_EVERY_EPOCH}" == "1" ]]; then
  EXTRA_ARGS+=(--save-every-epoch)
fi

srun "${PYTHON_BIN}" -u scripts/train_baseline.py \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-sequences "${TRAIN_SEQUENCES}" \
  --val-sequences "${VAL_SEQUENCES}" \
  --context-frames "${CONTEXT_FRAMES}" \
  --pred-frames "${PRED_FRAMES}" \
  --image-size "${IMAGE_SIZE}" \
  --num-digits "${NUM_DIGITS}" \
  --latent-dim "${LATENT_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-workers "${NUM_WORKERS}" \
  --pred-l1-weight "${PRED_L1_WEIGHT}" \
  --pred-bce-weight "${PRED_BCE_WEIGHT}" \
  --pred-mse-weight "${PRED_MSE_WEIGHT}" \
  --teacher-force-start "${TEACHER_FORCE_START}" \
  --teacher-force-end "${TEACHER_FORCE_END}" \
  --grad-clip "${GRAD_CLIP}" \
  --device cuda \
  --output-dir "outputs/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
