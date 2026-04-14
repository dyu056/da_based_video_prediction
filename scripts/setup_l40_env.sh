#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/ssd/da_final_project}"
ENV_FILE="${2:-environment-l40.yml}"
ENV_NAME="${ENV_NAME:-da-video-l40}"
ENV_TYPE="${ENV_TYPE:-conda}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-l40}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

cd "${PROJECT_DIR}"

activate_conda_env() {
  local conda_bin conda_base
  conda_bin="$(find_conda || true)"
  if [[ -z "${conda_bin}" ]]; then
    echo "conda was not found."
    echo "If this is your first login, initialize Conda first, reconnect, and rerun this script."
    exit 1
  fi

  conda_base="$("${conda_bin}" info --base)"
  source "${conda_base}/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Updating existing Conda environment ${ENV_NAME} from ${ENV_FILE}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    echo "Creating Conda environment ${ENV_NAME} from ${ENV_FILE}"
    conda env create -f "${ENV_FILE}"
  fi

  conda activate "${ENV_NAME}"
}

activate_venv() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter ${PYTHON_BIN} was not found."
    exit 1
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch
}

activate_user_site() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter ${PYTHON_BIN} was not found."
    exit 1
  fi

  "${PYTHON_BIN}" -m pip install --user --upgrade pip
  "${PYTHON_BIN}" -m pip install --user --upgrade --index-url https://download.pytorch.org/whl/cu124 torch
}

case "${ENV_TYPE}" in
  conda)
    activate_conda_env
    VERIFY_PYTHON="python"
    ;;
  venv)
    activate_venv
    VERIFY_PYTHON="python"
    ;;
  user)
    activate_user_site
    VERIFY_PYTHON="${PYTHON_BIN}"
    ;;
  *)
    echo "Unsupported ENV_TYPE=${ENV_TYPE}. Use conda, venv, or user."
    exit 1
    ;;
esac

"${VERIFY_PYTHON}" - <<'PY'
import sys
import torch
import numpy
import matplotlib
import sklearn
from PIL import Image

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
print("numpy:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
print("sklearn:", sklearn.__version__)
print("pillow:", Image.__version__)
PY

echo
if [[ "${ENV_TYPE}" == "conda" ]]; then
  echo "Environment ${ENV_NAME} is ready."
elif [[ "${ENV_TYPE}" == "venv" ]]; then
  echo "Virtualenv ${VENV_DIR} is ready."
else
  echo "User-site Python environment is ready via ${PYTHON_BIN}."
fi
echo "Project directory: ${PROJECT_DIR}"
