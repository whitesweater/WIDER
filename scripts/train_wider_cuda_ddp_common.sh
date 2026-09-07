#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCHER_BASENAME:?LAUNCHER_BASENAME must be set by the wrapper script}"
: "${MODEL_PATH:?MODEL_PATH must be set by the wrapper script}"

usage() {
  cat <<EOF
Usage: bash scripts/${LAUNCHER_BASENAME} [DATASET] [METHOD] [extra train.py args...]

Datasets:
  strategyqa | commonsense | asdiv-aug | aqua

Methods:
  baseline | wider

Examples:
  bash scripts/${LAUNCHER_BASENAME} asdiv-aug baseline
  NPROC_PER_NODE=4 bash scripts/${LAUNCHER_BASENAME} asdiv-aug wider
EOF
}

is_method() {
  case "${1:-}" in
    baseline|wider|spectral)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

DATA_NAME="${DATA_NAME:-${DATASET:-asdiv-aug}}"
METHOD="${METHOD:-baseline}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac

  if is_method "$1"; then
    METHOD="$1"
    shift
  else
    DATA_NAME="$1"
    shift
    if [[ $# -gt 0 ]]; then
      if is_method "$1"; then
        METHOD="$1"
        shift
      elif [[ "$1" != -* ]]; then
        echo "Unsupported method: $1" >&2
        usage >&2
        exit 1
      fi
    fi
  fi
fi

EXTRA_TRAIN_ARGS=("$@")

case "${DATA_NAME}" in
  commonsenseqa|common-senseqa|commensenqa)
    DATA_NAME="commonsense"
    ;;
  asdiv|asdiv_aug|asdivaug)
    DATA_NAME="asdiv-aug"
    ;;
esac

case "${DATA_NAME}" in
  strategyqa|commonsense|asdiv-aug|aqua)
    ;;
  *)
    echo "Unsupported dataset: ${DATA_NAME}" >&2
    usage >&2
    exit 1
    ;;
esac

case "${METHOD}" in
  baseline|wider|spectral)
    ;;
  *)
    echo "Unsupported method: ${METHOD}" >&2
    usage >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${WIDER_DATA_DIR:-${REPO_ROOT}/data}"

source_safely() {
  local script_path="$1"
  local status
  local had_errexit=0
  case "$-" in
    *e*)
      had_errexit=1
      set +e
      ;;
  esac
  set +u
  # shellcheck disable=SC1090
  source "${script_path}"
  status=$?
  set -u
  if [[ "${had_errexit}" == "1" ]]; then
    set -e
  fi
  return "${status}"
}

activate_cuda_env() {
  local venv_activate="${REPO_ROOT}/.venv/bin/activate"
  if [[ -f "${venv_activate}" ]]; then
    source_safely "${venv_activate}"
    return 0
  fi
  echo "[error] no uv-managed environment found; run: uv sync" >&2
  return 1
}

ensure_dataset_ready() {
  case "$1" in
    strategyqa)
      [[ -s "${DATA_DIR}/strategyqa_train_clean.json" && -s "${DATA_DIR}/strategyqa_test_clean.json" ]] || python "${REPO_ROOT}/dataset/prepare_strategyqa.py"
      ;;
    commonsense)
      [[ -s "${DATA_DIR}/commonsense_train_clean.json" && -s "${DATA_DIR}/commonsense_val_clean.json" ]] || python "${REPO_ROOT}/dataset/prepare_commonsenseqa.py"
      ;;
    asdiv-aug)
      [[ -s "${DATA_DIR}/asdiv_aug_train_clean.json" && -s "${DATA_DIR}/asdiv_aug_val_clean.json" ]] || python "${REPO_ROOT}/dataset/prepare_asdiv_aug.py"
      ;;
    aqua)
      [[ -s "${DATA_DIR}/aqua_train_clean.json" && -s "${DATA_DIR}/aqua_val_clean.json" ]] || python "${REPO_ROOT}/dataset/prepare_aqua.py"
      ;;
  esac
}

activate_cuda_env

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WIDER_DATA_DIR="${DATA_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-${HOME}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WIDER_DATASET_CACHE_DIR="${WIDER_DATASET_CACHE_DIR:-${REPO_ROOT}/cache/dataset}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

ensure_dataset_ready "${DATA_NAME}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUN_TRAIN="${RUN_TRAIN:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_TOKEN_NUM="${MAX_TOKEN_NUM:-256}"
DISTILL_LOSS_DIV_STD="${DISTILL_LOSS_DIV_STD:-False}"
DISTILL_LOSS_FACTOR="${DISTILL_LOSS_FACTOR:-1.0}"
USE_DECODER="${USE_DECODER:-False}"
BF16="${BF16:-True}"

MODEL_LOG_TAG="${MODEL_LOG_TAG:-$(basename "${MODEL_PATH}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
LOG_ROOT="${OUTPUT_DIR}/logs"
MODEL_SHORT="$(basename "${MODEL_PATH}")"
SEED="${SEED:-11}"
if [[ -z "${EXPT_NAME:-}" ]]; then
  if [[ -n "${WIDER_RUN_SUFFIX:-}" ]]; then
    EXPT_NAME="${DATA_NAME}-${METHOD}-${WIDER_RUN_SUFFIX}"
  else
    EXPT_NAME="${DATA_NAME}-${METHOD}"
  fi
fi

TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
NUM_LATENT="${NUM_LATENT:-6}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
PRJ_DIM="${PRJ_DIM:-2048}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"

SPECTRAL_FACTOR=0.0

case "${METHOD}" in
  baseline)
    ;;
  wider|spectral)
    SPECTRAL_FACTOR="${SPECTRAL_FACTOR_OVERRIDE:-${SPECTRAL_LOSS_FACTOR:-0.1}}"
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}" "${WIDER_DATASET_CACHE_DIR}"
cd "${REPO_ROOT}"

TRAIN_CKPT_ROOT="${OUTPUT_DIR}/${EXPT_NAME}/${MODEL_SHORT}"

TRAIN_DTYPE_ARGS=()
case "${BF16}" in
  1|true|True|TRUE|yes|YES)
    TRAIN_DTYPE_ARGS=(--bf16)
    ;;
esac

print_runtime_config() {
  local extra_args_rendered="<none>"
  if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
    extra_args_rendered="$(printf '%q ' "${EXTRA_TRAIN_ARGS[@]}")"
    extra_args_rendered="${extra_args_rendered% }"
  fi

  echo
  echo "========== runtime config =========="
  echo "LAUNCHER_BASENAME=${LAUNCHER_BASENAME}"
  echo "BACKEND=cuda"
  echo "DATA_NAME=${DATA_NAME}"
  echo "METHOD=${METHOD}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "MODEL_LOG_TAG=${MODEL_LOG_TAG}"
  echo "MODEL_SHORT=${MODEL_SHORT}"
  echo "EXPT_NAME=${EXPT_NAME}"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "WIDER_DATA_DIR=${WIDER_DATA_DIR}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_ROOT=${LOG_ROOT}"
  echo "TRAIN_CKPT_ROOT=${TRAIN_CKPT_ROOT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "RUN_TRAIN=${RUN_TRAIN}"
  echo "BF16=${BF16}"
  echo "MAX_TOKEN_NUM=${MAX_TOKEN_NUM}"
  echo "USE_DECODER=${USE_DECODER}"
  echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
  echo "NUM_LATENT=${NUM_LATENT}"
  echo "PRJ_DIM=${PRJ_DIM}"
  echo "SPECTRAL_FACTOR=${SPECTRAL_FACTOR}"
  echo "EXTRA_TRAIN_ARGS=${extra_args_rendered}"
  echo "===================================="
}

run_training() {
  echo
  echo "========== [${EXPT_NAME}] start training =========="
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_name "${DATA_NAME}" \
    --seed "${SEED}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    "${TRAIN_DTYPE_ARGS[@]}" \
    --lora_r 128 --lora_alpha 32 --lora_init True \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOG_ROOT}/${EXPT_NAME}-${MODEL_LOG_TAG}-logs" \
    --expt_name "${EXPT_NAME}" \
    --num_train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
    --num_latent "${NUM_LATENT}" \
    --use_prj True \
    --prj_dim "${PRJ_DIM}" \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --max_token_num "${MAX_TOKEN_NUM}" \
    --distill_loss_div_std "${DISTILL_LOSS_DIV_STD}" \
    --distill_loss_factor "${DISTILL_LOSS_FACTOR}" \
    --use_decoder "${USE_DECODER}" \
    --remove_eos True \
    --use_lora True \
    --spectral_loss_factor "${SPECTRAL_FACTOR}" \
    --report_to none \
    "${EXTRA_TRAIN_ARGS[@]}"
}

print_runtime_config

if [[ "${RUN_TRAIN}" == "1" ]]; then
  run_training
fi

echo
echo "========== ${EXPT_NAME} [${MODEL_SHORT}] training done =========="
