#!/usr/bin/env bash
# Reproduce ARCQuant on the full-precision SmolLM2-135M checkpoint.
#
# Usage (from the ARCQuant directory):
#   CONDA_ENV=<an-existing-conda-env> bash run_smollm2_135m_nvfp4.sh preflight
#   CONDA_ENV=<validated-conda-env> bash run_smollm2_135m_nvfp4.sh build
#   CONDA_ENV=<validated-conda-env> bash run_smollm2_135m_nvfp4.sh run
#
# The model and datasets are fetched through the standard Hugging Face cache.
# Set HF_HOME to the cache already used by MR-GPTQ to avoid duplicate downloads.

set -euo pipefail

MODE="${1:-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
MODEL_REPO="${MODEL_REPO:-HuggingFaceTB/SmolLM2-135M}"
MODEL_ALIAS="${ROOT}/models/llama-smollm2-135m"
CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
GPU_ID="${GPU_ID:-0}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
RUN_MMLU="${RUN_MMLU:-1}"

if [[ -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_DOWNLOAD_TIMEOUT="120"
export TOKENIZERS_PARALLELISM="false"

preflight() {
  command -v nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  command -v nvcc || true
  nvcc --version || true
  python - <<'PY'
import importlib
import sys

required = ("torch", "transformers", "datasets", "lm_eval", "pybind11")
print("python=", sys.version.replace("\n", " "))
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}=", getattr(module, "__version__", "installed"))
    except Exception as exc:
        print(f"{name}= MISSING ({exc})")

import torch
print("torch.cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
    print("compute_capability=", torch.cuda.get_device_capability(0))
PY
  echo "Expected for the official NVFP4 kernel: CUDA 12.8, a Blackwell GPU (sm_120), and a PyTorch build compatible with CUDA 12.8."
  echo "ARCQuant pins transformers==4.44.0; do not use an environment that has MR-GPTQ's transformers 5.8.0 unchanged."
}

prepare_model_alias() {
  mkdir -p "${ROOT}/models"
  local snapshot
  snapshot="$(MODEL_REPO="${MODEL_REPO}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=os.environ["MODEL_REPO"]))
PY
 )"
  ln -sfn "${snapshot}" "${MODEL_ALIAS}"
  echo "model_alias=${MODEL_ALIAS} -> ${snapshot}"
}

build_kernel() {
  git -C "${ROOT}" submodule update --init --recursive
  cmake -S "${ROOT}/kernels" -B "${ROOT}/kernels/build"
  cmake --build "${ROOT}/kernels/build" -j "${BUILD_JOBS:-$(nproc)}"
  export PYTHONPATH="${ROOT}/kernels/build:${PYTHONPATH:-}"
  python -c 'import agemm; print("agemm import: OK")'
}

run_arcquant() {
  export PYTHONPATH="${ROOT}/kernels/build:${PYTHONPATH:-}"
  cd "${ROOT}"

  # Strongest setting reported by the ARCQuant paper/repository:
  # WikiText-2 train, 128 random segments x 2048 tokens, seed 0, max sorting,
  # adaptive per-layer residual-channel selection, and W4A4 NVFP4.
  python reorder_indices.py \
    --model "${MODEL_ALIAS}" \
    --dataset "${CALIB_DATASET}" \
    --samples "${CALIB_SAMPLES}" \
    --seqlen "${SEQLEN}" \
    --act_sort_metric max

  python model/main.py "${MODEL_ALIAS}" \
    --seed 0 \
    --dataset "${CALIB_DATASET}" \
    --act_sort_metric max \
    --quant_type NVFP4 \
    --eval_ppl \
    --tasks piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy \
    --lm_eval_num_fewshot 0 \
    --lm_eval_limit "${EVAL_LIMIT}"

  if [[ "${RUN_MMLU}" == "1" ]]; then
    python model/main.py "${MODEL_ALIAS}" \
      --seed 0 \
      --dataset "${CALIB_DATASET}" \
      --act_sort_metric max \
      --quant_type NVFP4 \
      --tasks mmlu \
      --lm_eval_num_fewshot 5 \
      --lm_eval_limit "${EVAL_LIMIT}"
  fi
}

case "${MODE}" in
  preflight) preflight ;;
  build) preflight; build_kernel ;;
  run) preflight; prepare_model_alias; build_kernel; run_arcquant ;;
  *) echo "Usage: $0 {preflight|build|run}" >&2; exit 2 ;;
esac
