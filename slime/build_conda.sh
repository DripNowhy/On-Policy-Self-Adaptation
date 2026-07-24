#!/usr/bin/env bash

set -e
# set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${SCRIPT_DIR}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-slime}"
CONDA_ROOT="${CONDA_ROOT:-/scratch/gautschi/ding432/anaconda3}"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # Prefer the explicit Miniforge install used on this machine.
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Could not find conda. Set CONDA_ROOT or add conda to PATH first." >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"
export CUDA_HOME="${CONDA_PREFIX}"
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
mkdir -p "${HOME}/.cargo"
touch "${HOME}/.cargo/env"

cd "${BASE_DIR}"

# # install cuda 12.9 as it's the default cuda version for torch
# conda install -n "${CONDA_ENV_NAME}" cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
# conda install -n "${CONDA_ENV_NAME}" -c conda-forge cudnn -y

# # prevent installing cuda 13.0 for sglang
# pip install cuda-python==13.1.0
# pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# # install sglang
# if [ ! -d "${BASE_DIR}/sglang" ]; then
#   git clone https://github.com/sgl-project/sglang.git
# fi
# cd "${BASE_DIR}/sglang"
# git checkout "${SGLANG_COMMIT}"
# # Install the python packages
# pip install -e "python[all]"


# pip install cmake ninja

# flash-attn must be installed on a compute node.
# The newest version Megatron supports is v2.7.4.post1.
# if [[ -n "${SLURM_JOB_ID:-}" ]]; then
#   MAX_JOBS="${MAX_JOBS:-14}" pip -v install flash-attn==2.7.4.post1 --no-build-isolation
# else
#   cat <<EOF
# Skipping flash-attn on the login node.
# Run the following on a compute node after this script finishes:

# srun --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=14 --partition=ai -A ruqiz --qos=preemptible --time=12:00:00 --mem=128G \\
#   bash -lc 'source "${CONDA_ROOT}/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV_NAME}" && MAX_JOBS=14 pip -v install flash-attn==2.7.4.post1 --no-build-isolation'
# EOF
# fi

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.1
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# megatron
cd "${BASE_DIR}"
if [ ! -d "${BASE_DIR}/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd "${BASE_DIR}/Megatron-LM"
git checkout "${MEGATRON_COMMIT}"
pip install -e .

# install slime and apply patches

cd "${SLIME_DIR}"
pip install -e .

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# apply patch
cd "${BASE_DIR}/sglang"
git apply "${SLIME_DIR}/docker/patch/v0.5.9/sglang.patch"
cd "${BASE_DIR}/Megatron-LM"
git apply "${SLIME_DIR}/docker/patch/v0.5.9/megatron.patch"