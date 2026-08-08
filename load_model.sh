#!/usr/bin/env bash

# ==============================================================================
# Model Downloader Shell Script (load_model.sh)
# Uses `huggingface-cli download` to download Hugging Face models directly
# into the project's local cache folder (./hf_cache/hub).
#
# Optimized for transfer.lanta.nstda.or.th (LANTA HPC Transfer Node)
# No heavy Python / PyTorch / CUDA runtime required.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- HuggingFace Cache Directory Setup ---
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRIPT_DIR}/hf_cache}"
HUB_CACHE="${HF_CACHE_DIR}/hub"
mkdir -p "${HUB_CACHE}"

export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"

echo "================================================================="
echo "  [HF Cache] HF_HOME              = ${HF_HOME}"
echo "  [HF Cache] HUGGINGFACE_HUB_CACHE= ${HUGGINGFACE_HUB_CACHE}"
echo "================================================================="

# Find huggingface-cli executable or fallback to python module
if command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
elif command -v python3 &> /dev/null && python3 -m huggingface_hub.cli.huggingface_cli --help &> /dev/null; then
    HF_CMD="python3 -m huggingface_hub.cli.huggingface_cli"
elif command -v python &> /dev/null && python -m huggingface_hub.cli.huggingface_cli --help &> /dev/null; then
    HF_CMD="python -m huggingface_hub.cli.huggingface_cli"
else
    echo "[!] Warning: 'huggingface-cli' was not found."
    echo "[*] Attempting to install 'huggingface_hub' via pip..."
    pip install huggingface_hub
    if command -v huggingface-cli &> /dev/null; then
        HF_CMD="huggingface-cli"
    else
        HF_CMD="python3 -m huggingface_hub.cli.huggingface_cli"
    fi
fi

# Default models if no argument provided
DEFAULT_MODELS=(
    "unsloth/whisper-large-v3"
    "unsloth/orpheus-3b-0.1-ft"
    "hubertsiuzdak/snac_24khz"
)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    echo "[*] No model specified. Downloading default models:"
    for m in "${DEFAULT_MODELS[@]}"; do
        echo "  - $m"
    done
    MODELS=("${DEFAULT_MODELS[@]}")
fi

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "[*] Downloading model: ${MODEL} to ${HUB_CACHE}..."
    $HF_CMD download "${MODEL}" --cache-dir "${HUB_CACHE}"
done

echo ""
echo "[✓] All model downloads finished! Saved in: ${HUB_CACHE}"
