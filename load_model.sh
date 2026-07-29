#!/usr/bin/env bash

# ==============================================================================
# Model Loader Shell Script (load_model.sh)
# Wrapper for load_model.py providing interactive CLI model selection.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect python executable (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo "[!] Error: Python is not installed or not found in system PATH."
    exit 1
fi

# Execute python script passing all CLI arguments
"$PYTHON_BIN" "$SCRIPT_DIR/load_model.py" "$@"
