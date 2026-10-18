#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python 3.11 or newer is required.")
print("Python version:", sys.version)
PY

mkdir -p books models data/extracted data/cleaned \
    data/chunks artifacts/index evaluation tests

if [[ ! -d .venv ]]; then
    python -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    pymupdf pyyaml numpy sentence-transformers \
    faiss-cpu streamlit pytest psutil huggingface-hub \
    llama-cpp-python

python -m pip freeze > requirements-linux.lock.txt

echo
echo "Setup finished."
echo "Activate with: source .venv/bin/activate"
