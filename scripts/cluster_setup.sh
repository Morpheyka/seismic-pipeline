#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r seismic_pipeline_standalone/requirements.txt
pip install mkl mkl-service
pip install pymc arviz pytensor numpy scipy pandas matplotlib joblib tqdm

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Smoke test:  cd seismic_pipeline_standalone && python scripts/smoke_test_parallel_search.py"
echo "Full search: cd seismic_pipeline_standalone && python scripts/run_parallel_search.py"
