#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ROOT=$(dirname "${SCRIPT_DIR}")
REGISTRY="${ROOT}/configs/models/final_models.json"
V1_CONFIG="${1:?usage: run_training_pipeline.sh V1_CONFIG V2_CONFIG}"
V2_CONFIG="${2:?usage: run_training_pipeline.sh V1_CONFIG V2_CONFIG}"

bash "${ROOT}/preprocessing/plan_and_preprocess.sh"

python "${ROOT}/training/run_nnunet.py" \
  --registry "${REGISTRY}" --model m --action train
python "${ROOT}/training/run_nnunet.py" \
  --registry "${REGISTRY}" --model ft --action train
python "${ROOT}/training/run_nnunet.py" \
  --registry "${REGISTRY}" --model xl --action train

for model in m ft xl; do
  python "${ROOT}/training/run_nnunet.py" \
    --registry "${REGISTRY}" --model "${model}" --action validate
done

bash "${ROOT}/training/train_learned_gates.sh" "${V1_CONFIG}" "${V2_CONFIG}"
