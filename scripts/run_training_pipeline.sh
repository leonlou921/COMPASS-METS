#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT}/scripts/00_setup_environment.sh"
bash "${ROOT}/scripts/01_prepare_dataset.sh"
bash "${ROOT}/scripts/02_plan_and_preprocess.sh"
bash "${ROOT}/scripts/03_train_resencm_5fold.sh"
bash "${ROOT}/scripts/04_train_resencxl_5fold.sh"
bash "${ROOT}/scripts/05_train_small_lesion_ft_5fold.sh"
bash "${ROOT}/scripts/06_generate_oof_probabilities.sh"
bash "${ROOT}/scripts/07_train_learned_gates.sh"

if [[ "${RUN_TEST_INFERENCE:-0}" == 1 ]]; then
  bash "${ROOT}/scripts/08_predict_test_probabilities.sh"
  bash "${ROOT}/scripts/09_build_n03_utility_v4.sh"
fi
