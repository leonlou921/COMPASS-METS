#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
require_env nnUNet_raw nnUNet_preprocessed nnUNet_results

DATASET_ID=501
CONFIGURATION=3d_fullres

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -pl nnUNetPlannerResEncM \
  -c "${CONFIGURATION}" \
  -overwrite_plans_name nnUNetResEncUNetMPlans \
  --verify_dataset_integrity

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -pl nnUNetPlannerResEncXL \
  -c "${CONFIGURATION}" \
  -overwrite_plans_name nnUNetResEncUNetXL30GBPlans \
  --verify_dataset_integrity
