#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_raw:?set nnUNet_raw}"
: "${nnUNet_preprocessed:?set nnUNet_preprocessed}"
: "${nnUNet_results:?set nnUNet_results}"

DATASET_ID="${DATASET_ID:-501}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
M_PLANNER="${M_PLANNER:-nnUNetPlannerResEncM}"
M_PLANS="${M_PLANS:-nnUNetResEncUNetMPlans}"
XL_PLANNER="${XL_PLANNER:-nnUNetPlannerResEncXL}"
XL_PLANS="${XL_PLANS:-nnUNetResEncUNetXL30GBPlans}"

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -pl "${M_PLANNER}" \
  -c "${CONFIGURATION}" \
  -overwrite_plans_name "${M_PLANS}" \
  --verify_dataset_integrity

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -pl "${XL_PLANNER}" \
  -c "${CONFIGURATION}" \
  -overwrite_plans_name "${XL_PLANS}" \
  --verify_dataset_integrity
