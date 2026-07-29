# Training

## Environment

The pinned nnU-Net source revision used for inference is vendored under
`third_party/nnUNet`. Install it together with this repository and its training
dependencies. Export `nnUNet_raw`,
`nnUNet_preprocessed`, and `nnUNet_results`.

The final system contains three independent five-fold sources:

1. ResEncM (`m`);
2. ResEncM DiceCE/Focal-Tversky (`ft`), initialized from the matching M fold;
3. ResEncXL (`xl`).

The authoritative model identities are in
`configs/models/final_models.json`. Do not substitute `checkpoint_final.pth`
for `checkpoint_best.pth`.

## Five-fold order

Run M before FT so each FT fold receives the matching M best checkpoint:

```bash
python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model m --action train

python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model ft --action train

python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model xl --action train
```

Use `--folds 0 1` to select a subset. Use `--dry-run` to audit commands without
starting training.

## OOF probability export

After all folds finish, explicitly validate the best checkpoint and retain NPZ
probabilities:

```bash
for model in m ft xl; do
  python training/run_nnunet.py \
    --registry configs/models/final_models.json \
    --model "$model" --action validate
done
```

Fold membership is the leakage boundary: a case's learned-gate features must
come only from the model for which that case was held out.

## Learned LCv1/LCv2 gates

Copy the two JSON templates under `configs/learned_gate/` to a non-versioned
working directory and replace every `/path/to/...` value. Then run:

```bash
SHARDS=1 bash training/train_learned_gates.sh \
  /path/to/v1.json \
  /path/to/v2.json
```

The ordered stages are:

1. extract OOF case and component features;
2. consolidate the five-fold universe;
3. cross-fit and train the final LCv1 case model;
4. cross-fit and train the final LCv2 component model;
5. construct and consolidate calibration proposals;
6. evaluate the learned gate on held-out folds.

The final Docker needs only:

- LCv1 `models/lightgbm/final/models.joblib`;
- LCv2 `models/lightgbm/final/models.joblib`.

## RGv3-ET and utility-v4

`scripts/03_train_learned_gates.sh` also preserves the exact downstream
training order. Set:

```bash
export RGV3_COMPONENT_FEATURES=/path/to/component_features.parquet
export RGV3_CASE_PREDICTIONS=/path/to/lcv1_case_predictions.parquet
export RGV3_V2_PREDICTIONS=/path/to/lcv2_component_predictions.parquet
export RGV3_OUTPUT_ROOT=/path/to/rgv3-output

export UTILITY_FEATURES="$RGV3_COMPONENT_FEATURES"
export UTILITY_V2_PREDICTIONS="$RGV3_V2_PREDICTIONS"
export UTILITY_OUTPUT_ROOT=/path/to/utility-v4-output
```

The script cross-fits RGv3 by held-out case fold and then trains two
utility-v4 classifiers over ET components missing from the N03 baseline:
existence and geometry safety. Their source implementations are in
`training/learned_gates/`.

Retrained models are research artifacts. `04_export_inference_assets.sh`
accepts only the fixed release hashes; replacing a frozen model requires a new
candidate identity and a new evaluation.

Training labels, OOF predictions, and evaluation masks must never be included
as inference assets.

## One ordered launcher

After Dataset501 exists and both learned-gate configuration copies are ready:

```bash
bash scripts/01_prepare_dataset.sh /path/to/training /path/to/validation
bash scripts/02_train_models.sh
bash scripts/03_train_learned_gates.sh /path/to/v1.json /path/to/v2.json
```

This is an orchestration convenience, not a distributed scheduler. On a
multi-GPU cluster, launch folds with the same frozen commands through the local
scheduler while preserving model and handoff order.
