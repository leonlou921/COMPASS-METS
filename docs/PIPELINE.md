# Pipeline map

```text
official BraTS-METS data
  -> 01 Dataset501 conversion
  -> 02 ResEncM/ResEncXL planning and preprocessing
  -> 03 ResEncM folds 0-4
  -> 04 ResEncXL folds 0-4
  -> 05 FT donor and retained small-lesion ablations
  -> 06 held-out best-checkpoint probabilities
  -> 07 LCv1 -> LCv2 -> RGv3 -> UTILITY_V4
  -> 08 test probabilities from XL -> M -> FT
  -> 09 XF12 -> N03 parent-supported -> UTILITY_V4 labels
```

## Handoffs

| Producer | Required consumer artifact |
|---|---|
| Dataset conversion | Dataset501 raw tree and `dataset.json` |
| Planning | M and XL plans in `nnUNet_preprocessed` |
| ResEncM training | fold-matched best checkpoints for FT initialization |
| Three model sources | OOF/test four-region probability NPZ files |
| LCv1 | case model and OOF case probabilities |
| LCv2 | component model, OOF component probabilities, fixed cutoffs |
| RGv3 | ET final model and OOF ET probabilities |
| UTILITY_V4 | existence model, geometry model, feature-name list |
| Final stage | one BraTS label NIfTI per input case |

All handoff directories are external artifacts and are ignored by Git.
