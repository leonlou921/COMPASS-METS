# Configuration reference

## Dataset

`configs/dataset501/dataset.json` freezes channel names, region definitions,
label order, and file ending.

## Trainers

`configs/trainers/` contains:

- `resencm.json`: final ResEncM source;
- `resencxl.json`: final ResEncXL source;
- `focal_tversky.json`: final FT donor initialized from ResEncM;
- `small_lesion_os.json`: retained small-lesion oversampling ablation;
- `component_small_lesion_os.json`: retained component-aware OS ablation.

Every final source uses folds `0,1,2,3,4` and `checkpoint_best.pth`.

## Learned gates

`configs/learned_gates/` contains portable templates for LCv1 and LCv2 and
the frozen numerical contracts for RGv3 and UTILITY_V4. Private paths belong
only in copies outside Git.

## Fusion and final policy

`configs/fusion/xf12.json` records the structured-probability anchor,
component filtering, TC boundary completion, and strict RC settings.

`configs/final/n03_utility_v4.json` records the parent-supported ET-only N03
policy, region-specific LCv2 cutoffs, RGv3 cutoff, three-state utility gate,
anchor preservation, RC priority, and the prohibition on a second global V2
postprocessing pass.
