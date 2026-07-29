# Reproducibility

## Frozen

- Dataset501 channel and label semantics;
- upstream nnU-Net version and commit;
- ResEncM, ResEncXL, and FT trainer/plan/fold identities;
- best-checkpoint five-fold probability policy;
- LCv1, LCv2, RGv3, XF12, N03, and UTILITY_V4 implementations;
- numerical thresholds and add-only final update policy;
- Python dependency constraints and ordered entry scripts.

## External artifacts

The repository does not contain challenge data, checkpoints, learned gate
models, OOF/test probabilities, predictions, or platform scores. Exact
inference requires the corresponding trained artifacts. Retraining reproduces
the method but may not produce byte-identical weights because of stochastic
optimization and hardware-dependent kernels.

## Source verification

```bash
python -m pytest -q
python -m compileall -q compass_mets third_party/nnUNet/nnunetv2
python scripts/verify_release.py . --output release_audit.json
git status --short
```

The release audit checks required source/config/script entry points, the frozen
final policy, accidental model/data binaries, credential-like assignments,
private hostnames, and workspace-specific absolute paths. It does not require
challenge test outputs.
