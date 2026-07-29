# Learned gate training sources

This directory preserves the exact project-side training implementations used
for the downstream learned gates that are not part of upstream nnU-Net.

- `rgv3/region_gate_v3.py` trains the separated ET and TC/WT regional
  component models with five-fold, case-disjoint OOF predictions.
- `utility_v4/utility_v4.py` defines the N03-missing ET proposal universe,
  leakage exclusions, targets, and deterministic LightGBM classifiers.
- `utility_v4/train_utility_v4.py` cross-fits and trains the final existence
  and geometry-safety models.

The generated binaries are deliberately excluded from Git. The final image
export accepts only the frozen model hashes listed in
`inference/src/n03_docker/asset_inventory.py`. Retraining is a scientific
reproduction path, not permission to silently replace the release assets.
