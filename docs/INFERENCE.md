# Inference

## Input contract

Each case must provide four geometrically aligned NIfTI files. Either naming
style is accepted:

```text
CASE-t1c.nii.gz     CASE_0000.nii.gz
CASE-t1n.nii.gz     CASE_0001.nii.gz
CASE-t2f.nii.gz     CASE_0002.nii.gz
CASE-t2w.nii.gz     CASE_0003.nii.gz
```

Do not mix naming styles within one case. The output directory must be empty.

## Frozen chain

The runtime performs:

```text
input
  -> XL five-fold best-checkpoint probabilities
  -> M five-fold best-checkpoint probabilities
  -> FT five-fold best-checkpoint probabilities
  -> LCv1 case features + LCv2 component scores
  -> XF12 structured-probability/V2-strict anchor
  -> N03 ET-only parent-supported baseline
  -> disconnected LCv2 structured-union ET proposals
  -> RGv3-ET + utility-v4 existence/geometry scoring
  -> accepted add-only ET updates
  -> one flat uint8 BraTS label map
```

The preserved N03 baseline uses union proposals at probability `0.25` and the
original parent-support rules. The final utility-v4 stage considers only
disconnected ET components absent from that baseline. A component is accepted
only when:

1. it belongs to the fixed LCv2 structured-union candidate pool;
2. its RGv3-ET score is at least `0.7702616034384248`;
3. its LCv2 component probability is at least `0.75`;
4. its utility-v4 existence probability is at least `0.75`;
5. its utility-v4 geometry-safety probability is at least `0.75`.

The update is add-only and preserves every N03 anchor voxel and RC priority.
The earlier component, TC-boundary, and strict-RC operations are not run a
second time after the ET addition.

## Output contract

The image writes exactly one root-level `CASE.nii.gz` per input case:

- reference shape, affine, and spacing are preserved;
- dtype is `uint8`;
- labels are restricted to `0,1,2,3,4`;
- there are no nested directories or auxiliary outputs.

The machine-readable runtime report is written to the configured temporary
work directory, not the submission output.

The frozen prediction ZIP is not a runtime input. It is consumed only by
`verification/verify_frozen_equivalence.py` after inference has finished.
