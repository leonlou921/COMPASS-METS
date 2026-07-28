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
  -> ET-only parent-supported additions
  -> one flat uint8 BraTS label map
```

N03 uses union proposals at probability `0.25`. An ET component is eligible
only when:

1. its LCv2 score is at least `0.5497123599`;
2. at least two of XL/M/FT support its ET region at `0.25`;
3. at least two of XL/M/FT support its TC parent at `0.25`;
4. at least two of XL/M/FT support its WT parent at `0.25`.

The update is add-only and preserves the XF12 anchor. The earlier component,
TC-boundary, and strict-RC operations are not run a second time after N03.

## Output contract

The image writes exactly one root-level `CASE.nii.gz` per input case:

- reference shape, affine, and spacing are preserved;
- dtype is `uint8`;
- labels are restricted to `0,1,2,3,4`;
- there are no nested directories or auxiliary outputs.

The machine-readable runtime report is written to the configured temporary
work directory, not the submission output.
