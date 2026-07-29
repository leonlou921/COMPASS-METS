# Third-party notices

## nnU-Net

This repository vendors the complete nnU-Net source tree from
[MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), version 2.6.2, pinned to
commit `86606c53ef9f556d6f024a304b52a48378453641`.

nnU-Net is distributed under the Apache License 2.0. Its original license is
retained at `third_party/nnUNet/LICENSE`.

The vendored tree differs from that upstream commit only by the challenge
trainer variants in
`nnunetv2/training/nnUNetTrainer/variants/brats_mets/`. Those local additions
implement the focal-Tversky donor and the two retained small-lesion
oversampling ablations used by this project.
