"""Frozen model-training and decision constants for the learned gate chain."""

from __future__ import annotations


FOLDS = (0, 1, 2, 3, 4)
SEED = 20260727
PROPOSAL_THRESHOLD = 0.25
RGV3_PRECISION_FLOORS = {"ET": 0.97, "TC": 0.98, "WT": 0.98}
N01_LCV2_ET_CUTOFF = 0.5497123599
RGV3_ET_CUTOFF = 0.7702616034384248
UTILITY_ACCEPT_THRESHOLD = 0.75
UTILITY_REJECT_THRESHOLD = 0.50
GEOMETRY_PRECISION_FLOOR = 0.50


def utility_decision(
    v2_probability: float,
    existence_probability: float,
    geometry_probability: float,
) -> str:
    scores = (
        float(v2_probability),
        float(existence_probability),
        float(geometry_probability),
    )
    if all(score >= UTILITY_ACCEPT_THRESHOLD for score in scores):
        return "accept"
    if (
        existence_probability < UTILITY_REJECT_THRESHOLD
        or geometry_probability < UTILITY_REJECT_THRESHOLD
    ):
        return "reject"
    return "rgv3_fallback"
