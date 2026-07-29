"""Frozen utility-v4 three-state gate for disconnected ET additions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


ACCEPT_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.50
RGV3_ET_CUTOFF = 0.7702616034384248
SCORE_COLUMNS = (
    "v2_component_probability",
    "v4_existence_probability",
    "v4_geometry_probability",
)


def three_state_decision(frame: pd.DataFrame) -> pd.Series:
    """Apply the frozen accept/abstain/reject rule to component scores."""
    missing = [column for column in SCORE_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"missing gate scores: {missing}")
    values = frame.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("gate scores must be finite")
    decision = np.full(len(frame), "abstain", dtype=object)
    decision[np.any(values[:, 1:] < REJECT_THRESHOLD, axis=1)] = "reject"
    decision[np.all(values >= ACCEPT_THRESHOLD, axis=1)] = "accept"
    return pd.Series(decision, index=frame.index, name="gate_decision")


def add_accepted_et(
    base_label_zyx: np.ndarray,
    coordinates_zyx: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    """Add ET voxels without mutating the input or overriding RC."""
    base = np.asarray(base_label_zyx)
    if base.ndim != 3:
        raise ValueError(f"expected a 3D label map, got {base.shape}")
    labels = set(int(value) for value in np.unique(base))
    if not labels.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"invalid BraTS labels: {sorted(labels)}")
    output = base.astype(np.uint8, copy=True)
    seen: set[tuple[int, int, int]] = set()
    for coordinate in coordinates_zyx:
        value = tuple(int(part) for part in coordinate)
        if len(value) != 3:
            raise ValueError(f"invalid ET coordinate: {coordinate}")
        if value in seen:
            raise RuntimeError(f"duplicate ET coordinate: {value}")
        seen.add(value)
        if any(part < 0 or part >= size for part, size in zip(value, output.shape)):
            raise ValueError(f"ET coordinate is out of bounds: {value}")
        if output[value] != 4:
            output[value] = 3
    return output


def predict_component_probability(
    bundle: Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Score a feature frame with an audited LightGBM component bundle."""
    features = list(bundle["component_features"])
    missing = [column for column in features if column not in frame]
    if missing:
        raise KeyError(f"missing component features: {missing}")
    probability = np.asarray(
        bundle["component_model"].predict_proba(frame.loc[:, features])[:, 1],
        dtype=np.float64,
    )
    if probability.shape != (len(frame),) or not np.isfinite(probability).all():
        raise RuntimeError("component model returned invalid probabilities")
    return probability


def score_utility_features(
    component_frame: pd.DataFrame,
    *,
    rgv3_et_bundle: Mapping[str, Any],
    existence_model: Any,
    geometry_model: Any,
    utility_feature_names: list[str],
) -> pd.DataFrame:
    """Attach RGv3-ET and dual utility-v4 probabilities to ET rows."""
    required = {"region", "v2_component_probability"}
    missing = sorted(required.difference(component_frame.columns))
    if missing:
        raise KeyError(f"missing utility input columns: {missing}")
    output = component_frame.copy()
    output["v3_component_probability"] = np.nan
    output["v4_existence_probability"] = np.nan
    output["v4_geometry_probability"] = np.nan
    output["gate_decision"] = None
    et = output["region"].astype(str).str.upper().eq("ET")
    if not et.any():
        return output
    et_frame = output.loc[et].copy()
    output.loc[et, "v3_component_probability"] = predict_component_probability(
        rgv3_et_bundle,
        et_frame,
    )
    missing_features = [
        column for column in utility_feature_names if column not in output
    ]
    if missing_features:
        raise KeyError(f"missing utility-v4 features: {missing_features}")
    utility_values = output.loc[et, utility_feature_names].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(utility_values).all():
        raise ValueError("utility-v4 features must be finite")
    output.loc[et, "v4_existence_probability"] = np.asarray(
        existence_model.predict_proba(output.loc[et, utility_feature_names])[:, 1],
        dtype=np.float64,
    )
    output.loc[et, "v4_geometry_probability"] = np.asarray(
        geometry_model.predict_proba(output.loc[et, utility_feature_names])[:, 1],
        dtype=np.float64,
    )
    output.loc[et, "gate_decision"] = three_state_decision(output.loc[et])
    return output

