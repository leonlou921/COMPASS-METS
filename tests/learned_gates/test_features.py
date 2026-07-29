from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass_mets.learned_gates.lcv1.build_features import (
    checkpoint_is_complete,
    write_case_checkpoint,
)
from compass_mets.learned_gates.lcv1.features import extract_case_features


def synthetic_case():
    shape = (4, 8, 8, 8)
    arrays = {name: np.full(shape, 0.01, dtype=np.float32) for name in ("XL", "M", "FT")}
    for source, peak in zip(arrays.values(), (0.9, 0.8, 0.7)):
        source[0, 1:4, 1:4, 1:4] = peak
        source[1, 1:3, 1:3, 1:3] = peak
        source[2, 1:2, 1:2, 1:2] = peak
    arrays["XL"][0, 6, 6, 6] = 0.6
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    gt[1:2, 1:2, 1:2] = 3
    gt[2:3, 2:3, 2:3] = 1
    gt[3:4, 3:4, 3:4] = 2
    return arrays, gt


def test_feature_extraction_is_finite_and_complete():
    arrays, gt = synthetic_case()
    case_rows, component_rows = extract_case_features(
        "BraTS-MET-00001-000", 0, arrays["XL"], arrays["M"], arrays["FT"], gt
    )

    assert len(case_rows) == 4
    assert {row["region"] for row in case_rows} == {"WT", "TC", "ET", "RC"}
    assert all(row["case_id"] == "BraTS-MET-00001-000" and row["fold"] == 0 for row in case_rows)
    assert any(row["region"] == "RC" and row["target"] == 0 for row in case_rows)
    assert component_rows

    required = {
        "XL_mean",
        "M_peak",
        "FT_p95",
        "fused_mean",
        "support2_t025_fraction",
        "support3_t050_fraction",
        "mean_entropy",
        "model_disagreement",
        "volume_voxels",
        "bbox_fill_fraction",
        "surface_voxels",
        "compactness",
        "within_WT_t050_fraction",
        "nearest_large_distance",
        "nearest_large_distance_missing",
        "volume_rank",
        "peak_rank",
        "target",
        "overlap_voxels",
    }
    assert required.issubset(component_rows[0])
    main_wt = next(
        row for row in component_rows if row["region"] == "WT" and row["volume_voxels"] == 27
    )
    expected_fused_peak = 1.0 / (
        1.0
        + np.exp(
            -np.mean(
                np.log(np.array([0.9, 0.8, 0.7]) / (1.0 - np.array([0.9, 0.8, 0.7])))
            )
        )
    )
    assert main_wt["fused_peak"] == pytest.approx(expected_fused_peak, rel=1e-6)
    assert "baseline_keep" in main_wt
    for row in case_rows + component_rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.number)):
                assert np.isfinite(value), (key, value)


def test_empty_region_has_finite_case_features_and_no_component_row():
    arrays, gt = synthetic_case()
    case_rows, component_rows = extract_case_features(
        "case-empty-rc", 2, arrays["XL"], arrays["M"], arrays["FT"], gt
    )
    rc_case = next(row for row in case_rows if row["region"] == "RC")
    assert rc_case["candidate_component_count"] == 0
    assert rc_case["target"] == 0
    assert not [row for row in component_rows if row["region"] == "RC"]


def test_atomic_checkpoint_prevents_duplicates_and_rejects_incomplete(tmp_path: Path):
    case_id = "case-001"
    assert not checkpoint_is_complete(tmp_path, case_id)
    incomplete = tmp_path / f"{case_id}.components.parquet"
    pd.DataFrame([{"case_id": case_id}]).to_parquet(incomplete, index=False)
    assert not checkpoint_is_complete(tmp_path, case_id)

    incomplete.unlink()
    case_rows = [{"case_id": case_id, "region": "WT", "target": 1}]
    component_rows = [{"case_id": case_id, "region": "WT", "target": 1}]
    write_case_checkpoint(tmp_path, case_id, case_rows, component_rows)
    assert checkpoint_is_complete(tmp_path, case_id)

    with pytest.raises(FileExistsError):
        write_case_checkpoint(tmp_path, case_id, case_rows, component_rows)
