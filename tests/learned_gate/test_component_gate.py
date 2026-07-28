import numpy as np

from component_gate import (
    REGION_NAMES,
    label_to_regions,
    match_component,
    propose_components,
    regions_to_label,
)


def test_region_order_and_label_conversion_roundtrip():
    label = np.zeros((4, 5, 6), dtype=np.uint8)
    label[0, 0, 0] = 1
    label[1, 1, 1] = 2
    label[2, 2, 2] = 3
    label[3, 3, 3] = 4

    regions = label_to_regions(label)

    assert REGION_NAMES == ("WT", "TC", "ET", "RC")
    assert tuple(regions) == REGION_NAMES
    assert regions["WT"][0, 0, 0]
    assert regions["WT"][1, 1, 1]
    assert regions["WT"][2, 2, 2]
    assert not regions["WT"][3, 3, 3]
    assert regions["TC"][0, 0, 0]
    assert not regions["TC"][1, 1, 1]
    assert regions["TC"][2, 2, 2]
    assert regions["ET"][2, 2, 2]
    assert regions["RC"][3, 3, 3]

    restored = regions_to_label(regions)
    np.testing.assert_array_equal(restored, label)
    assert set(np.unique(restored)).issubset({0, 1, 2, 3, 4})


def test_regions_to_label_enforces_et_nesting_and_rc_independence():
    shape = (3, 3, 3)
    regions = {name: np.zeros(shape, dtype=bool) for name in REGION_NAMES}
    regions["ET"][0, 0, 0] = True
    regions["TC"][1, 1, 1] = True
    regions["WT"][2, 2, 2] = True
    regions["RC"][0, 2, 0] = True

    restored = regions_to_label(regions)
    assert restored[0, 0, 0] == 3
    assert restored[1, 1, 1] == 1
    assert restored[2, 2, 2] == 2
    assert restored[0, 2, 0] == 4


def test_proposals_use_26_connectivity_and_local_masks():
    shape = (4, 7, 7, 7)
    xl = np.zeros(shape, dtype=np.float32)
    m = np.zeros(shape, dtype=np.float32)
    ft = np.zeros(shape, dtype=np.float32)
    # WT: diagonal voxels are a single 26-connected component.
    xl[0, 1, 1, 1] = 0.7
    m[0, 2, 2, 2] = 0.6
    # A separate component.
    ft[0, 6, 6, 6] = 0.8

    proposals = propose_components({"XL": xl, "M": m, "FT": ft}, threshold=0.25)

    assert len(proposals["WT"]) == 2
    first = proposals["WT"][0]
    assert first["voxel_count"] == 2
    assert first["local_mask"].shape == (2, 2, 2)
    assert first["local_mask"].sum() == 2
    assert first["full_shape"] == (7, 7, 7)
    assert first["local_mask"].size < np.prod(first["full_shape"])


def test_component_gt_matching_counts_one_voxel_overlap_as_positive():
    probs = {name: np.zeros((4, 5, 5, 5), dtype=np.float32) for name in ("XL", "M", "FT")}
    probs["XL"][2, 1:3, 1:3, 1:3] = 0.9
    component = propose_components(probs, threshold=0.25)["ET"][0]

    gt = np.zeros((5, 5, 5), dtype=bool)
    gt[2, 2, 2] = True
    matched = match_component(component, gt)
    assert matched["target"] == 1
    assert matched["overlap_voxels"] == 1
    assert matched["component_precision"] == 1 / 8
    assert matched["gt_coverage"] == 1.0
    assert matched["iou"] == 1 / 8

    empty = match_component(component, np.zeros_like(gt))
    assert empty["target"] == 0
    assert empty["overlap_voxels"] == 0
    assert empty["gt_coverage"] == 0.0
    assert empty["iou"] == 0.0
