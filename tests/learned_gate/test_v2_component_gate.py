import numpy as np

from component_gate import REGION_NAMES, propose_components
from v2_component_gate import (
    LC_V2_STRUCTURED_FILTER,
    LC_V2_STRUCTURED_PROTECTED_RESCUE,
    LC_V2_XLFT_FILTER,
    apply_gate_preserving_v2,
    candidate_score_maps,
    filter_scored_components,
    reconstruct_scored_candidate,
    scored_final_component_rows,
)


def _arrays(shape=(4, 3, 3, 3), fill=0.01):
    return {
        name: np.full(shape, fill, dtype=np.float32)
        for name in ("XL", "M", "FT")
    }


def test_candidate_score_maps_reproduce_structured_and_equal_xlft_bases():
    arrays = _arrays(shape=(4, 1, 1, 1))
    arrays["XL"][:] = 0.8
    arrays["FT"][:] = 0.2
    arrays["M"][:] = 0.99

    structured = candidate_score_maps(arrays, LC_V2_STRUCTURED_FILTER)
    xlft = candidate_score_maps(arrays, LC_V2_XLFT_FILTER)

    # XF10/XF12 use FT directly for WT and TC.
    assert np.isclose(structured["basis"][0, 0, 0, 0], 0.2)
    assert np.isclose(structured["basis"][1, 0, 0, 0], 0.2)
    # ET uses calibrated 85% XL + 15% FT logits.
    expected_et = 1.0 / (
        1.0
        + np.exp(
            -(
                0.85 * np.log(0.8 / 0.2)
                + 0.15 * np.log(0.2 / 0.8)
            )
        )
    )
    assert np.isclose(structured["basis"][2, 0, 0, 0], expected_et)
    # Equal opposite logits produce exactly 0.5 for XF03/XF16.
    assert np.allclose(xlft["basis"], 0.5)
    # M raw is auxiliary only and cannot change either fixed output basis.
    arrays["M"][:] = 0.01
    assert np.allclose(
        candidate_score_maps(arrays, LC_V2_STRUCTURED_FILTER)["basis"],
        structured["basis"],
    )


def test_protected_rescue_adds_only_seed_connected_m_ft_consensus():
    arrays = _arrays(shape=(4, 4, 4, 4))
    # Structured WT basis is FT. One seed is >=0.5; its neighbor is only 0.30.
    arrays["FT"][0, 1, 1, 1] = 0.60
    arrays["FT"][0, 1, 1, 2] = 0.30
    arrays["M"][0, 1, 1, 1] = 0.70
    arrays["M"][0, 1, 1, 2] = 0.70
    # Disconnected M/FT-supported voxel is a separate proposal and must not join.
    arrays["FT"][0, 3, 3, 3] = 0.30
    arrays["M"][0, 3, 3, 3] = 0.70
    proposals = propose_components(arrays, threshold=0.25)
    scores = {
        (region, int(component["component_id"])): 0.2
        for region in REGION_NAMES
        for component in proposals[region]
    }

    filtered, _, _ = reconstruct_scored_candidate(
        arrays, proposals, scores, LC_V2_STRUCTURED_FILTER
    )
    rescued, rescued_scores, protected = reconstruct_scored_candidate(
        arrays, proposals, scores, LC_V2_STRUCTURED_PROTECTED_RESCUE
    )

    assert filtered["WT"][1, 1, 1]
    assert not filtered["WT"][1, 1, 2]
    assert rescued["WT"][1, 1, 1]
    assert rescued["WT"][1, 1, 2]
    assert not rescued["WT"][3, 3, 3]
    assert protected >= 1
    # Protection opens the M/M-FT-supported geometry but never overwrites the
    # learned probability used by final-component calibration.
    assert np.isclose(rescued_scores["WT"][1, 1, 2], 0.2)


def test_final_component_filtering_uses_reconstructed_component_score():
    shape = (4, 5, 5, 5)
    regions = {region: np.zeros(shape[1:], dtype=bool) for region in REGION_NAMES}
    scores = {region: np.zeros(shape[1:], dtype=np.float32) for region in REGION_NAMES}
    regions["WT"][1, 1, 1] = True
    regions["WT"][3, 3, 3] = True
    scores["WT"][1, 1, 1] = 0.9
    scores["WT"][3, 3, 3] = 0.4

    filtered = filter_scored_components(
        regions,
        scores,
        {"WT": 0.8, "TC": 0.8, "ET": 0.8, "RC": 0.8},
    )

    assert filtered["WT"][1, 1, 1]
    assert not filtered["WT"][3, 3, 3]


def test_gate_preserving_v2_never_creates_rc_outside_learned_mask():
    shape = (4, 5, 5, 5)
    regions = {region: np.zeros(shape[1:], dtype=bool) for region in REGION_NAMES}
    regions["RC"][1, 1, 1] = True
    score_maps = {
        "basis": np.zeros(shape, dtype=np.float32),
        "tc_v2": np.zeros(shape[1:], dtype=np.float32),
        "rc_v2": np.zeros(shape[1:], dtype=np.float32),
    }
    score_maps["rc_v2"][1, 1, 1] = 0.95
    # High raw score outside the learned mask must not create a new RC component.
    score_maps["rc_v2"][4, 4, 4] = 0.99
    output = apply_gate_preserving_v2(
        regions,
        score_maps,
        spacing_zyx=(1.0, 1.0, 1.0),
        settings={
            "tc_boundary_threshold": 0.4,
            "tc_boundary_budget": 20,
            "rc_threshold": 0.5,
            "rc_minimum_volume_mm3": 1.0,
            "rc_minimum_mean": 0.7,
            "rc_minimum_peak": 0.85,
        },
    )

    assert output["RC"][1, 1, 1]
    assert not output["RC"][4, 4, 4]


def test_scored_final_component_rows_use_final_fragment_grain():
    shape = (5, 5, 5)
    regions = {region: np.zeros(shape, dtype=bool) for region in REGION_NAMES}
    scores = {region: np.zeros(shape, dtype=np.float32) for region in REGION_NAMES}
    gt = {region: np.zeros(shape, dtype=bool) for region in REGION_NAMES}
    regions["RC"][1, 1, 1] = True
    regions["RC"][3, 3, 3] = True
    scores["RC"][1, 1, 1] = 0.8
    scores["RC"][3, 3, 3] = 0.4
    gt["RC"][1, 1, 1] = True

    rows = scored_final_component_rows(
        regions, scores, gt, case_id="case-a", fold=2, candidate="candidate-a"
    )
    rc = [row for row in rows if row["region"] == "RC"]

    assert len(rc) == 2
    assert np.allclose(sorted(row["score"] for row in rc), [0.4, 0.8])
    assert sorted(row["target"] for row in rc) == [0, 1]
    assert {row["fold"] for row in rc} == {2}
