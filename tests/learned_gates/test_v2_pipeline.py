import numpy as np
import pandas as pd

from compass_mets.learned_gates.lcv1.v2_pipeline import (
    aligned_anchor,
    bbox_from_component_frame,
    calibrate_nested_cutoffs,
    candidate_submission_verdict,
    load_cropped_probabilities,
    proposal_score_map,
    proposal_scores_from_predictions,
    restore_cropped_label,
)


def test_aligned_anchor_routes_structured_and_xlft_candidates():
    assert aligned_anchor("LCv2_structured_filter_V2") == "anchor_structured_V2"
    assert (
        aligned_anchor("LCv2_structured_protected_rescue_V2")
        == "anchor_structured_V2"
    )
    assert aligned_anchor("LCv2_XLFT_filter_V2") == "anchor_XLFT_V2"


def test_nested_cutoffs_never_use_the_heldout_fold():
    candidate_rows = []
    anchor_rows = []
    for fold in range(5):
        for region in ("WT", "TC", "ET", "RC"):
            candidate_rows.extend(
                [
                    {
                        "fold": fold,
                        "region": region,
                        "score": 0.9,
                        "target": 1,
                    },
                    {
                        "fold": fold,
                        "region": region,
                        "score": 0.1 + 0.01 * fold,
                        "target": 0,
                    },
                ]
            )
            anchor_rows.append(
                {"fold": fold, "region": region, "target": int(fold % 2 == 0)}
            )

    result = calibrate_nested_cutoffs(
        pd.DataFrame(candidate_rows), pd.DataFrame(anchor_rows)
    )

    assert set(result["folds"]) == {"0", "1", "2", "3", "4"}
    for fold, record in result["folds"].items():
        assert int(fold) not in record["training_folds"]
        assert set(record["cutoffs"]) == {"WT", "TC", "ET", "RC"}
    # Hierarchy-safe ordering prevents a kept child from being lost in a parent.
    for record in [*result["folds"].values(), result["final"]]:
        cutoffs = record["cutoffs"]
        assert cutoffs["WT"] <= cutoffs["TC"] <= cutoffs["ET"]


def test_submission_verdict_applies_every_guardrail():
    anchor = {
        "candidate": "anchor",
        "small_f1": 0.40,
        "all_f1": 0.60,
        "macro_dsc": 0.70,
        "macro_hd95": 10.0,
        "false_positive_components": 100,
        "true_positive_components": 200,
    }
    good = {
        **anchor,
        "candidate": "good",
        "small_f1": 0.45,
        "macro_dsc": 0.71,
        "macro_hd95": 10.2,
        "false_positive_components": 104,
        "true_positive_components": 205,
        "worst_case_macro_dsc_delta": -0.05,
    }
    bad = {
        **good,
        "candidate": "bad",
        "false_positive_components": 106,
        "worst_case_macro_dsc_delta": -0.11,
    }
    contract = {
        "max_fp_relative_increase": 0.05,
        "max_macro_hd95_increase": 0.5,
        "min_worst_case_macro_dsc_delta": -0.10,
    }

    assert candidate_submission_verdict(anchor, good, contract)["worth_submitting"]
    rejected = candidate_submission_verdict(anchor, bad, contract)
    assert not rejected["worth_submitting"]
    assert set(rejected["reasons"]) == {
        "false_positive_components",
        "worst_case_macro_dsc_delta",
    }


def test_proposal_score_map_handles_zero_components_and_rejects_duplicates():
    empty = pd.DataFrame(columns=["region", "component_id", "v2_component_probability"])
    assert proposal_score_map(empty) == {}

    duplicate = pd.DataFrame(
        [
            {"region": "ET", "component_id": 1, "v2_component_probability": 0.8},
            {"region": "ET", "component_id": 1, "v2_component_probability": 0.7},
        ]
    )
    try:
        proposal_score_map(duplicate)
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate proposal IDs must be rejected")


def test_test_proposal_scores_accept_a_schema_less_empty_component_frame():
    assert proposal_scores_from_predictions(pd.DataFrame(), np.array([])) == {}


def test_probability_crop_unions_model_support_and_ground_truth(tmp_path):
    shape = (4, 8, 9, 10)
    paths = {}
    for name in ("XL", "M", "FT"):
        array = np.zeros(shape, dtype=np.float32)
        paths[name] = tmp_path / f"{name}.npz"
        if name == "XL":
            array[0, 2, 3, 4] = 0.8
        if name == "FT":
            array[3, 4, 5, 6] = 0.9
        np.savez_compressed(paths[name], probabilities=array)
    include = np.zeros(shape[1:], dtype=bool)
    include[6, 7, 8] = True

    arrays, bbox, full_shape = load_cropped_probabilities(
        paths, threshold=0.25, include_mask=include
    )

    assert full_shape == shape[1:]
    assert tuple((sl.start, sl.stop) for sl in bbox) == (
        (2, 7),
        (3, 8),
        (4, 9),
    )
    assert all(array.shape == (4, 5, 5, 5) for array in arrays.values())


def test_restore_cropped_label_embeds_at_original_location():
    crop = np.ones((2, 2, 2), dtype=np.uint8)
    bbox = (slice(1, 3), slice(2, 4), slice(3, 5))

    restored = restore_cropped_label(crop, bbox, (5, 6, 7))

    assert restored.shape == (5, 6, 7)
    assert restored.sum() == 8
    assert np.all(restored[bbox] == 1)


def test_component_feature_bboxes_collapse_to_one_support_bbox():
    frame = pd.DataFrame(
        [
            {
                "bbox_z0": 2,
                "bbox_z1": 4,
                "bbox_y0": 3,
                "bbox_y1": 7,
                "bbox_x0": 5,
                "bbox_x1": 8,
            },
            {
                "bbox_z0": 1,
                "bbox_z1": 6,
                "bbox_y0": 4,
                "bbox_y1": 5,
                "bbox_x0": 2,
                "bbox_x1": 9,
            },
        ]
    )

    bbox = bbox_from_component_frame(frame)

    assert tuple((sl.start, sl.stop) for sl in bbox) == (
        (1, 6),
        (3, 7),
        (2, 9),
    )
    assert bbox_from_component_frame(frame.iloc[0:0]) is None
    assert bbox_from_component_frame(pd.DataFrame()) is None
