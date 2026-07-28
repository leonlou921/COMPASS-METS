import numpy as np
import pandas as pd

from train_models import (
    fixed_component_keep,
    iter_crossfit_splits,
    select_fp_constrained_cutoff,
    train_crossfit,
)


def test_fixed_component_keep_accepts_an_empty_component_table():
    config = {
        "fixed_component_conf": {
            "minimum_volume": {"WT": 10, "TC": 10, "ET": 5, "RC": 20},
            "minimum_mean": {"WT": 0.0, "TC": 0.0, "ET": 0.55, "RC": 0.65},
            "minimum_peak": {"WT": 0.0, "TC": 0.0, "ET": 0.70, "RC": 0.80},
        }
    }

    keep = fixed_component_keep(pd.DataFrame(), config)

    assert keep.dtype == bool
    assert keep.size == 0


def test_crossfit_splits_isolate_fold_and_case_groups():
    rows = []
    for fold in range(5):
        for case_number in range(3):
            case_id = f"fold{fold}-case{case_number}"
            rows.extend(
                {"case_id": case_id, "fold": fold, "component_id": component}
                for component in range(2)
            )
    table = pd.DataFrame(rows)

    seen_validation = set()
    for heldout_fold, train_index, validation_index in iter_crossfit_splits(table):
        train = table.loc[train_index]
        validation = table.loc[validation_index]
        assert set(validation["fold"]) == {heldout_fold}
        assert heldout_fold not in set(train["fold"])
        assert set(train["case_id"]).isdisjoint(validation["case_id"])
        seen_validation.update(validation_index)
    assert seen_validation == set(table.index)


def test_fp_constrained_cutoff_maximizes_recall_within_budget():
    probability = np.array([0.90, 0.80, 0.70, 0.60, 0.20])
    target = np.array([1, 1, 0, 1, 0])
    result = select_fp_constrained_cutoff(probability, target, allowed_fp=0)

    assert result["feasible"]
    assert result["false_positives"] == 0
    assert result["true_positives"] == 2
    assert result["recall"] == 2 / 3
    assert result["threshold"] == 0.80


def test_cutoff_fallback_minimizes_fp_excess_then_maximizes_recall():
    probability = np.array([0.90, 0.80, 0.70, 0.60, 0.20])
    target = np.array([1, 1, 0, 1, 0])
    result = select_fp_constrained_cutoff(probability, target, allowed_fp=-1)

    assert not result["feasible"]
    assert result["fp_excess"] == 1
    assert result["false_positives"] == 0
    assert result["true_positives"] == 2
    assert result["threshold"] == 0.80


def test_cutoff_is_deterministic_and_handles_empty_positives():
    probability = np.array([0.8, 0.3, 0.1])
    target = np.zeros(3, dtype=int)
    first = select_fp_constrained_cutoff(probability, target, allowed_fp=0)
    second = select_fp_constrained_cutoff(probability, target, allowed_fp=0)
    assert first == second
    assert first["recall"] == 1.0
    assert first["false_positives"] == 0
    assert first["kept"] == 0


def test_crossfit_writes_complete_heldout_predictions_and_models(tmp_path):
    case_rows = []
    component_rows = []
    regions = ("WT", "TC", "ET", "RC")
    for fold in range(5):
        for case_number in range(4):
            case_id = f"f{fold}-c{case_number}"
            for region_index, region in enumerate(regions):
                target = int((fold + case_number + region_index) % 3 != 0)
                signal = 0.8 if target else 0.2
                case_rows.append(
                    {
                        "case_id": case_id,
                        "fold": fold,
                        "region": region,
                        "region_index": region_index,
                        "target": target,
                        "gt_voxels": target * 10,
                        "signal": signal,
                    }
                )
                component_rows.append(
                    {
                        "case_id": case_id,
                        "fold": fold,
                        "region": region,
                        "region_index": region_index,
                        "component_id": 1,
                        "target": target,
                        "overlap_voxels": target,
                        "component_precision": float(target),
                        "gt_coverage": float(target),
                        "iou": float(target),
                        "volume_voxels": 20,
                        "fused_mean": signal,
                        "fused_peak": signal + 0.05,
                        "signal": signal,
                    }
                )
    config = {
        "seed": 7,
        "fixed_component_conf": {
            "minimum_volume": {region: 1 for region in regions},
            "minimum_mean": {region: 0.5 for region in regions},
            "minimum_peak": {region: 0.5 for region in regions},
        },
    }

    case_output, component_output, audit = train_crossfit(
        pd.DataFrame(case_rows), pd.DataFrame(component_rows), config, tmp_path
    )

    assert len(case_output) == len(case_rows)
    assert len(component_output) == len(component_rows)
    for algorithm in ("logistic", "lightgbm"):
        assert np.isfinite(case_output[f"{algorithm}_case_probability"]).all()
        assert np.isfinite(component_output[f"{algorithm}_score"]).all()
        assert component_output[f"{algorithm}_keep"].dtype == bool
        assert all(fold["case_overlap"] == 0 for fold in audit["algorithms"][algorithm]["folds"])
        assert (tmp_path / algorithm / "final" / "models.joblib").is_file()
        assert (tmp_path / algorithm / "final" / "cutoffs.json").is_file()
