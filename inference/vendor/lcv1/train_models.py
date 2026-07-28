"""Leakage-safe five-fold case/component model training and cutoff selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REGIONS = ("WT", "TC", "ET", "RC")
ALGORITHMS = ("logistic", "lightgbm")
NON_FEATURE_COLUMNS = {
    "case_id",
    "fold",
    "region",
    "component_id",
    "target",
    "gt_voxels",
    "overlap_voxels",
    "component_precision",
    "gt_coverage",
    "iou",
    "baseline_keep",
    "baseline_case_keep",
}


def iter_crossfit_splits(table: pd.DataFrame):
    """Yield label-index splits with whole folds and cases isolated."""
    folds = sorted(int(value) for value in table["fold"].unique())
    for heldout_fold in folds:
        validation_mask = table["fold"].astype(int) == heldout_fold
        train_index = table.index[~validation_mask]
        validation_index = table.index[validation_mask]
        if set(table.loc[train_index, "case_id"]).intersection(
            table.loc[validation_index, "case_id"]
        ):
            raise RuntimeError(f"case leakage in fold {heldout_fold}")
        yield heldout_fold, train_index, validation_index


def select_fp_constrained_cutoff(
    probability: np.ndarray,
    target: np.ndarray,
    allowed_fp: int,
) -> dict[str, float | int | bool]:
    """Maximize recall subject to an FP budget with deterministic fallback."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    if probability.ndim != 1 or target.shape != probability.shape:
        raise ValueError("probability and target must be aligned one-dimensional arrays")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    if probability.size == 0:
        return {
            "threshold": 1.0,
            "feasible": allowed_fp >= 0,
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "kept": 0,
            "recall": 1.0,
            "precision": 1.0,
            "fp_excess": max(0, -int(allowed_fp)),
        }
    thresholds = np.r_[np.nextafter(probability.max(), np.inf), np.unique(probability)[::-1]]
    positives = int((target == 1).sum())
    candidates = []
    for threshold in thresholds:
        keep = probability >= threshold
        tp = int(np.logical_and(keep, target == 1).sum())
        fp = int(np.logical_and(keep, target == 0).sum())
        fn = positives - tp
        recall = tp / positives if positives else 1.0
        kept = int(keep.sum())
        precision = tp / kept if kept else (1.0 if positives == 0 else 0.0)
        candidates.append(
            {
                "threshold": float(threshold),
                "feasible": fp <= allowed_fp,
                "false_positives": fp,
                "true_positives": tp,
                "false_negatives": fn,
                "kept": kept,
                "recall": float(recall),
                "precision": float(precision),
                "fp_excess": max(0, fp - int(allowed_fp)),
            }
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    pool = feasible if feasible else candidates
    if feasible:
        key = lambda row: (
            row["recall"],
            -row["false_positives"],
            row["precision"],
            row["true_positives"],
            row["threshold"],
        )
    else:
        key = lambda row: (
            -row["fp_excess"],
            row["recall"],
            row["precision"],
            row["true_positives"],
            row["threshold"],
        )
    return max(pool, key=key)


def feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in table.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(table[column])
    ]
    if not columns:
        raise ValueError("no numeric feature columns")
    if table[columns].isna().any().any() or not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError("features must be finite before model fitting")
    return columns


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, values):
        count = len(values)
        p = np.full(count, self.probability, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def _fit_model(algorithm: str, x: pd.DataFrame, y: np.ndarray, seed: int):
    unique = np.unique(y)
    if len(unique) == 1:
        return ConstantProbabilityModel(float(unique[0]))
    if algorithm == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                solver="liblinear",
                random_state=seed,
            ),
        )
    elif algorithm == "lightgbm":
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    model.fit(x, y)
    return model


def _predict(model, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def fixed_component_keep(table: pd.DataFrame, config: dict) -> np.ndarray:
    if table.empty:
        return np.zeros(0, dtype=bool)
    if "baseline_keep" in table.columns:
        return table["baseline_keep"].astype(bool).to_numpy()
    settings = config["fixed_component_conf"]
    result = np.zeros(len(table), dtype=bool)
    for region in REGIONS:
        mask = table["region"].eq(region).to_numpy()
        result[mask] = (
            (table.loc[mask, "volume_voxels"].to_numpy() >= settings["minimum_volume"][region])
            & (table.loc[mask, "fused_mean"].to_numpy() >= settings["minimum_mean"][region])
            & (table.loc[mask, "fused_peak"].to_numpy() >= settings["minimum_peak"][region])
        )
    return result


def _add_case_baseline(case_table: pd.DataFrame, component_table: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        component_table.groupby(["case_id", "region"], sort=False)["baseline_keep"].any().to_dict()
    )
    result = case_table.copy()
    result["baseline_case_keep"] = [
        bool(baseline.get((case_id, region), False))
        for case_id, region in zip(result["case_id"], result["region"])
    ]
    return result


def _map_case_probability(component: pd.DataFrame, case: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    mapping = {
        (case_id, region): float(value)
        for case_id, region, value in zip(case["case_id"], case["region"], values)
    }
    return np.asarray(
        [mapping[(case_id, region)] for case_id, region in zip(component["case_id"], component["region"])],
        dtype=np.float64,
    )


def train_crossfit(
    case_table: pd.DataFrame,
    component_table: pd.DataFrame,
    config: dict,
    model_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train case and component models with fold-local cutoffs and final all-data fits."""
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    component = component_table.copy()
    component["baseline_keep"] = fixed_component_keep(component, config)
    case = _add_case_baseline(case_table, component)
    case_features = feature_columns(case)
    component_features = feature_columns(component)
    seed = int(config.get("seed", 20260723))
    audit: dict = {
        "case_features": case_features,
        "component_features": component_features,
        "algorithms": {},
    }
    case_output = case[["case_id", "fold", "region", "target", "baseline_case_keep"]].copy()
    component_output = component[
        [
            "case_id",
            "fold",
            "region",
            "component_id",
            "target",
            "overlap_voxels",
            "baseline_keep",
        ]
    ].copy()

    for algorithm in ALGORITHMS:
        case_oof = np.full(len(case), np.nan, dtype=np.float64)
        component_oof = np.full(len(component), np.nan, dtype=np.float64)
        score_oof = np.full(len(component), np.nan, dtype=np.float64)
        keep_oof = np.zeros(len(component), dtype=bool)
        fold_audit = []
        for heldout_fold in sorted(case["fold"].astype(int).unique()):
            case_train = case["fold"].astype(int) != heldout_fold
            case_validation = ~case_train
            component_train = component["fold"].astype(int) != heldout_fold
            component_validation = ~component_train
            train_cases = set(case.loc[case_train, "case_id"])
            validation_cases = set(case.loc[case_validation, "case_id"])
            if train_cases.intersection(validation_cases):
                raise RuntimeError(f"case leakage in fold {heldout_fold}")

            case_model = _fit_model(
                algorithm,
                case.loc[case_train, case_features],
                case.loc[case_train, "target"].to_numpy(dtype=np.int8),
                seed + heldout_fold,
            )
            component_model = _fit_model(
                algorithm,
                component.loc[component_train, component_features],
                component.loc[component_train, "target"].to_numpy(dtype=np.int8),
                seed + 100 + heldout_fold,
            )
            case_train_probability = _predict(case_model, case.loc[case_train, case_features])
            case_validation_probability = _predict(
                case_model, case.loc[case_validation, case_features]
            )
            component_train_probability = _predict(
                component_model, component.loc[component_train, component_features]
            )
            component_validation_probability = _predict(
                component_model, component.loc[component_validation, component_features]
            )
            train_case_factor = _map_case_probability(
                component.loc[component_train], case.loc[case_train], case_train_probability
            )
            validation_case_factor = _map_case_probability(
                component.loc[component_validation],
                case.loc[case_validation],
                case_validation_probability,
            )
            train_score = component_train_probability * train_case_factor
            validation_score = component_validation_probability * validation_case_factor
            cutoffs = {}
            validation_keep = np.zeros(int(component_validation.sum()), dtype=bool)
            validation_regions = component.loc[component_validation, "region"].to_numpy()
            train_regions = component.loc[component_train, "region"].to_numpy()
            for region in REGIONS:
                region_train = train_regions == region
                baseline_fp = int(
                    np.logical_and(
                        component.loc[component_train, "baseline_keep"].to_numpy()[region_train],
                        component.loc[component_train, "target"].to_numpy()[region_train] == 0,
                    ).sum()
                )
                selected = select_fp_constrained_cutoff(
                    train_score[region_train],
                    component.loc[component_train, "target"].to_numpy()[region_train],
                    allowed_fp=baseline_fp,
                )
                cutoffs[region] = selected
                validation_keep[validation_regions == region] = (
                    validation_score[validation_regions == region] >= selected["threshold"]
                )

            case_oof[case_validation.to_numpy()] = case_validation_probability
            component_oof[component_validation.to_numpy()] = component_validation_probability
            score_oof[component_validation.to_numpy()] = validation_score
            keep_oof[component_validation.to_numpy()] = validation_keep
            fold_dir = model_root / algorithm / f"fold_{heldout_fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "case_model": case_model,
                    "component_model": component_model,
                    "case_features": case_features,
                    "component_features": component_features,
                },
                fold_dir / "models.joblib",
            )
            (fold_dir / "cutoffs.json").write_text(
                json.dumps(cutoffs, indent=2, sort_keys=True), encoding="utf-8"
            )
            fold_audit.append(
                {
                    "heldout_fold": int(heldout_fold),
                    "train_case_count": len(train_cases),
                    "validation_case_count": len(validation_cases),
                    "case_overlap": 0,
                    "cutoffs": cutoffs,
                }
            )

        if not (
            np.isfinite(case_oof).all()
            and np.isfinite(component_oof).all()
            and np.isfinite(score_oof).all()
        ):
            raise RuntimeError(f"incomplete OOF predictions for {algorithm}")
        case_output[f"{algorithm}_case_probability"] = case_oof
        component_output[f"{algorithm}_component_probability"] = component_oof
        component_output[f"{algorithm}_score"] = score_oof
        component_output[f"{algorithm}_keep"] = keep_oof

        final_case_model = _fit_model(
            algorithm, case[case_features], case["target"].to_numpy(dtype=np.int8), seed + 1000
        )
        final_component_model = _fit_model(
            algorithm,
            component[component_features],
            component["target"].to_numpy(dtype=np.int8),
            seed + 1100,
        )
        final_cutoffs = {}
        for region in REGIONS:
            region_mask = component["region"].eq(region).to_numpy()
            baseline_fp = int(
                np.logical_and(
                    component.loc[region_mask, "baseline_keep"].to_numpy(),
                    component.loc[region_mask, "target"].to_numpy() == 0,
                ).sum()
            )
            final_cutoffs[region] = select_fp_constrained_cutoff(
                score_oof[region_mask],
                component.loc[region_mask, "target"].to_numpy(),
                baseline_fp,
            )
        final_dir = model_root / algorithm / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "case_model": final_case_model,
                "component_model": final_component_model,
                "case_features": case_features,
                "component_features": component_features,
            },
            final_dir / "models.joblib",
        )
        (final_dir / "cutoffs.json").write_text(
            json.dumps(final_cutoffs, indent=2, sort_keys=True), encoding="utf-8"
        )
        audit["algorithms"][algorithm] = {
            "folds": fold_audit,
            "final_cutoffs_from_oof": final_cutoffs,
        }
    return case_output, component_output, audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    case = pd.read_parquet(root / "features" / "case_features.parquet")
    component = pd.read_parquet(root / "features" / "component_features.parquet")
    case_predictions, component_predictions, audit = train_crossfit(
        case, component, config, root / "models"
    )
    predictions = root / "oof_predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    case_predictions.to_parquet(predictions / "case_predictions.parquet", index=False)
    component_predictions.to_parquet(
        predictions / "component_predictions.parquet", index=False
    )
    (predictions / "training_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "training_complete",
                "case_rows": len(case),
                "component_rows": len(component),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
