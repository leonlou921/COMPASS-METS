"""Leakage-safe ET and TC/WT component classifiers trained from frozen 5-fold OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


FAMILY_REGIONS = {"ET": ("ET",), "TCWT": ("TC", "WT")}
CASE_PROBABILITY_COLUMN = "case_probability_feature"
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
DEFAULT_PRECISION_FLOORS = {"ET": 0.97, "TC": 0.98, "WT": 0.98}


def family_for_region(region: str) -> str:
    for family, regions in FAMILY_REGIONS.items():
        if region in regions:
            return family
    raise ValueError(f"unsupported region for v3 component gate: {region}")


def attach_case_probability(
    component_table: pd.DataFrame,
    case_predictions: pd.DataFrame,
    source_column: str = "lightgbm_case_probability",
) -> pd.DataFrame:
    required = {"case_id", "region", source_column}
    missing = required.difference(case_predictions.columns)
    if missing:
        raise KeyError(f"missing case prediction columns: {sorted(missing)}")
    mapping = case_predictions.loc[:, ["case_id", "region", source_column]].copy()
    duplicated = mapping.duplicated(["case_id", "region"], keep=False)
    if duplicated.any():
        examples = mapping.loc[duplicated, ["case_id", "region"]].head()
        raise RuntimeError(
            f"duplicate case-region probabilities: {examples.to_dict('records')}"
        )
    mapping = mapping.rename(columns={source_column: CASE_PROBABILITY_COLUMN})
    joined = component_table.merge(
        mapping,
        on=["case_id", "region"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(joined) != len(component_table):
        raise RuntimeError("case-probability join changed the component row count")
    if joined[CASE_PROBABILITY_COLUMN].isna().any():
        missing_rows = joined.loc[
            joined[CASE_PROBABILITY_COLUMN].isna(), ["case_id", "region"]
        ].head()
        raise RuntimeError(
            f"missing case probability: {missing_rows.to_dict('records')}"
        )
    return joined


def validate_crossfit_partition(
    table: pd.DataFrame, heldout_fold: int
) -> dict[str, object]:
    fold = table["fold"].astype(int)
    validation = fold.eq(int(heldout_fold))
    if not validation.any():
        raise ValueError(f"held-out fold {heldout_fold} is absent")
    train_cases = set(table.loc[~validation, "case_id"].astype(str))
    validation_cases = set(table.loc[validation, "case_id"].astype(str))
    overlap = train_cases.intersection(validation_cases)
    if overlap:
        raise RuntimeError(
            f"case leakage in fold {heldout_fold}: {sorted(overlap)[:5]}"
        )
    return {
        "heldout_fold": int(heldout_fold),
        "training_folds": sorted(int(value) for value in fold.loc[~validation].unique()),
        "validation_folds": sorted(
            int(value) for value in fold.loc[validation].unique()
        ),
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "case_overlap": 0,
    }


def feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in table.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(table[column])
    ]
    if not columns:
        raise ValueError("no numeric component features")
    values = table.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("component features must be finite")
    return columns


def select_precision_cutoff(
    probability: np.ndarray,
    target: np.ndarray,
    minimum_precision: float,
) -> dict[str, float | int | bool]:
    """Maximize recall while meeting a fixed precision floor."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    if probability.ndim != 1 or probability.shape != target.shape:
        raise ValueError("probability and target must be aligned one-dimensional arrays")
    if not 0.0 <= float(minimum_precision) <= 1.0:
        raise ValueError("minimum_precision must be in [0, 1]")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("target must be binary")
    if probability.size == 0:
        return {
            "threshold": 1.0,
            "minimum_precision": float(minimum_precision),
            "feasible": True,
            "kept": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
        }
    thresholds = np.r_[
        np.nextafter(float(probability.max()), np.inf),
        np.unique(probability)[::-1],
    ]
    positives = int(target.sum())
    candidates: list[dict[str, float | int | bool]] = []
    for threshold in thresholds:
        keep = probability >= threshold
        kept = int(keep.sum())
        tp = int(np.logical_and(keep, target == 1).sum())
        fp = int(np.logical_and(keep, target == 0).sum())
        fn = positives - tp
        precision = tp / kept if kept else 1.0
        recall = tp / positives if positives else 1.0
        candidates.append(
            {
                "threshold": float(threshold),
                "minimum_precision": float(minimum_precision),
                "feasible": precision >= minimum_precision,
                "kept": kept,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": float(precision),
                "recall": float(recall),
            }
        )
    feasible = [row for row in candidates if bool(row["feasible"])]
    # Empty keep is the only honest feasible choice when every proposal is negative.
    if positives == 0:
        return candidates[0]
    positive_keep = [row for row in feasible if int(row["kept"]) > 0]
    pool = positive_keep or feasible or candidates
    if positive_keep:
        return max(
            pool,
            key=lambda row: (
                float(row["recall"]),
                float(row["precision"]),
                -int(row["false_positives"]),
                float(row["threshold"]),
            ),
        )
    if feasible:
        return candidates[0]
    return max(
        pool,
        key=lambda row: (
            float(row["precision"]),
            float(row["recall"]),
            -int(row["false_positives"]),
            float(row["threshold"]),
        ),
    )


def _fit_model(x: pd.DataFrame, y: np.ndarray, seed: int) -> LGBMClassifier:
    if len(np.unique(y)) != 2:
        raise RuntimeError("each component family training split must contain both classes")
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
        random_state=int(seed),
        n_jobs=8,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(x, y)
    return model


def _predict(model: object, frame: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(frame)[:, 1], dtype=np.float64)
    if not np.isfinite(probability).all():
        raise RuntimeError("model returned non-finite probability")
    return probability


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    return {
        "row_count": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_region_gate(
    component_table: pd.DataFrame,
    case_predictions: pd.DataFrame,
    model_root: Path,
    v2_predictions: pd.DataFrame | None = None,
    seed: int = 20260727,
    precision_floors: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    precision_floors = dict(DEFAULT_PRECISION_FLOORS if precision_floors is None else precision_floors)
    eligible = component_table.loc[
        component_table["region"].isin(("ET", "TC", "WT"))
    ].copy()
    eligible = attach_case_probability(eligible, case_predictions)
    folds = sorted(int(value) for value in eligible["fold"].astype(int).unique())
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"expected folds [0, 1, 2, 3, 4], got {folds}")

    output_columns = [
        "case_id",
        "fold",
        "region",
        "component_id",
        "target",
        "overlap_voxels",
    ]
    output = eligible.loc[:, output_columns].copy()
    output["v3_component_probability"] = np.nan
    family_audit: dict[str, object] = {}

    for family, regions in FAMILY_REGIONS.items():
        family_mask = eligible["region"].isin(regions)
        family_table = eligible.loc[family_mask]
        features = feature_columns(family_table)
        fold_audits = []
        for heldout in folds:
            partition = validate_crossfit_partition(family_table, heldout)
            validation = family_table["fold"].astype(int).eq(heldout)
            training = ~validation
            model = _fit_model(
                family_table.loc[training, features],
                family_table.loc[training, "target"].to_numpy(dtype=np.int8),
                int(seed) + (1000 if family == "ET" else 2000) + heldout,
            )
            probability = _predict(model, family_table.loc[validation, features])
            output.loc[family_table.index[validation], "v3_component_probability"] = probability
            fold_dir = model_root / family / f"fold_{heldout}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "family": family,
                    "regions": regions,
                    "component_model": model,
                    "component_features": features,
                },
                fold_dir / "models.joblib",
            )
            partition["metrics"] = _binary_metrics(
                family_table.loc[validation, "target"].to_numpy(dtype=np.int8),
                probability,
            )
            fold_audits.append(partition)

        final_model = _fit_model(
            family_table.loc[:, features],
            family_table["target"].to_numpy(dtype=np.int8),
            int(seed) + (3000 if family == "ET" else 4000),
        )
        final_dir = model_root / family / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "family": family,
                "regions": regions,
                "component_model": final_model,
                "component_features": features,
            },
            final_dir / "models.joblib",
        )
        family_audit[family] = {
            "regions": list(regions),
            "row_count": int(len(family_table)),
            "case_count": int(family_table["case_id"].nunique()),
            "component_features": features,
            "folds": fold_audits,
        }

    if output["v3_component_probability"].isna().any():
        raise RuntimeError("v3 OOF probabilities are incomplete")

    metrics: dict[str, object] = {"v3": {}, "v2": {}, "delta_average_precision": {}}
    if v2_predictions is not None:
        keys = ["case_id", "fold", "region", "component_id"]
        old = v2_predictions.loc[
            v2_predictions["region"].isin(("ET", "TC", "WT")),
            keys + ["v2_component_probability"],
        ]
        output = output.merge(old, on=keys, validate="one_to_one", how="left")
        if output["v2_component_probability"].isna().any():
            raise RuntimeError("v2 comparison probabilities are incomplete")

    cutoffs: dict[str, object] = {}
    for region in ("ET", "TC", "WT"):
        mask = output["region"].eq(region)
        target = output.loc[mask, "target"].to_numpy(dtype=np.int8)
        probability = output.loc[mask, "v3_component_probability"].to_numpy(dtype=float)
        metrics["v3"][region] = _binary_metrics(target, probability)
        cutoffs[region] = select_precision_cutoff(
            probability, target, float(precision_floors[region])
        )
        if v2_predictions is not None:
            old_probability = output.loc[
                mask, "v2_component_probability"
            ].to_numpy(dtype=float)
            metrics["v2"][region] = _binary_metrics(target, old_probability)
            metrics["delta_average_precision"][region] = float(
                metrics["v3"][region]["average_precision"]
                - metrics["v2"][region]["average_precision"]
            )

    family_metrics = {}
    for family, regions in FAMILY_REGIONS.items():
        mask = output["region"].isin(regions)
        family_metrics[family] = _binary_metrics(
            output.loc[mask, "target"].to_numpy(dtype=np.int8),
            output.loc[mask, "v3_component_probability"].to_numpy(dtype=float),
        )

    deltas = list(metrics["delta_average_precision"].values())
    stable = (
        all(bool(cutoffs[region]["feasible"]) for region in ("ET", "TC", "WT"))
        and (not deltas or min(deltas) >= -0.005)
        and (not deltas or sum(deltas) > 0.0)
    )
    audit: dict[str, object] = {
        "run": "region_component_gate_v3_20260727",
        "seed": int(seed),
        "families": family_audit,
        "metrics": metrics,
        "family_metrics": family_metrics,
        "precision_floors": precision_floors,
        "add_cutoffs": cutoffs,
        "stability_gate": {
            "passed": bool(stable),
            "requirements": {
                "all_precision_cutoffs_feasible": True,
                "minimum_region_ap_delta_vs_v2": -0.005,
                "sum_region_ap_delta_vs_v2_strictly_positive": True,
            },
        },
        "row_count": int(len(output)),
        "case_count": int(output["case_id"].nunique()),
    }
    return output, audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-features", required=True)
    parser.add_argument("--case-predictions", required=True)
    parser.add_argument("--v2-predictions")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args(argv)

    component_path = Path(args.component_features)
    case_path = Path(args.case_predictions)
    v2_path = Path(args.v2_predictions) if args.v2_predictions else None
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions, audit = train_region_gate(
        pd.read_parquet(component_path),
        pd.read_parquet(case_path),
        output_root / "models",
        pd.read_parquet(v2_path) if v2_path is not None else None,
        seed=int(args.seed),
    )
    prediction_root = output_root / "oof_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / "component_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    audit["inputs"] = {
        "component_features": {
            "path": str(component_path),
            "sha256": _sha256(component_path),
        },
        "case_predictions": {"path": str(case_path), "sha256": _sha256(case_path)},
        "v2_predictions": (
            {"path": str(v2_path), "sha256": _sha256(v2_path)}
            if v2_path is not None
            else None
        ),
    }
    audit["outputs"] = {
        "component_predictions": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path),
        }
    }
    audit_path = output_root / "training_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "region_component_gate_v3_complete",
                "rows": len(predictions),
                "cases": int(predictions["case_id"].nunique()),
                "stability_gate_passed": audit["stability_gate"]["passed"],
                "audit": str(audit_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
