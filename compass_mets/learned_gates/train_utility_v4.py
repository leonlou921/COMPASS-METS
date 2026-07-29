from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from compass_mets.learned_gates.utility_v4 import (
    candidate_pool_mask,
    crossfit_binary_classifier,
    model_feature_columns,
    new_binary_classifier,
    utility_targets,
)


KEY_COLUMNS = ["case_id", "fold", "region", "component_id"]
SEED = 20260728


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if len(target) == 0:
        raise ValueError("cannot score an empty target")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    metrics: dict[str, Any] = {
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(target, probability))
        if np.unique(target).size == 2
        else None
    )
    return metrics


def merge_sources(
    feature_path: Path, v2_path: Path, v3_path: Path
) -> pd.DataFrame:
    features = pd.read_parquet(feature_path)
    v2 = pd.read_parquet(
        v2_path, columns=KEY_COLUMNS + ["v2_component_probability"]
    )
    v3 = pd.read_parquet(
        v3_path, columns=KEY_COLUMNS + ["v3_component_probability"]
    )
    merged = features.merge(v2, on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = merged.merge(v3, on=KEY_COLUMNS, how="left", validate="one_to_one")
    et = merged.loc[merged["region"].astype(str).str.upper().eq("ET")].copy()
    if et[["v2_component_probability", "v3_component_probability"]].isna().any().any():
        missing = et[
            et[["v2_component_probability", "v3_component_probability"]]
            .isna()
            .any(axis=1)
        ]
        raise RuntimeError(f"ET model probabilities missing for {len(missing)} rows")
    targets = utility_targets(et)
    et[targets.columns] = targets
    et["candidate_pool"] = candidate_pool_mask(et).astype(np.int8)
    return et


def train_one_target(
    training: pd.DataFrame,
    *,
    feature_names: list[str],
    target_column: str,
    output_root: Path,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability, models, audit = crossfit_binary_classifier(
        training,
        feature_names=feature_names,
        target_column=target_column,
        seed=seed,
    )
    model_root = output_root / "models" / target_column
    for fold, model in enumerate(models):
        fold_root = model_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, fold_root / "model.joblib")
    final_model = new_binary_classifier(seed + 100)
    final_model.fit(
        training.loc[:, feature_names].to_numpy(dtype=np.float64),
        training[target_column].to_numpy(dtype=np.int8),
    )
    final_root = model_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, final_root / "model.joblib")
    return probability, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    et = merge_sources(args.features, args.v2, args.v3)
    training = et.loc[et["baseline_keep"].eq(0)].copy().reset_index(drop=True)
    if sorted(training["fold"].astype(int).unique()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("training rows do not cover exactly five folds")
    feature_names = model_feature_columns(training)

    existence_probability, existence_audit = train_one_target(
        training,
        feature_names=feature_names,
        target_column="existence_target",
        output_root=args.output,
        seed=SEED,
    )
    geometry_probability, geometry_audit = train_one_target(
        training,
        feature_names=feature_names,
        target_column="geometry_safe_target",
        output_root=args.output,
        seed=SEED + 1000,
    )
    training["v4_existence_probability"] = existence_probability
    training["v4_geometry_probability"] = geometry_probability
    training["v4_utility_probability"] = (
        existence_probability * geometry_probability
    )

    pool = training["candidate_pool"].eq(1).to_numpy()
    existence_target = training["existence_target"].to_numpy(dtype=np.int8)
    geometry_target = training["geometry_safe_target"].to_numpy(dtype=np.int8)
    metrics = {
        "all_n03_missing_et_components": {
            "v2_vs_existence": binary_metrics(
                existence_target,
                training["v2_component_probability"].to_numpy(dtype=np.float64),
            ),
            "v3_vs_existence": binary_metrics(
                existence_target,
                training["v3_component_probability"].to_numpy(dtype=np.float64),
            ),
            "v4_existence": binary_metrics(
                existence_target, existence_probability
            ),
            "v4_geometry": binary_metrics(geometry_target, geometry_probability),
            "v4_utility_vs_geometry": binary_metrics(
                geometry_target, training["v4_utility_probability"].to_numpy()
            ),
        },
        "n01_rgv3_aligned_candidate_pool": {
            "v2_vs_existence": binary_metrics(
                existence_target[pool],
                training.loc[pool, "v2_component_probability"].to_numpy(),
            ),
            "v3_vs_existence": binary_metrics(
                existence_target[pool],
                training.loc[pool, "v3_component_probability"].to_numpy(),
            ),
            "v4_existence": binary_metrics(
                existence_target[pool], existence_probability[pool]
            ),
            "v4_geometry": binary_metrics(
                geometry_target[pool], geometry_probability[pool]
            ),
            "v4_utility_vs_geometry": binary_metrics(
                geometry_target[pool],
                training.loc[pool, "v4_utility_probability"].to_numpy(),
            ),
        },
    }

    prediction_path = args.output / "oof_predictions.parquet"
    output_columns = (
        KEY_COLUMNS
        + [
            "baseline_keep",
            "candidate_pool",
            "target",
            "component_precision",
            "existence_target",
            "geometry_safe_target",
            "v2_component_probability",
            "v3_component_probability",
            "v4_existence_probability",
            "v4_geometry_probability",
            "v4_utility_probability",
        ]
    )
    training.loc[:, output_columns].to_parquet(prediction_path, index=False)
    (args.output / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "name": "N03_ET_add_utility_v4",
        "seed": SEED,
        "training_scope": "ET components with baseline_keep == 0",
        "candidate_evaluation_scope": (
            "N01-like or RGv3-like ET candidates within baseline_keep == 0"
        ),
        "important_limit": (
            "This is component-table five-fold OOF, not a full N03 mask-level OOF."
        ),
        "targets": {
            "existence_target": "any GT overlap",
            "geometry_safe_target": (
                "any GT overlap and component precision >= 0.50"
            ),
            "utility_probability": (
                "v4_existence_probability * v4_geometry_probability"
            ),
        },
        "source_files": {
            "features": {
                "path": str(args.features),
                "sha256": file_sha256(args.features),
            },
            "v2": {"path": str(args.v2), "sha256": file_sha256(args.v2)},
            "v3": {"path": str(args.v3), "sha256": file_sha256(args.v3)},
        },
        "rows": {
            "all_et": int(len(et)),
            "training_n03_missing": int(len(training)),
            "candidate_pool": int(pool.sum()),
            "cases_training": int(training["case_id"].nunique()),
            "cases_candidate_pool": int(training.loc[pool, "case_id"].nunique()),
            "fold_rows": {
                str(int(fold)): int(count)
                for fold, count in training.groupby("fold").size().items()
            },
            "candidate_fold_rows": {
                str(int(fold)): int(count)
                for fold, count in training.loc[pool].groupby("fold").size().items()
            },
        },
        "feature_count": len(feature_names),
        "features": feature_names,
        "crossfit": {
            "existence": existence_audit,
            "geometry": geometry_audit,
        },
        "metrics": metrics,
        "artifacts": {
            "oof_predictions": str(prediction_path),
        },
    }
    audit_path = args.output / "training_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
