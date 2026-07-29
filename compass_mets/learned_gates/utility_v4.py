from __future__ import annotations

from typing import Any, Iterable

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd


N01_LCV2_ET_CUTOFF = 0.5497123599
RGV3_ET_CUTOFF = 0.7702616034384248
GEOMETRY_PRECISION_FLOOR = 0.50
NON_FEATURE_COLUMNS = frozenset(
    {
        "case_id",
        "fold",
        "region",
        "component_id",
        "proposal_id",
        "target",
        "overlap_voxels",
        "component_precision",
        "gt_coverage",
        "iou",
        "baseline_keep",
        "existence_target",
        "geometry_safe_target",
        "utility_target",
        "candidate_pool",
    }
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def candidate_pool_mask(frame: pd.DataFrame) -> pd.Series:
    required = (
        "region",
        "baseline_keep",
        "v2_component_probability",
        "support2_t025_fraction",
    )
    _require_columns(frame, required)
    n01_like = frame["v2_component_probability"].ge(
        N01_LCV2_ET_CUTOFF
    ) & frame["support2_t025_fraction"].gt(0.0)
    rgv3_like = frame["v2_component_probability"].ge(RGV3_ET_CUTOFF)
    return (
        frame["region"].astype(str).str.upper().eq("ET")
        & frame["baseline_keep"].eq(0)
        & (n01_like | rgv3_like)
    )


def utility_targets(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("target", "component_precision"))
    existence = frame["target"].astype(np.int8)
    if not existence.isin((0, 1)).all():
        raise ValueError("target must be binary")
    geometry_safe = (
        existence.eq(1)
        & frame["component_precision"].astype(float).ge(
            GEOMETRY_PRECISION_FLOOR
        )
    ).astype(np.int8)
    return pd.DataFrame(
        {
            "existence_target": existence.to_numpy(dtype=np.int8),
            "geometry_safe_target": geometry_safe.to_numpy(dtype=np.int8),
        },
        index=frame.index,
    )


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not features:
        raise ValueError("no numeric model features remain after leakage exclusion")
    values = frame.loc[:, features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("model features must be finite")
    return features


def new_binary_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def crossfit_binary_classifier(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    target_column: str,
    seed: int,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]]]:
    _require_columns(frame, ("case_id", "fold", target_column, *feature_names))
    folds = sorted(int(value) for value in frame["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"expected folds [0, 1, 2, 3, 4], found {folds}")
    target = frame[target_column].to_numpy(dtype=np.int8)
    if not np.isin(target, (0, 1)).all():
        raise ValueError(f"{target_column} must be binary")
    features = frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("crossfit features must be finite")

    probability = np.full(len(frame), np.nan, dtype=np.float64)
    models: list[Any] = []
    audit: list[dict[str, Any]] = []
    for heldout in folds:
        validation = frame["fold"].astype(int).eq(heldout).to_numpy()
        training = ~validation
        train_cases = set(frame.loc[training, "case_id"].astype(str))
        validation_cases = set(frame.loc[validation, "case_id"].astype(str))
        overlap = train_cases & validation_cases
        if overlap:
            raise RuntimeError(
                f"case leakage in fold {heldout}: {sorted(overlap)[:5]}"
            )
        if np.unique(target[training]).size != 2:
            raise RuntimeError(f"training fold {heldout} lacks both target classes")
        model = new_binary_classifier(seed + heldout)
        model.fit(features[training], target[training])
        probability[validation] = model.predict_proba(features[validation])[:, 1]
        models.append(model)
        audit.append(
            {
                "heldout_fold": heldout,
                "training_folds": sorted(
                    int(value)
                    for value in frame.loc[training, "fold"].unique()
                ),
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "training_cases": len(train_cases),
                "validation_cases": len(validation_cases),
                "case_overlap": len(overlap),
            }
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("crossfit probabilities are incomplete")
    return probability, models, audit
