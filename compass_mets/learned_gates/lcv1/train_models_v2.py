"""Leakage-safe LightGBM component model with case probability as a feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from compass_mets.learned_gates.lcv1.train_models import (
    _fit_model,
    _predict,
    feature_columns,
)


CASE_PROBABILITY_COLUMN = "case_probability_feature"


def join_oof_case_probability(
    component_table: pd.DataFrame,
    case_predictions: pd.DataFrame,
    source_column: str = "lightgbm_case_probability",
) -> pd.DataFrame:
    """Join exactly one leakage-safe OOF case probability per case and region."""
    required = {"case_id", "region", source_column}
    missing = required.difference(case_predictions.columns)
    if missing:
        raise KeyError(f"missing case prediction columns: {sorted(missing)}")
    mapping = case_predictions[["case_id", "region", source_column]].copy()
    duplicates = mapping.duplicated(["case_id", "region"], keep=False)
    if duplicates.any():
        keys = mapping.loc[duplicates, ["case_id", "region"]].head().to_dict("records")
        raise RuntimeError(f"duplicate case-region predictions: {keys}")
    mapping = mapping.rename(columns={source_column: CASE_PROBABILITY_COLUMN})
    output = component_table.merge(
        mapping,
        on=["case_id", "region"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(output) != len(component_table):
        raise RuntimeError("case-probability join changed component row count")
    if output[CASE_PROBABILITY_COLUMN].isna().any():
        missing_keys = output.loc[
            output[CASE_PROBABILITY_COLUMN].isna(), ["case_id", "region"]
        ].head()
        raise RuntimeError(
            f"missing case probability for components: {missing_keys.to_dict('records')}"
        )
    return output


def train_component_crossfit_v2(
    component_table: pd.DataFrame,
    case_predictions: pd.DataFrame,
    model_root: str | Path,
    seed: int = 20260723,
) -> tuple[pd.DataFrame, dict]:
    """Train five held-out-fold component models and one all-data final model."""
    component = join_oof_case_probability(component_table, case_predictions)
    features = feature_columns(component)
    if CASE_PROBABILITY_COLUMN not in features:
        raise RuntimeError("case probability is not present in v2 component features")
    output_columns = [
        "case_id",
        "fold",
        "region",
        "component_id",
        "target",
        "overlap_voxels",
    ]
    output = component[output_columns].copy()
    probability_oof = np.full(len(component), np.nan, dtype=np.float64)
    audit_folds = []
    root = Path(model_root)
    for heldout_fold in sorted(component["fold"].astype(int).unique()):
        validation = component["fold"].astype(int) == heldout_fold
        training = ~validation
        train_cases = set(component.loc[training, "case_id"])
        validation_cases = set(component.loc[validation, "case_id"])
        if train_cases.intersection(validation_cases):
            raise RuntimeError(f"case leakage in fold {heldout_fold}")
        model = _fit_model(
            "lightgbm",
            component.loc[training, features],
            component.loc[training, "target"].to_numpy(dtype=np.int8),
            int(seed) + 2000 + int(heldout_fold),
        )
        probability_oof[validation.to_numpy()] = _predict(
            model, component.loc[validation, features]
        )
        fold_dir = root / "lightgbm" / f"fold_{heldout_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"component_model": model, "component_features": features},
            fold_dir / "models.joblib",
        )
        audit_folds.append(
            {
                "heldout_fold": int(heldout_fold),
                "training_folds": sorted(
                    int(value)
                    for value in component.loc[training, "fold"].astype(int).unique()
                ),
                "train_case_count": len(train_cases),
                "validation_case_count": len(validation_cases),
                "case_overlap": 0,
            }
        )
    if not np.isfinite(probability_oof).all():
        raise RuntimeError("incomplete v2 OOF component probabilities")
    output["v2_component_probability"] = probability_oof

    final_model = _fit_model(
        "lightgbm",
        component[features],
        component["target"].to_numpy(dtype=np.int8),
        int(seed) + 3000,
    )
    final_dir = root / "lightgbm" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"component_model": final_model, "component_features": features},
        final_dir / "models.joblib",
    )
    audit = {
        "algorithm": "lightgbm",
        "case_probability_usage": "input_feature_only",
        "component_features": features,
        "folds": audit_folds,
        "row_count": len(component),
        "case_count": int(component["case_id"].nunique()),
    }
    return output, audit


def _case_probability_map(
    component_frame: pd.DataFrame,
    case_frame: pd.DataFrame,
    probability: np.ndarray,
) -> np.ndarray:
    mapping = {
        (case_id, region): float(value)
        for case_id, region, value in zip(
            case_frame["case_id"], case_frame["region"], probability
        )
    }
    return np.asarray(
        [
            mapping[(case_id, region)]
            for case_id, region in zip(
                component_frame["case_id"], component_frame["region"]
            )
        ],
        dtype=np.float64,
    )


def predict_test_component_probability(
    v2_component_bundle: dict,
    v1_case_bundle: dict,
    case_frame: pd.DataFrame,
    component_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict test components without multiplying by the case probability."""
    case_probability = _predict(
        v1_case_bundle["case_model"],
        case_frame[v1_case_bundle["case_features"]],
    )
    if component_frame.empty:
        return np.zeros(0, dtype=np.float64), case_probability
    component = component_frame.copy()
    component[CASE_PROBABILITY_COLUMN] = _case_probability_map(
        component, case_frame, case_probability
    )
    probability = _predict(
        v2_component_bundle["component_model"],
        component[v2_component_bundle["component_features"]],
    )
    return probability, case_probability


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    v1_root = Path(config["v1_output_root"])
    output_root = Path(config["output_root"])
    component = pd.read_parquet(
        v1_root / "features" / "component_features.parquet"
    )
    case_predictions = pd.read_parquet(
        v1_root / "oof_predictions" / "case_predictions.parquet"
    )
    predictions, audit = train_component_crossfit_v2(
        component,
        case_predictions,
        output_root / "models",
        seed=int(config.get("seed", 20260723)),
    )
    oof_root = output_root / "oof_predictions"
    oof_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        oof_root / "component_predictions.parquet", index=False
    )
    (oof_root / "training_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "v2_training_complete",
                "component_rows": len(predictions),
                "case_count": int(predictions["case_id"].nunique()),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
