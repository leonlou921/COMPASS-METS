"""Final-fit inference, orchestration, validation, and flat ZIP packaging."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

from build_features import consolidate_checkpoints
from component_gate import (
    label_to_regions,
    load_probabilities,
    propose_components,
    reconstruct_regions,
    regions_to_label,
)
from features import extract_case_features
from reconstruct_and_evaluate import apply_v2_final, evaluate_oof
from train_models import fixed_component_keep, train_crossfit


def _evaluation_shard_command(
    python_executable: str,
    source_root: Path,
    config_path: Path,
    shard_count: int,
    shard_index: int,
    max_cases: int | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(source_root / "reconstruct_and_evaluate.py"),
        "--config",
        str(config_path),
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
        "--checkpoint-only",
    ]
    if max_cases is not None:
        command.extend(["--max-cases", str(max_cases)])
    return command


def _test_inference_command(
    python_executable: str,
    source_root: Path,
    config_path: Path,
    candidate: str,
    max_cases: int,
) -> list[str]:
    return [
        python_executable,
        str(source_root / "run_pipeline.py"),
        "--config",
        str(config_path),
        "--stage",
        "infer",
        "--candidate",
        candidate,
        "--max-cases",
        str(max_cases),
    ]


def _run_parallel_oof(config: dict) -> dict:
    """Evaluate disjoint case shards, then consolidate only after all succeed."""
    root = Path(config["output_root"])
    source_root = Path(config.get("source_root", Path(__file__).resolve().parent))
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "evaluation_config.json"
    temporary = config_path.with_name(config_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, config_path)
    worker_count = int(config.get("evaluation_workers", 3))
    if worker_count < 1:
        raise ValueError("evaluation_workers must be >= 1")
    batch_value = config.get("evaluation_batch_size")
    batch_size = None if batch_value is None else int(batch_value)
    if batch_size is not None and batch_size < 1:
        raise ValueError("evaluation_batch_size must be >= 1")
    expected_case_ids = None
    if batch_size is not None:
        prediction_cases = pd.read_parquet(
            root / "oof_predictions" / "component_predictions.parquet",
            columns=["case_id"],
        )
        expected_case_ids = set(prediction_cases["case_id"].astype(str).unique())

    round_index = 0
    while True:
        metric_dir = root / "metrics" / "per_case"
        completed_before = {path.stem for path in metric_dir.glob("*.json")}
        if expected_case_ids is not None and completed_before == expected_case_ids:
            break

        processes: list[tuple[int, subprocess.Popen, object, Path]] = []
        for shard_index in range(worker_count):
            log_path = run_dir / f"oof_evaluation_shard{shard_index}.log"
            log_handle = log_path.open("a" if round_index else "w", encoding="utf-8")
            log_handle.write(f"evaluation_round={round_index}\n")
            log_handle.flush()
            command = _evaluation_shard_command(
                sys.executable,
                source_root,
                config_path,
                worker_count,
                shard_index,
                max_cases=batch_size,
            )
            process = subprocess.Popen(
                command,
                cwd=source_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_index, process, log_handle, log_path))

        failures = []
        for shard_index, process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                failures.append(
                    {"shard_index": shard_index, "return_code": return_code, "tail": tail}
                )
        if failures:
            raise RuntimeError(f"parallel OOF evaluation failed: {failures}")
        if expected_case_ids is None:
            break
        completed_after = {path.stem for path in metric_dir.glob("*.json")}
        if len(completed_after) <= len(completed_before):
            missing = sorted(expected_case_ids - completed_after)
            raise RuntimeError(
                f"batched OOF evaluation made no progress; missing={missing[:5]}"
            )
        round_index += 1
    return evaluate_oof(config, consolidate_only=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_submission_package(zip_path: str | Path, expected: int = 179) -> dict:
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad_crc = archive.testzip()
    flat = all("/" not in name and "\\" not in name for name in names)
    legal_names = all(name.endswith(".nii.gz") for name in names)
    unique = len(set(names)) == len(names)
    if len(names) != expected or not flat or not legal_names or not unique or bad_crc is not None:
        raise RuntimeError(
            f"invalid submission ZIP: entries={len(names)}, flat={flat}, legal={legal_names}, "
            f"unique={unique}, bad_crc={bad_crc}"
        )
    return {
        "entry_count": len(names),
        "flat": flat,
        "legal_names": legal_names,
        "unique": unique,
        "crc_ok": bad_crc is None,
    }


def package_submission(
    prediction_dir: str | Path,
    submission_root: str | Path,
    run_id: str,
    expected: int,
    selected_candidate: str,
) -> dict[str, Path]:
    predictions = Path(prediction_dir)
    output = Path(submission_root)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(predictions.glob("*.nii.gz"))
    if len(files) != expected or len({path.name for path in files}) != expected:
        raise RuntimeError(f"expected {expected} unique predictions, found {len(files)}")
    zip_path = output / f"{run_id}_{selected_candidate}.zip"
    temporary = output / f".{zip_path.name}.tmp.{os.getpid()}"
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    os.replace(temporary, zip_path)
    validation = validate_submission_package(zip_path, expected=expected)
    digest = _sha256(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "selected_candidate": selected_candidate,
        "prediction_count": len(files),
        "zip_path": str(zip_path),
        "zip_sha256": digest,
        "validation": validation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output / f"{zip_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "zip_path": zip_path,
        "sha256_path": sha_path,
        "manifest_path": manifest_path,
    }


def _prediction_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame)[:, 1], dtype=np.float64)


def _case_probability_map(
    component: pd.DataFrame, case: pd.DataFrame, case_probability: np.ndarray
) -> np.ndarray:
    mapping = {
        (case_id, region): float(value)
        for case_id, region, value in zip(case["case_id"], case["region"], case_probability)
    }
    return np.asarray(
        [mapping[(case_id, region)] for case_id, region in zip(component["case_id"], component["region"])],
        dtype=np.float64,
    )


def _parse_candidate(candidate: str) -> tuple[str | None, str, bool]:
    if candidate == "fixed_component_conf":
        return None, "filter_only", False
    v2 = candidate.endswith("_V2final")
    base = candidate.removesuffix("_V2final") if v2 else candidate
    for algorithm in ("logistic", "lightgbm"):
        prefix = f"{algorithm}_"
        if base.startswith(prefix):
            policy = base.removeprefix(prefix)
            if policy not in {"filter_only", "consensus_rescue"}:
                break
            return algorithm, policy, v2
    raise ValueError(f"cannot parse selected candidate: {candidate}")


def _test_case_sets(test_roots: dict[str, Path]) -> list[str]:
    sets = {name: {path.stem for path in root.glob("*.npz")} for name, root in test_roots.items()}
    if not sets or any(values != sets["XL"] for values in sets.values()):
        raise RuntimeError("XL/M/FT test case sets are not identical")
    cases = sorted(sets["XL"])
    if len(cases) != 179:
        raise RuntimeError(f"expected 179 test cases, found {len(cases)}")
    return cases


def _learned_test_decisions(
    model_bundle: dict,
    cutoffs: dict,
    case_frame: pd.DataFrame,
    component_frame: pd.DataFrame,
) -> tuple[dict[tuple[str, int], bool], np.ndarray, np.ndarray, np.ndarray]:
    case_probability = _prediction_probability(
        model_bundle["case_model"], case_frame[model_bundle["case_features"]]
    )
    if component_frame.empty:
        empty = np.zeros(0, dtype=np.float64)
        return {}, case_probability, empty, empty
    component_probability = _prediction_probability(
        model_bundle["component_model"],
        component_frame[model_bundle["component_features"]],
    )
    score = component_probability * _case_probability_map(
        component_frame, case_frame, case_probability
    )
    decisions = {
        (row.region, int(row.component_id)): bool(
            value >= float(cutoffs[row.region]["threshold"])
        )
        for row, value in zip(component_frame.itertuples(), score)
    }
    return decisions, case_probability, component_probability, score


def infer_test(
    config: dict, selected_candidate: str, max_cases: int | None = None
) -> Path:
    root = Path(config["output_root"])
    test_roots = {name: Path(config["test_roots"][name]) for name in ("XL", "M", "FT")}
    cases = _test_case_sets(test_roots)
    algorithm, policy, use_v2 = _parse_candidate(selected_candidate)
    model_bundle = cutoffs = None
    if algorithm is not None:
        final_dir = root / "models" / algorithm / "final"
        model_bundle = joblib.load(final_dir / "models.joblib")
        cutoffs = json.loads((final_dir / "cutoffs.json").read_text(encoding="utf-8"))
    prediction_dir = root / "test_predictions" / selected_candidate
    prediction_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = root / "test_features" / "per_case"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for index, case_id in enumerate(cases, start=1):
        destination = prediction_dir / f"{case_id}.nii.gz"
        if destination.is_file():
            continue
        arrays = {
            name: load_probabilities(test_roots[name] / f"{case_id}.npz")
            for name in ("XL", "M", "FT")
        }
        empty_gt = np.zeros(arrays["XL"].shape[1:], dtype=np.uint8)
        case_rows, component_rows = extract_case_features(
            case_id, -1, arrays["XL"], arrays["M"], arrays["FT"], empty_gt
        )
        case_frame = pd.DataFrame(case_rows)
        component_frame = pd.DataFrame(component_rows)
        if algorithm is None:
            baseline = fixed_component_keep(component_frame, config)
            decisions = {
                (row.region, int(row.component_id)): bool(keep)
                for row, keep in zip(component_frame.itertuples(), baseline)
            }
        else:
            assert model_bundle is not None and cutoffs is not None
            decisions, case_probability, component_probability, score = (
                _learned_test_decisions(
                    model_bundle,
                    cutoffs,
                    case_frame,
                    component_frame,
                )
            )
            component_frame[f"{algorithm}_component_probability"] = component_probability
            component_frame[f"{algorithm}_score"] = score
            case_frame[f"{algorithm}_case_probability"] = case_probability
        proposals = propose_components(arrays, threshold=float(config["proposal_threshold"]))
        regions = reconstruct_regions(arrays, proposals, decisions, policy=policy)
        reference = nib.load(test_roots["XL"] / f"{case_id}.nii.gz")
        spacing = tuple(float(x) for x in reference.header.get_zooms()[:3][::-1])
        if use_v2:
            regions = apply_v2_final(regions, arrays, spacing, config["v2_final"])
        segmentation_xyz = regions_to_label(regions).transpose(2, 1, 0)
        if segmentation_xyz.shape != reference.shape:
            raise RuntimeError(
                f"{case_id}: output shape {segmentation_xyz.shape} != reference {reference.shape}"
            )
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(segmentation_xyz.astype(np.uint8), reference.affine, header)
        temporary = prediction_dir / f".{case_id}.tmp.nii.gz"
        nib.save(image, temporary)
        os.replace(temporary, destination)
        case_frame.to_parquet(feature_dir / f"{case_id}.case.parquet", index=False)
        component_frame.to_parquet(
            feature_dir / f"{case_id}.components.parquet", index=False
        )
        print(
            json.dumps(
                {"event": "test_case_complete", "case_id": case_id, "index": index, "total": len(cases)}
            ),
            flush=True,
        )
        if max_cases is not None and index >= max_cases:
            break
    return prediction_dir


def _run_batched_test_inference(config: dict, selected_candidate: str) -> Path:
    batch_value = config.get("test_inference_batch_size")
    if batch_value is None:
        return infer_test(config, selected_candidate)
    batch_size = int(batch_value)
    if batch_size < 1:
        raise ValueError("test_inference_batch_size must be >= 1")
    root = Path(config["output_root"])
    source_root = Path(config.get("source_root", Path(__file__).resolve().parent))
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "test_inference_config.json"
    temporary = config_path.with_name(config_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, config_path)
    prediction_dir = root / "test_predictions" / selected_candidate
    prediction_dir.mkdir(parents=True, exist_ok=True)
    expected = int(config.get("expected_test_cases", 179))
    log_path = run_dir / "test_inference_batches.log"
    batch_index = 0
    while True:
        completed_before = {path.name for path in prediction_dir.glob("*.nii.gz")}
        if len(completed_before) == expected:
            return prediction_dir
        log_handle = log_path.open("a" if batch_index else "w", encoding="utf-8")
        log_handle.write(f"test_inference_batch={batch_index}\n")
        log_handle.flush()
        process = subprocess.Popen(
            _test_inference_command(
                sys.executable,
                source_root,
                config_path,
                selected_candidate,
                batch_size,
            ),
            cwd=source_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            raise RuntimeError(
                f"batched test inference failed: return_code={return_code}, tail={tail}"
            )
        completed_after = {path.name for path in prediction_dir.glob("*.nii.gz")}
        if len(completed_after) <= len(completed_before):
            raise RuntimeError(
                f"batched test inference made no progress: completed={len(completed_after)}"
            )
        if len(completed_after) > expected:
            raise RuntimeError(
                f"batched test inference exceeded expected universe: {len(completed_after)}>{expected}"
            )
        batch_index += 1


def _write_run_manifest(config: dict, selected_candidate: str) -> Path:
    source_root = Path(config["source_root"])
    sources = sorted(source_root.glob("*.py")) + [source_root / "config.json"]
    manifest = {
        "run_id": config["run_id"],
        "selected_candidate": selected_candidate,
        "workspace_has_git": False,
        "raw_mri_features": bool(config["raw_mri_features"]),
        "workers": int(config["workers"]),
        "sources": {path.name: _sha256(path) for path in sources if path.is_file()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(config["output_root"]) / "run_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def run_after_features(config: dict) -> dict[str, str]:
    root = Path(config["output_root"])
    case_path, component_path = consolidate_checkpoints(root)
    case = pd.read_parquet(case_path)
    component = pd.read_parquet(component_path)
    if case["case_id"].nunique() != 1295 or component["case_id"].nunique() > 1295:
        raise RuntimeError(
            f"feature universe invalid: case unique={case['case_id'].nunique()}, "
            f"component unique={component['case_id'].nunique()}"
        )
    case_predictions, component_predictions, audit = train_crossfit(
        case, component, config, root / "models"
    )
    oof_predictions = root / "oof_predictions"
    oof_predictions.mkdir(parents=True, exist_ok=True)
    case_predictions.to_parquet(oof_predictions / "case_predictions.parquet", index=False)
    component_predictions.to_parquet(
        oof_predictions / "component_predictions.parquet", index=False
    )
    (oof_predictions / "training_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    del case_predictions, component_predictions, case, component
    gc.collect()
    report = _run_parallel_oof(config)
    selected = str(report["selection"]["selected"]["candidate"])
    predictions = _run_batched_test_inference(config, selected)
    artifacts = package_submission(
        predictions,
        config["submission_root"],
        run_id=config["run_id"],
        expected=179,
        selected_candidate=selected,
    )
    _write_run_manifest(config, selected)
    return {key: str(value) for key, value in artifacts.items()}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("after-features", "evaluate", "infer", "package"),
        default="after-features",
    )
    parser.add_argument("--candidate")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "after-features":
        result = run_after_features(config)
    elif args.stage == "evaluate":
        result = evaluate_oof(config)
    else:
        candidate = args.candidate
        if not candidate:
            selection = json.loads(
                (Path(config["output_root"]) / "selection.json").read_text(encoding="utf-8")
            )
            candidate = selection["selected"]["candidate"]
        predictions = infer_test(config, candidate, max_cases=args.max_cases)
        if args.stage == "infer":
            result = {"prediction_dir": str(predictions)}
        else:
            result = {
                key: str(value)
                for key, value in package_submission(
                    predictions,
                    config["submission_root"],
                    config["run_id"],
                    179,
                    candidate,
                ).items()
            }
    print(json.dumps({"event": "pipeline_stage_complete", "stage": args.stage, **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
