"""Resumable two-worker remote controller for the complete v2 pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from v2_component_gate import CANDIDATES


def worker_command(
    python_executable: str,
    source_root: Path,
    config_path: Path,
    stage: str,
    shard_count: int,
    shard_index: int,
    max_cases: int,
    case_ids: list[str] | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(source_root / "v2_pipeline.py"),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
        "--max-cases",
        str(max_cases),
    ]
    if case_ids:
        command.extend(["--case-ids", *case_ids])
    return command


def adaptive_case_batches(
    case_costs: dict[str, int],
    max_workers: int,
    max_cases_per_worker: int,
    peak_cost_budget: int,
) -> list[list[str]]:
    """Build one memory-bounded wave using uncompressed NPZ bytes as cost."""
    if not case_costs:
        return []
    if max_workers < 1 or max_cases_per_worker < 1 or peak_cost_budget < 1:
        raise ValueError("adaptive scheduling limits must all be positive")
    ordered = sorted(case_costs, key=lambda case_id: (-case_costs[case_id], case_id))
    seeds: list[str] = []
    deferred: list[str] = []
    peak_cost = 0
    for case_id in ordered:
        cost = int(case_costs[case_id])
        if len(seeds) < max_workers and (
            not seeds or peak_cost + cost <= peak_cost_budget
        ):
            seeds.append(case_id)
            peak_cost += cost
        else:
            deferred.append(case_id)
    batches = [[case_id] for case_id in seeds]
    batch_peaks = [int(case_costs[case_id]) for case_id in seeds]
    batch_loads = batch_peaks.copy()
    target_load = int(max(batch_peaks) * 1.25)
    for case_id in deferred:
        cost = int(case_costs[case_id])
        eligible = [
            index
            for index, batch in enumerate(batches)
            if len(batch) < max_cases_per_worker
            and cost <= batch_peaks[index]
            and batch_loads[index] + cost <= target_load
        ]
        if not eligible:
            continue
        index = min(eligible, key=lambda value: (batch_loads[value], value))
        batches[index].append(case_id)
        batch_loads[index] += cost
    return batches


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        sizes = [
            int(info.file_size)
            for info in archive.infolist()
            if info.filename.endswith(".npy")
        ]
    if not sizes:
        raise RuntimeError(f"{path} contains no NPY payload")
    return max(sizes)


def _stage_case_paths(config: dict, stage: str) -> dict[str, Path]:
    if stage in {"build-calibration", "evaluate"}:
        table = pd.read_parquet(
            Path(config["v1_output_root"])
            / "oof_predictions"
            / "case_predictions.parquet",
            columns=["case_id", "fold"],
        ).drop_duplicates()
        root = Path(config["oof_roots"]["XL"])
        return {
            str(row.case_id): root
            / f"fold_{int(row.fold)}"
            / "validation"
            / f"{row.case_id}.npz"
            for row in table.itertuples(index=False)
        }
    if stage == "infer":
        return {
            path.stem: path
            for path in Path(config["test_roots"]["XL"]).glob("*.npz")
        }
    raise ValueError(f"unknown checkpoint stage: {stage}")


def _completed_case_ids(config: dict, stage: str) -> set[str]:
    root = Path(config["output_root"])
    if stage == "build-calibration":
        return {
            path.stem for path in (root / "calibration" / "per_case").glob("*.parquet")
        }
    if stage == "evaluate":
        return {path.stem for path in (root / "metrics" / "per_case").glob("*.json")}
    if stage == "infer":
        return set.intersection(
            *[
                {
                    path.name.removesuffix(".nii.gz")
                    for path in (root / "test_predictions" / candidate).glob("*.nii.gz")
                }
                for candidate in CANDIDATES
            ]
        )
    raise ValueError(f"unknown checkpoint stage: {stage}")


def _run_checked(command: list[str], cwd: Path) -> None:
    print(json.dumps({"event": "command_started", "command": command}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)
    print(json.dumps({"event": "command_complete", "command": command}), flush=True)


def _checkpoint_count(config: dict, stage: str) -> int:
    root = Path(config["output_root"])
    if stage == "build-calibration":
        return len(list((root / "calibration" / "per_case").glob("*.parquet")))
    if stage == "evaluate":
        return len(list((root / "metrics" / "per_case").glob("*.json")))
    if stage == "infer":
        return min(
            len(list((root / "test_predictions" / candidate).glob("*.nii.gz")))
            for candidate in CANDIDATES
        )
    raise ValueError(f"unknown checkpoint stage: {stage}")


def _parallel_batches(
    config: dict,
    config_path: Path,
    stage: str,
    expected: int,
    workers: int,
    batch_size: int,
) -> None:
    source_root = Path(config["source_root"])
    python_executable = str(config["python"])
    log_root = Path(config["output_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    round_index = 0
    while _checkpoint_count(config, stage) < expected:
        before = _checkpoint_count(config, stage)
        processes = []
        for shard_index in range(workers):
            log_path = log_root / f"{stage}_shard{shard_index}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"round={round_index}\n")
            handle.flush()
            process = subprocess.Popen(
                worker_command(
                    python_executable,
                    source_root,
                    config_path,
                    stage,
                    workers,
                    shard_index,
                    batch_size,
                ),
                cwd=source_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_index, process, handle, log_path))
        failures = []
        for shard_index, process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-30:]
                failures.append(
                    {
                        "shard_index": shard_index,
                        "return_code": return_code,
                        "tail": tail,
                    }
                )
        if failures:
            raise RuntimeError(f"{stage} workers failed: {failures}")
        after = _checkpoint_count(config, stage)
        print(
            json.dumps(
                {
                    "event": "parallel_round_complete",
                    "stage": stage,
                    "round": round_index,
                    "before": before,
                    "after": after,
                    "expected": expected,
                }
            ),
            flush=True,
        )
        if after <= before:
            raise RuntimeError(
                f"{stage} made no progress: completed={after}, expected={expected}"
            )
        round_index += 1


def _adaptive_parallel_batches(
    config: dict,
    config_path: Path,
    stage: str,
    expected: int,
    workers: int,
    batch_size: int,
    peak_cost_budget: int,
) -> None:
    source_root = Path(config["source_root"])
    python_executable = str(config["python"])
    log_root = Path(config["output_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    case_paths = _stage_case_paths(config, stage)
    if len(case_paths) != expected:
        raise RuntimeError(
            f"{stage} case universe differs: found={len(case_paths)} expected={expected}"
        )
    case_costs = {
        case_id: _npz_uncompressed_bytes(path)
        for case_id, path in case_paths.items()
    }
    round_index = 0
    while _checkpoint_count(config, stage) < expected:
        before = _checkpoint_count(config, stage)
        completed = _completed_case_ids(config, stage)
        pending = {
            case_id: cost
            for case_id, cost in case_costs.items()
            if case_id not in completed
        }
        batches = adaptive_case_batches(
            pending,
            max_workers=workers,
            max_cases_per_worker=batch_size,
            peak_cost_budget=peak_cost_budget,
        )
        if not batches:
            raise RuntimeError(
                f"{stage} scheduler found no batch: completed={before} expected={expected}"
            )
        planned_peak = sum(
            max(case_costs[case_id] for case_id in batch) for batch in batches
        )
        print(
            json.dumps(
                {
                    "event": "adaptive_round_started",
                    "stage": stage,
                    "round": round_index,
                    "workers": len(batches),
                    "case_count": sum(len(batch) for batch in batches),
                    "planned_peak_uncompressed_bytes": planned_peak,
                    "batches": batches,
                }
            ),
            flush=True,
        )
        processes = []
        for worker_index, batch in enumerate(batches):
            log_path = log_root / f"{stage}_adaptive_worker{worker_index}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"round={round_index} cases={json.dumps(batch)}\n")
            handle.flush()
            process = subprocess.Popen(
                worker_command(
                    python_executable,
                    source_root,
                    config_path,
                    stage,
                    1,
                    0,
                    len(batch),
                    case_ids=batch,
                ),
                cwd=source_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker_index, process, handle, log_path))
        failures = []
        for worker_index, process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                failures.append(
                    {
                        "worker_index": worker_index,
                        "return_code": return_code,
                        "tail": log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-30:],
                    }
                )
        if failures:
            raise RuntimeError(f"{stage} adaptive workers failed: {failures}")
        after = _checkpoint_count(config, stage)
        print(
            json.dumps(
                {
                    "event": "adaptive_round_complete",
                    "stage": stage,
                    "round": round_index,
                    "before": before,
                    "after": after,
                    "expected": expected,
                }
            ),
            flush=True,
        )
        if after <= before:
            raise RuntimeError(
                f"{stage} made no progress: completed={after}, expected={expected}"
            )
        round_index += 1


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(config.get("parallel_workers", 2))
    batch_size = int(config.get("batch_size", 3))
    peak_cost_budget = int(
        config.get("peak_uncompressed_npz_budget_bytes", 4_500_000_000)
    )
    expected_oof = int(config.get("expected_oof_cases", 1295))
    expected_test = int(config.get("expected_test_cases", 179))
    python_executable = str(config["python"])

    final_model = (
        output_root / "models" / "lightgbm" / "final" / "models.joblib"
    )
    if not final_model.is_file():
        _run_checked(
            [
                python_executable,
                str(source_root / "train_models_v2.py"),
                "--config",
                str(config_path),
            ],
            cwd=source_root,
        )
    _adaptive_parallel_batches(
        config,
        config_path,
        "build-calibration",
        expected_oof,
        workers,
        batch_size,
        peak_cost_budget,
    )
    _run_checked(
        [
            python_executable,
            str(source_root / "v2_pipeline.py"),
            "--config",
            str(config_path),
            "--stage",
            "consolidate-calibration",
        ],
        cwd=source_root,
    )
    _adaptive_parallel_batches(
        config,
        config_path,
        "evaluate",
        expected_oof,
        workers,
        batch_size,
        peak_cost_budget,
    )
    _run_checked(
        [
            python_executable,
            str(source_root / "v2_pipeline.py"),
            "--config",
            str(config_path),
            "--stage",
            "consolidate-evaluation",
        ],
        cwd=source_root,
    )
    _adaptive_parallel_batches(
        config,
        config_path,
        "infer",
        expected_test,
        workers,
        int(config.get("inference_batch_size", 1)),
        peak_cost_budget,
    )
    _run_checked(
        [
            python_executable,
            str(source_root / "v2_pipeline.py"),
            "--config",
            str(config_path),
            "--stage",
            "package",
        ],
        cwd=source_root,
    )
    print(
        json.dumps(
            {
                "event": "v2_pipeline_complete",
                "candidates": list(CANDIDATES),
            }
        ),
        flush=True,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
