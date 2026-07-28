"""Resumable one-case-at-a-time feature-table builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import pandas as pd

from component_gate import load_probabilities
from features import extract_case_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(output_dir: Path, case_id: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{case_id}.case.parquet",
        output_dir / f"{case_id}.components.parquet",
        output_dir / f"{case_id}.complete.json",
    )


def checkpoint_is_complete(output_dir: str | Path, case_id: str) -> bool:
    output = Path(output_dir)
    case_path, component_path, status_path = _checkpoint_paths(output, case_id)
    if not (case_path.is_file() and component_path.is_file() and status_path.is_file()):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        status.get("complete")
        and status.get("case_id") == case_id
        and status.get("case_sha256") == _sha256(case_path)
        and status.get("components_sha256") == _sha256(component_path)
    )


def write_case_checkpoint(
    output_dir: str | Path,
    case_id: str,
    case_rows: list[dict],
    component_rows: list[dict],
) -> dict[str, str | int | bool]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_is_complete(output, case_id):
        raise FileExistsError(f"complete checkpoint already exists for {case_id}")
    case_path, component_path, status_path = _checkpoint_paths(output, case_id)
    token = f".tmp.{os.getpid()}"
    case_tmp = output / f"{case_path.name}{token}"
    component_tmp = output / f"{component_path.name}{token}"
    status_tmp = output / f"{status_path.name}{token}"
    pd.DataFrame(case_rows).to_parquet(case_tmp, index=False)
    pd.DataFrame(component_rows).to_parquet(component_tmp, index=False)
    os.replace(case_tmp, case_path)
    os.replace(component_tmp, component_path)
    status = {
        "complete": True,
        "case_id": case_id,
        "case_rows": len(case_rows),
        "component_rows": len(component_rows),
        "case_sha256": _sha256(case_path),
        "components_sha256": _sha256(component_path),
    }
    status_tmp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(status_tmp, status_path)
    return status


def _fold_cases(root: Path, fold: int) -> list[str]:
    validation = root / f"fold_{fold}" / "validation"
    return sorted(path.stem for path in validation.glob("*.npz"))


def _label_path(labels_root: Path, case_id: str) -> Path:
    path = labels_root / f"{case_id}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_oof_features(
    config: dict,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    folds: set[int] | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> int:
    roots = {name: Path(config["oof_roots"][name]) for name in ("XL", "M", "FT")}
    labels_root = Path(config["labels_root"])
    output_dir = Path(config["output_root"]) / "features" / "per_case"
    processed = 0
    for fold in range(5):
        if folds is not None and fold not in folds:
            continue
        xl_cases = _fold_cases(roots["XL"], fold)
        for name in ("M", "FT"):
            if _fold_cases(roots[name], fold) != xl_cases:
                raise RuntimeError(f"fold {fold} case mismatch between XL and {name}")
        for case_id in xl_cases:
            shard = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big") % shard_count
            if shard != shard_index:
                continue
            if case_ids is not None and case_id not in case_ids:
                continue
            if checkpoint_is_complete(output_dir, case_id):
                continue
            arrays = {
                name: load_probabilities(roots[name] / f"fold_{fold}" / "validation" / f"{case_id}.npz")
                for name in ("XL", "M", "FT")
            }
            gt = nib.load(_label_path(labels_root, case_id)).get_fdata(dtype="float32").transpose(2, 1, 0)
            case_rows, component_rows = extract_case_features(
                case_id, fold, arrays["XL"], arrays["M"], arrays["FT"], gt
            )
            status = write_case_checkpoint(output_dir, case_id, case_rows, component_rows)
            processed += 1
            print(
                json.dumps(
                    {"event": "case_complete", "fold": fold, "processed": processed, **status},
                    sort_keys=True,
                ),
                flush=True,
            )
            if max_cases is not None and processed >= max_cases:
                return processed
    return processed


def consolidate_checkpoints(output_root: str | Path) -> tuple[Path, Path]:
    root = Path(output_root)
    per_case = root / "features" / "per_case"
    statuses = sorted(per_case.glob("*.complete.json"))
    case_frames = []
    component_frames = []
    for status_path in statuses:
        case_id = status_path.name.removesuffix(".complete.json")
        if not checkpoint_is_complete(per_case, case_id):
            raise RuntimeError(f"invalid checkpoint: {case_id}")
        case_path, component_path, _ = _checkpoint_paths(per_case, case_id)
        case_frames.append(pd.read_parquet(case_path))
        component_frames.append(pd.read_parquet(component_path))
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    case_out = feature_dir / "case_features.parquet"
    component_out = feature_dir / "component_features.parquet"
    pd.concat(case_frames, ignore_index=True).to_parquet(case_out, index=False)
    pd.concat(component_frames, ignore_index=True).to_parquet(component_out, index=False)
    return case_out, component_out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--fold", action="append", type=int, dest="folds")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.consolidate:
        paths = consolidate_checkpoints(config["output_root"])
        print(json.dumps({"event": "consolidated", "paths": [str(x) for x in paths]}))
        return 0
    count = build_oof_features(
        config,
        max_cases=args.max_cases,
        case_ids=set(args.case_ids) if args.case_ids else None,
        folds=set(args.folds) if args.folds else None,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(json.dumps({"event": "feature_build_complete", "processed": count}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
