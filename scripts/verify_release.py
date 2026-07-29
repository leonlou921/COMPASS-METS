#!/usr/bin/env python3
"""Audit the public COMPASS-METS source tree without private artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".gitignore",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__"}
IGNORED_GENERATED_ROOTS = {"artifacts", "logs", "output", "outputs", "work"}
PRIVATE_BINARY_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".npz",
    ".npy",
    ".pth",
    ".pt",
    ".pkl",
    ".joblib",
    ".zip",
    ".tar",
)
PUBLIC_THIRD_PARTY_BINARY_PREFIXES = (
    Path("third_party/nnUNet/nnunetv2/tests/example_data"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:Users|brats_challenge|MICCAI)\\", re.I),
    re.compile(r"/data/" + r"coding/challenge(?:/|\b)", re.I),
    re.compile(r"\broot@[A-Za-z0-9.-]+", re.I),
    re.compile(r"\b(?:deepln|funhpc)\.com\b", re.I),
)
CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)"
    r"\s*[:=]\s*[\"']?[^\"'\s]{8,}"
)
REQUIRED_PATHS = (
    "compass_mets/__init__.py",
    "compass_mets/data/prepare_dataset501.py",
    "compass_mets/training/run_nnunet.py",
    "compass_mets/learned_gates/lcv1/run_pipeline.py",
    "compass_mets/learned_gates/rgv3.py",
    "compass_mets/learned_gates/train_utility_v4.py",
    "compass_mets/inference/predict.py",
    "compass_mets/postprocessing/final.py",
    "compass_mets/pipeline.py",
    "configs/fusion/xf12.json",
    "configs/final/n03_utility_v4.json",
    "scripts/01_prepare_dataset.sh",
    "scripts/03_train_resencm_5fold.sh",
    "scripts/04_train_resencxl_5fold.sh",
    "scripts/07_train_learned_gates.sh",
    "scripts/08_predict_test_probabilities.sh",
    "scripts/09_build_n03_utility_v4.sh",
    "third_party/nnUNet/nnunetv2/__init__.py",
)


def _is_ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts)
        or (relative.parts and relative.parts[0] in IGNORED_GENERATED_ROOTS)
    )


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE"}


def _validate_final_config(root: Path) -> bool:
    path = root / "configs" / "final" / "n03_utility_v4.json"
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding="utf-8"))
    utility = config.get("utility_v4", {})
    return (
        config.get("candidate_id") == "N03_FINAL_UTILITY_V4"
        and config.get("baseline_candidate_id")
        == "N03_XF12_LCv3_ET_parent_supported"
        and config.get("anchor_id")
        == "XF12_XLM_structured_probability_V2_strict"
        and config.get("proposal_threshold") == 0.25
        and config.get("allowed_add_regions") == ["ET"]
        and config.get("allowed_delete_regions") == []
        and config.get("minimum_parent_model_support") == 2
        and config.get("parent_models") == ["XL", "M", "FT"]
        and config.get("preserve_anchor") is True
        and config.get("preserve_rc_priority") is True
        and config.get("global_v2_rerun") is False
        and utility.get("candidate_scope")
        == "disconnected_et_from_lcv2_structured_union"
        and utility.get("rgv3_et_cutoff") == 0.7702616034384248
        and utility.get("accept_threshold") == 0.75
        and utility.get("reject_threshold") == 0.5
        and utility.get("operation") == "et_add_only"
    )


def inspect_release(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    findings: list[dict[str, Any]] = []
    missing_required = [
        relative for relative in REQUIRED_PATHS if not (root / relative).is_file()
    ]
    file_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored(path, root):
            continue
        relative = path.relative_to(root)
        file_count += 1
        lower_name = relative.as_posix().lower()
        if lower_name.endswith(PRIVATE_BINARY_SUFFIXES):
            if not any(
                relative.is_relative_to(prefix)
                for prefix in PUBLIC_THIRD_PARTY_BINARY_PREFIXES
            ):
                findings.append(
                    {"kind": "private_binary", "path": str(relative), "line": None}
                )
            continue
        if not _is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if CREDENTIAL_PATTERN.search(line):
                findings.append(
                    {"kind": "credential", "path": str(relative), "line": number}
                )
            if any(pattern.search(line) for pattern in PRIVATE_PATH_PATTERNS):
                findings.append(
                    {
                        "kind": "private_path_or_host",
                        "path": str(relative),
                        "line": number,
                    }
                )
    config_valid = _validate_final_config(root)
    return {
        "root": str(root),
        "file_count": file_count,
        "required_paths_present": not missing_required,
        "missing_required_paths": missing_required,
        "final_config_valid": config_valid,
        "findings": findings,
        "passed": config_valid and not missing_required and not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_release(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
