#!/usr/bin/env python3
"""Check the public release for frozen N03 metadata and accidental secrets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".dockerignore",
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
PUBLIC_THIRD_PARTY_BINARY_PREFIXES = (
    Path("third_party/nnUNet/nnunetv2/tests/example_data"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:Users|brats_challenge|MICCAI)\\", re.I),
    re.compile(r"/data/" r"coding/challenge(?:/|\b)"),
)
CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)"
    r"\s*[:=]\s*[\"']?[^\"'\s]{8,}"
)


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        "Dockerfile",
        "LICENSE",
    }


def _validate_n03_config(root: Path) -> bool:
    path = root / "configs" / "n03" / "final.json"
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding="utf-8"))
    utility = config.get("utility_v4", {})
    return (
        config.get("candidate_id")
        == "N03_FINAL_UTILITY_V4"
        and config.get("baseline_candidate_id")
        == "N03_XF12_LCv3_ET_parent_supported"
        and config.get("proposal_threshold") == 0.25
        and config.get("et_component_cutoff") == 0.5497123599
        and config.get("minimum_parent_model_support") == 2
        and config.get("parent_models") == ["XL", "M", "FT"]
        and config.get("policy") == "add_only_et_parent_supported"
        and config.get("preserve_anchor") is True
        and config.get("rerun_anchor_postprocess_after_addition") is False
        and utility.get("candidate_scope")
        == "disconnected_et_from_lcv2_structured_union"
        and utility.get("rgv3_et_cutoff") == 0.7702616034384248
        and utility.get("accept_all_scores_gte") == 0.75
        and utility.get("reject_any_utility_score_lt") == 0.5
        and utility.get("operation") == "et_add_only"
        and utility.get("preserve_rc_priority") is True
    )


def inspect_release(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    findings: list[dict[str, Any]] = []
    file_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        file_count += 1
        if path.suffix.lower() in {".pth", ".pt", ".npz", ".nii", ".gz", ".tar"}:
            relative = path.relative_to(root)
            if not any(
                relative.is_relative_to(prefix)
                for prefix in PUBLIC_THIRD_PARTY_BINARY_PREFIXES
            ):
                findings.append(
                    {
                        "kind": "private_binary",
                        "path": str(relative),
                        "line": None,
                    }
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
                    {
                        "kind": "credential",
                        "path": str(path.relative_to(root)),
                        "line": number,
                    }
                )
            if any(pattern.search(line) for pattern in PRIVATE_PATH_PATTERNS):
                findings.append(
                    {
                        "kind": "private_absolute_path",
                        "path": str(path.relative_to(root)),
                        "line": number,
                    }
                )
    config_valid = _validate_n03_config(root)
    return {
        "root": str(root),
        "file_count": file_count,
        "n03_config_valid": config_valid,
        "findings": findings,
        "passed": config_valid and not findings,
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
