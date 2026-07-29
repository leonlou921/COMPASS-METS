"""Public repository boundary checks shared by tests and release tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


FORBIDDEN_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".npz",
    ".pth",
    ".pt",
    ".pkl",
    ".joblib",
    ".zip",
    ".tar",
)

FORBIDDEN_PATH_PARTS = {
    ".dockerignore",
    "docker",
    "submissions",
}


def find_forbidden_paths(paths: Iterable[str | Path]) -> list[str]:
    """Return normalized public-tree paths that violate the source-only boundary."""

    violations: list[str] = []
    for raw_path in paths:
        normalized = Path(raw_path).as_posix()
        lower = normalized.lower()
        parts = {part.lower() for part in Path(normalized).parts}
        if parts.intersection(FORBIDDEN_PATH_PARTS) or lower.endswith(
            FORBIDDEN_SUFFIXES
        ):
            violations.append(normalized)
    return sorted(set(violations))
