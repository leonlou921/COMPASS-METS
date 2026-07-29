"""Build an auditable image release manifest from external verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


CANDIDATE_ID = "N03_FINAL_UTILITY_V4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(
    report: dict[str, Any],
    archive_path: Path,
    *,
    accept_recorded_difference: bool = False,
) -> dict[str, Any]:
    """Validate release evidence and preserve any accepted voxel difference."""
    archive_path = Path(archive_path)
    if report.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("verification candidate is not N03_FINAL_UTILITY_V4")
    if report.get("case_count") != 179:
        raise ValueError("release requires exactly 179 reference cases")
    if report.get("candidate_case_count") != 179:
        raise ValueError("release requires exactly 179 candidate cases")
    if report.get("candidate_case_set_equal") is not True:
        raise ValueError("release candidate case set differs from reference")
    if report.get("structural_passed") is not True:
        raise ValueError("release structural verification did not pass")

    exact = report.get("passed") is True
    if not exact and not accept_recorded_difference:
        raise ValueError("exact frozen equivalence did not pass")
    if not archive_path.is_file():
        raise FileNotFoundError(f"image archive is absent: {archive_path}")

    acceptance_mode = (
        "exact_frozen_equivalence"
        if exact
        else "explicit_recorded_difference"
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "release_accepted": True,
        "acceptance_mode": acceptance_mode,
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "verification_mode": report.get("verification_mode"),
        "frozen_equivalence_passed": exact,
        "structural_verification_passed": True,
        "case_count": 179,
        "different_case_count": int(
            report.get("different_case_count") or 0
        ),
        "different_voxels": int(report.get("different_voxels") or 0),
    }
