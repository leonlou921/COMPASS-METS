from __future__ import annotations

import pytest

from n03_docker.asset_inventory import (
    EXPECTED_FINAL_LEARNED_HASHES,
    validate_source_inventory,
)


def test_inventory_rejects_wrong_frozen_utility_hash() -> None:
    learned = {
        role: {
            "path": f"/private/{role}",
            "role": f"{role}_final",
            "size_bytes": 1,
            "sha256": digest,
        }
        for role, digest in EXPECTED_FINAL_LEARNED_HASHES.items()
    }
    learned.update(
        {
            role: {
                "path": f"/private/{role}",
                "role": f"{role}_final",
                "size_bytes": 1,
                "sha256": "1" * 64,
            }
            for role in ("lcv1_case", "lcv2_component")
        }
    )
    learned["utility_v4_existence"]["sha256"] = "0" * 64
    inventory = {
        "candidate": "N03_FINAL_UTILITY_V4",
        "models": {},
        "learned_models": learned,
        "assets": [],
        "frozen_asset_hashes_verified": True,
    }

    with pytest.raises(ValueError, match="utility_v4_existence.*sha256"):
        validate_source_inventory(
            inventory,
            require_model_assets=False,
            require_frozen_hashes=True,
        )


def test_expected_final_hashes_cover_all_new_gate_assets() -> None:
    assert EXPECTED_FINAL_LEARNED_HASHES == {
        "rgv3_et": "5fff6d9c7ef31bf4ce33bad211abe017fa1e0235d4e7f78d264c50c2c2a9fac1",
        "utility_v4_existence": "8a8c8f02ed652861b949b9a47aedaa8faff8e9904e4d9645f48a1a743ec0e3e0",
        "utility_v4_geometry": "bfd2d2be7e9349d4cde41f1e4682f87b9f0108216d69597cb80d0a6c9991f87e",
        "utility_v4_feature_names": "87b6523508688f52ad1cee6d600d7d353991f0d1451ec29ce0aed500ee07699d",
    }
    assert all(len(value) == 64 for value in EXPECTED_FINAL_LEARNED_HASHES.values())
