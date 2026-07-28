from __future__ import annotations

from docker.build_asset_inventory import parse_args


def test_asset_inventory_cli_requires_all_frozen_model_roles() -> None:
    args = parse_args(
        [
            "--xl-root",
            "/path/to/xl",
            "--m-root",
            "/path/to/m",
            "--ft-root",
            "/path/to/ft",
            "--lcv1-root",
            "/path/to/lcv1",
            "--lcv2-root",
            "/path/to/lcv2",
            "--source-root",
            "/path/to/source",
            "--output",
            "/path/to/inventory.json",
        ]
    )
    assert args.xl_root.as_posix() == "/path/to/xl"
    assert args.m_root.as_posix() == "/path/to/m"
    assert args.ft_root.as_posix() == "/path/to/ft"
    assert args.output.name == "inventory.json"
