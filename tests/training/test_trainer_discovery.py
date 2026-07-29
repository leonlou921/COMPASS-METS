from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = (
    ROOT
    / "third_party"
    / "nnUNet"
    / "nnunetv2"
    / "training"
    / "nnUNetTrainer"
)
VARIANT_ROOT = TRAINER_ROOT / "variants" / "brats_mets"
EXPECTED = {
    "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT",
    "nnUNetTrainer_ResEncM_SmallLesionOS",
    "nnUNetTrainer_ResEncM_ComponentSmallLesionOS",
}


def _class_bases(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result[node.name] = {
                base.id for base in node.bases if isinstance(base, ast.Name)
            }
    return result


def test_retained_trainers_have_one_recursive_discovery_source() -> None:
    assert (VARIANT_ROOT / "__init__.py").is_file()
    discovered: dict[str, list[Path]] = {name: [] for name in EXPECTED}
    for path in TRAINER_ROOT.rglob("*.py"):
        for name in EXPECTED.intersection(_class_bases(path)):
            discovered[name].append(path)
    assert set(discovered) == EXPECTED
    for name, paths in discovered.items():
        assert paths == [VARIANT_ROOT / f"{name}.py"]
        assert "nnUNetTrainer" in _class_bases(paths[0])[name]


def test_duplicate_and_synthetic_trainers_are_absent() -> None:
    duplicate_root = ROOT / "training" / "nnunet_trainers"
    assert not list(duplicate_root.glob("*.py"))
    production_roots = [TRAINER_ROOT, ROOT / "brats_mets"]
    offending = [
        path
        for production_root in production_roots
        for path in production_root.rglob("*SyntheticTumorAug*")
    ]
    assert offending == []
