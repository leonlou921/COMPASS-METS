#!/usr/bin/env python
"""Prepare BraTS 2025 MET batch 1 for nnU-Net v2.

The script can either consume already extracted BraTS case directories:

  python preprocessing/prepare_dataset501.py \
    --train-dir /path/to/BraTS-training \
    --valid-dir /path/to/BraTS-validation

or extract the Synapse ZIP files from the same folder by default:

  MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip
  MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip
  MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip

It creates:

  nnUNet_raw/Dataset501_BraTS2025MET/imagesTr
  nnUNet_raw/Dataset501_BraTS2025MET/labelsTr
  nnUNet_raw/Dataset501_BraTS2025MET/imagesTs
  nnUNet_raw/Dataset501_BraTS2025MET/dataset.json

Training is used as nnU-Net training data. Validation is used as nnU-Net test
images, because the challenge validation zip does not contain labels.

By default, dataset.json uses nnU-Net region-based labels matching the BraTS
evaluation regions: WT, TC, ET, and RC.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent
TRAIN_ZIP = "MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip"
VALID_ZIP = "MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip"
CORRECTED_LABELS_ZIP = "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip"

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
FLAT_LABELS = {
    "background": 0,
    "NETC": 1,
    "SNFH": 2,
    "ET": 3,
    "RC": 4,
}
REGION_LABELS = {
    "background": 0,
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": 3,
    "RC": 4,
}
# The output label map is reconstructed in this order from WT, TC, ET, RC.
# For BraTS-MET labels, WT-only voxels are SNFH=2 and TC-only voxels are NETC=1.
REGIONS_CLASS_ORDER = (2, 1, 3, 4)
NIFTI_DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    modalities: dict[str, Path]
    seg: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BraTS 2025 MET batch 1 and convert it to nnU-Net v2 raw format."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"ZIP folder. Default: {DEFAULT_ROOT}")
    parser.add_argument("--train-zip", default=TRAIN_ZIP, help=f"Training ZIP filename. Default: {TRAIN_ZIP}")
    parser.add_argument("--valid-zip", default=VALID_ZIP, help=f"Validation ZIP filename. Default: {VALID_ZIP}")
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Already extracted training root containing BraTS-MET-* case folders. Skips training ZIP extraction.",
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        help="Already extracted validation root containing unlabeled BraTS-MET-* case folders. Skips validation ZIP extraction.",
    )
    parser.add_argument(
        "--corrected-labels-dir",
        type=Path,
        help="Optional already extracted corrected-labels root. Overrides --corrected-labels-zip for labels that match training cases.",
    )
    parser.add_argument(
        "--corrected-labels-zip",
        default=CORRECTED_LABELS_ZIP,
        help=f"Corrected labels ZIP filename. Default: {CORRECTED_LABELS_ZIP}",
    )
    parser.add_argument("--extract-dir", type=Path, help="Extraction folder. Default: <root>/extracted")
    parser.add_argument("--nnunet-raw", type=Path, help="nnU-Net raw folder. Default: env nnUNet_raw or <root>/nnUNet_raw")
    parser.add_argument("--dataset-id", type=int, default=501, help="nnU-Net dataset ID. Default: 501")
    parser.add_argument("--dataset-name", default="BraTS2025MET", help="nnU-Net dataset name. Default: BraTS2025MET")
    parser.add_argument(
        "--modalities",
        nargs=4,
        default=MODALITIES,
        metavar=("CH0", "CH1", "CH2", "CH3"),
        help="Modality/channel order. Default: t1c t1n t2f t2w.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("regions", "flat"),
        default="regions",
        help="Write region-based or flat multiclass labels in dataset.json. Default: regions.",
    )
    parser.add_argument(
        "--label-remap",
        nargs="+",
        default=("6:4",),
        metavar="FROM:TO",
        help="Remap unexpected label values while writing labelsTr. Default: 6:4. Use 'none' to disable.",
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=[],
        help="Training case ID to exclude, for example BraTS-MET-01094-002. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-case-file",
        type=Path,
        help="Optional text file with one training case ID per line to exclude. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="Use hard links where possible, symlinks, or copy files. Default: hardlink.",
    )
    parser.add_argument("--force-extract", action="store_true", help="Re-extract ZIPs even if extraction folder exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing nnU-Net dataset folder.")
    parser.add_argument("--check-only", action="store_true", help="Only check ZIP presence/completion and list counts.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def require_complete_zip(path: Path) -> None:
    part_path = path.with_suffix(path.suffix + ".part")
    if path.exists():
        if not zipfile.is_zipfile(path):
            fail(f"Found {path}, but it is not a valid ZIP file.")
        print(f"OK ZIP: {path} ({format_bytes(path.stat().st_size)})")
        return

    if part_path.exists():
        fail(
            f"Training ZIP is not complete yet:\n"
            f"  partial: {part_path} ({format_bytes(part_path.stat().st_size)})\n"
            f"Resume with:\n"
            f"  python {DEFAULT_ROOT / 'download_brats2025_met_training_batch1.py'} --synid syn64919665"
        )

    fail(f"Missing ZIP: {path}")


def require_optional_zip(path: Path) -> bool:
    if not path.exists():
        print(f"Optional ZIP missing, skipping: {path}")
        return False
    if not zipfile.is_zipfile(path):
        fail(f"Found {path}, but it is not a valid ZIP file.")
    print(f"OK ZIP: {path} ({format_bytes(path.stat().st_size)})")
    return True


def safe_extract(zip_path: Path, extract_dir: Path, force: bool) -> None:
    stamp = extract_dir / f".{zip_path.stem}.done"
    if stamp.exists() and not force:
        print(f"Already extracted, skipping: {zip_path.name}")
        return

    print(f"Extracting {zip_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if extract_dir.resolve() not in target.parents and target != extract_dir.resolve():
                fail(f"Unsafe path in ZIP: {member.filename}")
            zf.extract(member, extract_dir)
    stamp.write_text("ok\n", encoding="utf-8")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        fail(f"{label} directory does not exist or is not a directory: {path}")


def iter_case_dirs(base: Path) -> list[Path]:
    candidates = []
    if base.name.startswith("BraTS-MET-") and base.is_dir():
        candidates.append(base)
    candidates.extend(item for item in base.rglob("BraTS-MET-*") if item.is_dir())
    return sorted(set(candidates))


def find_case_dirs(base: Path, require_seg: bool, split_name: str, modalities: tuple[str, ...]) -> list[CaseFiles]:
    cases: list[CaseFiles] = []
    incomplete: list[str] = []
    for case_dir in iter_case_dirs(base):
        case_id = case_dir.name
        modality_paths = {mod: case_dir / f"{case_id}-{mod}.nii.gz" for mod in modalities}
        missing_modalities = [mod for mod, path in modality_paths.items() if not path.exists()]
        if missing_modalities:
            incomplete.append(f"{case_id}: missing modalities {', '.join(missing_modalities)}")
            continue

        seg = case_dir / f"{case_id}-seg.nii.gz"
        if require_seg and not seg.exists():
            incomplete.append(f"{case_id}: missing label {seg.name}")
            continue
        cases.append(CaseFiles(case_id=case_id, modalities=modality_paths, seg=seg if seg.exists() else None))

    if incomplete:
        sample = "\n  ".join(incomplete[:10])
        extra = "" if len(incomplete) <= 10 else f"\n  ... and {len(incomplete) - 10} more"
        fail(f"Found incomplete {split_name} cases under {base}:\n  {sample}{extra}")
    return cases


def find_corrected_labels(base: Path, require_corrected_path_marker: bool = True) -> dict[str, Path]:
    corrected: dict[str, Path] = {}
    for seg in sorted(base.rglob("BraTS-MET-*-seg.nii.gz")):
        if require_corrected_path_marker and "corrected-labels" not in str(seg).lower():
            continue
        case_id = seg.name.removesuffix("-seg.nii.gz")
        corrected[case_id] = seg
    return corrected


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "symlink":
        dst.symlink_to(src)
        return
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def parse_label_remap(entries: list[str] | tuple[str, ...]) -> dict[int, int]:
    if len(entries) == 1 and entries[0].lower() in {"none", "off", "false", "disable", "disabled"}:
        return {}

    remap: dict[int, int] = {}
    for entry in entries:
        if ":" not in entry:
            fail(f"Invalid --label-remap entry '{entry}'. Expected FROM:TO, for example 6:4.")
        src, dst = entry.split(":", 1)
        try:
            remap[int(src)] = int(dst)
        except ValueError:
            fail(f"Invalid --label-remap entry '{entry}'. FROM and TO must be integers.")
    return remap


def read_excluded_cases(case_ids: list[str], case_file: Path | None) -> set[str]:
    excluded = {case_id.strip() for case_id in case_ids if case_id.strip()}
    if case_file is not None:
        if not case_file.is_file():
            fail(f"Exclude case file does not exist: {case_file}")
        for line in case_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                excluded.add(value)
    return excluded


def remap_nifti_gz_labels(src: Path, dst: Path, remap: dict[int, int], copy_mode: str) -> bool:
    if not remap:
        link_or_copy(src, dst, copy_mode)
        return False

    with gzip.open(src, "rb") as f:
        data = bytearray(f.read())

    if len(data) < 352:
        fail(f"Label file is too small to be a NIfTI image: {src}")

    little_endian_header = int.from_bytes(data[0:4], byteorder="little", signed=True)
    big_endian_header = int.from_bytes(data[0:4], byteorder="big", signed=True)
    if little_endian_header == 348:
        endian = "<"
    elif big_endian_header == 348:
        endian = ">"
    else:
        fail(f"Unsupported NIfTI header in label file: {src}")

    datatype = int(np.frombuffer(data, dtype=np.dtype(endian + "i2"), count=1, offset=70)[0])
    bitpix = int(np.frombuffer(data, dtype=np.dtype(endian + "i2"), count=1, offset=72)[0])
    vox_offset = int(float(np.frombuffer(data, dtype=np.dtype(endian + "f4"), count=1, offset=108)[0]))
    if datatype not in NIFTI_DTYPES:
        fail(f"Unsupported NIfTI datatype {datatype} in label file: {src}")
    if vox_offset <= 0 or vox_offset >= len(data):
        fail(f"Invalid NIfTI vox_offset {vox_offset} in label file: {src}")

    dtype = np.dtype(endian + NIFTI_DTYPES[datatype])
    if dtype.itemsize * 8 != bitpix:
        fail(f"NIfTI datatype/bitpix mismatch in label file: {src}")

    arr = np.frombuffer(data, dtype=dtype, offset=vox_offset)
    needs_remap = False
    for from_label, to_label in remap.items():
        mask = arr == from_label
        if bool(mask.any()):
            arr[mask] = to_label
            needs_remap = True

    if not needs_remap:
        link_or_copy(src, dst, copy_mode)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with gzip.open(dst, "wb", compresslevel=6) as f:
        f.write(data)
    return True


def reset_dataset_dir(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            fail(f"Output dataset already exists: {dataset_dir}\nUse --overwrite to rebuild it.")
        shutil.rmtree(dataset_dir)
    (dataset_dir / "imagesTr").mkdir(parents=True)
    (dataset_dir / "labelsTr").mkdir()
    (dataset_dir / "imagesTs").mkdir()


def convert_cases(
    train_cases: list[CaseFiles],
    valid_cases: list[CaseFiles],
    corrected_labels: dict[str, Path],
    dataset_dir: Path,
    copy_mode: str,
    modalities: tuple[str, ...],
    label_remap: dict[int, int],
) -> None:
    remapped_count = 0
    for case in train_cases:
        for channel, modality in enumerate(modalities):
            link_or_copy(case.modalities[modality], dataset_dir / "imagesTr" / f"{case.case_id}_{channel:04d}.nii.gz", copy_mode)
        label_src = corrected_labels.get(case.case_id, case.seg)
        if label_src is None:
            fail(f"Missing training label for {case.case_id}")
        remapped = remap_nifti_gz_labels(label_src, dataset_dir / "labelsTr" / f"{case.case_id}.nii.gz", label_remap, copy_mode)
        remapped_count += int(remapped)

    for case in valid_cases:
        for channel, modality in enumerate(modalities):
            link_or_copy(case.modalities[modality], dataset_dir / "imagesTs" / f"{case.case_id}_{channel:04d}.nii.gz", copy_mode)

    if label_remap:
        print(f"Labels remapped while writing labelsTr: {remapped_count} cases")


def json_ready_labels(labels: dict[str, int | tuple[int, ...]]) -> dict[str, int | list[int]]:
    return {name: list(value) if isinstance(value, tuple) else value for name, value in labels.items()}


def write_dataset_json(dataset_dir: Path, train_count: int, modalities: tuple[str, ...], label_mode: str) -> None:
    labels = REGION_LABELS if label_mode == "regions" else FLAT_LABELS
    data = {
        "channel_names": {str(i): name for i, name in enumerate(modalities)},
        "labels": json_ready_labels(labels),
        "numTraining": train_count,
        "file_ending": ".nii.gz",
    }
    if label_mode == "regions":
        data["regions_class_order"] = list(REGIONS_CLASS_ORDER)
    (dataset_dir / "dataset.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    modalities = tuple(args.modalities)
    label_remap = parse_label_remap(args.label_remap)
    excluded_cases = read_excluded_cases(args.exclude_case, args.exclude_case_file)
    root = args.root.resolve()
    extract_dir = (args.extract_dir or root / "extracted").resolve()
    nnunet_raw = (args.nnunet_raw or Path(os.environ.get("nnUNet_raw", root / "nnUNet_raw"))).resolve()
    dataset_dir = nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    train_zip = root / args.train_zip
    valid_zip = root / args.valid_zip
    corrected_zip = root / args.corrected_labels_zip

    has_direct_input = args.train_dir is not None or args.valid_dir is not None
    if has_direct_input and (args.train_dir is None or args.valid_dir is None):
        fail("--train-dir and --valid-dir must be provided together.")

    corrected_labels: dict[str, Path] = {}
    if has_direct_input:
        train_base = args.train_dir.resolve()
        valid_base = args.valid_dir.resolve()
        require_dir(train_base, "Training")
        require_dir(valid_base, "Validation")

        if args.corrected_labels_dir is not None:
            corrected_base = args.corrected_labels_dir.resolve()
            require_dir(corrected_base, "Corrected labels")
            corrected_labels = find_corrected_labels(corrected_base, require_corrected_path_marker=False)
        elif corrected_zip.exists():
            has_corrected = require_optional_zip(corrected_zip)
            if has_corrected and not args.check_only:
                safe_extract(corrected_zip, extract_dir, args.force_extract)
                corrected_labels = find_corrected_labels(extract_dir)
    else:
        require_complete_zip(train_zip)
        require_complete_zip(valid_zip)
        has_corrected = require_optional_zip(corrected_zip)

        if args.check_only:
            print("ZIP check passed.")
            return

        safe_extract(train_zip, extract_dir, args.force_extract)
        safe_extract(valid_zip, extract_dir, args.force_extract)
        if has_corrected:
            safe_extract(corrected_zip, extract_dir, args.force_extract)

        train_base = extract_dir
        valid_base = extract_dir
        corrected_labels = find_corrected_labels(extract_dir) if has_corrected else {}

    train_cases = find_case_dirs(train_base, require_seg=True, split_name="training", modalities=modalities)
    valid_cases = [
        case
        for case in find_case_dirs(valid_base, require_seg=False, split_name="validation", modalities=modalities)
        if case.seg is None
    ]
    if excluded_cases:
        before_count = len(train_cases)
        available_train_case_ids = {case.case_id for case in train_cases}
        train_cases = [case for case in train_cases if case.case_id not in excluded_cases]
        missing_exclusions = sorted(excluded_cases - available_train_case_ids)
        if missing_exclusions:
            print(f"Warning: excluded case IDs not found in training data: {', '.join(missing_exclusions)}")
        print(f"Excluded training cases: {before_count - len(train_cases)}")

    if not train_cases:
        fail(f"No labeled training cases found under {train_base}")
    if not valid_cases:
        fail(f"No unlabeled validation/test cases found under {valid_base}")

    if args.check_only:
        print(f"Training cases: {len(train_cases)}")
        print(f"Validation-as-test cases: {len(valid_cases)}")
        if corrected_labels:
            print(f"Corrected labels found: {len(corrected_labels)}")
        print("Input check passed.")
        return

    print(f"Training cases: {len(train_cases)}")
    print(f"Validation-as-test cases: {len(valid_cases)}")
    if corrected_labels:
        used = sum(1 for case in train_cases if case.case_id in corrected_labels)
        print(f"Corrected labels found: {len(corrected_labels)}; used for training cases: {used}")

    reset_dataset_dir(dataset_dir, args.overwrite)
    convert_cases(train_cases, valid_cases, corrected_labels, dataset_dir, args.copy_mode, modalities, label_remap)
    write_dataset_json(dataset_dir, len(train_cases), modalities, args.label_mode)

    print(f"Done: {dataset_dir}")
    print(f"Set nnUNet_raw to: {nnunet_raw}")


if __name__ == "__main__":
    main()
