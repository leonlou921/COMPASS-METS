import os
from os.path import join
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_pickle

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


ClassKey = Union[int, Tuple[int, ...]]


class nnUNetTrainer_ResEncM_SmallLesionOS(nnUNetTrainer):
    """ResEncM trainer variant with foreground-centered, small-lesion-biased sampling."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = float(
            os.environ.get("SMALL_LESION_OVERSAMPLE_FG_PERCENT", "0.85")
        )
        self.small_lesion_weight_power = float(os.environ.get("SMALL_LESION_WEIGHT_POWER", "0.5"))
        self.small_lesion_max_weight = float(os.environ.get("SMALL_LESION_MAX_WEIGHT", "8.0"))
        self.small_lesion_uniform_blend = float(os.environ.get("SMALL_LESION_UNIFORM_BLEND", "0.25"))
        self.small_lesion_ref_percentile = float(os.environ.get("SMALL_LESION_REF_PERCENTILE", "50"))
        self.small_lesion_min_voxels = float(os.environ.get("SMALL_LESION_MIN_VOXELS", "1"))

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        probabilities, summary = self._build_small_lesion_sampling_probabilities(
            dataset_tr.identifiers, dataset_tr.source_folder
        )
        self._safe_log(
            "SmallLesionOS: "
            f"foreground_crop={self.oversample_foreground_percent:.3f}, "
            f"weight_power={self.small_lesion_weight_power:.3f}, "
            f"max_weight={self.small_lesion_max_weight:.3f}, "
            f"uniform_blend={self.small_lesion_uniform_blend:.3f}, "
            f"{summary}"
        )

        dl_tr = nnUNetDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=probabilities,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def _build_small_lesion_sampling_probabilities(self, identifiers: Iterable[str], source_folder: str):
        identifiers = list(identifiers)
        if len(identifiers) == 0:
            return None, "no training identifiers"

        lesion_counts = self._collect_lesion_counts(source_folder, identifiers)
        positive_counts = np.asarray([c for c in lesion_counts if c > 0], dtype=np.float64)
        if positive_counts.size == 0:
            return None, "no positive foreground counts in class_locations"

        ref_count = float(np.percentile(positive_counts, self.small_lesion_ref_percentile))
        ref_count = max(ref_count, self.small_lesion_min_voxels)

        raw_weights: List[float] = []
        for count in lesion_counts:
            if count <= 0:
                raw_weights.append(0.1)
                continue
            effective_count = max(float(count), self.small_lesion_min_voxels)
            weight = (ref_count / effective_count) ** self.small_lesion_weight_power
            raw_weights.append(float(np.clip(weight, 1.0 / self.small_lesion_max_weight, self.small_lesion_max_weight)))

        raw = np.asarray(raw_weights, dtype=np.float64)
        raw = raw / raw.sum()
        uniform = np.ones_like(raw) / raw.size
        blend = float(np.clip(self.small_lesion_uniform_blend, 0.0, 1.0))
        probabilities = (1.0 - blend) * raw + blend * uniform
        probabilities = probabilities / probabilities.sum()

        nonzero = int(positive_counts.size)
        return probabilities, (
            f"cases={len(identifiers)}, positive_cases={nonzero}, "
            f"ref_count={ref_count:.1f}, "
            f"count_q10/q50/q90={np.percentile(positive_counts, [10, 50, 90]).round(1).tolist()}, "
            f"prob_min/max={probabilities.min():.6f}/{probabilities.max():.6f}"
        )

    @staticmethod
    def _collect_lesion_counts(source_folder: str, identifiers: Iterable[str]) -> List[int]:
        counts: List[int] = []
        for identifier in identifiers:
            properties = load_pickle(join(source_folder, identifier + ".pkl"))
            class_locations: Dict[ClassKey, np.ndarray] = properties.get("class_locations", {})
            positive_region_sizes = []
            for key, locations in class_locations.items():
                if key == -1:
                    continue
                n_locations = len(locations)
                if n_locations > 0:
                    positive_region_sizes.append(n_locations)
            counts.append(min(positive_region_sizes) if positive_region_sizes else 0)
        return counts

    def _safe_log(self, message: str) -> None:
        try:
            self.print_to_log_file(message)
        except Exception:
            print(message)
