import os
import pickle
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from scipy.ndimage import find_objects, generate_binary_structure, label as cc_label
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class ComponentSmallLesionDataLoader(nnUNetDataLoader):
    """Foreground oversampling that can center patches on small connected components."""

    def __init__(
        self,
        *args,
        small_lesion_centroids_by_case: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
        small_lesion_patch_prob: float = 0.75,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.small_lesion_centroids_by_case = small_lesion_centroids_by_case or {}
        self.small_lesion_patch_prob = float(np.clip(small_lesion_patch_prob, 0.0, 1.0))

    def _bbox_from_center(self, data_shape: np.ndarray, center_zyx: np.ndarray):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]
        bbox_lbs = [
            int(min(max(lbs[i], int(center_zyx[i]) - self.patch_size[i] // 2), ubs[i]))
            for i in range(dim)
        ]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def _get_bbox_for_case(self, data_shape, force_fg: bool, class_locations, case_id: str):
        if force_fg and np.random.uniform() < self.small_lesion_patch_prob:
            centroids = self.small_lesion_centroids_by_case.get(case_id)
            if centroids:
                center = np.asarray(centroids[np.random.choice(len(centroids))], dtype=np.int64)
                return self._bbox_from_center(np.asarray(data_shape), center)

        return self.get_bbox(data_shape, force_fg, class_locations)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, case_id in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(case_id)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self._get_bbox_for_case(
                shape, force_fg, properties["class_locations"], case_id
            )
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {"data": data_all, "target": seg_all, "keys": selected_keys}

        return {"data": data_all, "target": seg_all, "keys": selected_keys}


class nnUNetTrainer_ResEncM_ComponentSmallLesionOS(nnUNetTrainer):
    """ResEncM trainer with default nnU-Net loss plus component-level small lesion oversampling."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = float(
            os.environ.get("SMALL_LESION_INITIAL_LR", os.environ.get("FT_INITIAL_LR", "0.001"))
        )
        self.num_epochs = int(
            os.environ.get("SMALL_LESION_NUM_EPOCHS", os.environ.get("FT_NUM_EPOCHS", "400"))
        )
        self.oversample_foreground_percent = float(
            os.environ.get("SMALL_LESION_OVERSAMPLE_FG_PERCENT", "0.85")
        )
        self.small_lesion_patch_prob = float(os.environ.get("SMALL_LESION_PATCH_PROB", "0.75"))
        self.small_lesion_component_min_mm3 = float(
            os.environ.get("SMALL_LESION_COMPONENT_MIN_MM3", "0")
        )
        self.small_lesion_component_max_mm3 = float(
            os.environ.get("SMALL_LESION_COMPONENT_MAX_MM3", "275")
        )
        self.small_lesion_case_weight_per_component = float(
            os.environ.get("SMALL_LESION_CASE_WEIGHT_PER_COMPONENT", "2.0")
        )
        self.small_lesion_max_case_weight = float(
            os.environ.get("SMALL_LESION_MAX_CASE_WEIGHT", "8.0")
        )
        self.small_lesion_uniform_blend = float(
            os.environ.get("SMALL_LESION_UNIFORM_BLEND", "0.25")
        )
        self.small_lesion_component_mode = os.environ.get(
            "SMALL_LESION_COMPONENT_MODE", "union"
        ).lower()

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
        (
            centroids_by_case,
            sampling_probabilities,
            component_summary,
        ) = self._build_component_small_lesion_sampling(dataset_tr)

        self._safe_log(
            "ComponentSmallLesionOS: "
            f"num_epochs={self.num_epochs}, fg_crop={self.oversample_foreground_percent:.3f}, "
            f"small_patch_prob={self.small_lesion_patch_prob:.3f}, "
            f"component_mm3=[{self.small_lesion_component_min_mm3:.1f}, "
            f"{self.small_lesion_component_max_mm3:.1f}], "
            f"mode={self.small_lesion_component_mode}, {component_summary}"
        )

        dl_tr = ComponentSmallLesionDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            small_lesion_centroids_by_case=centroids_by_case,
            small_lesion_patch_prob=self.small_lesion_patch_prob,
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

    def _build_component_small_lesion_sampling(self, dataset_tr):
        identifiers = list(dataset_tr.identifiers)
        if len(identifiers) == 0:
            return {}, None, "no training identifiers"

        cache_file = self._component_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            centroids_by_case = {
                k: [tuple(map(int, c)) for c in v] for k, v in cache["centroids_by_case"].items()
            }
            counts = {k: int(v) for k, v in cache["counts"].items()}
            summary_prefix = "loaded_cache"
        else:
            centroids_by_case, counts = self._compute_component_centroids(dataset_tr, identifiers)
            maybe_mkdir_p(os.path.dirname(cache_file))
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {"centroids_by_case": centroids_by_case, "counts": counts},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            summary_prefix = "built_cache"

        count_array = np.asarray([counts.get(i, 0) for i in identifiers], dtype=np.float64)
        raw_weights = 1.0 + np.minimum(
            count_array * self.small_lesion_case_weight_per_component,
            max(self.small_lesion_max_case_weight - 1.0, 0.0),
        )
        raw_weights = np.maximum(raw_weights, 1e-6)
        probabilities = raw_weights / raw_weights.sum()
        uniform = np.ones_like(probabilities) / probabilities.size
        blend = float(np.clip(self.small_lesion_uniform_blend, 0.0, 1.0))
        probabilities = (1.0 - blend) * probabilities + blend * uniform
        probabilities = probabilities / probabilities.sum()

        positive_counts = count_array[count_array > 0]
        if positive_counts.size:
            count_summary = (
                f"cases={len(identifiers)}, cases_with_small={positive_counts.size}, "
                f"small_components={int(count_array.sum())}, "
                f"components_q10/q50/q90="
                f"{np.percentile(positive_counts, [10, 50, 90]).round(1).tolist()}, "
                f"prob_min/max={probabilities.min():.6f}/{probabilities.max():.6f}"
            )
        else:
            count_summary = f"cases={len(identifiers)}, no small components found"

        return centroids_by_case, probabilities, f"{summary_prefix}, {count_summary}"

    def _component_cache_file(self):
        safe_max = str(self.small_lesion_component_max_mm3).replace(".", "p")
        safe_min = str(self.small_lesion_component_min_mm3).replace(".", "p")
        return os.path.join(
            self.output_folder,
            f"fold_{self.fold}_component_small_lesions_{self.small_lesion_component_mode}_"
            f"min{safe_min}_max{safe_max}.pkl",
        )

    def _compute_component_centroids(self, dataset_tr, identifiers: Iterable[str]):
        centroids_by_case: Dict[str, List[Tuple[int, int, int]]] = {}
        counts: Dict[str, int] = {}
        structure = generate_binary_structure(3, 1)

        identifiers = list(identifiers)
        total = len(identifiers)
        for idx, identifier in enumerate(identifiers):
            try:
                _, seg, _, properties = dataset_tr.load_case(identifier)
                masks = self._component_masks_from_segmentation(np.asarray(seg))
                spacing = np.asarray(properties.get("spacing", [1.0, 1.0, 1.0]), dtype=np.float64)
                voxel_volume = float(np.prod(spacing))
                centroids: List[Tuple[int, int, int]] = []

                for mask in masks:
                    labeled, num_components = cc_label(mask, structure=structure)
                    if num_components == 0:
                        continue
                    for component_id, slices in enumerate(find_objects(labeled), start=1):
                        if slices is None:
                            continue
                        component = labeled[slices] == component_id
                        voxel_count = int(component.sum())
                        volume_mm3 = voxel_count * voxel_volume
                        if not (
                            self.small_lesion_component_min_mm3
                            <= volume_mm3
                            <= self.small_lesion_component_max_mm3
                        ):
                            continue
                        coords = np.argwhere(component)
                        if coords.size == 0:
                            continue
                        starts = np.asarray([s.start for s in slices], dtype=np.float64)
                        centroid = np.rint(coords.mean(axis=0) + starts).astype(np.int64)
                        centroids.append(tuple(int(x) for x in centroid.tolist()))

                centroids_by_case[identifier] = centroids
                counts[identifier] = len(centroids)
                if (idx + 1) % 100 == 0:
                    self._safe_log(
                        f"ComponentSmallLesionOS: scanned {idx + 1}/{total} cases"
                    )
            except Exception as e:
                warnings.warn(f"Failed to compute small lesion components for {identifier}: {e}")
                centroids_by_case[identifier] = []
                counts[identifier] = 0

        return centroids_by_case, counts

    def _component_masks_from_segmentation(self, seg: np.ndarray):
        if seg.ndim == 4 and seg.shape[0] == 1:
            label_volume = seg[0]
            foreground = label_volume > 0
            if self.small_lesion_component_mode == "per_label":
                labels = [i for i in np.unique(label_volume) if i > 0]
                return [(label_volume == label) for label in labels]
            return [foreground]

        if seg.ndim == 4:
            masks = [(seg[c] > 0) for c in range(seg.shape[0])]
            if self.small_lesion_component_mode == "union":
                return [np.logical_or.reduce(masks)] if masks else []
            return masks

        if seg.ndim == 3:
            return [seg > 0]

        raise ValueError(f"Unsupported segmentation shape for component extraction: {seg.shape}")

    def _safe_log(self, message: str) -> None:
        try:
            self.print_to_log_file(message)
        except Exception:
            print(message)

    def on_train_start(self):
        super().on_train_start()
        self._safe_log(
            "ComponentSmallLesionOS uses default nnU-Net loss; no Focal/Tversky terms are enabled. "
            f"num_epochs={self.num_epochs}, initial_lr={self.initial_lr}"
        )
