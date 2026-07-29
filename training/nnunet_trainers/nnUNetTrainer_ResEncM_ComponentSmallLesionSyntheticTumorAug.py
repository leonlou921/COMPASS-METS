import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_ResEncM_ComponentSmallLesionOS import (
    ComponentSmallLesionDataLoader,
    nnUNetTrainer_ResEncM_ComponentSmallLesionOS,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class SyntheticTumorComponentSmallLesionDataLoader(ComponentSmallLesionDataLoader):
    """Adds MET-like synthetic enhancing lesions before nnU-Net transforms are applied."""

    def __init__(
        self,
        *args,
        synth_prob: float = 0.35,
        synth_max_lesions: int = 2,
        synth_radius_min: float = 1.5,
        synth_radius_max: float = 4.5,
        synth_edema_prob: float = 0.65,
        synth_edema_scale_min: float = 1.6,
        synth_edema_scale_max: float = 2.6,
        synth_blend: float = 0.75,
        synth_noise_std: float = 0.15,
        channel_names: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.synth_prob = float(np.clip(synth_prob, 0.0, 1.0))
        self.synth_max_lesions = max(1, int(synth_max_lesions))
        self.synth_radius_min = float(synth_radius_min)
        self.synth_radius_max = float(max(synth_radius_max, synth_radius_min))
        self.synth_edema_prob = float(np.clip(synth_edema_prob, 0.0, 1.0))
        self.synth_edema_scale_min = float(max(synth_edema_scale_min, 1.0))
        self.synth_edema_scale_max = float(max(synth_edema_scale_max, self.synth_edema_scale_min))
        self.synth_blend = float(np.clip(synth_blend, 0.0, 1.0))
        self.synth_noise_std = float(max(synth_noise_std, 0.0))
        self.tumor_boost, self.edema_boost = self._make_channel_boosts(channel_names or {})

    @staticmethod
    def _make_channel_boosts(channel_names: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        if not channel_names:
            return (
                np.asarray([2.6, -0.1, 1.0, 0.9], dtype=np.float32),
                np.asarray([0.2, 0.0, 1.8, 1.4], dtype=np.float32),
            )

        ordered = [channel_names[str(i)].lower() for i in sorted(map(int, channel_names.keys()))]
        tumor = []
        edema = []
        for name in ordered:
            if "t1c" in name or "t1ce" in name or "ce" in name:
                tumor.append(2.8)
                edema.append(0.25)
            elif "t1" in name:
                tumor.append(-0.15)
                edema.append(0.0)
            elif "flair" in name or "t2f" in name:
                tumor.append(1.0)
                edema.append(1.9)
            elif "t2" in name:
                tumor.append(0.9)
                edema.append(1.5)
            else:
                tumor.append(0.6)
                edema.append(0.4)
        return np.asarray(tumor, dtype=np.float32), np.asarray(edema, dtype=np.float32)

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

            cropped_data = crop_and_pad_nd(data, bbox, 0).astype(np.float32, copy=False)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1).astype(np.int16, copy=False)
            cropped_data, seg_cropped = self._maybe_add_synthetic_tumors(cropped_data, seg_cropped)

            data_all[j] = cropped_data
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

    def _maybe_add_synthetic_tumors(self, data: np.ndarray, seg: np.ndarray):
        if np.random.uniform() >= self.synth_prob:
            return data, seg
        if seg.ndim != 4 or seg.shape[0] < 1:
            return data, seg

        seg0 = seg[0]
        brain_mask = np.any(np.abs(data) > 1e-6, axis=0)
        candidate_mask = (seg0 == 0) & brain_mask
        if int(candidate_mask.sum()) < 64:
            return data, seg

        n_lesions = 1
        if self.synth_max_lesions > 1:
            n_lesions += int(np.random.binomial(self.synth_max_lesions - 1, 0.35))

        for _ in range(n_lesions):
            coords = np.argwhere(candidate_mask)
            if coords.size == 0:
                break
            center = coords[np.random.randint(0, len(coords))]
            radii = np.random.uniform(self.synth_radius_min, self.synth_radius_max, size=3)
            self._paint_one_lesion(data, seg0, candidate_mask, center, radii)

        return data, seg

    def _paint_one_lesion(self, data, seg0, candidate_mask, center, radii):
        shape = np.asarray(seg0.shape)
        max_radius = np.ceil(radii * self.synth_edema_scale_max).astype(int) + 1
        lo = np.maximum(center - max_radius, 0)
        hi = np.minimum(center + max_radius + 1, shape)
        slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        local_shape = np.asarray([s.stop - s.start for s in slices])
        zz, yy, xx = np.meshgrid(
            np.arange(local_shape[0]),
            np.arange(local_shape[1]),
            np.arange(local_shape[2]),
            indexing="ij",
        )
        local_coords = np.stack([zz, yy, xx], axis=0).astype(np.float32)
        local_center = (center - lo).astype(np.float32).reshape(3, 1, 1, 1)

        ellipsoid = (((local_coords - local_center) / radii.reshape(3, 1, 1, 1)) ** 2).sum(axis=0)
        tumor_local = ellipsoid <= 1.0
        target_bg = candidate_mask[slices]
        tumor_local &= target_bg
        if int(tumor_local.sum()) < 4:
            return

        edema_local = np.zeros_like(tumor_local, dtype=bool)
        if np.random.uniform() < self.synth_edema_prob:
            scale = np.random.uniform(self.synth_edema_scale_min, self.synth_edema_scale_max)
            edema_ellipsoid = (
                ((local_coords - local_center) / (radii * scale).reshape(3, 1, 1, 1)) ** 2
            ).sum(axis=0)
            edema_local = (edema_ellipsoid <= 1.0) & target_bg & ~tumor_local

        self._blend_intensity(data[:, slices[0], slices[1], slices[2]], tumor_local, self.tumor_boost)
        if edema_local.any():
            self._blend_intensity(data[:, slices[0], slices[1], slices[2]], edema_local, self.edema_boost)
            seg0[slices][edema_local] = 2
        seg0[slices][tumor_local] = 3
        candidate_mask[slices][tumor_local | edema_local] = False

    def _blend_intensity(self, data_patch: np.ndarray, mask: np.ndarray, boost: np.ndarray):
        n_channels = data_patch.shape[0]
        if boost.size != n_channels:
            boost = np.resize(boost, n_channels).astype(np.float32)
        noise = np.random.normal(0.0, self.synth_noise_std, size=(n_channels, int(mask.sum()))).astype(np.float32)
        for c in range(n_channels):
            original = data_patch[c][mask]
            synthetic = np.clip(original + boost[c] + noise[c], -5.0, 5.0)
            data_patch[c][mask] = (1.0 - self.synth_blend) * original + self.synth_blend * synthetic


class nnUNetTrainer_ResEncM_ComponentSmallLesionSyntheticTumorAug(
    nnUNetTrainer_ResEncM_ComponentSmallLesionOS
):
    """Component small-lesion oversampling plus MET-like synthetic tumor augmentation."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.synth_tumor_prob = float(os.environ.get("SYNTH_TUMOR_PROB", "0.35"))
        self.synth_tumor_max_lesions = int(os.environ.get("SYNTH_TUMOR_MAX_LESIONS", "2"))
        self.synth_tumor_radius_min = float(os.environ.get("SYNTH_TUMOR_RADIUS_MIN", "1.5"))
        self.synth_tumor_radius_max = float(os.environ.get("SYNTH_TUMOR_RADIUS_MAX", "4.5"))
        self.synth_tumor_edema_prob = float(os.environ.get("SYNTH_TUMOR_EDEMA_PROB", "0.65"))
        self.synth_tumor_edema_scale_min = float(os.environ.get("SYNTH_TUMOR_EDEMA_SCALE_MIN", "1.6"))
        self.synth_tumor_edema_scale_max = float(os.environ.get("SYNTH_TUMOR_EDEMA_SCALE_MAX", "2.6"))
        self.synth_tumor_blend = float(os.environ.get("SYNTH_TUMOR_BLEND", "0.75"))
        self.synth_tumor_noise_std = float(os.environ.get("SYNTH_TUMOR_NOISE_STD", "0.15"))

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

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
        centroids_by_case, sampling_probabilities, component_summary = (
            self._build_component_small_lesion_sampling(dataset_tr)
        )
        self._safe_log(
            "ComponentSmallLesionSyntheticTumorAug: "
            f"synth_prob={self.synth_tumor_prob:.3f}, max_lesions={self.synth_tumor_max_lesions}, "
            f"radius=[{self.synth_tumor_radius_min:.2f},{self.synth_tumor_radius_max:.2f}], "
            f"edema_prob={self.synth_tumor_edema_prob:.3f}, "
            f"edema_scale=[{self.synth_tumor_edema_scale_min:.2f},{self.synth_tumor_edema_scale_max:.2f}], "
            f"blend={self.synth_tumor_blend:.3f}, noise_std={self.synth_tumor_noise_std:.3f}, "
            f"{component_summary}"
        )

        dl_tr = SyntheticTumorComponentSmallLesionDataLoader(
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
            synth_prob=self.synth_tumor_prob,
            synth_max_lesions=self.synth_tumor_max_lesions,
            synth_radius_min=self.synth_tumor_radius_min,
            synth_radius_max=self.synth_tumor_radius_max,
            synth_edema_prob=self.synth_tumor_edema_prob,
            synth_edema_scale_min=self.synth_tumor_edema_scale_min,
            synth_edema_scale_max=self.synth_tumor_edema_scale_max,
            synth_blend=self.synth_tumor_blend,
            synth_noise_std=self.synth_tumor_noise_std,
            channel_names=self.dataset_json.get("channel_names", {}),
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

    def on_train_start(self):
        super().on_train_start()
        self._safe_log(
            "SyntheticTumorAug active for MET: ET label=3, edema halo label=2, "
            f"channel_names={self.dataset_json.get('channel_names', {})}"
        )
