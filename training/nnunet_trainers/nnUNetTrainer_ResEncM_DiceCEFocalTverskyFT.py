import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class BinaryFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        batch_dice: bool,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        smooth: float = 1e-5,
        use_ignore_label: bool = False,
    ):
        super().__init__()
        self.batch_dice = batch_dice
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth
        self.use_ignore_label = use_ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            mask = None
            target_regions = target

        target_regions = target_regions.float()
        probabilities = torch.sigmoid(net_output)

        focal = self._focal_loss(net_output, probabilities, target_regions, mask)
        tversky = self._tversky_loss(probabilities, target_regions, mask)
        return self.focal_weight * focal + self.tversky_weight * tversky

    def _focal_loss(self, logits, probabilities, target, mask):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = probabilities * target + (1 - probabilities) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal = alpha_t * torch.pow(torch.clamp(1 - p_t, min=0.0, max=1.0), self.focal_gamma) * bce
        if mask is not None:
            mask = mask.expand_as(focal)
            return (focal * mask).sum() / torch.clamp(mask.sum(), min=1)
        return focal.mean()

    def _tversky_loss(self, probabilities, target, mask):
        if mask is not None:
            mask = mask.float().expand_as(probabilities)
            probabilities = probabilities * mask
            target = target * mask

        axes = tuple(range(2, probabilities.ndim))
        if self.batch_dice:
            axes = (0, *axes)

        tp = (probabilities * target).sum(dim=axes)
        fp = (probabilities * (1 - target)).sum(dim=axes)
        fn = ((1 - probabilities) * target).sum(dim=axes)
        tversky = (tp + self.smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )
        return 1 - tversky.mean()


class DCAndBCEFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        batch_dice: bool,
        focal_weight: float,
        tversky_weight: float,
        focal_alpha: float,
        focal_gamma: float,
        tversky_alpha: float,
        tversky_beta: float,
        use_ignore_label: bool,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.extra_loss = BinaryFocalTverskyLoss(
            batch_dice=batch_dice,
            focal_weight=focal_weight,
            tversky_weight=tversky_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            use_ignore_label=use_ignore_label,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base_loss(net_output, target) + self.extra_loss(net_output, target)


class nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT(nnUNetTrainer):
    """ResEncM finetune trainer using the default Dice/BCE plus Focal and Tversky terms."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = float(os.environ.get("FT_INITIAL_LR", "0.001"))
        self.num_epochs = int(os.environ.get("FT_NUM_EPOCHS", "300"))
        self.focal_weight = float(os.environ.get("FT_FOCAL_WEIGHT", "0.5"))
        self.tversky_weight = float(os.environ.get("FT_TVERSKY_WEIGHT", "0.5"))
        self.focal_alpha = float(os.environ.get("FT_FOCAL_ALPHA", "0.25"))
        self.focal_gamma = float(os.environ.get("FT_FOCAL_GAMMA", "2.0"))
        self.tversky_alpha = float(os.environ.get("FT_TVERSKY_ALPHA", "0.3"))
        self.tversky_beta = float(os.environ.get("FT_TVERSKY_BETA", "0.7"))

    def _build_loss(self):
        if not self.label_manager.has_regions:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            raise RuntimeError(
                "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT is intended for region-based MET labels. "
                "This dataset is not using region labels, so non-region Focal/Tversky was not enabled."
            )

        base_loss = DC_and_BCE_loss(
            {},
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "do_bg": True,
                "smooth": 1e-5,
                "ddp": self.is_ddp,
            },
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        loss = DCAndBCEFocalTverskyLoss(
            base_loss=base_loss,
            batch_dice=self.configuration_manager.batch_dice,
            focal_weight=self.focal_weight,
            tversky_weight=self.tversky_weight,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            tversky_alpha=self.tversky_alpha,
            tversky_beta=self.tversky_beta,
            use_ignore_label=self.label_manager.ignore_label is not None,
        )

        if self._do_i_compile():
            loss.base_loss.dc = torch.compile(loss.base_loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "DiceCEFocalTverskyFT: "
            f"num_epochs={self.num_epochs}, initial_lr={self.initial_lr}, "
            f"focal_weight={self.focal_weight}, tversky_weight={self.tversky_weight}, "
            f"focal_alpha={self.focal_alpha}, focal_gamma={self.focal_gamma}, "
            f"tversky_alpha={self.tversky_alpha}, tversky_beta={self.tversky_beta}"
        )
