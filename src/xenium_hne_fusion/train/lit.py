from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from glom import glom
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from ai4bmr_learn.utils.pooling import pool


class RegressionLit(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module | None = None,
        embed_dim: int | None = None,
        num_outputs: int = 1,
        batch_key: str | None = "modalities",
        target_key: str = "target",
        lr_head: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-3,
        eta: float = 1e-6,
        schedule: str | None = "cosine",
        max_epochs: int = 35,
        num_warmup_epochs: int = 5,
        freeze_backbone: bool = False,
        pooling: str | None = None,
        loss: str = "mse",
        save_hparams: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if head is None:
            input_dim = embed_dim or backbone.output_dim
            head = nn.Linear(input_dim, 1)

        self.head = head

        self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.criterion = self.configure_loss(loss=loss)

        self.num_outputs = num_outputs
        metrics = self.get_metrics(num_outputs=num_outputs)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.schedule = schedule
        self.eta = eta
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs

        if save_hparams:
            self.save_hyperparameters(ignore=["head", "backbone"])

    def configure_loss(self, loss: str) -> nn.Module:
        if loss == "mse":
            return nn.MSELoss()
        if loss == "huber":
            return nn.SmoothL1Loss()
        raise ValueError(f"Unknown loss: {loss}")

    def get_metrics(self, num_outputs: int) -> MetricCollection:
        return MetricCollection(
            {
                "mse": MeanSquaredError(num_outputs=num_outputs, squared=True),
                "spearman": SpearmanCorrCoef(num_outputs=num_outputs),
                "pearson": PearsonCorrCoef(num_outputs=num_outputs),
            }
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Cast float64 tensors to float32 before device transfer (MPS has no float64 support)."""
        from lightning_utilities.core.apply_func import apply_to_collection
        batch = apply_to_collection(
            batch, dtype=torch.Tensor,
            function=lambda t: t.to(torch.float32) if t.dtype == torch.float64 else t,
        )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def reduce_log_reset(self, metrics: MetricCollection) -> None:
        scores = metrics.compute()
        metrics.reset()
        self.log_dict({f"{k}_mean": v.mean() for k, v in scores.items()})
        self.log_dict({f"{k}_std": v.std() for k, v in scores.items()})

    def _log_alpha(self, *, stage: str, batch_size: int) -> None:
        alpha_param = getattr(self.backbone, "fusion_alpha", None)
        if alpha_param is None:
            return
        alpha = torch.sigmoid(alpha_param.detach()).squeeze()
        self.log(
            f"{stage}/alpha",
            alpha,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)
        return self.head(z)

    def shared_step(self, batch: dict, batch_idx: int):
        x = glom(batch, self.batch_key) if self.batch_key is not None else batch
        y = glom(batch, self.target_key)

        y = y.unsqueeze(1) if y.ndim == 1 else y
        y = y.float()

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)

        y_hat = self.head(z)
        assert y_hat.ndim == 2

        loss = self.criterion(y_hat, y)
        return z, y_hat, y, loss

    def training_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self._log_alpha(stage="train", batch_size=batch_size)

        self.train_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_train_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            if self.num_outputs == 1:
                self.log_dict(self.train_metrics)
            else:
                self.reduce_log_reset(self.train_metrics)

    def validation_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/val", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_alpha(stage="val", batch_size=batch_size)

        self.valid_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            if self.num_outputs == 1:
                self.log_dict(self.valid_metrics)
            else:
                self.reduce_log_reset(self.valid_metrics)

    def test_step(self, batch, batch_idx: int):
        z, y_hat, y, loss = self.shared_step(batch, batch_idx)
        batch_size = int(y.shape[0])

        self.log("loss/test", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_alpha(stage="test", batch_size=batch_size)
        self.test_metrics.update(y_hat, y)

        batch["loss"] = loss
        batch["y_hat"] = y_hat.detach().cpu()
        batch["y"] = y.detach().cpu()
        batch["z"] = z.detach().cpu()
        return batch

    def on_test_epoch_end(self) -> None:
        if not self.trainer.fast_dev_run:
            if self.num_outputs == 1:
                self.log_dict(self.test_metrics)
            else:
                self.reduce_log_reset(self.test_metrics)

    def predict_step(self, batch, batch_idx: int):
        x = glom(batch, self.batch_key) if self.batch_key is not None else batch

        z = self.backbone(x)
        z = pool(z, strategy=self.pooling)
        y_hat = self.head(z)

        batch["prediction"] = y_hat.detach().cpu()
        batch["y_hat"] = y_hat.detach().cpu()
        batch["z"] = z.detach().cpu()

        y = glom(batch, self.target_key)
        assert isinstance(y, torch.Tensor), f"Expected target tensor at '{self.target_key}', got {type(y)}"
        if y.ndim == 1:
            y = y.unsqueeze(1)
        batch["y"] = y.detach().cpu()

        return batch

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {"params": self.head.parameters(), "lr": self.lr_head},
                {"params": filter(lambda p: p.requires_grad, self.backbone.parameters()), "lr": self.lr_backbone},
            ],
            weight_decay=self.weight_decay,
        )

        if self.schedule is None:
            return optimizer

        max_epochs = getattr(self.trainer, "max_epochs", None) or self.max_epochs

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.num_warmup_epochs,
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - self.num_warmup_epochs,
            eta_min=self.eta,
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.num_warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
