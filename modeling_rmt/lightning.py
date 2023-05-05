import math
import warnings
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing lr with linear warmup
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        T_max: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            T_max (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [self.warmup_start_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_epochs:
            return [
                group['lr'] + (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs


        last_epoch = self.last_epoch - self.warmup_epochs
        T_max = self.T_max - self.warmup_epochs

        if last_epoch >= T_max:
            return [self.eta_min for _ in self.optimizer.param_groups]

        if (last_epoch - 1 - T_max) % (2 * T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * last_epoch / T_max)) /
                (1 + math.cos(math.pi * (last_epoch - 1) / T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class LinearWarmupLinearLR(_LRScheduler):
    """
    Cosine annealing lr with linear warmup
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        T_max: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            T_max (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [self.warmup_start_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_epochs:
            return [
                group['lr'] + (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs


        last_epoch = self.last_epoch - self.warmup_epochs
        T_max = self.T_max - self.warmup_epochs

        return [
            max(self.eta_min, float(T_max - last_epoch) / float(max(1, T_max)) * group['initial_lr'])
            for group in self.optimizer.param_groups
        ]


def get_parameter_names(
    model: torch.nn.Module, forbidden_layer_types: List[torch.nn.Module]
) -> List[str]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


class RMTModelPL(LightningModule):
    def __init__(self, rmt_model, cfg):
        super().__init__()
        self._module = rmt_model
        self.cfg = cfg

        try:
            self.save_hyperparameters(ignore=['rmt_model'])
        except KeyError:
            warnings.warn(
                "Can't save hyperparameters. You are probably calling `.load_from_checkpoint()` method. "
                + "In this case call save_hyperparameters() manually.",
                UserWarning,
            )

    def forward(self, x, **kwargs):
        return self._module(**x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        self._log(
            "train",
            {
                "loss": out['loss'],
                "perplexity": torch.exp(out["loss"].detach())
            },
            batch_size=batch['input_ids'].shape[0]
        )
        return out['loss']

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        metrics = {
            "loss": out["loss"],
            "perplexity": torch.exp(out["loss"])
        }
        self._log("val", metrics, on_step=False, on_epoch=True, sync_dist=True,
                  batch_size=batch['input_ids'].shape[0])

    def configure_optimizers(self):

        decay_parameters = get_parameter_names(self, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.cfg.optimizer.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, **self.cfg["optimizer"]
        )

        options = self.cfg.lr_scheduler.as_dict()

        interval = options.pop("interval", "epoch")
        monitor = options.pop("monitor", "val/loss")

        lr_scheduler = {
            # "scheduler": LinearWarmupCosineAnnealingLR(optimizer, **options),
            "scheduler": LinearWarmupLinearLR(optimizer, **options),
            "interval": interval,
            "monitor": monitor,
        }

        return [optimizer], [lr_scheduler]

    def _log(self, prefix, log_dict, **kwargs):
        for k, v in log_dict.items():
            self.log(f"{prefix}/{k}", v, **kwargs)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, map_location=None, strict=True, **init_kwargs):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_dir():
            model = super(LightningModule, cls).load_from_checkpoint(
                ckpt_path, map_location=map_location, strict=strict
            )
        else:
            state = torch.load(
                ckpt_path / "checkpoint" / "mp_rank_00_model_states.pt",
                map_location=map_location,
            )
            state["state_dict"] = state.pop("module")

            for key in list(state["state_dict"].keys()):
                state["state_dict"][key.replace("_forward_module.", "")] = state[
                    "state_dict"
                ].pop(key)

            model = cls(**init_kwargs, **state["hyper_parameters"])
            model._set_hparams(state["hyper_parameters"])
            model.load_state_dict(state["state_dict"], strict=strict)
        return model