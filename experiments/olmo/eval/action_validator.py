"""Held-out action validation during training (mirrors evo/openpi's `eval_step`).

Runs the full flow-matching sampler (`generate_actions`) on held-out LeRobot
episodes and compares the sampled action chunk to ground truth in normalized
action units:

    val/<task>/mse            mean squared error over valid (timestep, dim) entries
    val/<task>/mse_vs_hold    mse / mse of "hold the chunk's first action for every step";
                              < 1 means the model predicts motion better than standing still
    val/<task>/accuracy@<tau> fraction of valid entries with |pred - target| < tau
    val/<metric>              mean over tasks, each task weighted equally

Absolute joint targets over a ~1s chunk barely move, so raw mse/accuracy look good even for a
policy that does nothing; mse_vs_hold is the number that shows real learning.

Every round evaluates the same frames with the same sampler noise, so curves are
comparable across steps.
"""

import inspect
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

from olmo.config import BaseConfig
from olmo.data.data_loader import DataLoaderConfig
from olmo.torch_util import get_global_rank, get_world_size, move_to_device

log = logging.getLogger(__name__)


@dataclass
class ActionValidatorConfig(BaseConfig):
    datasets: List[str] = field(default_factory=list)
    """LeRobot dataset specs to validate on, e.g. "lerobot:repo@<held-out episodes>"."""

    interval: int = 0
    """Run every `interval` steps (and once before training). 0 disables validation."""

    examples_per_dataset: int = 64
    device_batch_size: int = 2
    num_steps: int = 10
    """Flow-matching integration steps for sampling."""

    taus: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1])
    """In normalized units; min-max scales each joint to [-1, 1], so 0.1 = 5% of its range."""
    seed: int = 0
    num_workers: int = 0
    """Validation stops each loader early (after `examples_per_dataset`). With workers, that
    tears them down mid video-decode and pyav's threads abort the process (SIGABRT,
    "terminate called without an active exception"). Decoding in the main process avoids it."""
    sequence_length: Optional[int] = None

    def build(self, model_config, mesh: Optional[DeviceMesh], device: torch.device) -> "ActionValidator":
        loaders = {}
        for spec in self.datasets:
            label = spec.removeprefix("lerobot:").split("@", 1)[0].rsplit("/", 1)[-1]
            t0 = time.perf_counter()
            loaders[label] = DataLoaderConfig(
                dataset=spec,
                split="validation",
                seed=self.seed,
                pad=None,
                sequence_length=self.sequence_length,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=2 if self.num_workers > 0 else None,
            ).build_eval_dataloader(
                model_config=model_config,
                mesh=mesh,
                batch_size=self.device_batch_size,
                for_inference=False,
                include_metadata=False,
            )
            log.info(f"Action validation set '{label}' built in {time.perf_counter() - t0:.1f}s")
        return ActionValidator(cfg=self, loaders=loaders, device=device)


@dataclass
class ActionValidator:
    cfg: ActionValidatorConfig
    loaders: Dict[str, torch.utils.data.DataLoader]
    device: torch.device

    def run(self, fsdp_model: torch.nn.Module, autocast_precision: torch.dtype) -> Dict[str, float]:
        # Under FSDP2 only forward() triggers the parameter all-gather; other entry points
        # must be registered. Every rank calls generate_actions the same number of times.
        if isinstance(fsdp_model, FSDPModule) and not getattr(fsdp_model, "_action_val_registered", False):
            register_fsdp_forward_method(fsdp_model, "generate_actions")
            fsdp_model._action_val_registered = True
        # Inspect the class method: the FSDP-registered instance wrapper takes (*args, **kwargs).
        accepted = set(inspect.signature(type(fsdp_model).generate_actions).parameters)

        world_size = get_world_size()
        taus = list(self.cfg.taus)
        t_start = time.perf_counter()
        fsdp_model.eval()
        per_task: Dict[str, Dict[str, float]] = {}
        with torch.no_grad():
            for task_idx, (label, loader) in enumerate(self.loaders.items()):
                num_batches = min(
                    len(loader),
                    math.ceil(self.cfg.examples_per_dataset / (self.cfg.device_batch_size * world_size)),
                )
                # [sum sq err, valid count, hold-baseline sum sq err, correct@tau...]
                totals = torch.zeros(3 + len(taus), dtype=torch.float64, device=self.device)
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= num_batches:
                        break
                    batch = move_to_device(batch, self.device)
                    target = batch["actions"].float()
                    kwargs = {k: v for k, v in batch.items() if k in accepted and k not in ("labels", "loss_masks")}
                    kwargs["response_mask"] = batch["loss_masks"] > 0
                    generator = torch.Generator(device=self.device).manual_seed(
                        self.cfg.seed + 1_000_000 * get_global_rank() + 1_000 * task_idx + batch_idx
                    )
                    with torch.autocast("cuda", dtype=autocast_precision):
                        pred = fsdp_model.generate_actions(
                            **kwargs, num_steps=self.cfg.num_steps, generator=generator
                        )
                    pred = pred.float()

                    valid = torch.ones_like(target, dtype=torch.bool)
                    if batch.get("action_horizon_is_pad") is not None:
                        valid &= ~batch["action_horizon_is_pad"].bool()[:, :, None]
                    if batch.get("action_dim_is_pad") is not None:
                        valid &= ~batch["action_dim_is_pad"].bool()[:, None, :]
                    err = (pred - target)[valid]
                    hold_err = (target[:, :1] - target)[valid]
                    totals[0] += (err ** 2).sum().double()
                    totals[1] += err.numel()
                    totals[2] += (hold_err ** 2).sum().double()
                    for i, tau in enumerate(taus):
                        totals[3 + i] += (err.abs() < tau).sum().double()

                if dist.is_initialized():
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                count = max(totals[1].item(), 1.0)
                per_task[label] = {
                    "mse": totals[0].item() / count,
                    "mse_vs_hold": totals[0].item() / max(totals[2].item(), 1e-12),
                }
                for i, tau in enumerate(taus):
                    per_task[label][f"accuracy@{tau}"] = totals[3 + i].item() / count

        metrics: Dict[str, float] = {}
        keys = list(next(iter(per_task.values())).keys())
        for key in keys:
            metrics[f"val/{key}"] = sum(m[key] for m in per_task.values()) / len(per_task)
        for label, task_metrics in per_task.items():
            for key, value in task_metrics.items():
                metrics[f"val/{label}/{key}"] = value
        log.info(f"Action validation over {len(per_task)} tasks took {time.perf_counter() - t_start:.1f}s")
        return metrics
