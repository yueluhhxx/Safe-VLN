"""Operationally safer checkpointing for the released ETPNav trainer.

This module deliberately does not alter rollout, mapping, planning, or loss
code. It gates the existing rank-zero save calls with an optional publication
window, then writes atomically so that an interrupted job cannot expose a
partially-written ``ckpt.iter*.pth`` as a resumable checkpoint.
"""

import os
import tempfile
from pathlib import Path

import torch
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.common.checkpoint_cadence import (
    should_save_checkpoint,
    validate_checkpoint_cadence,
    validate_checkpoint_save_calls,
)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for a rename on local POSIX filesystems."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@baseline_registry.register_trainer(name="SS-ETP-Managed")
class ManagedCheckpointRLTrainer(RLTrainer):
    """ETPNav trainer with atomic, resume-safe checkpoint publication."""

    def _checkpoint_cadence(self):
        return validate_checkpoint_cadence(
            getattr(self.config.IL, "checkpoint_start_iter", 0),
            getattr(self.config.IL, "checkpoint_end_iter", 0),
            getattr(self.config.IL, "checkpoint_interval", 0),
        )

    def train(self):
        # Fail before parent train() initializes CUDA, workers, or model state.
        start, end, interval = self._checkpoint_cadence()
        if not getattr(self.config.IL, "load_from_ckpt", False):
            validate_checkpoint_save_calls(
                start, end, interval, getattr(self.config.IL, "log_every", 0)
            )
        return super().train()

    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        checkpoint = super().load_checkpoint(checkpoint_path, *args, **kwargs)
        if isinstance(checkpoint, dict):
            self._managed_amp_scaler_state = checkpoint.get("amp_scaler_state")
        return checkpoint

    def _train_interval(self, interval, ml_weight, sample_ratio):
        # ``RLTrainer.train`` creates GradScaler after it restores the model and
        # optimizer.  Defer scaler restoration until the first resumed interval
        # without duplicating or changing the released training loop.
        scaler_state = getattr(self, "_managed_amp_scaler_state", None)
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
            self._managed_amp_scaler_state = None
        return super()._train_interval(interval, ml_weight, sample_ratio)

    def save_checkpoint(self, iteration: int) -> None:
        # This pure gate precedes all directory creation and state extraction.
        if not should_save_checkpoint(iteration, *self._checkpoint_cadence()):
            return
        checkpoint_dir = Path(self.config.CHECKPOINT_FOLDER)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_dir / f"ckpt.iter{iteration}.pth"

        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
            "optim_state": self.optimizer.state_dict(),
            "iteration": iteration,
            "checkpoint_format_version": 1,
        }
        if hasattr(self, "scaler"):
            checkpoint["amp_scaler_state"] = self.scaler.state_dict()

        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(checkpoint_dir),
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        try:
            torch.save(checkpoint, temporary_name)
            with open(temporary_name, "rb") as checkpoint_file:
                os.fsync(checkpoint_file.fileno())
            os.replace(temporary_name, destination)
            _fsync_directory(checkpoint_dir)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
