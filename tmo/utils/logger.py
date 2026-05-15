"""Logging utilities for training and validation.

Provides WandB and TensorBoard integration as well as console logging.
"""

import logging
import sys
from typing import Optional, Any


def setup_logger(name: str = 'tmo', level: int = logging.INFO) -> logging.Logger:
    """Set up a console logger with basic formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class WandBLogger:
    """Lightweight wrapper for Weights & Biases."""

    def __init__(self, project_name: str = 'tmo', run_name: Optional[str] = None, config: Optional[dict] = None):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")
        self.run = self.wandb.init(project=project_name, name=run_name, config=config)

    def log(self, metrics: dict, step: int = None):
        self.wandb.log(metrics, step=step)

    def finish(self):
        self.wandb.finish()


class TensorBoardLogger:
    """Wrapper for TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str = './logs'):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        self.writer.close()