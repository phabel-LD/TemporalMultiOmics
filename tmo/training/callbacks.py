"""Callbacks for TMO training.

Provides:
    - LoggerCallback: logs metrics to console and optionally to TensorBoard/WandB.
    - LCSValidator: computes Lag Concordance Score on validation data at specified intervals.
    - CheckpointCallback: saves model checkpoints based on best validation LCS.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LoggerCallback:
    """Logs training metrics to console and optional experiment trackers."""

    def __init__(
        self,
        log_interval: int = 10,
        use_tensorboard: bool = False,
        tensorboard_writer: Optional[Any] = None,
        use_wandb: bool = False,
    ):
        """
        Parameters
        ----------
        log_interval : int, default=10
            Number of steps between logging.
        use_tensorboard : bool, default=False
            Whether to log to TensorBoard.
        tensorboard_writer : optional
            TensorBoard SummaryWriter instance.
        use_wandb : bool, default=False
            Whether to log to Weights & Biases.
        """
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard
        self.tensorboard_writer = tensorboard_writer
        self.use_wandb = use_wandb
        self.step = 0

    def on_train_step(self, step: int, metrics: Dict[str, float]):
        """Called after each training step."""
        self.step = step
        if step % self.log_interval == 0:
            log_msg = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(log_msg)

            if self.use_tensorboard and self.tensorboard_writer:
                for k, v in metrics.items():
                    self.tensorboard_writer.add_scalar(f"train/{k}", v, step)

            if self.use_wandb:
                import wandb
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def on_validation(self, step: int, metrics: Dict[str, float]):
        """Called after validation."""
        log_msg = f"Validation at step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_msg)

        if self.use_tensorboard and self.tensorboard_writer:
            for k, v in metrics.items():
                self.tensorboard_writer.add_scalar(f"val/{k}", v, step)

        if self.use_wandb:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)


class LCSValidator:
    """Computes Lag Concordance Score (LCS) on validation data.

    LCS = Spearman correlation between learned Δτ̂_g (per gene) and
    CCF-derived Δτ_g^CCF for a held-out set of genes or cells.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        val_dataloader: DataLoader,
        ccf_prior_dict: Dict[int, float],   # gene_id -> CCF delta_tau prior
        val_interval: int = 500,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            TMO model (must have forward_phase_b or forward_phase_c to get predictions).
        device : torch.device
            Device for computation.
        val_dataloader : DataLoader
            Validation data loader.
        ccf_prior_dict : dict
            Mapping from gene ID (int) to CCF-derived Δτ_g^CCF (float).
        val_interval : int, default=500
            Number of training steps between LCS evaluations.
        """
        self.model = model
        self.device = device
        self.val_dataloader = val_dataloader
        self.ccf_prior = ccf_prior_dict
        self.val_interval = val_interval
        self.best_lcs = -np.inf
        self.best_step = 0

    def compute_lcs(self, step: int, phase: str = 'phase_b') -> Optional[float]:
        """Compute LCS on validation data.

        Parameters
        ----------
        step : int
            Current training step.
        phase : str, default='phase_b'
            Which model forward to use ('phase_b' or 'phase_c').

        Returns
        -------
        lcs : float or None
            Spearman correlation, or None if no valid predictions.
        """
        self.model.eval()
        all_pred_lags = []
        all_target_lags = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                # Get predicted lags per gene (Phase B or C)
                if phase == 'phase_b':
                    outputs = self.model.forward_phase_b(
                        atac_ids=batch['atac_ids'],
                        rna_ids=batch['rna_ids'],
                        atac_values=batch['atac_values'],
                        rna_values=batch['rna_values'],
                        pseudotime=batch['pseudotime'],
                        modality_atac=batch['modality_atac'],
                        modality_rna=batch['modality_rna'],
                        attention_mask=batch.get('attention_mask'),
                        freeze_encoder=True,
                    )
                    delta_tau_pred = outputs['delta_tau']  # (batch, seq_len_rna)
                else:  # phase_c
                    outputs = self.model.forward_phase_c(
                        atac_ids=batch['atac_ids'],
                        rna_ids=batch['rna_ids'],
                        atac_values=batch['atac_values'],
                        rna_values=batch['rna_values'],
                        pseudotime=batch['pseudotime'],
                        modality_atac=batch['modality_atac'],
                        modality_rna=batch['modality_rna'],
                        attention_mask=batch.get('attention_mask'),
                    )
                    delta_tau_pred = outputs['delta_tau']

                # Average predictions per gene over cells
                rna_ids = batch['rna_ids']  # (batch, seq_len_rna)
                flat_genes = rna_ids.flatten()
                flat_pred = delta_tau_pred.flatten()
                # Gather unique genes and average
                unique_genes = torch.unique(flat_genes)
                for g in unique_genes:
                    g_int = g.item()
                    if g_int not in self.ccf_prior:
                        continue
                    mask = (flat_genes == g)
                    if mask.sum() > 0:
                        avg_pred = flat_pred[mask].mean().item()
                        all_pred_lags.append(avg_pred)
                        all_target_lags.append(self.ccf_prior[g_int])

        if len(all_pred_lags) < 2:
            logger.warning(f"Not enough genes for LCS computation at step {step}")
            return None

        # Compute Spearman correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(all_pred_lags, all_target_lags)
        logger.info(f"Step {step}: LCS = {corr:.4f} (p={p_value:.4e})")

        return corr

    def should_validate(self, step: int, last_val_step: int) -> bool:
        """Check if validation should be performed."""
        return (step - last_val_step) >= self.val_interval


class CheckpointCallback:
    """Saves model checkpoints based on best validation LCS."""

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best_only: bool = True,
        save_every_n_steps: Optional[int] = None,
        maximize: bool = True,
    ):
        """
        Parameters
        ----------
        checkpoint_dir : Path
            Directory to save checkpoints.
        save_best_only : bool, default=True
            Only save when validation metric improves.
        save_every_n_steps : int, optional
            If provided, save a checkpoint every N steps regardless of metric.
        maximize : bool, default=True
            Whether higher validation metric is better (True for LCS).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.save_every_n_steps = save_every_n_steps
        self.maximize = maximize
        self.best_metric = -np.inf if maximize else np.inf

    def on_validation(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metric: Optional[float] = None,
    ):
        """Called after validation.

        Saves checkpoint if metric improves (when save_best_only=True) or
        every N steps.
        """
        # Save periodic checkpoint if requested
        if self.save_every_n_steps and step % self.save_every_n_steps == 0:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            self._save_checkpoint(path, step, model, optimizer)
            logger.info(f"Periodic checkpoint saved to {path}")

        # Save best checkpoint
        if metric is not None and self.save_best_only:
            improved = (self.maximize and metric > self.best_metric) or \
                       (not self.maximize and metric < self.best_metric)
            if improved:
                self.best_metric = metric
                path = self.checkpoint_dir / "best.pt"
                self._save_checkpoint(path, step, model, optimizer)
                logger.info(f"New best model saved to {path} (metric={metric:.4f})")

    def _save_checkpoint(
        self,
        path: Path,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        additional: Optional[Dict] = None,
    ):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if additional:
            checkpoint.update(additional)
        torch.save(checkpoint, path)