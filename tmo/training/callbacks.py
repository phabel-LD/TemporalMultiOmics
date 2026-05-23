"""Callbacks for TMO training.

This module provides optional callback classes that can be used during
training to log metrics, compute validation scores, and save model
checkpoints.  They were designed for the original multi‑phase training
pipeline but are not required by the final single‑script training
procedure.  They are kept here as a reference and may be useful in future
extensions that require more complex training orchestration.

Provided callbacks:
    - ``LoggerCallback`` : prints metrics to the console and optionally
      logs them to TensorBoard or Weights & Biases.
    - ``LCSValidator`` : periodically evaluates the Lag Concordance
      Score (LCS) on a held‑out validation set.
    - ``CheckpointCallback`` : saves model checkpoints when a monitored
      metric improves.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LoggerCallback:
    """Logs training metrics to the console and optional experiment trackers.

    This callback is called after each training step (``on_train_step``)
    and after each validation run (``on_validation``).  It prints a
    formatted message to the standard Python logger and, if configured,
    writes scalar values to TensorBoard or Weights & Biases.

    Parameters
    ----------
    log_interval : int, default=10
        Number of training steps between log outputs.
    use_tensorboard : bool, default=False
        If True, use a ``SummaryWriter`` to log to TensorBoard.
    tensorboard_writer : optional
        An instance of ``torch.utils.tensorboard.SummaryWriter``.
    use_wandb : bool, default=False
        If True, use Weights & Biases for logging.  Requires ``wandb`` to
        be installed and initialized.
    """

    def __init__(
        self,
        log_interval: int = 10,
        use_tensorboard: bool = False,
        tensorboard_writer: Optional[Any] = None,
        use_wandb: bool = False,
    ):
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard
        self.tensorboard_writer = tensorboard_writer
        self.use_wandb = use_wandb
        self.step = 0

    def on_train_step(self, step: int, metrics: Dict[str, float]):
        """Called after each training step.

        Parameters
        ----------
        step : int
            Current training step (global step counter).
        metrics : dict
            Dictionary of metric names and their current values.
        """

        self.step = step
        if step % self.log_interval == 0:
            # Build a console message
            log_msg = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(log_msg)

            # Optionally: write to TensorBoard
            if self.use_tensorboard and self.tensorboard_writer:
                for k, v in metrics.items():
                    self.tensorboard_writer.add_scalar(f"train/{k}", v, step)

            # Optionally: write to Weights & Biases
            if self.use_wandb:
                import wandb
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def on_validation(self, step: int, metrics: Dict[str, float]):
        """Called after a validation run.

        Parameters
        ----------
        step : int
            Current training step.
        metrics : dict
            Validation metrics.
        """

        log_msg = f"Validation at step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_msg)

        if self.use_tensorboard and self.tensorboard_writer:
            for k, v in metrics.items():
                self.tensorboard_writer.add_scalar(f"val/{k}", v, step)

        if self.use_wandb:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)


class LCSValidator:
    """Periodically computes the Lag Concordance Score (LCS) on a validation set.

    LCS is the Spearman correlation between the model's predicted per‑gene
    lags and the CCF‑derived target lags.  This validator is intended for
    use with the older ``TMOModel`` that operates on raw gene identifiers.

    Because the final pipeline uses a latent‑space model and computes LCS
    directly in the training loop (on the full dataset), this class is
    **not** used in the current workflow.  It is preserved for completeness.

    Parameters
    ----------
    model : torch.nn.Module
        The TMO model.  Must implement ``forward_phase_b`` or
        ``forward_phase_c``.
    device : torch.device
        Device for computation.
    val_dataloader : DataLoader
        DataLoader for the validation set.
    ccf_prior_dict : dict
        Mapping from gene ID (int) to the CCF‑derived Δτ target (float).
    val_interval : int, default=500
        Number of training steps between LCS evaluations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        val_dataloader: DataLoader,
        ccf_prior_dict: Dict[int, float],   # gene_id -> CCF delta_tau prior
        val_interval: int = 500,
    ):
        self.model = model
        self.device = device
        self.val_dataloader = val_dataloader
        self.ccf_prior = ccf_prior_dict
        self.val_interval = val_interval
        self.best_lcs = -np.inf
        self.best_step = 0

    def compute_lcs(self, step: int, phase: str = 'phase_b') -> Optional[float]:
        """Compute LCS on the validation data.

        Parameters
        ----------
        step : int
            Current training step (for logging only).
        phase : str, default='phase_b'
            Which model forward method to use; must be ``'phase_b'`` or
            ``'phase_c'``.

        Returns
        -------
        lcs : float or None
            Spearman correlation, or None if fewer than two genes have
            valid targets.
        """

        self.model.eval()
        all_pred_lags = []
        all_target_lags = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Transfer batch tensors to the correct device
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                # Obtain predicted lags using the specified phase
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

                # Flatten and average per gene across all cells in the batch
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
        """Check if a validation run should be performed.

        Parameters
        ----------
        step : int
            Current training step.
        last_val_step : int
            Step at which the last validation was performed.

        Returns
        -------
        bool
            True if ``step - last_val_step >= self.val_interval``.
        """
        return (step - last_val_step) >= self.val_interval


class CheckpointCallback:
    """Saves model checkpoints based on the best validation metric.

    This callback is called after validation.  It can save:
        - the best model seen so far (when ``save_best_only`` is True),
        - a periodic checkpoint every N steps (optional).

    The metric to maximise (or minimise) is passed to ``on_validation``.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where checkpoints will be saved.  Created if needed.
    save_best_only : bool, default=True
        If True, only save a checkpoint when the monitored metric improves.
    save_every_n_steps : int, optional
        If provided, a checkpoint is saved every N steps regardless of
        metric improvement.
    maximize : bool, default=True
        If True, a higher metric is considered better (suitable for LCS).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best_only: bool = True,
        save_every_n_steps: Optional[int] = None,
        maximize: bool = True,
    ):
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
        """Called after each validation run.

        Parameters
        ----------
        step : int
            Current training step.
        model : torch.nn.Module
            The model whose parameters are to be saved.
        optimizer : torch.optim.Optimizer
            The optimizer whose state is saved together with the model.
        metric : float, optional
            The validation metric value.  If provided and ``save_best_only``
            is True, a checkpoint is saved when this metric improves.
        """

        # Save periodic checkpoint (if requested)
        if self.save_every_n_steps and step % self.save_every_n_steps == 0:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            self._save_checkpoint(path, step, model, optimizer)
            logger.info(f"Periodic checkpoint saved to {path}")

        # Save best‑metric checkpoint
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
        """Internal helper that writes a checkpoint to disk.

        The checkpoint contains the model and optimizer state dictionaries,
        the current step, and any additional information.

        Parameters
        ----------
        path : Path
            Output file path.
        step : int
            Current training step.
        model : torch.nn.Module
        optimizer : torch.optim.Optimizer
        additional : dict, optional
            Extra data to include in the checkpoint.
        """
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if additional:
            checkpoint.update(additional)
        torch.save(checkpoint, path)