"""Training curves plot (Figure 3).

Shows LCS and loss components (reconstruction, alignment, lag consistency)
over training steps, comparing TMO vs. symmetric baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def plot_training_curves(
    steps: np.ndarray,
    lcs_values: np.ndarray,
    loss_values: Optional[Dict[str, np.ndarray]] = None,
    baseline_lcs: Optional[np.ndarray] = None,
    baseline_label: str = "Symmetric baseline",
    model_label: str = "TMO (asymmetric)",
    title: str = "Training dynamics",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    ylim_lcs: Tuple[float, float] = (0, 1),
) -> plt.Figure:
    """Plot LCS and loss curves.

    Parameters
    ----------
    steps : np.ndarray, shape (n_steps,)
        Training steps at which metrics were logged.
    lcs_values : np.ndarray, shape (n_steps,)
        Lag Concordance Score at each step.
    loss_values : dict, optional
        Mapping loss name -> array of values (same length as steps).
        Common keys: 'recon', 'align', 'lag', 'total'.
    baseline_lcs : np.ndarray, optional
        LCS values for symmetric baseline (no lag bias).
    baseline_label, model_label : str
        Labels for legend.
    title, figsize, save_path, ylim_lcs.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2 if loss_values else 1, figsize=figsize)
    if not loss_values:
        axes = [axes]

    # Left: LCS plot
    ax_lcs = axes[0]
    ax_lcs.plot(steps, lcs_values, 'b-', linewidth=2, label=model_label)
    if baseline_lcs is not None:
        # Ensure baseline_lcs has matching steps (may need interpolation)
        if len(baseline_lcs) == len(steps):
            ax_lcs.plot(steps, baseline_lcs, 'r--', linewidth=2, label=baseline_label)
        else:
            # Assume baseline steps are aligned
            ax_lcs.plot(steps, baseline_lcs, 'r--', linewidth=2, label=baseline_label)
    ax_lcs.set_xlabel("Training step")
    ax_lcs.set_ylabel("Lag Concordance Score (LCS)")
    ax_lcs.set_title("LCS")
    ax_lcs.set_ylim(ylim_lcs)
    ax_lcs.legend()
    ax_lcs.grid(True, alpha=0.3)

    # Right: loss curves (if provided)
    if loss_values:
        ax_loss = axes[1]
        colors = {'recon': 'green', 'align': 'orange', 'lag': 'red', 'total': 'black'}
        for loss_name, values in loss_values.items():
            if loss_name in colors:
                ax_loss.plot(steps, values, label=loss_name, color=colors[loss_name], linewidth=1.5)
            else:
                ax_loss.plot(steps, values, label=loss_name, linewidth=1.5)
        ax_loss.set_xlabel("Training step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss components")
        ax_loss.set_yscale('log')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_phase_transition(
    phase_a_steps: np.ndarray,
    phase_a_lcs: np.ndarray,
    phase_b_steps: np.ndarray,
    phase_b_lcs: np.ndarray,
    phase_c_steps: np.ndarray,
    phase_c_lcs: np.ndarray,
    phase_boundaries: List[float],
    labels: List[str] = ["Phase A (warmup)", "Phase B (LagMLP)", "Phase C (fine-tune)"],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot LCS across three training phases.

    Parameters
    ----------
    phase_a_steps, phase_a_lcs : arrays for Phase A.
    phase_b_steps, phase_b_lcs : arrays for Phase B.
    phase_c_steps, phase_c_lcs : arrays for Phase C.
    phase_boundaries : list of floats [step_at_end_phaseA, step_at_end_phaseB]
        Steps where Phase A ends and Phase B ends.
    labels : list of three strings.
    figsize, save_path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(phase_a_steps, phase_a_lcs, 'b-', linewidth=2, label=labels[0])
    ax.plot(phase_b_steps, phase_b_lcs, 'g-', linewidth=2, label=labels[1])
    ax.plot(phase_c_steps, phase_c_lcs, 'r-', linewidth=2, label=labels[2])

    # Mark phase boundaries
    for bound in phase_boundaries:
        ax.axvline(x=bound, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Lag Concordance Score (LCS)")
    ax.set_title("LCS progression through training phases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig