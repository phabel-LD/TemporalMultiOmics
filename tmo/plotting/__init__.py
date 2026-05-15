"""Publication‑ready plotting utilities for TMO results."""

from .heatmap import (
    plot_delta_tau_heatmap,
    plot_delta_tau_with_trajectory,
    plot_gene_signature_heatmaps,
)
from .attention_offset import (
    plot_attention_vs_offset,
    plot_ridge_attention_by_gene_class,
)
from .training_curves import (
    plot_training_curves,
    plot_phase_transition,
)
from .violin_adstau import (
    plot_lag_shift_violin,
    plot_ads_violin,
    plot_multi_perturbation_comparison,
)

__all__ = [
    "plot_delta_tau_heatmap",
    "plot_delta_tau_with_trajectory",
    "plot_gene_signature_heatmaps",
    "plot_attention_vs_offset",
    "plot_ridge_attention_by_gene_class",
    "plot_training_curves",
    "plot_phase_transition",
    "plot_lag_shift_violin",
    "plot_ads_violin",
    "plot_multi_perturbation_comparison",
]