================================================================================
TMO: TEMPORAL MULTI‑OMICS (ASYM) – DOCUMENTATION  (final version, May 2026)
================================================================================

Project overview: Deep learning framework for single‑cell ATAC+RNA multi‑omics
that learns cell‑state‑conditional regulatory lags (Δτ) using asymmetric
cross‑modal attention. The pipeline includes data preparation, gene‑level ATAC
matrix building, PCA/LSI latent projection, training of symmetric and asymmetric
models, evaluation, plotting, and ChIP‑seq validation.

**Uniform interpretation**: After running the pipeline, all Δτ heatmaps follow
the same biological convention: **positive Δτ (red) means ATAC leads RNA
(priming), negative Δτ (blue) means RNA leads ATAC (activity‑dependent).**
This is achieved by automatically flipping the sign for datasets where the
pseudotime was reversed (using an early marker gene).

================================================================================
1. DIRECTORY STRUCTURE (actual)
================================================================================
ProjectRoot/
├── tmo/                       # Core library
│   ├── __init__.py
│   ├── __version__.py
│   ├── data/                  # Data loading (empty, kept for compatibility)
│   ├── ccf/                   # CCF estimation
│   │   ├── __init__.py
│   │   ├── local_ccf.py
│   │   └── gp_smoothing.py
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # LatentTokenizer: proj(value) + modality + pseudotime
│   │   ├── attention.py      # AsymmetricCrossAttention (fixed reverse bias)
│   │   ├── lagmlp.py         # LagMLP / WidthMLP (LayerNorm first)
│   │   ├── tmo_block.py      # Transformer block with cross‑attention
│   │   └── tmo_model.py      # TMOLatentModelAsymmetric (two‑pass, use_bias flag)
│   ├── training/              # Losses
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── validation/            # Metrics
│   │   ├── __init__.py
│   │   └── lcs.py
│   ├── plotting/              # Figure generation
│   │   ├── __init__.py
│   │   ├── heatmap.py
│   │   ├── attention_offset.py
│   │   ├── training_curves.py
│   │   └── violin_adstau.py
│   └── utils/                 # Helpers (sparse attention kept for future)
│       ├── __init__.py
│       └── sparse_attention.py
├── scripts/                   # User‑facing analysis scripts
│   ├── 1_create_gene_level_tmo_data.py
│   ├── 2_annotate_components.py
│   ├── 3_train_tmo_asymmetric.py    # single training script (use --symmetric for baseline)
│   ├── 4_evaluate_asymmetric.py
│   ├── 5_plot_correlation_heatmap.py
│   ├── 6_plot_training_curves.py
│   ├── 7_lcs_analysis.py
│   ├── 8_plot_marker_delta_tau_profiles.py
│   ├── 9_gene_program_ordering.py
│   ├── 10_validate_multigene.py      # Perturb‑seq validation
│   └── 11_chipseq_validation.py      # ChIP‑seq validation with top‑500 targets
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md

================================================================================
2. REQUIRED INPUT FILES (per dataset)
================================================================================
For each 10x Multiome dataset you need:

   A) filtered_feature_bc_matrix.h5 (HDF5)
        Contains RNA counts and ATAC peak counts. Read by muon.

   B) atac_peak_annotation.tsv (TSV)
        Map from ATAC peaks to proximal genes. Columns: 'peak' (or
        'chrom','start','end') and 'gene'. Optional: 'distance','peak_type'.

Place these two files in a dedicated folder (e.g., ./10x_dataset_name/).

================================================================================
3. OVERVIEW OF ANALYSIS SCRIPTS
================================================================================

3.1 1_create_gene_level_tmo_data.py
----------------------------------------------------------------------
Purpose: Build gene‑level ATAC matrix, compute pseudotime (first diffusion
component), optionally reverse orientation with a root marker. Saves an AnnData
object ready for TMO.

Usage:
  python 1_create_gene_level_tmo_data.py --data_dir <folder>
        --output_file <filename.h5ad> [--root_marker <gene>]

Parameters:
  --data_dir      : Folder with filtered_feature_bc_matrix.h5 and
                    atac_peak_annotation.tsv.
  --output_file   : Output .h5ad file.
  --root_marker   : (Optional) Early marker gene symbol. If its expression
                    correlates negatively with pseudotime, the axis is reversed.

Output:
  - AnnData with .X (raw RNA), .obsm['ATAC_gene'] (cells × genes),
    .obs['pseudotime'] ∈ [0,1], .uns['pseudotime_reversed'] (bool).

----------------------------------------------------------------------
3.2 2_annotate_components.py
----------------------------------------------------------------------
Purpose: PCA on RNA, extract top genes per component, run GO enrichment
(Enrichr). Saves component_annotation.csv.

Usage:
  python 2_annotate_components.py --data_path <file.h5ad>
        [--n_components 50] [--n_top_genes 100] [--n_top_go 5]
        [--delay 2.0] [--organism human|mouse]

Parameters:
  --data_path    : Input AnnData.
  --n_components : Number of PCA components (default 50).
  --n_top_genes  : Number of top genes (abs loading) per component (default 100).
  --n_top_go     : Number of top GO terms to store (default 5).
  --delay        : Seconds between Enrichr requests (default 2.0).
  --organism     : Species for GO enrichment (human/mouse).

Output:
  - component_annotation.csv with columns Component, Top_genes, GO_terms.

----------------------------------------------------------------------
3.3 3_train_tmo_asymmetric.py   (asymmetric & symmetric training)
----------------------------------------------------------------------
Purpose: Train the TMO model on the full dataset. Uses asymmetric attention by
default; with --symmetric, trains the baseline without attention bias and
without lag loss. Saves best model by LCS (in‑set metric).

Usage (asymmetric):
  python 3_train_tmo_asymmetric.py --data_path <file.h5ad>
        --epochs 50 --val_interval 5 --lambda_lag 0.1

Usage (symmetric):
  python 3_train_tmo_asymmetric.py --data_path <file.h5ad>
        --epochs 50 --val_interval 5 --lambda_lag 0.1 --symmetric

Parameters:
  --data_path      : Input AnnData.
  --output_dir     : Directory for checkpoints and metrics
                     (default ./tmo_results).
  --epochs         : Number of epochs (default 50).
  --val_interval   : Compute LCS every N epochs (default 5).
  --batch_size     : Batch size (default 32).
  --lambda_lag     : Weight for lag consistency loss (default 0.1).
  --lr             : Learning rate (default 1e-3).
  --symmetric      : Train symmetric baseline (no attention bias, no lag loss).

Outputs (in output_dir):
  - tmo_asymmetric_best.pt   (or tmo_symmetric_best.pt)
  - tmo_asymmetric_last.pt   (or tmo_symmetric_last.pt)
  - training_metrics_tmo_asymmetric.csv   (epoch, lcs, recon, lag, loss)
  - pca.pkl, tfidf.pkl, lsi.pkl           (for external validation)

Note:
  - The LCS is computed on the entire dataset (in‑set) every val_interval
    epochs; the best model is selected by this score.
  - Training time ~10‑20 min on CPU for ~3000–5000 cells.

----------------------------------------------------------------------
3.4 4_evaluate_asymmetric.py
----------------------------------------------------------------------
Purpose: Compute CCF Δτ surface, cluster components, generate Δτ heatmap
(Figure 1), and print LCS on the full dataset.

Usage:
  python 4_evaluate_asymmetric.py --data_path <file.h5ad>
        --model_path <model.pt> --annotation_mode cluster_aggregate
        --dist_threshold 1.0 --standardize

Parameters:
  --data_path        : Input AnnData.
  --model_path       : Trained model (.pt).
  --output_dir       : Output folder (default ./tmo_results).
  --annotation_mode  : "component" (raw) or "cluster_aggregate"
                       (median per cluster, rows sorted by similarity).
  --dist_threshold   : Distance threshold for hierarchical clustering
                       (default 1.0).
  --standardize      : Standardise profiles before clustering (True).

Output:
  - fig1_latent_delta_tau.pdf : Δτ heatmap (positive = ATAC‑led).
  - Console: LCS (Spearman correlation between predicted and CCF targets).

----------------------------------------------------------------------
3.5 5_plot_correlation_heatmap.py   (Figure 2)
----------------------------------------------------------------------
Purpose: ATAC‑RNA correlation heatmap (U‑shape of developmental coupling).

Usage:
  python 5_plot_correlation_heatmap.py --data_path <file.h5ad>
        --output_dir <dir> [--n_bins 10]

Output:
  - fig2_correlation_max_over_pseudotime.pdf

----------------------------------------------------------------------
3.6 6_plot_training_curves.py   (Figure 3)
----------------------------------------------------------------------
Purpose: Plot training loss and LCS from the metrics CSV.

Usage:
  python 6_plot_training_curves.py --metrics_path <path_to_csv>

Output:
  - <csv_prefix>.pdf

----------------------------------------------------------------------
3.7 7_lcs_analysis.py
----------------------------------------------------------------------
Purpose: LCS scatter plots for asymmetric & symmetric models, plus a bar chart
using best validation LCS from the metrics CSVs.

Usage:
  python 7_lcs_analysis.py --data_path <file.h5ad>
        --asym_model <model.pt> --sym_model <model.pt>
        --output_dir <dir>

Output:
  - lcs_scatter_asymmetric.png
  - lcs_scatter_symmetric.png
  - lcs_comparison.png

----------------------------------------------------------------------
3.8 8_plot_marker_delta_tau_profiles.py   (optional)
----------------------------------------------------------------------
Purpose: Plot Δτ profiles for a list of marker genes (from CCF).

Usage:
  python 8_plot_marker_delta_tau_profiles.py --data_path <file.h5ad>
        --markers Gene1 Gene2 ... --output_dir <dir>

----------------------------------------------------------------------
3.9 9_gene_program_ordering.py   (optional)
----------------------------------------------------------------------
Purpose: Rank components by late Δτ, perform GO enrichment for top‑positive
(ATAC‑led) and top‑negative (RNA‑led) groups, generate bar plot.

Usage:
  python 9_gene_program_ordering.py --data_path <file.h5ad>
        --output_dir <dir> --n_top_components 10 --organism human|mouse

Output:
  - gene_program_go_comparison.pdf
  - component_ordering_by_late_lag.csv
  - go_enrichment_negative_lag.csv, go_enrichment_positive_lag.csv

----------------------------------------------------------------------
3.10 10_validate_multigene.py   (Perturb‑seq, if available)
----------------------------------------------------------------------
Purpose: Causal validation using known target genes of a perturbation.
Computes δΔτ for target‑associated components vs. background (one‑sided
Mann‑Whitney U). Requires pre‑fitted PCA/TF‑IDF/LSI pickles.

Usage:
  python 10_validate_multigene.py --data_path <file.h5ad>
        --model_path <model.pt> --control_label NTC
        --perturb_label <perturbation> --target_genes_file <file.txt>
        --output_dir <dir> --pca_path <pca.pkl> … --lsi_path <lsi.pkl>

Output:
  - validation_multigene_results.txt : p‑value, mean shifts.

----------------------------------------------------------------------
3.11 11_chipseq_validation.py   (ChIP‑seq validation)
----------------------------------------------------------------------
Purpose: Validate that genes bound by a transcription factor have a
systematically different Δτ than background genes.  The background gene set
consists of **all eligible non‑target genes** (no random subsampling), making
the test fully deterministic.  Uses a two‑sided Mann‑Whitney U test.  The TF
name is extracted from the target‑gene filename.

Usage:
  python 11_chipseq_validation.py --data_path <file.h5ad>
        --model_path <model.pt> --target_genes_file <TF_top500.txt>
        --output_dir <dir> [--alternative two‑sided]

Output:
  - chipseq_validation_<TF>_delta_tau.png (violin plot with p‑value)

================================================================================
4. TYPICAL WORKFLOW FOR A NEW DATASET
================================================================================

1. Preprocessing:
   python 1_create_gene_level_tmo_data.py --data_dir ./my_dataset
         --output_file ./my_dataset/tmo_ready.h5ad --root_marker <early_gene>

2. Annotation:
   python 2_annotate_components.py --data_path ./my_dataset/tmo_ready.h5ad
         --n_components 50 --organism <human/mouse>

3. Train asymmetric model:
   python 3_train_tmo_asymmetric.py --data_path ./my_dataset/tmo_ready.h5ad
         --epochs 50 --val_interval 5 --lambda_lag 0.1

4. Train symmetric baseline:
   python 3_train_tmo_asymmetric.py --data_path ./my_dataset/tmo_ready.h5ad
         --epochs 50 --val_interval 5 --lambda_lag 0.1 --symmetric

5. Generate figures:
   python 4_evaluate_asymmetric.py --data_path ./my_dataset/tmo_ready.h5ad
         --model_path ./tmo_results/tmo_asymmetric_best.pt
         --annotation_mode cluster_aggregate --dist_threshold 1.0 --standardize
   python 5_plot_correlation_heatmap.py --data_path ./my_dataset/tmo_ready.h5ad
   python 6_plot_training_curves.py --metrics_path ./tmo_results/training_metrics_tmo_asymmetric.csv
   python 7_lcs_analysis.py --data_path ./my_dataset/tmo_ready.h5ad
         --asym_model ./tmo_results/tmo_asymmetric_best.pt
         --sym_model ./tmo_results/tmo_symmetric_best.pt
   python 8_plot_marker_delta_tau_profiles.py --data_path ./my_dataset/tmo_ready.h5ad
         --markers GENE1 GENE2 …
   python 9_gene_program_ordering.py --data_path ./my_dataset/tmo_ready.h5ad
         --n_top_components 10 --organism <…>

6. ChIP‑seq validation (if target gene list available):
   python 11_chipseq_validation.py --data_path ./my_dataset/tmo_ready.h5ad
         --model_path ./tmo_results/tmo_asymmetric_best.pt
         --target_genes_file ./my_dataset/<TF>_top500_genes.txt

All outputs are written to `./tmo_results/`.

================================================================================
5. FINAL RESULTS ON FOUR BENCHMARK DATASETS  (fully reproducible)
================================================================================

| Dataset       | Asym LCS | Sym LCS | ChIP‑seq TF | p‑value    | Target Δτ mean | Backgr. Δτ mean |
|---------------|----------|---------|-------------|------------|----------------|-----------------|
| Human PBMC    | 0.9990   | 0.1018  | PAX5        | 3.00×10⁻²³ |  0.0146        |  0.0106         |
| Mouse brain   | 0.9920   | 0.1080  | Pax6        | 2.38×10⁻¹⁸ | −0.0206        | −0.0158         |
| Human brain   | 0.9877   | 0.0841  | ASCL1       | 1.02×10⁻³  |  0.0167        |  0.0150         |
| Mouse kidney  | 0.9984   | 0.0477  | Hnf4a       | 1.98×10⁻⁴  | −0.0056        | −0.0044         |

Key points:
- **Asymmetric LCS** (in‑set, full dataset) is >0.98 for all four tissues; the symmetric baseline (identical architecture, no attention bias, no lag loss) yields LCS <0.11 (effectively zero). This confirms that the asymmetric attention bias is essential for learning regulatory lags.
- The ChIP‑seq validation uses a **deterministic, two‑sided Mann‑Whitney U test** against the full set of eligible background genes (all non‑target genes, optionally expression‑matched).  No random subsampling is performed. All four transcription factors show highly significant differences.
- The main‑paper figures use human Mouse Kidney as the primary exemplar (high LCS and strong ChIP‑seq signal); equivalent figures for the other three datasets are provided as supplementary material.
- **Perturb‑seq causal validation:**cSMARCB1 knockout (1,144 NTC, 147 perturbed): directional trend, target δΔτ 0.0003 vs. background 0.0002, one‑sided p = 0.056. SMARCE1 knockout (25,125 NTC, 3,394 perturbed): significant shift, one‑sided p = 0.0089.
- **Generalisation (held‑out LCS):** When trained on an 80% stratified split and evaluated on the held‑out 20% using the training‑set CCF target, TMO retains high LCS: PBMC 0.9885, mouse brain 0.9484, human brain 0.8483, mouse kidney 0.9369, confirming that the learned component‑lag ordering transfers to unseen cells of the same tissue.
- - **Cell‑state ablation:** Removing cell‑state information from the LagMLP causes a consistent drop in LCS (e.g. 0.9984->0.7136 in mouse kidney) and collapses per‑component temporal dynamics (KS p < 10^-18 in three of four tissues), proving that TMO's dynamic lag patterns depend on cell‑state conditioning.
- Training time ~10‑20 min on CPU for ~3000–5000 cells.

================================================================================
6. DEPENDENCIES
================================================================================
torch, scanpy, anndata, muon, scikit‑learn, scipy, matplotlib, seaborn,
pandas, numpy, tqdm, gseapy. Install with: pip install -r requirements.txt

================================================================================
7. TROUBLESHOOTING
================================================================================
- If peak‑gene match rate is <30%, check peak name format (':' vs '_').
- Enrichr 429 errors: increase --delay or wait.
- Training loss does not decrease: ensure pseudotime is sorted and
  log1p normalized.
- LCS stays negative: verify you are using the best checkpoint and
  that pseudotime orientation is correct.
- For large datasets (>10k cells), consider fewer PCA components
  or use the provided sparse attention implementation.

================================================================================
8. OUTPUT FILE TYPES
================================================================================
.h5ad : AnnData object (HDF5)
.pt   : PyTorch model checkpoint
.csv  : Tabular data
.pdf / .png : Figures

================================================================================
9. CORE LIBRARY MODULES (for advanced users)
================================================================================
- tmo.ccf.local_ccf : sliding window CCF with interpolation
- tmo.ccf.gp_smoothing : GP smoothing
- tmo.models.tokenizer : LatentTokenizer (value projection + modality + τ)
- tmo.models.attention : AsymmetricCrossAttention with correct reverse bias
- tmo.models.lagmlp : LagMLP / WidthMLP (LayerNorm first)
- tmo.models.tmo_block : Transformer block with self‑ and cross‑attention
- tmo.models.tmo_model : TMOLatentModelAsymmetric (two‑pass, use_bias flag)
- tmo.training.losses : mse_loss
- tmo.validation.lcs : LCS computation
- tmo.plotting : heatmaps, training curves, violin plots

================================================================================
10. CITATION
================================================================================
If you use TMO, please cite:

@article{LopezDelgado2026,
  title={TMO: Asymmetric cross-modal attention for learning
         cell-state-dependent regulatory lags from single-cell
         multi-omic data},
  author={P. A. Lopez-Delgado},
  journal={Nature Methods (submitted)},
  year={2026}
}

================================================================================
11. LICENSE
================================================================================
MIT License. See LICENSE.

================================================================================
END OF DOCUMENTATION
================================================================================