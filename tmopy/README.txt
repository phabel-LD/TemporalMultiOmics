================================================================================
tmopy – High‑level Python API for TMO  (final version, May 2026)
================================================================================

tmopy provides a Scanpy‑like interface to the TMO (Temporal Multi‑Omics)
framework for single‑cell ATAC+RNA multi‑omics.  It operates on AnnData objects
and offers intuitive functions for preprocessing, training, evaluation, and
validation.  Every function reproduces the command‑line pipeline exactly.

================================================================================
1. Installation
================================================================================

# Clone the repository
git clone https://github.com/phabel-LD/TemporalMultiOmics.git
cd tmo
conda env create -f environment.yaml
conda activate tmo_env
pip install -e .

After installation, import as:

import tmopy as tmo

================================================================================
2. Package structure
================================================================================

tmopy/
├── __init__.py
├── __version__.py
├── pp/                      # preprocessing
│   ├── __init__.py
│   ├── read.py
│   └── utils.py
├── tl/                      # tools (training, evaluation, validation)
│   ├── __init__.py
│   ├── annotation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validation.py
│   ├── gene_program_ordering.py
│   └── lcs_analysis.py
└── pl/                      # plotting
    ├── __init__.py
    ├── heatmap.py
    ├── correlation.py
    ├── training_curves.py
    ├── violin.py
    └── marker_profiles.py

================================================================================
3. Preprocessing (tmopy.pp)
================================================================================

3.1  tmopy.pp.read_10x_multiome(h5_file, atac_annotation_file,
         min_genes=200, min_atac_frags=1000)
     - Loads raw 10x Multiome data (filtered_feature_bc_matrix.h5 and
       atac_peak_annotation.tsv).
     - Returns an AnnData object with:
         .X              : raw RNA counts
         .obsm['ATAC']   : sparse ATAC peak matrix (cells × peaks)
         .uns['atac_annotation_file'] : path to annotation file
     - Filters cells by min_genes and min_atac_frags.

3.2  tmopy.pp.compute_gene_level_atac(adata, tsv_file=None)
     - Converts peak‑level ATAC to gene‑level accessibility stored in
       .obsm['ATAC_gene'] (cells × genes), using the peak annotation.

3.3  tmopy.pp.compute_pseudotime(adata, root_marker=None)
     - Computes pseudotime as the first diffusion component.
     - If a root_marker is given, checks its correlation with pseudotime;
       if negative, the axis is reversed and .uns['pseudotime_reversed']
       is set to True.

3.4  tmopy.pp.filter_genes_and_peaks(adata, n_top_genes=2000,
         n_top_peaks=20000, atac_key='ATAC')
     - Selects highly variable genes and the most frequent ATAC peaks.

================================================================================
4. Tools (tmopy.tl)
================================================================================

4.1  tmopy.tl.annotate_components(adata, n_components=50,
         n_top_genes=100, n_top_go=5, delay=2.0, organism='human',
         save_csv=None)
     - PCA on RNA, extracts top genes per component, runs GO enrichment
       (Enrichr).  Stores the result in .uns['component_annotation'].

4.2  tmopy.tl.train_asymmetric(adata, epochs=50, val_interval=5,
         lambda_lag=0.1, batch_size=32, lr=1e-3, device='cpu',
         save_model=None, output_dir='./tmo_results')
     - Trains the asymmetric TMO model on the full dataset.
     - Saves the best checkpoint (by in‑set LCS), training metrics CSV,
       and the PCA/TF‑IDF/LSI transformers (pca.pkl, tfidf.pkl, lsi.pkl).
     - Returns the trained model.

4.3  tmopy.tl.train_symmetric(adata, epochs=50, val_interval=5,
         batch_size=32, lr=1e-3, device='cpu', save_model=None,
         output_dir='./tmo_results')
     - Trains the symmetric baseline (identical architecture, no attention
       bias, no lag loss).  Still saves the pickles and metrics CSV.

4.4  tmopy.tl.evaluate_asymmetric(adata, model_path, n_components=50,
         cluster_aggregate=False, dist_threshold=1.0, standardize=True)
     - Computes the CCF‑based Δτ surface, applies sign‑flip if needed.
     - If cluster_aggregate=True, returns cluster‑level medians and labels
       (matching the command‑line Figure 1).
     - Returns (delta_tau_matrix, pseudotime_grid, LCS) or
       (cluster_medians, pseudotime_grid, LCS, row_labels).

4.5  tmopy.tl.validate_perturbseq(adata, model_path, control_label,
         perturb_label, target_genes, background_genes=None,
         match_expression=True, n_components=50,
         perturbation_column='guide_target', save_plot=None,
         pca=None, tfidf=None, lsi=None)
     - Causal validation using δΔτ.  Optionally loads pre‑fitted
       transformers for a consistent latent space.  Returns dict with
       p_value, target_shifts, background_shifts, mean_target,
       mean_background.

4.6  tmopy.tl.validate_chipseq(adata, model_path, target_genes_file,
         background_genes_file=None, match_expression=False,
         n_components=50, alternative='two-sided', save_plot=None,
         pca=None, tfidf=None, lsi=None)
     - ChIP‑seq validation using CCF‑derived gene‑level Δτ.
     - The background set consists of **all eligible non‑target genes**
       (optionally expression‑matched); no random subsampling is performed,
       making the test fully deterministic.
     - Returns dict with p_value, target_lags, background_lags,
       mean_target, mean_background.

4.7  tmopy.tl.gene_program_ordering(adata, n_top_components=10,
         organism='human', save_plots=None)
     - Ranks components by late Δτ, performs GO enrichment for the most
       positive and most negative groups, and generates a bar plot.

4.8  tmopy.tl.lcs_analysis(adata, asym_model_path, sym_model_path=None,
         n_components=50, save_plots=None)
     - Produces scatter plots of predicted vs CCF lags and a bar chart
       comparing the two models.

================================================================================
5. Plotting (tmopy.pl)
================================================================================

5.1  tmopy.pl.plot_delta_tau(delta_tau_matrix, pseudotime_grid,
         row_labels, title="Δτ heatmap", save=None, show=True, **kwargs)
     - Heatmap of Δτ values (positive = ATAC‑led, red).

5.2  tmopy.pl.plot_correlation(adata, n_bins=10, save=None, show=True)
     - ATAC‑RNA correlation heatmap (U‑shape, Figure 2).

5.3  tmopy.pl.plot_training_curves(metrics_csv, save=None, show=True)
     - Training loss and LCS from the metrics CSV.

5.4  tmopy.pl.plot_violin_validation(target_values, background_values,
         target_label="Target", background_label="Background",
         title="Validation", ylabel="Δτ shift / ADS", save=None, show=True)
     - Violin plot for two‑group comparisons.

5.5  tmopy.pl.plot_marker_profiles(adata, marker_genes, save=None,
         show=True)
     - Δτ profiles of specific marker genes (from the CCF surface).

================================================================================
6. Typical usage example (mirrors the validated pipeline)
================================================================================

import tmopy as tmo

# 1. Load and preprocess
adata = tmo.pp.read_10x_multiome("filtered_feature_bc_matrix.h5",
           "atac_peak_annotation.tsv")
adata = tmo.pp.compute_gene_level_atac(adata)
adata = tmo.pp.compute_pseudotime(adata, root_marker="Pax6")

# 2. Annotate components (requires internet)
tmo.tl.annotate_components(adata, n_components=50, organism='mouse')

# 3. Train asymmetric model (saves best model + pickles)
model = tmo.tl.train_asymmetric(adata, epochs=50, val_interval=5,
            lambda_lag=0.1, save_model="best_model.pt")

# 4. Train symmetric baseline (optional)
sym_model = tmo.tl.train_symmetric(adata, epochs=50, val_interval=5,
                save_model="symmetric_best.pt")

# 5. Evaluate and plot (clustered heatmap)
delta_tau, pt_grid, lcs, labels = tmo.tl.evaluate_asymmetric(
    adata, "best_model.pt", cluster_aggregate=True, dist_threshold=1.0,
    standardize=True)
tmo.pl.plot_delta_tau(delta_tau, pt_grid, labels,
     save="fig1_delta_tau.pdf")
tmo.pl.plot_correlation(adata, save="fig2_correlation.pdf")

# 6. ChIP‑seq validation (deterministic, full background)
ch_res = tmo.tl.validate_chipseq(adata, "best_model.pt",
     "PAX5_top500_genes.txt", match_expression=True, alternative='two-sided')
tmo.pl.plot_violin_validation(ch_res['target_lags'],
     ch_res['background_lags'],
     title=f"PAX5 ChIP‑seq (p={ch_res['p_value']:.2e})",
     save="fig4_chipseq.pdf")

================================================================================
7. Final results (four benchmark datasets)
================================================================================

| Dataset       | Asym LCS | Sym LCS | ChIP‑seq TF | p‑value    |
|---------------|----------|---------|-------------|------------|
| Human PBMC    | 0.9990   | 0.1018  | PAX5        | 3.00×10⁻²³ |
| Mouse brain   | 0.9920   | 0.1080  | Pax6        | 2.38×10⁻¹⁸ |
| Human brain   | 0.9877   | 0.0841  | ASCL1       | 1.02×10⁻³  |
| Mouse kidney  | 0.9984   | 0.0477  | Hnf4a       | 1.98×10⁻⁴  |

All asymmetric LCS values are > 0.98, while the symmetric baseline remains
near zero (< 0.11).  ChIP‑seq validations are fully deterministic (full
background, two‑sided Mann‑Whitney U) and all highly significant.

**Perturb‑seq causal validation:** SMARCB1 knockout (1,144 NTC, 147 perturbed):
directional trend, target δΔτ 0.0003 vs. background 0.0002,
one‑sided p = 0.056. SMARCE1 knockout (25,125 NTC, 3,394 perturbed):
significant shift, one‑sided p = 0.0089.

- **Generalization (held‑out LCS):** When trained on an 80% stratified split
and evaluated on the held‑out 20% using the training‑set CCF target,
TMO retains high LCS: PBMC 0.9885, mouse brain 0.9484, human brain 0.8483,
mouse kidney 0.9369, confirming that the learned component‑lag ordering
transfers to unseen cells of the same tissue.

================================================================================
8. Dependencies
================================================================================

torch, scanpy, anndata, muon, scikit‑learn, scipy, matplotlib, seaborn,
pandas, numpy, tqdm, gseapy.  See environment.yaml for exact versions.

================================================================================
9. Citing tmopy / TMO
================================================================================

@article{LopezDelgado2026,
  title={TMO: Asymmetric cross-modal attention for learning
         cell-state-dependent regulatory lags from single-cell
         multi-omic data},
  author={Lopez-Delgado, P. A.},
  journal={Nature Methods (submitted)},
  year={2026}
}

================================================================================
10. License
================================================================================

MIT License. See LICENSE.
