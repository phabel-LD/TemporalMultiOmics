#============================================================================
# TMO Pipeline – Mouse Brain (line‑by‑line for Anaconda Prompt)
# Run these from the scripts/ folder.
#============================================================================

cd C:\Users\phabe\OneDrive\Escritorio\PhD_MUDS\TemporalMultiOmics\scripts
conda activate tmo_env

# ----------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------
python 1_create_gene_level_tmo_data.py --data_dir "../data/10x_mouse_brain" --output_file "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --root_marker Pax6

# ----------------------------------------------------------------------
# 2. Component annotation
# ----------------------------------------------------------------------
python 2_annotate_components.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --n_components 50 --n_top_genes 100 --n_top_go 5 --delay 2.0 --output_dir "../results/tmo_results_mouse_brain_cmmd" --organism mouse

# ----------------------------------------------------------------------
# 3. Train asymmetric model
# ----------------------------------------------------------------------
python 3_train_tmo_asymmetric.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --output_dir "../results/tmo_results_mouse_brain_cmmd" --epochs 50 --val_interval 5 --lambda_lag 0.1

# ----------------------------------------------------------------------
# 4. Train symmetric baseline
# ----------------------------------------------------------------------
python 3_train_tmo_asymmetric.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --output_dir "../results/tmo_results_mouse_brain_cmmd" --epochs 50 --val_interval 5 --lambda_lag 0.1 --symmetric

# ----------------------------------------------------------------------
# 5. Δτ heatmap (asymmetric)
# ----------------------------------------------------------------------
python 4_evaluate_asymmetric.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --model_path "../results/tmo_results_mouse_brain_cmmd/tmo_asymmetric_best.pt" --output_dir "../results/tmo_results_mouse_brain_cmmd" --annotation_mode cluster_aggregate --dist_threshold 1.0 --standardize

# ----------------------------------------------------------------------
# 6. Correlation heatmap
# ----------------------------------------------------------------------
python 5_plot_correlation_heatmap.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --output_dir "../results/tmo_results_mouse_brain_cmmd"

# ----------------------------------------------------------------------
# 7. Training curves
# ----------------------------------------------------------------------
python 6_plot_training_curves.py --metrics_path "../results/tmo_results_mouse_brain_cmmd/training_metrics_tmo_asymmetric.csv"

# ----------------------------------------------------------------------
# 8. LCS analysis
# ----------------------------------------------------------------------
python 7_lcs_analysis.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --asym_model "../results/tmo_results_mouse_brain_cmmd/tmo_asymmetric_best.pt" --sym_model "../results/tmo_results_mouse_brain_cmmd/tmo_symmetric_best.pt" --output_dir "../results/tmo_results_mouse_brain_cmmd"

# ----------------------------------------------------------------------
# 9. Marker profiles
# ----------------------------------------------------------------------
python 8_plot_marker_delta_tau_profiles.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --markers Pax6 Tubb3 Sox2 --output_dir "../results/tmo_results_mouse_brain_cmmd"

# ----------------------------------------------------------------------
# 10. Gene program ordering
# ----------------------------------------------------------------------
python 9_gene_program_ordering.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --output_dir "../results/tmo_results_mouse_brain_cmmd" --n_top_components 10 --organism mouse

# ----------------------------------------------------------------------
# 11. Perturb‑seq validation  (only if you have the data)
# ----------------------------------------------------------------------
python 10_validate_multigene.py --data_path "../data/10x_mouse_brain/multip_seq_combined.h5ad" --model_path "../results/tmo_results_mouse_brain_cmmd/tmo_asymmetric_best.pt" --control_label NTC --perturb_label SMARCB1 --target_genes_file "../data/10x_mouse_brain/target_genes.txt" --output_dir "../results/tmo_results_mouse_brain_cmmd" --pca_path "../results/tmo_results_mouse_brain_cmmd/pca.pkl" --tfidf_path "../results/tmo_results_mouse_brain_cmmd/tfidf.pkl" --lsi_path "../results/tmo_results_mouse_brain_cmmd/lsi.pkl"

# ----------------------------------------------------------------------
# 12. ChIP‑seq validation (top 500 Pax6 targets)
# ----------------------------------------------------------------------
python 11_chipseq_validation.py --data_path "../data/10x_mouse_brain/mouse_brain_gene_level_tmo_ready.h5ad" --model_path "../results/tmo_results_mouse_brain_cmmd/tmo_asymmetric_best.pt" --target_genes_file "../data/10x_mouse_brain/Pax6_top500_genes.txt" --output_dir "../results/tmo_results_mouse_brain_cmmd"