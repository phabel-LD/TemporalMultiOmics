#!/bin/bash
# =============================================================================
# TMO PIPELINE – corrected version
# Run this script from the `scripts/` directory.
# Adjust DATASET_DIR and OUTPUT_DIR as needed.
# =============================================================================

# Environment
conda activate tmo_env

# Paths – edit these for each dataset
DATASET_DIR="../data/10x_mouse_brain"               # folder containing H5 + annotation TSV
H5AD_FILE="${DATASET_DIR}/mouse_brain_gene_level_tmo_ready.h5ad"
OUTPUT_DIR="../results/tmo_results_mouse_brain_cmmd"

# Variables
ORGANISM="mouse"
ROOT_MARKER="Pax6"
MARKER_GENES="Pax6 Tubb3 Sox2"
TF_TopGenes="Pax6_top500_genes.txt"
CONTROL_LABEL="NTC"
PERTURB_LABEL="SMARCB1"
PERTURB_TARGET_GENES="target_genes.txt"

# Create Output Directory
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 1. Create gene‑level ATAC matrix & pseudotime
# ----------------------------------------------------------------------
echo "Step 1: Preprocessing..."
python 1_create_gene_level_tmo_data.py \
    --data_dir "${DATASET_DIR}" \
    --output_file "${H5AD_FILE}" \
    --root_marker "${ROOT_MARKER}"              # change per dataset (e.g. CD34, Pax2, ...)

# ----------------------------------------------------------------------
# 2. Annotate components with GO terms (internet required)
# ----------------------------------------------------------------------
echo "Step 2: Component annotation..."
python 2_annotate_components.py \
    --data_path "${H5AD_FILE}" \
    --n_components 50 \
    --n_top_genes 100 \
    --n_top_go 5 \
    --delay 2.0 \
    --output_dir "${OUTPUT_DIR}" \
    --organism "${ORGANISM}"

# ----------------------------------------------------------------------
# 3. Train asymmetric model (saves best checkpoint + PCA/LSI pickles)
# ----------------------------------------------------------------------
echo "Step 3: Training asymmetric model..."
python 3_train_tmo_asymmetric.py \
    --data_path "${H5AD_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 50 \
    --val_interval 5 \
    --lambda_lag 0.1

# ----------------------------------------------------------------------
# 4. Train symmetric baseline (identical architecture, no bias)
# ----------------------------------------------------------------------
echo "Step 4: Training symmetric baseline..."
python 3_train_tmo_symmetric.py \
    --data_path "${H5AD_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 50 \
    --val_interval 5 \
    --lambda_lag 0.1

# ----------------------------------------------------------------------
# 5. Evaluate asymmetric model → Δτ heatmap (Figure 1)
# ----------------------------------------------------------------------
echo "Step 5: Asymmetric evaluation..."
python 4_evaluate_asymmetric.py \
    --data_path "${H5AD_FILE}" \
    --model_path "${OUTPUT_DIR}/tmo_asymmetric_best.pt" \
    --output_dir "${OUTPUT_DIR}" \
    --annotation_mode cluster_aggregate \
    --dist_threshold 1.0 \
    --standardize

# ----------------------------------------------------------------------
# 6. ATAC‑RNA correlation heatmap (Figure 2)
# ----------------------------------------------------------------------
echo "Step 6: Correlation heatmap..."
python 5_plot_correlation_heatmap.py \
    --data_path "${H5AD_FILE}" \
    --output_dir "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 7. Training curves (Figure 3)
# ----------------------------------------------------------------------
echo "Step 7: Training curves..."
python 6_plot_training_curves.py \
    --metrics_path "${OUTPUT_DIR}/training_metrics.csv"

# ----------------------------------------------------------------------
# 8. LCS comparison (scatter plots + bar chart)
# ----------------------------------------------------------------------
echo "Step 8: LCS analysis..."
python 7_lcs_analysis.py \
    --data_path "${H5AD_FILE}" \
    --asym_model "${OUTPUT_DIR}/tmo_asymmetric_best.pt" \
    --sym_model "${OUTPUT_DIR}/tmo_symmetric_best.pt" \
    --output_dir "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 9. (Optional) Marker gene Δτ profiles
#    Choose markers appropriate for your tissue.
# ----------------------------------------------------------------------
echo "Step 9: Marker profiles (example for mouse brain)..."
python 8_plot_marker_delta_tau_profiles.py \
    --data_path "${H5AD_FILE}" \
    --markers "${MARKER_GENES}" \
    --output_dir "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 10. (Optional) Gene program ordering (GO enrichment of ATAC‑led vs RNA‑led)
# ----------------------------------------------------------------------
echo "Step 10: Gene program ordering..."
python 9_gene_program_ordering.py \
    --data_path "${H5AD_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_top_components 10 \
    --organism "${ORGANISM}"                  # change to 'human' for human datasets

# ----------------------------------------------------------------------
# 11. (Optional) Perturb‑seq validation
#     Requires the multi‑perturbation AnnData and a target gene list.
#     The pickle files saved during training (Step 3) are used here.
# ----------------------------------------------------------------------
echo "Step 11: Perturb‑seq validation..."
python 10_validate_multigene.py \
    --data_path "${DATASET_DIR}/multip_seq_combined.h5ad" \
    --model_path "${OUTPUT_DIR}/tmo_asymmetric_best.pt" \
    --control_label "${CONTROL_LABEL}" \
    --perturb_label "${PERTURB_LABEL}" \
    --target_genes_file "${DATASET_DIR}/${PERTURB_TARGET_GENES}" \
    --output_dir "${OUTPUT_DIR}" \
    --pca_path "${OUTPUT_DIR}/pca.pkl" \
    --tfidf_path "${OUTPUT_DIR}/tfidf.pkl" \
    --lsi_path "${OUTPUT_DIR}/lsi.pkl"

# ----------------------------------------------------------------------
# 12. ChIP‑seq validation (top 500 target genes)
# ----------------------------------------------------------------------
echo "Step 12: ChIP‑seq validation..."
python 11_chipseq_validation.py \
    --data_path "${H5AD_FILE}" \
    --model_path "${OUTPUT_DIR}/tmo_asymmetric_best.pt" \
    --target_genes_file "${DATASET_DIR}/${TF_TopGenes}" \
    --output_dir "${OUTPUT_DIR}"

echo "Pipeline complete. Outputs are in ${OUTPUT_DIR}/"

# In terminal, inside the project folder:
#chmod +x commands.sh
#conda activate tmo_env
# Edit commands.sh to set DATASET_DIR
#./commands.sh