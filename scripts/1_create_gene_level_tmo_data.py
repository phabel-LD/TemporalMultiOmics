#!/usr/bin/env python
"""
Create TMO-ready AnnData with gene-level ATAC matrix from 10x Multiome data.
Pseudotime is derived from the first diffusion component (no DPT, no root required).
Optionally, a marker gene can be used to reverse the direction.
"""

import argparse
import muon as mu
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing filtered_feature_bc_matrix.h5 and atac_peak_annotation.tsv")
    parser.add_argument("--output_file", type=str, default="gene_level_tmo_ready.h5ad",
                        help="Output filename")
    parser.add_argument("--root_marker", type=str, default=None,
                        help="Gene name to determine direction (e.g., CD34 for PBMC). If provided and the marker correlates negatively, pseudotime will be reversed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {data_dir}")
    h5_file = h5_files[0]
    tsv_files = list(data_dir.glob("*atac_peak_annotation.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No peak annotation TSV found in {data_dir}")
    annot_file = tsv_files[0]

    print(f"Using H5 file: {h5_file}")
    print(f"Using annotation file: {annot_file}")

    # 1. Load MuData
    print("Loading MuData...")
    mdata = mu.read_10x_h5(h5_file)
    print(mdata)

    # 2. Build gene-activity matrix
    print("Loading peak annotation...")
    annot = pd.read_csv(annot_file, sep='\t')
    print("Columns in annotation file:", annot.columns.tolist())

    if 'peak' in annot.columns:
        peak_col = 'peak'
    elif 'chrom' in annot.columns and 'start' in annot.columns and 'end' in annot.columns:
        annot['peak'] = annot['chrom'] + ':' + annot['start'].astype(str) + '-' + annot['end'].astype(str)
        peak_col = 'peak'
    else:
        raise ValueError("Annotation file must have a 'peak' column or 'chrom','start','end' columns.")

    if 'gene' not in annot.columns:
        raise ValueError("Annotation file does not contain a 'gene' column.")
    annot = annot.dropna(subset=['gene'])
    print(f"Annotation rows with gene: {len(annot)}")

    all_genes = sorted(set(annot['gene']))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n_genes = len(all_genes)
    print(f"Unique genes: {n_genes}")

    def norm_peak(p):
        return p.replace(':', '_').replace('-', '_')

    peak_to_gene_idx = {}
    for _, row in annot.iterrows():
        peak_norm = norm_peak(row[peak_col])
        gene = row['gene']
        g_idx = gene_to_idx[gene]
        if peak_norm not in peak_to_gene_idx:
            peak_to_gene_idx[peak_norm] = []
        peak_to_gene_idx[peak_norm].append(g_idx)

    print("Building binary peak‑gene matrix...")
    peak_names = mdata['atac'].var_names
    n_peaks = len(peak_names)
    M = lil_matrix((n_peaks, n_genes), dtype=np.float32)
    for i, peak in enumerate(peak_names):
        peak_norm = norm_peak(peak)
        g_idxs = peak_to_gene_idx.get(peak_norm, [])
        for g_idx in g_idxs:
            M[i, g_idx] = 1
    M = M.tocsr()
    print(f"M shape: {M.shape}, non‑zeros: {M.nnz}")

    # Count how many peaks have at least one gene mapped
    peak_has_gene = np.asarray(M.sum(axis=1)).flatten()
    matched_peaks = (peak_has_gene > 0).sum()
    print(f"Peaks matched to genes: {matched_peaks}/{n_peaks} ({100*matched_peaks/n_peaks:.1f}%)")
    if matched_peaks == 0:
        raise ValueError("No peaks matched to any gene. Check peak name format in annotation file.")
    elif matched_peaks / n_peaks < 0.3:
        print("Warning: Low peak-to-gene match rate (<30%). Check peak name formatting (':' vs '_' etc.).")

    print("Computing gene‑activity matrix...")
    peak_matrix = mdata['atac'].X
    gene_activity = peak_matrix @ M
    print(f"Gene‑activity shape: {gene_activity.shape}, non‑zeros: {gene_activity.nnz}")

    # 3. Pseudotime from first diffusion component
    print("Computing pseudotime from diffusion map...")
    rna_raw = mdata['rna'].copy()
    tmp = ad.AnnData(X=rna_raw.X.copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000, flavor='seurat')
    sc.tl.pca(tmp, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(tmp)
    sc.tl.diffmap(tmp, n_comps=15)
    pseudotime = tmp.obsm['X_diffmap'][:, 0]
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

    was_reversed = False
    if args.root_marker and args.root_marker in rna_raw.var_names:
        marker_idx = list(rna_raw.var_names).index(args.root_marker)
        marker_expr = rna_raw.X[:, marker_idx]
        if hasattr(marker_expr, 'toarray'):
            marker_expr = marker_expr.toarray().flatten()
        else:
            marker_expr = marker_expr.flatten()
        corr = np.corrcoef(marker_expr, pseudotime)[0,1]
        if corr < 0:
            pseudotime = 1 - pseudotime
            was_reversed = True
            print(f"Reversed pseudotime so that {args.root_marker} increases with pseudotime.")
        else:
            print(f"Marker {args.root_marker} correlates positively (r={corr:.3f}); keeping orientation.")
    print(f"Pseudotime range: {pseudotime.min():.3f} – {pseudotime.max():.3f}")

    # 4. Create final AnnData
    print("Creating final AnnData...")
    adata_tmo = ad.AnnData(
        X=rna_raw.X,
        obs=pd.DataFrame(index=rna_raw.obs_names),
        var=rna_raw.var,
    )
    adata_tmo.obs['pseudotime'] = pseudotime
    adata_tmo.obsm['ATAC_gene'] = gene_activity
    adata_tmo.uns['pseudotime_reversed'] = was_reversed   # store flag

    # 5. Save
    adata_tmo.write(args.output_file, compression='gzip')
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()