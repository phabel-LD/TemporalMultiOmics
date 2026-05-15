"""Preprocessing: load 10x Multiome, create gene‑level ATAC, compute pseudotime."""

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import muon as mu
from scipy.sparse import lil_matrix, csr_matrix, issparse
from pathlib import Path
from typing import Optional, Union
import warnings


def read_10x_multiome(
    h5_file: Union[str, Path],
    atac_annotation_file: Union[str, Path],
    min_genes: int = 200,
    min_atac_frags: int = 1000,
) -> ad.AnnData:
    """
    Load raw 10x Multiome data and return an AnnData object.

    RNA counts are stored in `.X`, ATAC peak matrix in `.obsm['ATAC']`.
    Peak names are stored in `.uns['peak_names']` and the annotation file path
    in `.uns['atac_annotation_file']`.

    Parameters
    ----------
    h5_file : str or Path
        Path to filtered_feature_bc_matrix.h5.
    atac_annotation_file : str or Path
        Path to atac_peak_annotation.tsv.
    min_genes, min_atac_frags : int, default 200, 1000
        Filtering thresholds.

    Returns
    -------
    AnnData
        Combined object.
    """
    h5_file = Path(h5_file)
    atac_annotation_file = Path(atac_annotation_file)

    mdata = mu.read_10x_h5(h5_file)
    adata_rna = mdata['rna'].copy()
    adata_atac = mdata['atac'].copy()

    sc.pp.filter_cells(adata_rna, min_genes=min_genes)
    sc.pp.filter_cells(adata_atac, min_counts=min_atac_frags)

    common = adata_rna.obs_names.intersection(adata_atac.obs_names)
    adata_rna = adata_rna[common, :]
    adata_atac = adata_atac[common, :]
    adata_atac = adata_atac[adata_rna.obs_names, :]

    adata_rna.obsm['ATAC'] = adata_atac.X
    adata_rna.uns['atac_annotation_file'] = str(atac_annotation_file)
    adata_rna.uns['peak_names'] = adata_atac.var_names.tolist()
    return adata_rna


def compute_gene_level_atac(
    adata: ad.AnnData,
    tsv_file: Optional[Union[str, Path]] = None,
) -> ad.AnnData:
    """
    Convert peak-level ATAC (`.obsm['ATAC']`) to gene-level accessibility
    (`.obsm['ATAC_gene']`) using the peak annotation TSV.

    The function expects that `.uns['peak_names']` contains the peak identifiers
    in the same order as the columns of `.obsm['ATAC']`. These are stored during
    `read_10x_multiome`. The annotation TSV is read from `.uns['atac_annotation_file']`
    or provided as `tsv_file`.

    Parameters
    ----------
    adata : AnnData
        Must contain `.obsm['ATAC']` and `.uns['peak_names']`.
    tsv_file : str or Path, optional
        Path to atac_peak_annotation.tsv. If not given, uses the path stored in
        `.uns['atac_annotation_file']`.

    Returns
    -------
    AnnData
        Same object with `.obsm['ATAC_gene']` added.
    """
    if tsv_file is None:
        tsv_file = adata.uns.get('atac_annotation_file')
    if tsv_file is None:
        raise ValueError("No annotation file provided and none found in .uns['atac_annotation_file']")
    tsv_file = Path(tsv_file)

    annot = pd.read_csv(tsv_file, sep='\t')
    if 'peak' in annot.columns:
        peak_col = 'peak'
    elif 'chrom' in annot.columns and 'start' in annot.columns and 'end' in annot.columns:
        annot['peak'] = annot['chrom'] + ':' + annot['start'].astype(str) + '-' + annot['end'].astype(str)
        peak_col = 'peak'
    else:
        raise ValueError("Annotation file must have a 'peak' column or chrom/start/end.")

    annot = annot.dropna(subset=['gene'])
    all_genes = sorted(set(annot['gene']))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    def norm_peak(p):
        # Convert peak identifier from MuData (chr:start-end) to annotation format (chr_start_end)
        # or vice versa. The annotation TSV uses 'chr:start-end'? Actually, we need to match.
        # In the original script, the annotation file had a 'peak' column with 'chr:start-end'.
        # The MuData `peak_names` are also 'chr:start-end'. So no normalisation needed.
        # But the original script used `norm_peak` to replace ':' and '-' with '_' because
        # the annotation file sometimes used underscores. To be robust, we keep the normalisation.
        # However, if your annotation file already uses 'chr:start-end', you can skip this.
        # We'll keep the original logic.
        return p.replace(':', '_').replace('-', '_')

    # Build mapping from normalised peak name to list of gene indices
    peak_to_gene_idx = {}
    for _, row in annot.iterrows():
        peak_norm = norm_peak(row[peak_col])
        gene = row['gene']
        g_idx = gene_to_idx[gene]
        peak_to_gene_idx.setdefault(peak_norm, []).append(g_idx)

    # Get peak names from .uns['peak_names']
    if 'peak_names' not in adata.uns:
        raise ValueError("Peak names not found in .uns['peak_names']. Please use read_10x_multiome first.")
    peak_names = adata.uns['peak_names']
    n_peaks = len(peak_names)
    n_genes = len(all_genes)

    # Build binary matrix M (peaks x genes) in LIL format
    print("Building binary peak‑gene matrix...")
    M = lil_matrix((n_peaks, n_genes), dtype=np.float32)
    for i, peak_name in enumerate(peak_names):
        peak_norm = norm_peak(peak_name)
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

    # Compute gene-activity matrix: cells x genes = (cells x peaks) @ (peaks x genes)
    print("Computing gene‑activity matrix...")
    peak_matrix = adata.obsm['ATAC']  # sparse cells x peaks
    gene_activity = peak_matrix @ M   # cells x genes
    adata.obsm['ATAC_gene'] = csr_matrix(gene_activity)
    print(f"Gene‑activity shape: {adata.obsm['ATAC_gene'].shape}, non‑zeros: {adata.obsm['ATAC_gene'].nnz}")
    return adata


def compute_pseudotime(
    adata: ad.AnnData,
    root_marker: Optional[str] = None,
) -> ad.AnnData:
    """
    Compute pseudotime as first diffusion component, optionally reversed, and sort cells.

    Parameters
    ----------
    adata : AnnData
        RNA counts in `.X` (raw – will be normalised internally).
    root_marker : str, optional
        Gene symbol of an early marker; if provided and its expression correlates negatively
        with the initial pseudotime, the axis is reversed.

    Returns
    -------
    AnnData
        Same object with `.obs['pseudotime']`, `.uns['pseudotime_reversed']`, and cells sorted.
    """
    # Normalise if not already
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.diffmap(adata, n_comps=15)
    pseudotime = adata.obsm['X_diffmap'][:, 0]
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    reversed_flag = False

    if root_marker and root_marker in adata.var_names:
        idx = list(adata.var_names).index(root_marker)
        expr = adata.X[:, idx]
        if issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()
        corr = np.corrcoef(expr, pseudotime)[0, 1]
        if corr < 0:
            pseudotime = 1 - pseudotime
            reversed_flag = True
            print(f"Reversed pseudotime so that {root_marker} increases with pseudotime.")
        else:
            print(f"Marker {root_marker} correlates positively (r={corr:.3f}); keeping orientation.")
    adata.obs['pseudotime'] = pseudotime
    adata.uns['pseudotime_reversed'] = reversed_flag
    # Sort cells by pseudotime (required for CCF)
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
    print(f"Pseudotime range: {pseudotime.min():.3f} – {pseudotime.max():.3f}")
    return adata


def filter_genes_and_peaks(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    n_top_peaks: int = 20000,
    atac_key: str = 'ATAC',
) -> ad.AnnData:
    """Select highly variable genes and most frequent peaks (modifies in place)."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    adata.var['highly_variable'] = adata.var.get('highly_variable', False)
    adata._inplace_subset_var(adata.var['highly_variable'])

    X = adata.obsm[atac_key]
    if issparse(X):
        total = np.asarray(X.sum(axis=0)).ravel()
    else:
        total = X.sum(axis=0)
    top_idx = np.argsort(-total)[:n_top_peaks]
    adata.obsm[atac_key] = X[:, top_idx]
    if 'peak_names' in adata.uns:
        adata.uns['peak_names'] = [adata.uns['peak_names'][i] for i in top_idx]
    return adata