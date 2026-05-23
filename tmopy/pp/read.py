"""Preprocessing: load 10x Multiome, create gene‑level ATAC, compute pseudotime.

This module provides the high‑level entry points for loading raw 10x
Multiome data, converting peak‑level ATAC counts to gene‑level
accessibility scores, and inferring pseudotime from the RNA modality.
It is the first step in any TMO analysis and prepares an AnnData object
that all downstream tools (training, evaluation, validation) expect.

Functions
---------
read_10x_multiome
    Read a filtered_feature_bc_matrix.h5 file and an atac_peak_annotation.tsv
    file, returning an AnnData with RNA in .X and ATAC peaks in .obsm['ATAC'].
compute_gene_level_atac
    Aggregate peak accessibility to gene‑level scores using a peak‑to‑gene
    mapping provided in the annotation TSV.
compute_pseudotime
    Compute pseudotime as the first diffusion component of the RNA data,
    optionally reversed so that an early marker gene increases along the
    axis.
filter_genes_and_peaks
    Select highly variable genes and the most frequently observed ATAC
    peaks to reduce dimensionality before downstream analysis.
"""

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
    """Load raw 10x Multiome data and return a single AnnData object.

    The function reads a Cell Ranger ARC ``filtered_feature_bc_matrix.h5``
    file using ``muon``, separates the RNA and ATAC modalities, filters
    cells by basic quality thresholds, and stores the ATAC peak matrix
    as a sparse array in ``.obsm['ATAC']``.  Peak names are saved in
    ``.uns['peak_names']`` so that later steps can map them to genes.

    Parameters
    ----------
    h5_file : str or Path
        Path to the ``filtered_feature_bc_matrix.h5`` file.
    atac_annotation_file : str or Path
        Path to the ``atac_peak_annotation.tsv`` file.  Its path is stored
        in ``.uns['atac_annotation_file']`` for later use.
    min_genes : int, default=200
        Minimum number of genes expressed per cell.  Cells with fewer
        genes are removed.
    min_atac_frags : int, default=1000
        Minimum number of ATAC fragments per cell.  Cells with fewer
        fragments are removed.

    Returns
    -------
    AnnData
        An object with:
        - ``.X`` : raw RNA counts (cells × genes)
        - ``.obsm['ATAC']`` : sparse ATAC peak counts (cells × peaks)
        - ``.uns['peak_names']`` : list of peak identifiers
        - ``.uns['atac_annotation_file']`` : path to the annotation TSV
    """

    h5_file = Path(h5_file)
    atac_annotation_file = Path(atac_annotation_file)

    mdata = mu.read_10x_h5(h5_file)
    adata_rna = mdata['rna'].copy()
    adata_atac = mdata['atac'].copy()

    # Basic cell filtering
    sc.pp.filter_cells(adata_rna, min_genes=min_genes)
    sc.pp.filter_cells(adata_atac, min_counts=min_atac_frags)

    # Keep only cells that pass both filters
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
    """Convert peak‑level ATAC to gene‑level accessibility.

    Peak counts stored in ``.obsm['ATAC']`` are aggregated to gene‑level
    scores by summing the counts of all peaks annotated to each gene.
    The mapping from peaks to genes is read from a tab‑separated file
    (the same ``atac_peak_annotation.tsv`` used during loading).  The
    resulting cells‑by‑genes matrix is stored in ``.obsm['ATAC_gene']``.

    Peak names in ``.uns['peak_names']`` are normalised (colons and
    hyphens replaced by underscores) to match the identifiers in the
    annotation file, making the mapping robust to slight formatting
    differences between Cell Ranger output and the TSV.

    Parameters
    ----------
    adata : AnnData
        Must contain ``.obsm['ATAC']`` (sparse cells × peaks) and
        ``.uns['peak_names']`` (list of peak identifiers).
    tsv_file : str or Path, optional
        Path to the ``atac_peak_annotation.tsv`` file.  If not provided,
        the path stored in ``.uns['atac_annotation_file']`` is used.

    Returns
    -------
    AnnData
        The same object with ``.obsm['ATAC_gene']`` added (sparse CSR matrix,
        cells × genes).
    """

    # Obtain annotation file path
    if tsv_file is None:
        tsv_file = adata.uns.get('atac_annotation_file')
    if tsv_file is None:
        raise ValueError("No annotation file provided and none found in .uns['atac_annotation_file']")
    tsv_file = Path(tsv_file)

    # Read and standardize the annotation table
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

    # Helper to normalise peak identifiers for robust matching
    def norm_peak(p):
        return p.replace(':', '_').replace('-', '_')

    # Build mapping from normalized peak name to list of gene indices
    peak_to_gene_idx = {}
    for _, row in annot.iterrows():
        peak_norm = norm_peak(row[peak_col])
        gene = row['gene']
        g_idx = gene_to_idx[gene]
        peak_to_gene_idx.setdefault(peak_norm, []).append(g_idx)

    # Get peak names from .uns['peak_names'], stored during loading
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

    # Count how many peaks have at least one gene mapped. Report matching statistics.
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
    """Compute pseudotime as the first diffusion component.

    Pseudotime is inferred from the RNA modality using diffusion maps
    after normalisation, highly‑variable gene selection, PCA, and
    nearest‑neighbour graph construction.  The first diffusion component
    provides a one‑dimensional ordering of cells that reflects the
    dominant transcriptional trajectory.

    If an early marker gene is provided (e.g., *Pax6* for neurogenesis,
    *CD34* for haematopoiesis), its expression is correlated with the
    initial pseudotime.  When the correlation is negative, the axis is
    reversed so that the marker's expression increases along pseudotime.
    The reversal decision is stored in ``.uns['pseudotime_reversed']``
    and later used to flip the sign of Δτ, ensuring that positive Δτ
    always means ATAC leads RNA (priming).

    Parameters
    ----------
    adata : AnnData
        Must contain raw RNA counts in ``.X``.  If the data are already
        log‑normalised, the normalisation step is skipped.
    root_marker : str, optional
        Gene symbol of an early developmental marker.  If provided and
        negatively correlated with the initial pseudotime, the axis is
        reversed.

    Returns
    -------
    AnnData
        The same object with:
        - ``.obs['pseudotime']`` : continuous pseudotime in [0, 1]
        - ``.uns['pseudotime_reversed']`` : boolean flag
        Cells are sorted by pseudotime.
    """

    # Normalize and log‑transform if not already done
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Select highly variable genes for trajectory inference
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.diffmap(adata, n_comps=15)

    # First diffusion component as pseudotime
    pseudotime = adata.obsm['X_diffmap'][:, 0]
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    reversed_flag = False

    # Optional reversal with a root marker
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

    # Sort cells by pseudotime; this is required for CCF computation
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
    print(f"Pseudotime range: {pseudotime.min():.3f} – {pseudotime.max():.3f}")
    return adata


def filter_genes_and_peaks(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    n_top_peaks: int = 20000,
    atac_key: str = 'ATAC',
) -> ad.AnnData:
    """Select highly variable genes and the most frequent ATAC peaks.

    This function modifies the AnnData object in place, keeping only the
    specified number of top genes and peaks.  It is an optional step that
    can reduce memory usage and focus the analysis on the most informative
    features.

    Parameters
    ----------
    adata : AnnData
        Must contain RNA counts in ``.X`` and an ATAC matrix in
        ``.obsm[atac_key]``.
    n_top_genes : int, default=2000
        Number of highly variable genes to retain.
    n_top_peaks : int, default=20000
        Number of most frequently observed peaks to retain.
    atac_key : str, default='ATAC'
        Key in ``.obsm`` where the ATAC matrix is stored.

    Returns
    -------
    AnnData
        The filtered object (modified in place).
    """

    # RNA: select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    adata.var['highly_variable'] = adata.var.get('highly_variable', False)
    adata._inplace_subset_var(adata.var['highly_variable'])

    # ATAC: keep the most frequently observed peaks
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