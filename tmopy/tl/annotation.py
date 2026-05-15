"""Component annotation using GO enrichment (Enrichr)."""

import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
import time
from tqdm import tqdm


def annotate_components(
    adata,
    n_components: int = 50,
    n_top_genes: int = 100,
    n_top_go: int = 5,
    delay: float = 2.0,
    organism: str = 'human',
    save_csv: str = None,
) -> pd.DataFrame:
    """
    Perform PCA on RNA, extract top genes per component, run GO enrichment.

    Parameters
    ----------
    adata : AnnData
        RNA counts (log‑normalised recommended). Will normalise if not already.
    n_components : int, default=50
        Number of PCA components.
    n_top_genes : int, default=100
        Number of top genes (by absolute loading) per component.
    n_top_go : int, default=5
        Number of top GO terms to store.
    delay : float, default=2.0
        Seconds to wait between Enrichr requests to avoid 429 errors.
    organism : str, default='human'
        'human', 'mouse', etc.
    save_csv : str, optional
        If provided, save the annotation table to this CSV file.

    Returns
    -------
    pd.DataFrame
        Table with columns Component, Top_genes, GO_terms.
    """
    # Ensure log‑normalised
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    # PCA (use all genes, not just HVG)
    sc.tl.pca(adata, n_comps=n_components, use_highly_variable=False)
    loadings = adata.varm['PCs']  # shape (n_genes, n_components) in scanpy
    # Note: scanpy stores PCs as (n_obs, n_comps) in .obsm and loadings as (n_vars, n_comps) in .varm
    # For each component, take top genes by absolute loading
    component_genes = {}
    for comp in range(n_components):
        abs_load = np.abs(loadings[:, comp])
        top_idx = np.argsort(-abs_load)[:n_top_genes]
        top_genes = adata.var_names[top_idx].tolist()
        component_genes[f"Comp{comp}"] = top_genes

    # GO enrichment
    go_results = {}
    print(f"Running GO enrichment via Enrichr (delay {delay}s between requests)...")
    for comp_name, genes in tqdm(component_genes.items()):
        try:
            enr = gp.enrichr(gene_list=genes,
                             gene_sets=['GO_Biological_Process_2023'],
                             organism=organism,
                             outdir=None,
                             cutoff=0.05)
            if enr.results is not None and not enr.results.empty:
                top_terms = enr.results['Term'].head(n_top_go).tolist()
            else:
                top_terms = []
        except Exception as e:
            print(f"Enrichment failed for {comp_name}: {e}")
            top_terms = []
        go_results[comp_name] = top_terms
        time.sleep(delay)

    # Build DataFrame
    summary = []
    for comp_name, genes in component_genes.items():
        go_str = "; ".join(go_results.get(comp_name, []))
        summary.append({
            'Component': comp_name,
            'Top_genes': ', '.join(genes[:10]),
            'GO_terms': go_str
        })
    df = pd.DataFrame(summary)
    adata.uns['component_annotation'] = df
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Saved annotation to {save_csv}")
    return df