#!/usr/bin/env python
"""Annotate RNA PCA components with top genes and GO terms. Supports rate‑limited requests.

Now ensures highly variable genes are computed before PCA.
"""

import argparse
import time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to AnnData with RNA counts (log1p recommended)")
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--n_components", type=int, default=50,
                        help="Number of PCA components to annotate (set to 50 for all)")
    parser.add_argument("--n_top_genes", type=int, default=100,
                        help="Number of top genes per component")
    parser.add_argument("--n_top_go", type=int, default=5,
                        help="Number of top GO terms to display")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait between Enrichr requests (avoid 429 errors)")
    parser.add_argument("--organism", type=str, default='human',
                        help="Species for GO enrichment: 'human', 'mouse', etc.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data (ensure log1p normalised)
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    if 'log1p' not in adata.uns and 'pseudotime' in adata.obs:
        print("Normalising RNA log1p...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Compute highly variable genes if not already present
    if 'highly_variable' not in adata.var or not adata.var['highly_variable'].any():
        print("Selecting highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')

    # Compute PCA if not already in .obsm / .varm
    if 'X_pca' not in adata.obsm or 'PCs' not in adata.varm:
        print(f"Computing PCA with {args.n_components} components...")
        sc.tl.pca(adata, n_comps=args.n_components, use_highly_variable=True)
    loadings = adata.varm['PCs']  # shape (n_genes, n_components)

    # For each component, get top genes by absolute loading
    component_genes = {}
    print("Extracting top genes per component...")
    for comp in range(args.n_components):
        abs_load = np.abs(loadings[:, comp])
        top_idx = np.argsort(-abs_load)[:args.n_top_genes]
        top_genes = adata.var_names[top_idx].tolist()
        component_genes[f"Comp{comp}"] = top_genes

    # Run GO enrichment using gseapy (Enrichr) with delays
    go_results = {}
    try:
        import gseapy as gp
        print(f"Running GO enrichment via Enrichr (delay {args.delay}s between requests)...")
        for comp_name, genes in component_genes.items():
            print(f"  Enriching {comp_name}...")
            try:
                enr = gp.enrichr(gene_list=genes,
                                 gene_sets=['GO_Biological_Process_2023'],
                                 organism=args.organism,
                                 outdir=None,
                                 cutoff=0.05)
                if enr.results is not None and not enr.results.empty:
                    top_terms = enr.results['Term'].head(args.n_top_go).tolist()
                else:
                    top_terms = []
            except Exception as e:
                print(f"    Enrichment failed: {e}")
                top_terms = []
            go_results[comp_name] = top_terms
            time.sleep(args.delay)  # rate limiting
    except ImportError:
        print("gseapy not installed. Install with: pip install gseapy")
        go_results = {}

    # Create summary table
    summary = []
    for comp_name, genes in component_genes.items():
        go_str = "; ".join(go_results.get(comp_name, [])) if go_results else ""
        summary.append({
            'Component': comp_name,
            'Top_genes': ', '.join(genes[:10]),
            'GO_terms': go_str
        })
    df = pd.DataFrame(summary)
    df.to_csv(output_dir / "component_annotation.csv", index=False)
    print(f"\nSaved annotation table to {output_dir / 'component_annotation.csv'}")

    # Print readable output
    print("\n=== Component annotations ===")
    for comp_name, genes in list(component_genes.items())[:args.n_components]:
        print(f"\n{comp_name}:")
        print(f"  Top genes: {', '.join(genes[:5])} ...")
        if go_results and go_results.get(comp_name):
            print(f"  GO: {', '.join(go_results[comp_name][:3])}")
        else:
            print("  GO: (not available)")

if __name__ == '__main__':
    main()