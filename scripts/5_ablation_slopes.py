#!/usr/bin/env python
"""Ablation: per‑component slope of predicted Δτ vs pseudotime."""
import argparse, torch, numpy as np, anndata as ad, pickle
import matplotlib.pyplot as plt
from scipy.stats import linregress, ks_2samp
from tmo.models import TMOLatentModelAsymmetric

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--full_model", required=True)
parser.add_argument("--ablated_model", required=True)
parser.add_argument("--full_pca", required=True)
parser.add_argument("--ablated_pca", required=True)
parser.add_argument("--full_tfidf", required=True)
parser.add_argument("--ablated_tfidf", required=True)
parser.add_argument("--full_lsi", required=True)
parser.add_argument("--ablated_lsi", required=True)
parser.add_argument("--output", default="ablation_slopes.pdf")
args = parser.parse_args()

# Load data
adata = ad.read_h5ad(args.data_path)
adata = adata[adata.obs['pseudotime'].argsort()].copy()
pt = adata.obs['pseudotime'].values

def compute_slopes(model_path, pca_path, tfidf_path, lsi_path):
    with open(pca_path,'rb') as f: pca = pickle.load(f)
    with open(tfidf_path,'rb') as f: tfidf = pickle.load(f)
    with open(lsi_path,'rb') as f: lsi = pickle.load(f)
    rna = pca.transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    atac_mat = adata.obsm['ATAC_gene'].toarray()
    atac_lsi = lsi.transform(tfidf.transform(atac_mat))
    model = TMOLatentModelAsymmetric(n_rna_components=50, n_atac_components=50,
                                     d_model=64, n_heads=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(rna, dtype=torch.float32),
                    torch.tensor(atac_lsi, dtype=torch.float32),
                    torch.tensor(pt, dtype=torch.float32), use_bias=True)
        pred = out['lag_per_token'][:, :50].numpy()   # (cells, 50)
    slopes = []
    for i in range(50):
        slope, _, _, _, _ = linregress(pt, pred[:, i])
        slopes.append(slope)
    return np.array(slopes)

# Get Slopes
slopes_full = compute_slopes(args.full_model, args.full_pca, args.full_tfidf, args.full_lsi)
slopes_abl  = compute_slopes(args.ablated_model, args.ablated_pca, args.ablated_tfidf, args.ablated_lsi)

# KS test
ks_stat, ks_pval = ks_2samp(slopes_full, slopes_abl)

# Histogram
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.06, 0.06, 30)
ax.hist(slopes_full, bins=bins, alpha=0.7, color='#4575b4', label='Full model')
ax.hist(slopes_abl,  bins=bins, alpha=0.7, color='#d73027', label='Ablated (no cell state)')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Slope of predicted Δτ vs pseudotime')
ax.set_ylabel('Number of components')
ax.set_title('Cell‑state ablation: per‑component slopes')
ax.legend()
# Annotate KS test result on the plot
ax.text(0.05, 0.10, f'KS $D$ = {ks_stat:.3f}\n$p$ = {ks_pval:.2e}',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
# Plot
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.close()

print(f"Mean |slope| full: {np.abs(slopes_full).mean():.4f}, ablated: {np.abs(slopes_abl).mean():.4f}")
print(f"KS test: statistic = {ks_stat:.4f}, p = {ks_pval:.4e}")
print(f"Saved slopes histogram to {args.output}")