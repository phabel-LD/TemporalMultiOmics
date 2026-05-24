#!/usr/bin/env python
"""Compute held‑out LCS using the training‑set CCF target."""
import argparse, torch, numpy as np, anndata as ad, pickle
from scipy.stats import spearmanr
from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--pca_path", required=True)
parser.add_argument("--tfidf_path", required=True)
parser.add_argument("--lsi_path", required=True)
parser.add_argument("--target_path", required=True)
parser.add_argument("--n_components", type=int, default=50)
args = parser.parse_args()

# 1. Load test data and sort by pseudotime
adata_test = ad.read_h5ad(args.data_path)
adata_test = adata_test[adata_test.obs['pseudotime'].argsort()].copy()
pt = adata_test.obs['pseudotime'].values

# 2. Load saved transformers
with open(args.pca_path, 'rb') as f: pca = pickle.load(f)
with open(args.tfidf_path, 'rb') as f: tfidf = pickle.load(f)
with open(args.lsi_path, 'rb') as f: lsi = pickle.load(f)

# 3. Transform test cells
rna = pca.transform(adata_test.X.toarray() if hasattr(adata_test.X, 'toarray') else adata_test.X)
atac_mat = adata_test.obsm['ATAC_gene'].toarray()
atac_tfidf = tfidf.transform(atac_mat)
atac_lsi = lsi.transform(atac_tfidf)

# 4. Load the training‑set CCF target
with open(args.target_path, 'rb') as f:
    ccf_target_train = pickle.load(f)

# 5. Load trained model
model = TMOLatentModelAsymmetric(n_rna_components=args.n_components,
                                 n_atac_components=args.n_components,
                                 d_model=64, n_heads=4, num_layers=2, dropout=0.1)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

# 6. Predict lags on test cells
with torch.no_grad():
    out = model(torch.tensor(rna, dtype=torch.float32),
                torch.tensor(atac_lsi, dtype=torch.float32),
                torch.tensor(pt, dtype=torch.float32), use_bias=True)
    pred = out['lag_per_token'][:, :args.n_components].mean(dim=0).numpy()

# 7. LCS against training‑set CCF target (the generalisation metric)
valid = ~np.isnan(ccf_target_train)
lcs_train_target = spearmanr(pred[valid], ccf_target_train[valid])[0]
print(f"Held‑out LCS (training‑set CCF target): {lcs_train_target:.4f}")

# 8. LCS against test‑set CCF target (for reference)
window_centers = np.linspace(0.1, 0.9, 11)
delta_tau_raw, _, _ = local_ccf_surface(
    atac_series=atac_lsi.T,
    rna_series=rna.T,
    pseudotime=pt,
    window_centers=window_centers,
    window_half_width=0.1, max_lag=0.3, n_lags=31, bootstrap_samples=0)
delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)
ccf_target_test = delta_tau_smooth[:, -1]

valid_test = ~np.isnan(ccf_target_test)
lcs_test_target = spearmanr(pred[valid_test], ccf_target_test[valid_test])[0]
print(f"Held‑out LCS (test‑set CCF target, unstable): {lcs_test_target:.4f}")