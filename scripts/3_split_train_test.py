#!/usr/bin/env python
"""Stratified 80/20 split of a TMO‑ready AnnData."""
import anndata as ad, numpy as np, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--output_train", required=True)
parser.add_argument("--output_test", required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

adata = ad.read_h5ad(args.data_path)
n_cells = adata.n_obs

# stratify by 10 equal pseudotime bins
bins = np.digitize(adata.obs['pseudotime'].values, np.linspace(0, 1, 11)) - 1
np.random.seed(args.seed)
train_idx, test_idx = [], []
for b in range(10):
    idx = np.where(bins == b)[0]
    np.random.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))   # ensure at least one cell per bin in test
    train_idx.extend(idx[:split])
    test_idx.extend(idx[split:])

adata[train_idx].copy().write(args.output_train)
adata[test_idx].copy().write(args.output_test)
print(f"Saved {len(train_idx)} training cells to {args.output_train}")
print(f"Saved {len(test_idx)} test cells to {args.output_test}")