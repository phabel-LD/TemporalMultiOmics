#!/usr/bin/env python
"""Plot training curves (loss and LCS) from CSV file.
Handles missing 'loss' column gracefully.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_path)
    has_loss = 'loss' in df.columns
    has_lcs = 'lcs' in df.columns

    if not has_lcs:
        raise KeyError("CSV file must contain at least an 'lcs' column (and optionally 'loss').")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if has_loss:
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss", color='blue')
        ax1.plot(df['epoch'], df['loss'], 'b-', label='Loss')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
    else:
        ax2 = ax1

    ax2.set_ylabel("LCS (validation)", color='red')
    lcs_data = df[['epoch', 'lcs']].dropna()
    ax2.plot(lcs_data['epoch'], lcs_data['lcs'], 'ro-', label='LCS')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Training curves")
    fig.tight_layout()

    save_path = Path(args.metrics_path).with_suffix('.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()