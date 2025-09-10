#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_chisq_selection.py
=====================

Purpose:
    Perform Chi-squared feature selection on very large genomic k-mer presence/absence matrices
    in a memory-efficient way using sparse matrices.

Inputs:
    --meta:       Path to metadata file (TSV containing sample identifier and label columns).
    --features1:  Path to first feature matrix (TSV, rows = isolates, cols = k-mers, first col = sample IDs).
    --features2:  Path to optional second feature matrix (same format).
    --output_dir: Directory to write output files.
    --name:       Base name for output files (no extension).

Parameters:
    --length_threshold:  Minimum k-mer string length to retain (default: 80).
    --id:                Column name in metadata for sample IDs (default: 'SRA').
    --label:             Column name in metadata for labels (default: 'SYMP').
    --k:                 Number of top features to select (default: 100000).

Outputs (written to --output_dir):
    <name>_top{k}_features.tsv    Top k features (full P/A matrix with isolates × selected features).
    <name>_pvalues.tsv            Table of all features and their chi2 p-values.
    <name>_pvalues_features.tsv   P/A matrix with features passing p ≤ 0.05 (isolates × features).
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def get_opts():
    parser = argparse.ArgumentParser(description="Chi2 feature selection from genomic data.")
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV with columns SRA, SYMP)')
    parser.add_argument('--features1', required=True, help='Path to first feature matrix (TSV)')
    parser.add_argument('--features2', required=False, help='Path to second feature matrix (TSV)')
    parser.add_argument('--output_dir', required=True, help='Directory to write output files')
    parser.add_argument('--name', required=True, help='Base name for output files (no extension)')
    parser.add_argument('--length_threshold', type=int, default=80,
                        help='Minimum k-mer string length to keep')
    parser.add_argument('--id', dest='id_col', default='SRA',
                        help="Metadata column name for sample IDs (default: 'SRA')")
    parser.add_argument('--label', dest='label_col', default='SYMP',
                        help="Metadata column name for labels (default: 'SYMP')")
    parser.add_argument('--k', type=int, default=100000,
                        help='Number of top features to select (default: 100000)')
    return parser.parse_args()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def read_meta(meta_file, id_col, label_col):
    """Read metadata file and return DataFrame indexed by sample IDs."""
    df = pd.read_csv(meta_file, sep="\t", header=0, dtype=str)
    missing = [c for c in [id_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in metadata file: {missing}. Found: {list(df.columns)}")
    return df[[id_col, label_col]].set_index(id_col)

def load_and_filter_features(feature_file, length_threshold):
    """Load feature matrix from TSV as a sparse binary matrix, return (X_sparse, index, feature_names)."""
    print(f"Loading features from {feature_file}")
    # Load all as string first to avoid sample IDs being forced numeric
    df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0, dtype=str)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Convert feature columns to numeric (int8), binarize
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int8)
    df = (df > 0).astype(np.int8)
    # Filter by k-mer length
    long_cols = [c for c in df.columns if len(str(c)) >= length_threshold]
    df = df[long_cols]
    # Convert to sparse matrix
    X = sparse.csr_matrix(df.values, dtype=np.float32)
    return X, df.index, pd.Index(long_cols)

def merge_feature_sets(X1, idx1, cols1, X2, idx2, cols2):
    """Align two feature matrices on sample IDs and concatenate horizontally."""
    df1 = pd.DataFrame.sparse.from_spmatrix(X1, index=idx1, columns=cols1)
    df2 = pd.DataFrame.sparse.from_spmatrix(X2, index=idx2, columns=cols2)

    merged = df1.join(df2, how="inner").astype(np.int8)
    X = sparse.csr_matrix(merged.values, dtype=np.float32)
    return X, merged.index, merged.columns

def write_matrix_tsv(path, X, index, cols):
    """Write sparse matrix X (csr) with row index and column names to TSV."""
    print(f"Writing matrix to {path}")
    with open(path, "w") as f:
        # header
        f.write("\t" + "\t".join(map(str, cols)) + "\n")
        for i, row_id in enumerate(index):
            row_data = X.getrow(i).toarray().ravel().astype(str)
            f.write(row_id + "\t" + "\t".join(row_data) + "\n")

def perform_chi2_analysis(X, y, feature_names, index, k, output_dir, name):
    """Run chi2 feature selection and save outputs in the same format as original script."""
    chi_scores, p_values = chi2(X, y)

    # Select top-k features
    selector = SelectKBest(chi2, k=min(k, X.shape[1]))
    X_kbest = selector.fit_transform(X, y)
    selected_features = feature_names[selector.get_support()]

    os.makedirs(output_dir, exist_ok=True)

    # Save full p-values table
    out_pvals = os.path.join(output_dir, f"{name}_pvalues.tsv")
    pd.DataFrame({"feature": feature_names, "p_value": p_values}).to_csv(
        out_pvals, sep="\t", index=False
    )

    # Save top-k feature matrix (isolates × selected features, TSV)
    out_k = os.path.join(output_dir, f"{name}_top{k}_features.tsv")
    X_kbest_df = pd.DataFrame(
        X_kbest.toarray().astype(np.int8), index=index, columns=selected_features
    )
    X_kbest_df.to_csv(out_k, sep="\t")

    # Save p-value ≤ 0.05 feature matrix
    good_cols = feature_names[p_values <= 0.05]
    if len(good_cols) > 0:
        X_pval = X[:, feature_names.get_indexer(good_cols)]
        X_pval_df = pd.DataFrame(
            X_pval.toarray().astype(np.int8), index=index, columns=good_cols
        )
        out_pval = os.path.join(output_dir, f"{name}_pvalues_features.tsv")
        X_pval_df.to_csv(out_pval, sep="\t")
    else:
        print("No features with p ≤ 0.05 found; skipping p-value feature matrix.")

    print(f"Original feature count: {X.shape[1]}")
    print(f"Reduced feature count (top {len(selected_features)}): {X_kbest.shape[1]}")
    print(f"Saved top {k} features to: {out_k}")
    print(f"Saved p-values to: {out_pvals}")
    if len(good_cols) > 0:
        print(f"Saved p ≤ 0.05 features to: {out_pval}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    args = get_opts()

    # Load metadata
    meta_df = read_meta(args.meta, args.id_col, args.label_col)

    # Load features
    X1, idx1, cols1 = load_and_filter_features(args.features1, args.length_threshold)

    if args.features2:
        X2, idx2, cols2 = load_and_filter_features(args.features2, args.length_threshold)
        X, idx, cols = merge_feature_sets(X1, idx1, cols1, X2, idx2, cols2)
    else:
        print("No second feature file provided; using only features1")
        X, idx, cols = X1, idx1, cols1

    # Align labels with features
    y = meta_df.loc[idx, args.label_col].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Run chi2 feature selection
    perform_chi2_analysis(X, y, cols, idx, args.k, args.output_dir, args.name)

