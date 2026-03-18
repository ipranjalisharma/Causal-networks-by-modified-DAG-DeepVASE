#!/usr/bin/env python3
"""
RNA-seq Expression Matrix Builder

Builds a gene-expression matrix from per-sample ENCODE/GENCODE TSV files,
maps Ensembl IDs → gene symbols via MyGene.info, filters low-expression genes,
and writes two outputs:
  1. Full log1p-TPM matrix
  2. Top-N most-variable genes (for causal network input)

Part of the DAG-deepVASE-RNA pipeline.

Requirements:
    pip install pandas numpy mygene tqdm

Usage:
    python build_rnaseq_matrix.py

--------------------------------------------
 USER CONFIGURATION — edit the block below

"""

import os
import glob
import pandas as pd
import numpy as np
import mygene
from tqdm import tqdm


#  CONFIGURATION  ← Edit these values before running


# Root folder containing one sub-folder per cell line / sample.
# Each sub-folder must contain exactly one .tsv file with a 'gene_id' column
# and a TPM column (see EXPRESSION_COL below).
# Example layout:
#   data/
#     K562/
#       ENCFF001.tsv
#     HepG2/
#       ENCFF002.tsv
DATA_ROOT = "/path/to/your/data/folder"

# Full path (including filename) for the main output matrix.
# A second file ending in _top5k_causal.txt is also written automatically.
OUTPUT_FILE = "/path/to/your/output/RNAseq_Matrix.txt"

# Column name in the TSV files that holds expression values.
# Common choices: "TPM", "FPKM", "expected_count"
EXPRESSION_COL = "TPM"

# Genes expressed as 0 in more than this fraction of samples are removed.
# 0.8 means: drop a gene if it is zero in >80% of samples.
MAX_ZERO_FRACTION = 0.8

# Whether to apply log1p transformation to the final matrix.
# Strongly recommended for downstream causal / DNN analysis.
LOG_TRANSFORM = True

# Number of Ensembl IDs sent to MyGene.info per API request.
# Reduce to 500 if you hit rate-limit errors.
MYGENE_BATCH_SIZE = 1000

# Number of top-variance genes to keep in the causal-network input matrix.
# 5000 is a reasonable default; increase if you have >1000 samples.
TOP_CAUSAL_GENES = 5000




def find_tsv_files(data_root: str) -> dict:
    """Return {cell_line_name: path_to_tsv} for every sub-folder in data_root."""
    cell_files = {}
    for cell_dir in sorted(os.listdir(data_root)):
        full_dir  = os.path.join(data_root, cell_dir)
        if not os.path.isdir(full_dir):
            continue
        tsv_files = glob.glob(os.path.join(full_dir, "*.tsv"))
        if not tsv_files:
            print(f"  [WARNING] No .tsv in {cell_dir}, skipping.")
            continue
        cell_files[cell_dir] = tsv_files[0]
    return cell_files


def load_expression_series(tsv_path: str, expr_col: str) -> pd.Series:
    """
    Load expression values from one TSV file.
    - Keeps only rows whose gene_id starts with 'ENSG'
    - Strips Ensembl version suffixes  (ENSG00000000419.12 → ENSG00000000419)
    """
    df = pd.read_csv(tsv_path, sep="\t", usecols=["gene_id", expr_col],
                     dtype={"gene_id": str})
    df = df[df["gene_id"].str.startswith("ENSG")].copy()
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    df = df.drop_duplicates(subset="gene_id")
    return df.set_index("gene_id")[expr_col]


def build_matrix(cell_files: dict, expr_col: str) -> pd.DataFrame:
    """Load all samples and assemble a (samples × genes) matrix."""
    print(f"\nLoading {expr_col} from {len(cell_files)} sample(s)...")
    series_list = {}
    for cell_name, tsv_path in tqdm(cell_files.items(), desc="  Loading TSVs"):
        try:
            series_list[cell_name] = load_expression_series(tsv_path, expr_col)
        except Exception as e:
            print(f"  [ERROR] {cell_name}: {e}")

    matrix = pd.DataFrame(series_list).T  # rows=samples, cols=gene_ids
    n_nan  = matrix.isna().sum().sum()
    if n_nan > 0:
        print(f"  Filling {n_nan:,} NaN values with 0 "
              f"(gene absent in a sample = unexpressed, not missing)")
        matrix = matrix.fillna(0)

    matrix.index.name   = "sample"
    matrix.columns.name = "gene_id"
    print(f"  Raw matrix shape: {matrix.shape}  (samples × genes)")
    return matrix


def map_gene_ids_to_symbols(gene_ids: list, batch_size: int = 1000) -> dict:
    """Batch-query MyGene.info to convert Ensembl IDs → HGNC gene symbols."""
    mg = mygene.MyGeneInfo()
    print(f"\n  Querying MyGene.info in batches of {batch_size}...")

    id_to_symbol = {}
    batches = [gene_ids[i:i+batch_size] for i in range(0, len(gene_ids), batch_size)]

    for batch in tqdm(batches, desc="  Mapping gene IDs", unit="batch"):
        try:
            results = mg.querymany(
                batch,
                scopes       = "ensembl.gene",
                fields       = "symbol",
                species      = "human",
                returnall    = False,
                verbose      = False,
                as_dataframe = True,
            )
            if isinstance(results, pd.DataFrame) and "symbol" in results.columns:
                batch_map = (
                    results["symbol"]
                    .dropna()
                    .reset_index()
                    .rename(columns={"query": "gene_id"})
                    .drop_duplicates(subset="gene_id")
                    .set_index("gene_id")["symbol"]
                    .to_dict()
                )
                id_to_symbol.update(batch_map)
        except Exception as e:
            print(f"\n  [WARNING] Batch failed: {e}")

    mapped   = len(id_to_symbol)
    unmapped = len(gene_ids) - mapped
    print(f"  Mapped: {mapped:,}   |   Unmapped / dropped: {unmapped:,}")
    return id_to_symbol


def filter_and_rename(matrix: pd.DataFrame,
                      id_to_symbol: dict,
                      max_zero_frac: float,
                      log_transform: bool) -> pd.DataFrame:
    """Rename columns, collapse duplicate symbols, filter low-expression genes."""
    print("\nRenaming Ensembl IDs → gene symbols...")
    keep_ids       = [c for c in matrix.columns if c in id_to_symbol]
    matrix         = matrix[keep_ids].copy()
    matrix.columns = [id_to_symbol[c] for c in matrix.columns]

    dup_count = matrix.columns.duplicated().sum()
    if dup_count:
        print(f"  Averaging {dup_count:,} duplicate gene symbols...")
        matrix = matrix.T.groupby(level=0).mean().T

    n_before   = matrix.shape[1]
    zero_frac  = (matrix == 0).mean(axis=0)
    keep_genes = zero_frac[zero_frac <= max_zero_frac].index
    matrix     = matrix[keep_genes]
    n_after    = matrix.shape[1]
    print(f"  After zero-filter (>{max_zero_frac*100:.0f}% zeros removed): "
          f"{n_before:,} → {n_after:,} genes  (dropped {n_before - n_after:,})")

    if log_transform:
        print("  Applying log1p transformation...")
        matrix = np.log1p(matrix)

    return matrix


def main():
    print("=" * 60)
    print("  RNA-seq Matrix Builder")
    print("=" * 60)
    print(f"  Data root  : {DATA_ROOT}")
    print(f"  Output     : {OUTPUT_FILE}")

    if DATA_ROOT == "/path/to/your/data/folder":
        raise SystemExit(
            "\n[ERROR] DATA_ROOT has not been set.\n"
            "  Open build_rnaseq_matrix.py and edit the CONFIGURATION block at the top."
        )

    cell_files = find_tsv_files(DATA_ROOT)
    if not cell_files:
        raise SystemExit(f"\n[ERROR] No sample sub-folders found in {DATA_ROOT}")
    print(f"\n  Found {len(cell_files)} sample(s): {', '.join(sorted(cell_files))}")

    raw_matrix   = build_matrix(cell_files, EXPRESSION_COL)
    gene_ids     = raw_matrix.columns.tolist()
    id_to_symbol = map_gene_ids_to_symbols(gene_ids, batch_size=MYGENE_BATCH_SIZE)

    final_matrix = filter_and_rename(
        raw_matrix, id_to_symbol,
        max_zero_frac = MAX_ZERO_FRACTION,
        log_transform = LOG_TRANSFORM,
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    final_matrix.to_csv(OUTPUT_FILE, sep="\t")
    print(f"\n[1/2] Full matrix  →  {OUTPUT_FILE}")
    print(f"      Shape: {final_matrix.shape}  (samples × genes)")

    print(f"\nGenerating variance-filtered matrix for causal analysis...")
    top_n         = min(TOP_CAUSAL_GENES, final_matrix.shape[1])
    gene_var      = final_matrix.var(axis=0)
    top_genes     = gene_var.nlargest(top_n).index
    causal_matrix = final_matrix[top_genes]
    causal_file   = OUTPUT_FILE.replace(".txt", f"_top{top_n}_causal.txt")
    causal_matrix.to_csv(causal_file, sep="\t")
    print(f"[2/2] Causal matrix (top {top_n} variable genes)  →  {causal_file}")
    print(f"      Shape: {causal_matrix.shape}")

    print(f"\nPreview — first 5 samples × first 5 genes:")
    print(final_matrix.iloc[:5, :5].round(3).to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
