"""
Build one spatial AnnData object per sample at native 2um bin resolution,
merging Space Ranger gene counts with SoloTE TE counts (both natively at
2um bin barcodes, e.g. s_002um_00999_01331-1) BEFORE any cell segmentation.

This way, downstream segmentation (bin2cell + StarDist, as in the workshop
notebook) aggregates genes and TEs together into pseudo-cells automatically
-- no separate bin-to-cell mapping is needed.

Usage (from terminal):
    python scripts/build_spatial_object.py --sample Control-GER --te-level family

Usage (from a notebook, before the destripe/segmentation steps):
    from scripts.build_spatial_object import build_sample
    adata = build_sample("Control-GER", te_level="family")
    # continue with: b2c.destripe(adata), stardist, bin_to_cell, etc.
"""

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

# ---- paths (adjust if your layout differs) ----
PROJECT_DIR = Path("/ibex/project/c2344")
SOLOTE_DIR = PROJECT_DIR / "Spatial" / "STE_results"

DATA_DIR = Path("/ibex/user/medinils/data")
SAMPLES_DIR = DATA_DIR / "samples"
OBJECTS_DIR = DATA_DIR / "objects"

LEVEL_SUFFIX = {
    "class": "classtes_MATRIX",
    "family": "familytes_MATRIX",
    "subfamily": "subfamilytes_MATRIX",
    "locus": "locustes_MATRIX",
}

BARCODE_RE = re.compile(r"s_002um_(\d+)_(\d+)-1")
BIN_RATIO = 4  # 8um / 2um


def load_gene_adata_8um(sample: str) -> ad.AnnData:
    """Load genes at 8um bin resolution from the already-copied sample data
    (samples/<sample>/filtered_feature_bc_matrix + spatial/), as prepared
    earlier when we copied square_008um outputs."""
    sample_dir = SAMPLES_DIR / sample
    adata = sc.read_10x_mtx(
        sample_dir / "filtered_feature_bc_matrix",
        var_names="gene_symbols",
        cache=True,
    )
    adata.var_names_make_unique()

    pos_path = sample_dir / "spatial" / "tissue_positions.parquet"
    if pos_path.exists():
        pos = pd.read_parquet(pos_path)
    else:
        pos = pd.read_csv(sample_dir / "spatial" / "tissue_positions.csv")
    pos = pos.set_index(pos.columns[0])
    pos = pos.reindex(adata.obs_names)
    coord_cols = [c for c in ["pxl_col_in_fullres", "pxl_row_in_fullres"] if c in pos.columns]
    adata.obsm["spatial"] = pos[coord_cols].values

    adata.obs["sample"] = sample
    return adata


def find_solote_matrix_dir(sample: str, level: str) -> Path:
    suffix = LEVEL_SUFFIX[level]
    sample_dir = SOLOTE_DIR / f"{sample}_SoloTE_output"
    if not sample_dir.exists():
        candidates = list(SOLOTE_DIR.glob(f"{sample}*_SoloTE_output"))
        if not candidates:
            raise FileNotFoundError(f"No SoloTE output folder found for sample '{sample}'")
        sample_dir = candidates[0]
    matches = list(sample_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No '{suffix}' matrix found under {sample_dir}")
    return matches[0]


def load_te_adata_2um(sample: str, level: str) -> ad.AnnData:
    """Load one SoloTE-level matrix at its native 2um bin resolution.
    SoloTE writes uncompressed matrix.mtx/barcodes.tsv/features.tsv (no .gz)."""
    mtx_dir = find_solote_matrix_dir(sample, level)

    matrix = sio.mmread(mtx_dir / "matrix.mtx").tocsr()
    barcodes = pd.read_csv(mtx_dir / "barcodes.tsv", header=None)[0].values
    features = pd.read_csv(mtx_dir / "features.tsv", header=None, sep="\t")

    if matrix.shape[0] == len(features) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()

    var = pd.DataFrame(index=features[0].values)
    if features.shape[1] > 1:
        var["gene_ids"] = features[1].values

    te = ad.AnnData(X=matrix, obs=pd.DataFrame(index=barcodes), var=var)
    # keep only genuine TE features (SoloTE matrices also include regular gene
    # rows -- we only want the ones actually flagged as TE by SoloTE)
    te = te[:, te.var_names.str.contains("SoloTE")].copy()
    te.var_names = ["TE_" + v for v in te.var_names]
    te.var_names_make_unique()
    return te


def aggregate_te_2um_to_8um(te: ad.AnnData) -> ad.AnnData:
    """Aggregate a 2um-barcode TE AnnData up to 8um bins (groups of 4x4 2um
    bins), by summing counts. Produces barcodes in the same
    's_008um_FILA_COLUMNA-1' format Space Ranger uses natively."""
    rows, cols, keep_idx = [], [], []
    for i, bc in enumerate(te.obs_names):
        m = BARCODE_RE.match(bc)
        if m is None:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        rows.append(r // BIN_RATIO)
        cols.append(c // BIN_RATIO)
        keep_idx.append(i)

    te = te[keep_idx].copy()
    bin8_barcodes = [f"s_008um_{r:05d}_{c:05d}-1" for r, c in zip(rows, cols)]

    df_bins = pd.Series(bin8_barcodes)
    unique_bins = df_bins.unique()
    bin_to_idx = {b: i for i, b in enumerate(unique_bins)}
    row_idx = df_bins.map(bin_to_idx).values

    n_bins8 = len(unique_bins)
    n_2um = te.n_obs
    grouping = sp.coo_matrix(
        (np.ones(n_2um, dtype="float32"), (row_idx, range(n_2um))),
        shape=(n_bins8, n_2um),
    ).tocsr()

    X8 = (grouping @ te.X).tocsr()
    te8 = ad.AnnData(X=X8, obs=pd.DataFrame(index=unique_bins), var=te.var.copy())
    return te8


def merge_genes_and_te(adata: ad.AnnData, te: ad.AnnData) -> ad.AnnData:
    """Concatenate TE features as extra columns, restricted to barcodes
    present in both objects (intersect + subset)."""
    common = adata.obs_names.intersection(te.obs_names)
    print(f"  {len(common)} barcodes shared out of {adata.n_obs} (genes) / {te.n_obs} (TE, 8um-aggregated)")

    adata_sub = adata[common].copy()
    te_sub = te[common].copy()

    merged_X = sp.hstack([adata_sub.X, te_sub.X], format="csr")
    merged_var = pd.concat([adata_sub.var, te_sub.var])
    merged = ad.AnnData(X=merged_X, obs=adata_sub.obs.copy(), var=merged_var)
    merged.obsm = adata_sub.obsm.copy()
    merged.uns = adata_sub.uns.copy()
    return merged


def build_sample(sample: str, te_level: str = "family") -> ad.AnnData:
    print(f"[{sample}] loading gene expression (8um bins) ...")
    adata = load_gene_adata_8um(sample)
    print(f"[{sample}] genes: {adata.shape}")

    print(f"[{sample}] loading SoloTE '{te_level}'-level matrix (native 2um) ...")
    te_2um = load_te_adata_2um(sample, te_level)
    print(f"[{sample}] TE features ({te_level}), 2um: {te_2um.shape}")

    print(f"[{sample}] aggregating TE counts 2um -> 8um bins ...")
    te_8um = aggregate_te_2um_to_8um(te_2um)
    print(f"[{sample}] TE features ({te_level}), 8um: {te_8um.shape}")

    print(f"[{sample}] merging genes + TE (8um bin level) ...")
    merged = merge_genes_and_te(adata, te_8um)
    print(f"[{sample}] combined: {merged.shape}")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--te-level", default="family", choices=list(LEVEL_SUFFIX.keys()))
    args = parser.parse_args()

    adata = build_sample(args.sample, te_level=args.te_level)

    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBJECTS_DIR / f"{args.sample}_{args.te_level}_8um.h5ad"
    adata.write(out_path)
    print(f"Saved {out_path}")