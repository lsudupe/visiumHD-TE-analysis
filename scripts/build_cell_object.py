"""
Build one spatial AnnData object per sample at REAL CELL resolution, using
Space Ranger's native Cellpose segmentation (segmented_outputs/), instead of
raw 8um square bins.

Pipeline:
  1. Load barcode_mappings.parquet -> maps each native 2um SoloTE bin to a
     cell_id (bins outside any detected cell are dropped).
  2. Load SoloTE TE matrix (native 2um) and aggregate (sum) up to cell_id.
  3. Load Space Ranger's cell-level gene matrix (segmented_outputs/filtered_feature_cell_matrix).
  4. Merge genes + TE by shared cell_id.
  5. Attach spatial coordinates: cell centroid, computed from the polygon in
     cell_segmentations.geojson (segmented_outputs/spatial/ only has
     scalefactors_json.json, no tissue_positions file).

Usage (from terminal):
    python scripts/build_cell_object.py --sample Control-GER --te-level family

Usage (from a notebook):
    from scripts.build_cell_object import build_sample
    adata = build_sample("Control-GER", te_level="family")
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# reuse the TE-loading logic already written and tested for the bin-level pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_spatial_object import load_te_adata_2um  # noqa: E402

PROJECT_DIR = Path("/ibex/project/c2344")
RUN_DIR = PROJECT_DIR / "20260402_LL00134_0018_A23J2Y7LT4"

DATA_DIR = Path("/ibex/user/medinils/data")
SAMPLES_DIR = DATA_DIR / "samples"
OBJECTS_DIR = DATA_DIR / "objects"


def load_bin_to_cell_mapping(sample: str) -> pd.DataFrame:
    """Load barcode_mappings.parquet, restricted to 2um bins with a cell assigned."""
    path = SAMPLES_DIR / sample / "barcode_mappings.parquet"
    if not path.exists():
        # fall back to reading directly from the shared project dir
        path = RUN_DIR / sample / "outs" / "barcode_mappings.parquet"
    mapping = pd.read_parquet(path)
    mapping = mapping[["square_002um", "cell_id"]].dropna(subset=["cell_id"])
    return mapping


def aggregate_te_2um_to_cell(te_2um: ad.AnnData, mapping: pd.DataFrame) -> ad.AnnData:
    """Aggregate (sum) native 2um TE counts up to cell_id, using the official
    Space Ranger bin->cell mapping."""
    te_barcodes = pd.Series(te_2um.obs_names, name="square_002um")
    te_map = te_barcodes.to_frame().merge(mapping, on="square_002um", how="inner")

    te_2um_sub = te_2um[te_map["square_002um"].values].copy()

    cell_ids = te_map["cell_id"].values
    unique_cells = pd.unique(cell_ids)
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    row_idx = [cell_to_idx[c] for c in cell_ids]

    n_cells = len(unique_cells)
    n_bins = te_2um_sub.n_obs
    grouping = sp.coo_matrix(
        (np.ones(n_bins, dtype="float32"), (row_idx, range(n_bins))),
        shape=(n_cells, n_bins),
    ).tocsr()

    te_cell_X = (grouping @ te_2um_sub.X).tocsr()
    te_cell = ad.AnnData(X=te_cell_X, obs=pd.DataFrame(index=unique_cells), var=te_2um_sub.var.copy())
    return te_cell


def load_gene_adata_cell(sample: str) -> ad.AnnData:
    """Load Space Ranger's native Cellpose-segmented cell-level gene matrix."""
    cell_matrix_dir = RUN_DIR / sample / "outs" / "segmented_outputs" / "filtered_feature_cell_matrix"
    adata = sc.read_10x_mtx(cell_matrix_dir, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.obs["sample"] = sample
    return adata


def load_cell_centroids(sample: str) -> pd.DataFrame:
    """Compute cell centroids from the segmentation polygons (cell_segmentations.geojson)."""
    geojson_path = RUN_DIR / sample / "outs" / "segmented_outputs" / "cell_segmentations.geojson"
    with open(geojson_path) as f:
        geo = json.load(f)

    centroids = []
    for feat in geo["features"]:
        coords = np.array(feat["geometry"]["coordinates"][0])
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        cell_id_raw = feat["properties"]["cell_id"]
        cell_id_fmt = f"cellid_{cell_id_raw:09d}-1"
        centroids.append({"cell_id": cell_id_fmt, "x": cx, "y": cy})

    return pd.DataFrame(centroids).set_index("cell_id")


def merge_genes_and_te(genes: ad.AnnData, te: ad.AnnData) -> ad.AnnData:
    """Intersect + subset both objects to shared cell_ids, then concatenate features."""
    common = genes.obs_names.intersection(te.obs_names)
    genes_sub = genes[common].copy()
    te_sub = te[common].copy()

    merged_X = sp.hstack([genes_sub.X, te_sub.X], format="csr")
    merged_var = pd.concat([genes_sub.var, te_sub.var])
    merged = ad.AnnData(X=merged_X, obs=genes_sub.obs.copy(), var=merged_var)
    return merged


def build_sample(sample: str, te_level: str = "family") -> ad.AnnData:
    print(f"[{sample}] loading bin->cell mapping ...")
    mapping = load_bin_to_cell_mapping(sample)
    print(f"[{sample}] bins with a cell assigned: {mapping.shape[0]}")

    print(f"[{sample}] loading SoloTE '{te_level}'-level matrix (native 2um) ...")
    te_2um = load_te_adata_2um(sample, te_level)

    print(f"[{sample}] aggregating TE counts to cell level ...")
    te_cell = aggregate_te_2um_to_cell(te_2um, mapping)
    print(f"[{sample}] TE at cell level: {te_cell.shape}")

    print(f"[{sample}] loading gene expression (Cellpose-segmented cells) ...")
    genes_cell = load_gene_adata_cell(sample)
    print(f"[{sample}] genes at cell level: {genes_cell.shape}")

    print(f"[{sample}] merging genes + TE by cell_id ...")
    adata = merge_genes_and_te(genes_cell, te_cell)
    print(f"[{sample}] combined: {adata.shape}")

    print(f"[{sample}] attaching cell centroid coordinates ...")
    centroids = load_cell_centroids(sample)
    common = adata.obs_names.intersection(centroids.index)
    print(f"[{sample}] {len(common)} cells with centroid, of {adata.n_obs}")
    adata = adata[common].copy()
    adata.obsm["spatial"] = centroids.loc[common, ["x", "y"]].values

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--te-level", default="family", choices=["class", "family", "subfamily", "locus"])
    args = parser.parse_args()

    adata = build_sample(args.sample, te_level=args.te_level)

    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBJECTS_DIR / f"{args.sample}_{args.te_level}_cell.h5ad"
    adata.write(out_path)
    print(f"Saved {out_path}")
