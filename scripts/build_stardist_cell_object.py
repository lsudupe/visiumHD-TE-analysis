"""
Build one spatial AnnData object per sample using bin2cell + StarDist
segmentation on the REAL H&E microscope image (not the lower-resolution
CytAssist image used as a fallback earlier).

Key design choice: genes + TE are merged at the native 2um bin level
BEFORE running bin_to_cell, so the aggregation step sums them together
automatically -- no separate TE aggregation step is needed (unlike
build_cell_object.py, which aggregates TE onto Space Ranger's own
Cellpose segmentation after the fact).

Follows Ana's QC convention: cells built from too few bins (<=2, i.e.
essentially un-segmented single-bin "cells") are dropped, since they are
not reliable cell-level measurements.

Usage (from terminal):
    python scripts/build_stardist_cell_object.py --sample Control-GER --te-level family

Usage (from a notebook):
    from scripts.build_stardist_cell_object import build_sample
    adata = build_sample("Control-GER", te_level="family")
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import bin2cell as b2c

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_spatial_object import load_te_adata_2um  # noqa: E402

PROJECT_DIR = Path("/ibex/project/c2344")
RUN_DIR = PROJECT_DIR / "20260402_LL00134_0018_A23J2Y7LT4"

DATA_DIR = Path("/ibex/user/medinils/data")
SAMPLES_DIR = DATA_DIR / "samples"
OBJECTS_DIR = DATA_DIR / "objects"
STARDIST_DIR = DATA_DIR / "stardist"

# Real H&E microscope image per sample (much higher resolution than the
# CytAssist fallback: ~8900x10600px vs ~3200x3000px). Copied to
# samples/<sample>/he_image.tif -- see repo notes for the copy command.
MIN_BINS_PER_CELL = 3  # Ana's rule: drop cells built from <=2 bins


def load_genes_2um_with_he(sample: str) -> ad.AnnData:
    """Load genes at native 2um bin resolution via bin2cell, using the real
    H&E image (not CytAssist) as the segmentation source image."""
    sample_dir = SAMPLES_DIR / sample
    path_002 = sample_dir / "square_002um_view"
    if not path_002.exists():
        path_002.mkdir(parents=True, exist_ok=True)
        (path_002 / "filtered_feature_bc_matrix.h5").symlink_to(
            sample_dir / "filtered_feature_bc_matrix_002um.h5"
        )
        (path_002 / "spatial").symlink_to(sample_dir / "spatial_002um")

    source_image = sample_dir / "he_image.tif"
    spaceranger_image_path = sample_dir / "spatial"  # unified spatial/ (hires/lowres), copied earlier

    adata = b2c.read_visium(
        str(path_002),
        source_image_path=str(source_image),
        spaceranger_image_path=str(spaceranger_image_path),
    )
    adata.var_names_make_unique()
    adata.obs["sample"] = sample
    return adata


def merge_genes_and_te_at_bin_level(adata: ad.AnnData, te: ad.AnnData) -> ad.AnnData:
    """Concatenate TE features as extra columns at the 2um bin level,
    restricted to barcodes present in both (intersect + subset)."""
    common = adata.obs_names.intersection(te.obs_names)
    print(f"  {len(common)} bins shared out of {adata.n_obs} (genes) / {te.n_obs} (TE)")

    adata_sub = adata[common].copy()
    te_sub = te[common].copy()

    merged_X = sp.hstack([adata_sub.X, te_sub.X], format="csr")
    merged_var = pd.concat([adata_sub.var, te_sub.var])
    merged = ad.AnnData(X=merged_X, obs=adata_sub.obs.copy(), var=merged_var)
    merged.obsm = adata_sub.obsm.copy()
    merged.uns = adata_sub.uns.copy()
    return merged


def run_stardist_segmentation(adata: ad.AnnData, sample: str) -> ad.AnnData:
    """Destripe, generate a scaled H&E crop, run StarDist nucleus detection,
    expand labels to approximate cell boundaries."""
    STARDIST_DIR.mkdir(parents=True, exist_ok=True)

    # destripe() expects adata.obs["n_counts"] to already exist
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()

    b2c.destripe(adata)

    he_scaled_path = STARDIST_DIR / f"{sample}_he_scaled.tiff"
    b2c.scaled_he_image(adata, mpp=0.5, save_path=str(he_scaled_path))

    labels_path = STARDIST_DIR / f"{sample}_stardist_labels.npz"
    b2c.stardist(
        image_path=str(he_scaled_path),
        labels_npz_path=str(labels_path),
        stardist_model="2D_versatile_he",
        prob_thresh=0.01,
    )
    b2c.insert_labels(
        adata, labels_npz_path=str(labels_path),
        basis="spatial", spatial_key="spatial_cropped_150_buffer",
        mpp=0.5, labels_key="labels_he",
    )

    # Compute BOTH expansion algorithms, as Ana does, rather than picking one
    # blindly -- distance-based (fixed N bins around each nucleus) vs.
    # volume_ratio (per-cell expansion derived from the nucleus/cell volume
    # relationship). Both are saved; the QC notebook compares them visually
    # before choosing which one feeds into bin_to_cell.
    b2c.expand_labels(adata, labels_key="labels_he", expanded_labels_key="labels_expanded_distance")
    b2c.expand_labels(adata, labels_key="labels_he", algorithm="volume_ratio", expanded_labels_key="labels_expanded_volume")

    return adata


def build_sample(sample: str, te_level: str = "family") -> ad.AnnData:
    print(f"[{sample}] loading genes (2um, real H&E image) ...")
    adata = load_genes_2um_with_he(sample)
    print(f"[{sample}] genes: {adata.shape}")

    print(f"[{sample}] loading SoloTE '{te_level}'-level matrix (native 2um) ...")
    te = load_te_adata_2um(sample, te_level)

    print(f"[{sample}] merging genes + TE at bin level (before segmentation) ...")
    merged = merge_genes_and_te_at_bin_level(adata, te)
    print(f"[{sample}] combined (bin level): {merged.shape}")

    print(f"[{sample}] running StarDist segmentation on real H&E image ...")
    merged = run_stardist_segmentation(merged, sample)

    # NOTE: bin_to_cell is intentionally NOT run here. We save the bin-level
    # object with BOTH expansion label sets (labels_expanded_distance,
    # labels_expanded_volume) so the QC notebook can compare them visually
    # first (as Ana does) before picking one and aggregating to cells.
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--te-level", default="family", choices=["class", "family", "subfamily", "locus"])
    args = parser.parse_args()

    adata = build_sample(args.sample, te_level=args.te_level)

    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBJECTS_DIR / f"{args.sample}_{args.te_level}_stardist_bins.h5ad"
    adata.write(out_path)
    print(f"Saved {out_path}")