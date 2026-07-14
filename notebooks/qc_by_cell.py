# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: visiumhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visium HD TE Analysis — QC by Cell (Cellpose segmentation)
#
# Uses Space Ranger's native Cellpose cell segmentation instead of raw 8um
# bins. Object construction (bin->cell mapping, TE aggregation, merge with
# Cellpose gene matrix, cell centroids) lives in `scripts/build_cell_object.py`,
# already run for all 6 samples via SLURM (`scripts/run_build_cell_object.sh`).
# This notebook starts directly from the resulting `.h5ad` files.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation

# %% [markdown]
# ## 1. Load the 6 samples (family-level, cell resolution)

# %%
samples = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs", "Injured-12hrs", "Injured-24hrs"]

cell_adatas = {}
for s in samples:
    a = sc.read_h5ad(f"/ibex/user/medinils/data/objects/{s}_family_cell.h5ad")
    cell_adatas[s] = a
    print(s, a.shape)

# %% [markdown]
# ## 2. QC metrics: counts, genes, mitochondrial %, and raw TE burden per cell
# Zero-count cells are dropped BEFORE computing pct_counts_mt (avoids the
# division-by-zero -> NaN issue we hit in the bin-level pipeline).

# %%
for s, a in cell_adatas.items():
    # Pass 1: compute QC metrics so total_counts exists
    a.var["mt"] = a.var_names.str.lower().str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    # Drop zero-count cells, then recompute cleanly
    a = a[a.obs["total_counts"] > 0].copy()
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]
    a.obs["TE_burden"] = np.asarray(a.X[:, te_idx].sum(axis=1)).flatten()
    a.obs["log1p_TE_burden"] = np.log1p(a.obs["TE_burden"])

    cell_adatas[s] = a
    print(
        s,
        "| cells:", a.n_obs,
        "| mt genes:", a.var["mt"].sum(),
        "| TE features:", len(te_features),
        "| median TE burden:", np.median(a.obs["TE_burden"]),
        "| median pct_counts_mt:", np.median(a.obs["pct_counts_mt"]),
    )

# %% [markdown]
# ## 3. Combine QC metrics across samples for comparison

# %%
qc_df = pd.concat([
    a.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "TE_burden", "log1p_TE_burden"]].assign(sample=s)
    for s, a in cell_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df.groupby("sample")[["total_counts", "n_genes_by_counts", "pct_counts_mt", "TE_burden"]].median())

# %% [markdown]
# ## 4. Violin plots — raw QC metrics, 6 samples side by side

# %%
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "log1p_TE_burden", "pct_counts_mt"],
    ["Total counts per cell", "Genes detected per cell", "log1p(TE burden) per cell", "% mitochondrial counts per cell"],
):
    sns.violinplot(data=qc_df, x="sample", y=metric, ax=ax, cut=0, inner="quartile")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Spatial plots — raw TE burden and mitochondrial % side by side

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    coords = a.obsm["spatial"]
    vals = np.log1p(a.obs["TE_burden"].values)
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=10, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="log1p(TE burden)")
plt.suptitle("TE burden (raw, log1p) — by cell", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["pct_counts_mt"].values
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=10, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% mt counts")
plt.suptitle("Mitochondrial % per cell", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. QC filtering — MAD-based outlier removal (per sample)

# %%
def mad_outlier(values, nmads=5):
    med = np.median(values)
    mad = median_abs_deviation(values)
    return (values < med - nmads * mad) | (values > med + nmads * mad)

filtered_cell_adatas = {}
for s, a in cell_adatas.items():
    outlier_counts = mad_outlier(a.obs["total_counts"], nmads=5)
    outlier_genes = mad_outlier(a.obs["n_genes_by_counts"], nmads=5)
    keep = ~(outlier_counts | outlier_genes)
    a_f = a[keep].copy()
    filtered_cell_adatas[s] = a_f
    print(s, a.shape, "->", a_f.shape, f"({keep.sum()/len(keep)*100:.1f}% kept)")

# %% [markdown]
# ## 6a. MAD thresholds — distributions and cutoffs, per sample

# %%
fig, axes = plt.subplots(2, 6, figsize=(28, 8))
for col, s in enumerate(samples):
    a = cell_adatas[s]
    for row, (metric, label) in enumerate([
        ("total_counts", "Total counts"),
        ("n_genes_by_counts", "Genes detected"),
    ]):
        ax = axes[row, col]
        values = a.obs[metric]
        med = np.median(values)
        mad = median_abs_deviation(values)
        lo, hi = med - 5 * mad, med + 5 * mad

        ax.hist(values, bins=60, color="#028090", alpha=0.75)
        ax.axvline(med, color="black", linestyle="-", linewidth=1, label="median")
        ax.axvline(max(lo, 0), color="#C97B2E", linestyle="--", linewidth=1.2, label="5 MADs")
        ax.axvline(hi, color="#C97B2E", linestyle="--", linewidth=1.2)
        if row == 0:
            ax.set_title(s, fontsize=10)
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6b. Spatial check — where are the discarded cells?

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    outlier_counts = mad_outlier(a.obs["total_counts"], nmads=5)
    outlier_genes = mad_outlier(a.obs["n_genes_by_counts"], nmads=5)
    is_outlier = (outlier_counts | outlier_genes).values
    coords = a.obsm["spatial"]
    ax.scatter(coords[~is_outlier, 0], coords[~is_outlier, 1], c="lightgray", s=8, alpha=0.5)
    ax.scatter(coords[is_outlier, 0], coords[is_outlier, 1], c="red", s=12, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{s} ({is_outlier.sum()} discarded)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Normalization — per sample (normalize_total + log1p)

# %%
normalized_cell_adatas = {}
for s, a in filtered_cell_adatas.items():
    a = a.copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    normalized_cell_adatas[s] = a
    print(s, "normalized:", a.shape)

for s, a in normalized_cell_adatas.items():
    a.write(f"/ibex/user/medinils/data/objects/{s}_family_cell_normalized.h5ad")

# %% [markdown]
# ## 8. TE burden, normalized

# %%
for s, a in normalized_cell_adatas.items():
    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]
    a.obs["TE_burden_norm"] = np.asarray(a.X[:, te_idx].sum(axis=1)).flatten()

qc_df_norm = pd.concat([
    a.obs[["total_counts", "n_genes_by_counts", "TE_burden_norm"]].assign(sample=s)
    for s, a in normalized_cell_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df_norm.groupby("sample")["TE_burden_norm"].median())

# %% [markdown]
# ## 9. Violin plots — normalized

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "TE_burden_norm"],
    ["Total counts per cell (normalized, log1p)", "Genes detected per cell", "TE burden per cell (normalized, log1p)"],
):
    sns.violinplot(data=qc_df_norm, x="sample", y=metric, ax=ax, cut=0, inner="quartile")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Spatial plots — TE burden, normalized

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = normalized_cell_adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["TE_burden_norm"].values
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=10, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="TE burden (normalized)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Merge all 6 samples (outer join) + single gene/TE filter

# %%
adata_merged_cell = ad.concat(
    normalized_cell_adatas, join="outer", label="sample", fill_value=0, index_unique="-"
)
sc.pp.filter_genes(adata_merged_cell, min_cells=3)
print(adata_merged_cell.shape)
print(adata_merged_cell.obs["sample"].value_counts())

adata_merged_cell.write("/ibex/user/medinils/data/objects/all_samples_family_cell_normalized_merged.h5ad")

# %% [markdown]
# ## 12. TE fraction per cell (raw TE UMIs / raw total UMIs)

# %%
for s, a in normalized_cell_adatas.items():
    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]

    te_counts_raw = np.asarray(a.layers["counts"][:, te_idx].sum(axis=1)).flatten()
    total_counts_raw = np.asarray(a.layers["counts"].sum(axis=1)).flatten()

    a.obs["TE_fraction"] = np.divide(
        te_counts_raw, total_counts_raw,
        out=np.zeros_like(te_counts_raw, dtype=float),
        where=total_counts_raw > 0,
    )

qc_df_frac = pd.concat([
    a.obs[["TE_fraction"]].assign(sample=s) for s, a in normalized_cell_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df_frac.groupby("sample")["TE_fraction"].median())

# %% [markdown]
# ## 13. Spatial plots — TE fraction (%)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = normalized_cell_adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["TE_fraction"].values * 100
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=10, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% UMIs from TE")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What to compare against the bin-level pipeline
# - Does the control-vs-early-injury "V" pattern in TE_burden hold up with
#   real cells (much more signal per unit than an 8um bin)?
# - Is the noise we saw in raw per-bin TE_fraction (isolated bright spots,
#   ~50% in single bins) reduced now that each unit aggregates ~350 bins
#   worth of counts?
# - Only ~2,994-5,700 cells per sample vs. 70,000+ bins -- much less spatial
#   resolution, but each unit is a real biological entity, not an arbitrary
#   square.
