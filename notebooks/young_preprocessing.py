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
# # Visium HD TE Analysis — Preprocessing (family level, 8um)
# QC filtering (MAD) -> normalization -> merge -> TE burden / TE fraction

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation

# %% [markdown]
# ## 1. Load the 6 samples (family-level, 8um objects)

# %%
samples = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs", "Injured-12hrs", "Injured-24hrs"]

adatas = {}
for s in samples:
    a = sc.read_h5ad(f"/ibex/user/medinils/data/objects/{s}_family_8um.h5ad")
    adatas[s] = a
    print(s, a.shape)

# %% [markdown]
# ## 2. QC metrics: counts, genes, mitochondrial %, and raw TE burden per bin
# All core QC metrics computed together, so filtering decisions (step 6) can
# be made looking at the full picture at once, not metric-by-metric.

# %%
for s, a in adatas.items():
    a.var["mt"] = a.var_names.str.lower().str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]
    a.obs["TE_burden"] = np.asarray(a.X[:, te_idx].sum(axis=1)).flatten()
    a.obs["log1p_TE_burden"] = np.log1p(a.obs["TE_burden"])

    print(
        s,
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
    for s, a in adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df.groupby("sample")[["total_counts", "n_genes_by_counts", "pct_counts_mt", "TE_burden"]].median())

# %% [markdown]
# ## 4. Violin plots — raw QC metrics, 6 samples side by side
# Four panels together: total counts, genes detected, TE burden, and
# mitochondrial % — the full picture in one view.

# %%
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "log1p_TE_burden", "pct_counts_mt"],
    ["Total counts per bin", "Genes detected per bin", "log1p(TE burden) per bin", "% mitochondrial counts per bin"],
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
    a = adatas[s]
    coords = a.obsm["spatial"]
    vals = np.log1p(a.obs["TE_burden"].values)
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=8, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="log1p(TE burden)")
plt.suptitle("TE burden (raw, log1p)", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["pct_counts_mt"].values
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=8, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% mt counts")
plt.suptitle("Mitochondrial % per bin", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. QC filtering — MAD-based outlier removal (per sample)
# Fixed thresholds don't transfer across samples with very different depth
# (e.g. Injured-1hrs vs Control-Old) -- MAD adapts to each sample's own
# distribution instead. Currently filtering on total_counts and n_genes;
# pct_counts_mt is visualized above but not yet used as a filter criterion
# -- decide after reviewing whether it shows a spatially localized artifact.

# %%
def mad_outlier(values, nmads=5):
    med = np.median(values)
    mad = median_abs_deviation(values)
    return (values < med - nmads * mad) | (values > med + nmads * mad)

filtered_adatas = {}
for s, a in adatas.items():
    outlier_counts = mad_outlier(a.obs["total_counts"], nmads=5)
    outlier_genes = mad_outlier(a.obs["n_genes_by_counts"], nmads=5)
    keep = ~(outlier_counts | outlier_genes)
    a_f = a[keep].copy()
    filtered_adatas[s] = a_f
    print(s, a.shape, "->", a_f.shape, f"({keep.sum()/len(keep)*100:.1f}% kept)")

# %% [markdown]
# ## 6b. Spatial check — where are the discarded bins?
# If discarded bins cluster in one region (tissue edge, fold, bubble) rather
# than being scattered randomly, that's a localized technical artifact worth
# flagging -- not just random noise.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = adatas[s]
    outlier_counts = mad_outlier(a.obs["total_counts"], nmads=5)
    outlier_genes = mad_outlier(a.obs["n_genes_by_counts"], nmads=5)
    is_outlier = (outlier_counts | outlier_genes).values
    coords = a.obsm["spatial"]
    ax.scatter(coords[~is_outlier, 0], coords[~is_outlier, 1], c="lightgray", s=4, alpha=0.4)
    ax.scatter(coords[is_outlier, 0], coords[is_outlier, 1], c="red", s=6, alpha=0.8)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{s} ({is_outlier.sum()} discarded)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Normalization — per sample (normalize_total + log1p)
# Done per-sample, before merging, consistent with model-based normalization
# best practice (see project notes on why-per-sample).

# %%
normalized_adatas = {}
for s, a in filtered_adatas.items():
    a = a.copy()
    a.layers["counts"] = a.X.copy()  # keep raw counts for TE_fraction etc.
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    normalized_adatas[s] = a
    print(s, "normalized:", a.shape)

# save each normalized sample to disk (safety checkpoint)
for s, a in normalized_adatas.items():
    a.write(f"/ibex/user/medinils/data/objects/{s}_family_8um_normalized.h5ad")

# %% [markdown]
# ## 8. TE burden, normalized

# %%
for s, a in normalized_adatas.items():
    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]
    a.obs["TE_burden_norm"] = np.asarray(a.X[:, te_idx].sum(axis=1)).flatten()

qc_df_norm = pd.concat([
    a.obs[["total_counts", "n_genes_by_counts", "TE_burden_norm"]].assign(sample=s)
    for s, a in normalized_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df_norm.groupby("sample")["TE_burden_norm"].median())

# %% [markdown]
# ## 9. Violin plots — normalized

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "TE_burden_norm"],
    ["Total counts per bin (normalized, log1p)", "Genes detected per bin", "TE burden per bin (normalized, log1p)"],
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
    a = normalized_adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["TE_burden_norm"].values
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=8, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
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
adata_merged = ad.concat(
    normalized_adatas, join="outer", label="sample", fill_value=0, index_unique="-"
)
sc.pp.filter_genes(adata_merged, min_cells=3)
print(adata_merged.shape)
print(adata_merged.obs["sample"].value_counts())

adata_merged.write("/ibex/user/medinils/data/objects/all_samples_family_8um_normalized_merged.h5ad")

# %% [markdown]
# ## 12. TE fraction per bin (raw TE UMIs / raw total UMIs)
# Requested by PI: real TE fraction per spatial zone, not just normalized burden.

# %%
for s, a in normalized_adatas.items():
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
    a.obs[["TE_fraction"]].assign(sample=s) for s, a in normalized_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df_frac.groupby("sample")["TE_fraction"].median())

# %% [markdown]
# ## 13. Spatial plots — TE fraction (%)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = normalized_adatas[s]
    coords = a.obsm["spatial"]
    vals = a.obs["TE_fraction"].values * 100
    vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=8, cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% UMIs from TE")
plt.tight_layout()
plt.show()
