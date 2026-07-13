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
    # Pasada 1: calcular QC metrics para poder ver total_counts
    a.var["mt"] = a.var_names.str.lower().str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    # Ahora sí existe total_counts -- quitar bins en cero (causan NaN en pct_counts_mt)
    a = a[a.obs["total_counts"] > 0].copy()

    # Pasada 2: recalcular QC metrics ya sobre datos limpios
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    te_features = [v for v in a.var_names if "SoloTE" in v]
    te_idx = [a.var_names.get_loc(f) for f in te_features]
    a.obs["TE_burden"] = np.asarray(a.X[:, te_idx].sum(axis=1)).flatten()
    a.obs["log1p_TE_burden"] = np.log1p(a.obs["TE_burden"])

    adatas[s] = a

    print(
        s,
        "| bins:", a.n_obs,
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
# ## 6a. MAD thresholds — distributions and cutoffs, per sample
# Visualizing where the MAD-based threshold actually falls on each sample's
# own distribution, since the cutoff value differs sample to sample by design.

# %%
fig, axes = plt.subplots(2, 6, figsize=(28, 8))

for col, s in enumerate(samples):
    a = adatas[s]
    for row, (metric, label) in enumerate([
        ("total_counts", "Total counts"),
        ("n_genes_by_counts", "Genes detected"),
    ]):
        ax = axes[row, col]
        values = a.obs[metric]
        med = np.median(values)
        mad = median_abs_deviation(values)
        lo, hi = med - 5 * mad, med + 5 * mad

        ax.hist(values, bins=80, color="#028090", alpha=0.75)
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

# %% [markdown]
# ## What to look for here
# `TE_fraction` is the direct, depth-independent metric the PI asked for:
# what fraction of a bin's UMIs are TE-derived, computed from raw counts
# (not the normalized/log burden used earlier). A bin at 5% means literally
# 1 in 20 molecules detected there came from a TE.
#
# Compare this against the normalized TE burden plots (section 10):
# - If the spatial pattern looks similar between the two, the earlier
#   normalization was already doing a reasonable job of capturing the same
#   signal.
# - If patterns diverge, TE_fraction is the more trustworthy one to report,
#   since it isn't affected by the choice of `target_sum` or the log
#   transform -- it's a direct ratio of real counts.

# %%

# %%
timepoint_order = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs", "Injured-12hrs", "Injured-24hrs"]
families_of_interest = ["TE_SoloTE|ERVK", "TE_SoloTE|ERVL", "TE_SoloTE|B2", "TE_SoloTE|L1"]

fig, axes = plt.subplots(1, len(families_of_interest), figsize=(20, 5))
for ax, feat in zip(axes, families_of_interest):
    means, sems = [], []
    for s in timepoint_order:
        a = normalized_adatas[s]
        if feat in a.var_names:
            vals = np.asarray(a[:, feat].X.todense()).flatten()
        else:
            vals = np.array([np.nan])
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
    ax.bar(timepoint_order, means, yerr=sems, color="#C97B2E")
    ax.set_title(feat.replace("TE_SoloTE|", ""))
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %%
# --- 1. Cargar los 6 objetos a nivel subfamily ---
subfamily_adatas = {}
for s in samples:
    a = sc.read_h5ad(f"/ibex/user/medinils/data/objects/{s}_subfamily_8um.h5ad")
    subfamily_adatas[s] = a
    print(s, a.shape)

# --- 2. QC (mismo criterio MAD que ya usamos) + normalización, por muestra ---
subfamily_normalized = {}
for s, a in subfamily_adatas.items():
    sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=True, inplace=True)
    outlier_counts = mad_outlier(a.obs["total_counts"], nmads=5)
    outlier_genes = mad_outlier(a.obs["n_genes_by_counts"], nmads=5)
    a_f = a[~(outlier_counts | outlier_genes)].copy()

    a_f.layers["counts"] = a_f.X.copy()
    sc.pp.normalize_total(a_f, target_sum=1e4)
    sc.pp.log1p(a_f)

    subfamily_normalized[s] = a_f
    print(s, "->", a_f.shape)

# --- 3. Localizar los nombres exactos de subfamilias L1 ---
l1_features = [v for v in subfamily_normalized["Control-GER"].var_names if "L1" in v]
#print("Subfamilias L1 encontradas:", l1_features)

# --- 4. Bar chart, mismo estilo que antes, una por subfamilia L1 ---
fig, axes = plt.subplots(1, len(l1_features), figsize=(5 * len(l1_features), 5))
if len(l1_features) == 1:
    axes = [axes]

for ax, feat in zip(axes, l1_features):
    means, sems = [], []
    for s in timepoint_order:
        a = subfamily_normalized[s]
        if feat in a.var_names:
            vals = np.asarray(a[:, feat].X.todense()).flatten()
        else:
            vals = np.array([np.nan])
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
    ax.bar(timepoint_order, means, yerr=sems, color="#C97B2E")
    ax.set_title(feat.replace("TE_SoloTE|", ""))
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %%
l1_groups = {
    "L1_A": ["TE_SoloTE|L1MdA_I", "TE_SoloTE|L1MdA_II", "TE_SoloTE|L1MdA_III",
             "TE_SoloTE|L1MdA_IV", "TE_SoloTE|L1MdA_V", "TE_SoloTE|L1MdA_VI", "TE_SoloTE|L1MdA_VII"],
    "L1_T": ["TE_SoloTE|L1MdTf_I", "TE_SoloTE|L1MdTf_II", "TE_SoloTE|L1MdTf_III"],
    "L1_G": ["TE_SoloTE|L1MdGf_I", "TE_SoloTE|L1MdGf_II"],
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (group_name, features) in zip(axes, l1_groups.items()):
    means, sems = [], []
    for s in timepoint_order:
        a = subfamily_normalized[s]
        present = [f for f in features if f in a.var_names]
        if not present:
            means.append(np.nan); sems.append(np.nan)
            continue
        idx = [a.var_names.get_loc(f) for f in present]
        vals = np.asarray(a.X[:, idx].sum(axis=1)).flatten()
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
    ax.bar(timepoint_order, means, yerr=sems, color="#C97B2E")
    ax.set_title(group_name)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %%
l1_groups = {
    "L1_A": ["TE_SoloTE|L1MdA_I", "TE_SoloTE|L1MdA_II", "TE_SoloTE|L1MdA_III",
             "TE_SoloTE|L1MdA_IV", "TE_SoloTE|L1MdA_V", "TE_SoloTE|L1MdA_VI", "TE_SoloTE|L1MdA_VII"],
    "L1_T": ["TE_SoloTE|L1MdTf_I", "TE_SoloTE|L1MdTf_II", "TE_SoloTE|L1MdTf_III"],
    "L1_G": ["TE_SoloTE|L1MdGf_I", "TE_SoloTE|L1MdGf_II"],
}

rows = []
for s in timepoint_order:
    a = subfamily_normalized[s]  # tiene layers["counts"] con los crudos
    for group_name, features in l1_groups.items():
        present = [f for f in features if f in a.var_names]
        if not present:
            continue
        idx = [a.var_names.get_loc(f) for f in present]
        raw_counts = np.asarray(a.layers["counts"][:, idx].sum(axis=1)).flatten()

        rows.append({
            "sample": s,
            "group": group_name,
            "mean_UMI_per_bin": raw_counts.mean(),
            "pct_bins_with_0_UMI": (raw_counts == 0).mean() * 100,
            "pct_bins_with_1plus_UMI": (raw_counts >= 1).mean() * 100,
            "total_UMIs_in_sample": raw_counts.sum(),
        })

umi_summary = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print(umi_summary.pivot(index="sample", columns="group", values="mean_UMI_per_bin"))
print()
print(umi_summary.pivot(index="sample", columns="group", values="pct_bins_with_0_UMI"))
