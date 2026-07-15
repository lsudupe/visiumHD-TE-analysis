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
# # Visium HD TE Analysis — QC by Cell (StarDist segmentation, real H&E image)
#
# Segmentation uses the real H&E microscope image (~8900x10600px) instead of
# the lower-resolution CytAssist fallback (~3200x3000px). Object construction
# up to segmentation (destripe, StarDist, both label expansions) lives in
# `scripts/build_stardist_cell_object.py`, run per sample via SLURM. This
# notebook picks up from the bin-level object: compares the raw image and
# the two expansion algorithms visually (as Ana does), aggregates to cells,
# checks segmentation quality, then runs the usual QC/normalization pipeline.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bin2cell as b2c
from scipy.stats import median_abs_deviation

# %% [markdown]
# ## Helper: spatial plot with low-first ordering + inverted magma
# Low values plotted first (high-TE cells render on top, not hidden
# underneath); magma_r so low = pale, high = dark purple/black.

# %%
def plot_spatial_te(ax, coords, vals, title, vmax=None, cmap="magma_r", label=""):
    order = np.argsort(vals)
    coords_sorted = coords[order]
    vals_sorted = vals[order]
    if vmax is None:
        vmax = np.percentile(vals, 99)
    sca = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1], c=vals_sorted,
                      s=10, cmap=cmap, vmin=0, vmax=vmax, alpha=0.9)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    return sca

# %% [markdown]
# ## 1. Load bin-level objects (post-StarDist, both expansion algorithms)

# %%
samples = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs", "Injured-12hrs", "Injured-24hrs"]

bin_adatas = {}
for s in samples:
    a = sc.read_h5ad(f"/ibex/user/medinils/data/objects/{s}_family_stardist_bins.h5ad")
    bin_adatas[s] = a
    n_nuclei = (a.obs["labels_he"] > 0).sum()
    print(s, a.shape, "| bins with a nucleus label:", n_nuclei)

# %% [markdown]
# ## 1a. Compare raw image vs. the two expansion algorithms, all 6 samples
# Columns: raw H&E crop (no segmentation overlay) -> raw nuclei (StarDist)
# -> expanded (distance-based) -> expanded (volume_ratio).

# %%
fig, axes = plt.subplots(6, 4, figsize=(24, 30), dpi=70)

for row, s in enumerate(samples):
    a = bin_adatas[s]
    row_mid = a.obs["array_row"].median()
    col_mid = a.obs["array_col"].median()
    mask = (
        (a.obs["array_row"] >= row_mid - 25) & (a.obs["array_row"] <= row_mid + 25) &
        (a.obs["array_col"] >= col_mid - 25) & (a.obs["array_col"] <= col_mid + 25)
    )

    bdata_raw = a[mask].copy()
    sc.pl.spatial(bdata_raw, color=[None], show=False, ax=axes[row, 0],
                  img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
    axes[row, 0].set_title(f"{s} — raw H&E", fontsize=9)

    for col, (label_key, title) in enumerate([
        ("labels_he", "Raw nuclei"),
        ("labels_expanded_distance", "Expanded: distance"),
        ("labels_expanded_volume", "Expanded: volume_ratio"),
    ], start=1):
        bdata = a[mask].copy()
        bdata = bdata[bdata.obs[label_key] > 0]
        bdata.obs[label_key] = bdata.obs[label_key].astype(str)
        sc.pl.spatial(bdata, color=[label_key], show=False, ax=axes[row, col],
                      img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer", legend_loc=None)
        axes[row, col].set_title(title, fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1b. Decision
# Based on the comparison above, we use `labels_expanded_volume` (same
# choice Ana made) -- biologically motivated per-cell expansion, rather
# than a fixed distance for every cell regardless of size.
# **Adjust EXPANSION_KEY below if the comparison suggests otherwise.**

# %%
EXPANSION_KEY = "labels_expanded_volume"

# %% [markdown]
# ## 1c. Aggregate bins -> cells (bin_to_cell), then apply Ana's bin_count filter

# %%
cell_adatas = {}
for s, a in bin_adatas.items():
    print(f"[{s}] aggregating bins -> cells ({EXPANSION_KEY}) ...")
    pseudo_sc = b2c.bin_to_cell(a, labels_key=EXPANSION_KEY, spatial_keys=["spatial"])
    print(f"[{s}] pseudo-cells before bin_count filter: {pseudo_sc.shape}")

    keep = pseudo_sc.obs["bin_count"] >= 3  # Ana's rule: drop cells built from <=2 bins
    pseudo_sc = pseudo_sc[keep].copy()
    print(f"[{s}] pseudo-cells after bin_count filter: {pseudo_sc.shape} ({keep.sum()}/{len(keep)} kept)")

    cell_adatas[s] = pseudo_sc
    pseudo_sc.write(f"/ibex/user/medinils/data/objects/{s}_family_stardist_cell.h5ad")

# %% [markdown]
# ## 1d. Cell size sanity check (bin_count distribution)
# Are there cells with an unusually high bin_count (possible fusion of two
# neighbouring cells into one label)?

# %%
fig, axes = plt.subplots(1, 6, figsize=(28, 4))
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    ax.hist(a.obs["bin_count"], bins=50, color="#02C39A", alpha=0.8)
    ax.axvline(3, color="#C97B2E", linestyle="--", linewidth=1.2, label="Ana's min=3")
    ax.set_title(s, fontsize=10)
    ax.set_xlabel("bins per cell", fontsize=8)
    ax.tick_params(labelsize=7)
    if ax is axes[0]:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

for s, a in cell_adatas.items():
    print(s, "bin_count: median =", a.obs["bin_count"].median(),
          "| p99 =", a.obs["bin_count"].quantile(0.99),
          "| max =", a.obs["bin_count"].max())

# %% [markdown]
# ## 2. QC metrics: counts, genes, mitochondrial %, and raw TE burden per cell
# Zero-count cells dropped before pct_counts_mt (avoids NaN-propagation bug
# hit in the earlier pipelines).

# %%
for s, a in cell_adatas.items():
    a.var["mt"] = a.var_names.str.lower().str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

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
# (qc_df is used by sections 3a and 4 below -- must run before both.)

# %%
qc_df = pd.concat([
    a.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt", "TE_burden", "log1p_TE_burden", "bin_count"]].assign(sample=s)
    for s, a in cell_adatas.items()
], axis=0).reset_index(drop=True)

print(qc_df.groupby("sample")[["total_counts", "n_genes_by_counts", "pct_counts_mt", "TE_burden", "bin_count"]].median())

# %% [markdown]
# ## 3a. Candidate thresholds + are high-QC cells just bigger cells?
# Defines MIN_COUNTS_FLOOR (applied later, in section 6) and bin_count
# upper-bound candidates (p99 / p99.5), and checks whether large bin_count
# correlates with high total_counts/n_genes/pct_counts_mt as expected for
# genuinely larger cells, or looks decoupled (possible segmentation error).

# %%
# %%
MIN_COUNTS_FLOOR = 10
BIN_COUNT_P99 = qc_df["bin_count"].quantile(0.99)
BIN_COUNT_P995 = qc_df["bin_count"].quantile(0.995)

TOTAL_COUNTS_P99 = qc_df["total_counts"].quantile(0.99)
N_GENES_P99 = qc_df["n_genes_by_counts"].quantile(0.99)
PCT_MT_P99 = qc_df["pct_counts_mt"].quantile(0.99)

# "grandes pero con poca señal" -- sospechosas de fusión/artefacto de segmentación
LOW_SIGNAL_THRESHOLD = qc_df["total_counts"].quantile(0.25)
suspicious = (qc_df["bin_count"] > BIN_COUNT_P99) & (qc_df["total_counts"] < LOW_SIGNAL_THRESHOLD)
print(f"Células grandes (bin_count>p99) con poca señal (total_counts<p25): {suspicious.sum()} de {len(qc_df)}")

print(f"bin_count: p99={BIN_COUNT_P99:.0f}, p99.5={BIN_COUNT_P995:.0f}")
print(f"total_counts p99={TOTAL_COUNTS_P99:.0f} | n_genes p99={N_GENES_P99:.0f} | pct_mt p99={PCT_MT_P99:.1f}")

palette = dict(zip(samples, sns.color_palette("tab10", n_colors=len(samples))))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, hline in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    [TOTAL_COUNTS_P99, N_GENES_P99, PCT_MT_P99],
):
    for s in samples:
        a = cell_adatas[s]
        ax.scatter(a.obs["bin_count"], a.obs[metric], s=5, alpha=0.4,
                   color=palette[s], edgecolors="none", label=s)

    # resalta las sospechosas: grandes + poca señal
    sub = qc_df[suspicious]
    ax.scatter(sub["bin_count"], sub[metric], s=18, color="black", marker="x",
               linewidths=1, label="grande + poca señal", zorder=5)

    ax.axvline(BIN_COUNT_P99, color="#C97B2E", linestyle="--", linewidth=1.2, label="p99 (bin_count)")
    ax.axvline(BIN_COUNT_P995, color="#8B0000", linestyle="--", linewidth=1.2, label="p99.5 (bin_count)")
    ax.axhline(hline, color="#028090", linestyle=":", linewidth=1.2, label="p99 (metric)")

    ax.set_xlabel("bin_count (cell size proxy)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. cell size")

axes[0].legend(markerscale=2, fontsize=6.5, loc="upper right")
plt.tight_layout()
plt.show()

# %%
print(f"Células grandes + poca señal: {suspicious.sum()}")

# aislar el cúmulo de alto % mitocondrial + célula pequeña
high_mt_small = (qc_df["pct_counts_mt"] > 90) & (qc_df["bin_count"] < BIN_COUNT_P99)
print(f"Células pequeñas con >90% mitocondrial: {high_mt_small.sum()} de {len(qc_df)} ({high_mt_small.sum()/len(qc_df)*100:.1f}%)")

# %%
# localizar algunas de las células más grandes de Control-GER
a = cell_adatas["Control-GER"]
largest = a.obs.sort_values("bin_count", ascending=False).head(5)
print(largest[["bin_count"]])

# coordenadas de esas células, para ubicarlas en la imagen
print(a[largest.index].obsm["spatial"])

# %%
a = cell_adatas["Control-GER"]
large_coords = np.array([
    [4927.93, 1694.47],
    [5140.54, 3638.81],
    [7277.20, 5811.10],
    [2814.45, 2861.17],
    [7443.07, 5290.40],
])

bin_a = bin_adatas["Control-GER"]
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, (cx, cy) in zip(axes, large_coords):
    # necesitamos array_row/array_col aproximados -- usamos un radio en píxeles sobre spatial directamente
    coords_bin = bin_a.obsm["spatial"]
    dist = np.sqrt((coords_bin[:, 0] - cx) ** 2 + (coords_bin[:, 1] - cy) ** 2)
    mask = dist < 150  # radio en píxeles, ajusta si hace falta
    bdata = bin_a[mask].copy()
    bdata = bdata[bdata.obs["labels_expanded_volume"] > 0]
    bdata.obs["labels_expanded_volume"] = bdata.obs["labels_expanded_volume"].astype(str)
    sc.pl.spatial(bdata, color=["labels_expanded_volume"], show=False, ax=ax,
                  img_key="0.5_mpp_150_buffer", basis="spatial", legend_loc=None)
plt.tight_layout()
plt.show()

# %%
a = cell_adatas["Control-GER"]  # o normalized_cell_adatas, si quieres ya normalizado
largest = a.obs.sort_values("bin_count", ascending=False).head(5)
print(largest.index.tolist())

marker_groups = {
    "macrophage": ["Adgre1", "Cd68", "Itgam", "Csf1r"],
    "mast": ["Cma1", "Tpsab1", "Kit", "Mcpt4"],
    "neutrophil": ["S100a8", "S100a9", "Ly6g", "Mpo"],
    "FAP": ["Pdgfra", "Ly6a", "Dcn"],
    "myonuclei": ["Myh1", "Myh2", "Acta1", "Ttn", "Des"],
}

for cell_id in largest.index:
    print(f"\n=== célula {cell_id} (bin_count={a.obs.loc[cell_id, 'bin_count']}) ===")
    for name, markers in marker_groups.items():
        present = [m for m in markers if m in a.var_names]
        if not present:
            continue
        idx = [a.var_names.get_loc(m) for m in present]
        vals = np.asarray(a[cell_id, idx].X.todense()).flatten()
        total = vals.sum()
        print(f"  {name}: {total:.2f}  (genes con señal: {[m for m,v in zip(present, vals) if v>0]})")

# %% [markdown]
# ## 3b. Where do the largest cells (candidate fused/oversized) sit spatially?
# Random scatter across the tissue = likely genuinely large cells (keep).
# Spatial clustering = possible segmentation artifact (fused neighbouring
# cells) -- would support adding an upper bin_count cutoff.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    is_large = a.obs["bin_count"] > BIN_COUNT_P99
    coords = a.obsm["spatial"]
    ax.scatter(coords[~is_large, 0], coords[~is_large, 1], c="lightgray", s=6, alpha=0.4)
    ax.scatter(coords[is_large, 0], coords[is_large, 1], c="#8B0000", s=14, alpha=0.85)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{s} ({is_large.sum()} cells > p99 bin_count)")
plt.suptitle("Largest cells (bin_count > p99), red", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Violin plots — raw QC metrics, 6 samples side by side
# With candidate thresholds marked (floor, Ana's bin_count minimum, p99).

# %%
MIN_COUNTS_FLOOR = 10
MAX_PCT_MT = 40

fig, axes = plt.subplots(1, 5, figsize=(32, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "log1p_TE_burden", "pct_counts_mt", "bin_count"],
    ["Total counts per cell", "Genes detected per cell", "log1p(TE burden) per cell", "% mitochondrial counts per cell", "Bins per cell"],
):
    sns.violinplot(data=qc_df, x="sample", y=metric, ax=ax, cut=0, inner="quartile")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

    if metric == "total_counts":
        ax.axhline(MIN_COUNTS_FLOOR, color="#C97B2E", linestyle="--", linewidth=1.2, label=f"floor={MIN_COUNTS_FLOOR}")
        ax.legend(fontsize=7)
    elif metric == "pct_counts_mt":
        ax.axhline(MAX_PCT_MT, color="#8B0000", linestyle="--", linewidth=1.2, label=f"max={MAX_PCT_MT}%")
        ax.legend(fontsize=7)
    elif metric == "bin_count":
        ax.axhline(3, color="#C97B2E", linestyle="--", linewidth=1.2, label="Ana's min=3")
        ax.axhline(BIN_COUNT_P99, color="#8B0000", linestyle="--", linewidth=1.2, label="p99")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Spatial plots — raw TE burden and mitochondrial % side by side

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], np.log1p(a.obs["TE_burden"].values), s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="log1p(TE burden)")
plt.suptitle("TE burden (raw, log1p) — by cell, StarDist segmentation", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], a.obs["pct_counts_mt"].values, s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% mt counts")
plt.suptitle("Mitochondrial % per cell", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. QC filtering — floor only (NOT MAD two-sided)
# Distributions may be bimodal (real biological populations, not just
# low-quality artifacts) -- a two-sided MAD filter would systematically
# remove one whole population. Only drop truly empty/degenerate cells.
# Ana's bin_count<=2 filter was already applied in section 1c.
# (MIN_COUNTS_FLOOR defined in section 3a.)

# %%
MIN_COUNTS_FLOOR = 10
MAX_PCT_MT = 40

filtered_cell_adatas = {}
for s, a in cell_adatas.items():
    keep = (a.obs["total_counts"] >= MIN_COUNTS_FLOOR) & (a.obs["pct_counts_mt"] <= MAX_PCT_MT)
    a_f = a[keep].copy()
    filtered_cell_adatas[s] = a_f
    print(s, a.shape, "->", a_f.shape, f"({keep.sum()/len(keep)*100:.1f}% kept)")

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
    a.write(f"/ibex/user/medinils/data/objects/{s}_family_stardist_cell_normalized.h5ad")

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
# ## 10. Spatial plots — which cells survive filtering, per sample

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]  # objeto ANTES de filtrar, con todas las células
    keep_mask = (a.obs["total_counts"] >= MIN_COUNTS_FLOOR) & (a.obs["pct_counts_mt"] <= MAX_PCT_MT)
    coords = a.obsm["spatial"]

    ax.scatter(coords[keep_mask, 0], coords[keep_mask, 1], c="lightgray", s=8, alpha=0.5, label="retained")
    ax.scatter(coords[~keep_mask, 0], coords[~keep_mask, 1], c="#8B0000", s=10, alpha=0.85, label="discarted")

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    n_discarded = (~keep_mask).sum()
    n_total = len(keep_mask)
    ax.set_title(f"{s} ({n_discarded}/{n_total} discarted, {n_discarded/n_total*100:.1f}%)")

axes[0].legend(markerscale=2, fontsize=8, loc="upper right")
plt.suptitle("Cells retained vs. discarted (floor + %mt)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. TE fraction per cell (raw TE UMIs / raw total UMIs)

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
# ## 12. Spatial plots — TE fraction (%)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = normalized_cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], a.obs["TE_fraction"].values * 100, s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% UMIs from TE")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Merge all 6 samples (outer join) + single gene/TE filter
# Last step before integration/clustering, after all per-sample
# QC/normalization/metrics are done.

# %%
adata_merged_cell = ad.concat(
    normalized_cell_adatas, join="outer", label="sample", fill_value=0, index_unique="-"
)
sc.pp.filter_genes(adata_merged_cell, min_cells=3)
print(adata_merged_cell.shape)
print(adata_merged_cell.obs["sample"].value_counts())

adata_merged_cell.write("/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_normalized_merged.h5ad")

# %% [markdown]
# ## 13a. Summary — cell counts through the pipeline, per sample

# %%
summary_rows = []
for s in samples:
    labels_col = bin_adatas[s].obs["labels_expanded_volume"]
    n_after_segmentation = labels_col[labels_col > 0].nunique()
    n_after_bincount = cell_adatas[s].n_obs
    n_after_floor = filtered_cell_adatas[s].n_obs
    n_final = normalized_cell_adatas[s].n_obs

    summary_rows.append({
        "sample": s,
        "cells_after_segmentation": n_after_segmentation,
        "cells_after_bin_count_filter": n_after_bincount,
        "cells_after_floor_filter": n_after_floor,
        "cells_final": n_final,
        "%_retained": round(n_final / n_after_segmentation * 100, 1),
    })

summary_df = pd.DataFrame(summary_rows).set_index("sample")
print(summary_df)

print("\nTotal cells, all samples, after full filtering:", summary_df["cells_final"].sum())
print("Merged object (section 13):", adata_merged_cell.shape)

# %% [markdown]
# ## Compare against the two earlier pipelines
# - Bin-level (8um, young_preprocessing.py): strong "V" pattern, likely
#   confounded by depth/composition.
# - Cell-level, Cellpose + real H&E image (per Core Labs, same image as
#   StarDist here): ~2,994 cells for Control-GER.
# - Cell-level, StarDist + real H&E image (this notebook): ~13,000 cells
#   for Control-GER -- same source image as Cellpose, so the ~4.3x
#   difference reflects the segmentation algorithms/thresholds themselves,
#   not image resolution as first assumed.
# - Nucleus-vs-cytoplasm TE localization check (done separately) found no
#   evidence of cytoplasmic enrichment beyond what compartment size alone
#   explains, consistently across all 6 samples.
