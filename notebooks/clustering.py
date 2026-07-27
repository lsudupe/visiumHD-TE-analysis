## ---
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
# # Visium HD TE Analysis — Clustering (StarDist cell object, all 8 samples)
#
# Continues from `qc_stardist_cell.py` (Sections 1-14a), which produces the
# merged object with QC + normalization + Level-A classification already
# done. This notebook covers PROJECT_CONTEXT.md Section 7's open item:
# batch-aware HVG -> Harmony -> Leiden -> annotate clusters with marker
# panels (Level B, cross-checked against the per-cell `final_call` from
# Section 13 of the QC notebook). Differential TE expression by
# cluster/condition is the next step, not covered here.
#
# No mitochondrial filter is applied here (Finding #9: decided after
# clustering).
#
# ## Change log (this revision)
# 1. **Removed the checkpoint auto-resume mechanism entirely.** The old
#    `USE_CHECKPOINT = os.path.exists(CHECKPOINT)` silently skipped Sections
#    1a-6 (QC filter, HVG, PCA, Harmony, Leiden) whenever a stale checkpoint
#    file happened to exist on disk. Confirmed on the object that had been
#    produced this way: recomputing each group's own 5th percentile of
#    `n_genes_by_counts` and checking what fraction fell below it came back
#    at ~5% everywhere, which is only possible if the Section 1a filter had
#    never actually run. Every section below now always executes, top to
#    bottom, from the raw unclustered merged object.
# 2. **Fixed `sample_to_condition`**: ST0001=Injured-6hrs, ST0002=
#    Control-young (was silently swapped).
# 3. **Section 1a filter extended**: quantile-5 cutoff on BOTH
#    `n_genes_by_counts` AND `total_counts` (previously genes only), each
#    computed independently per `(sample, labels_joint_source)` group, plus
#    a hard `MIN_GENES_FLOOR=20` on top. This resolved a KNN-graph
#    fragmentation problem (23-32 disconnected components) down to 1 --
#    see Section 5a.
# 4. **Section 6g (finalize clusters) generalized**: tiny clusters by
#    `MIN_CLUSTER_SIZE` AND clusters with no clean marker in the dotplot
#    (`NO_MARKER_CLUSTERS`, filled in by hand after inspecting Section 6f's
#    output on THIS run -- cluster IDs are not stable across reruns, do not
#    reuse IDs from a previous session) both get folded into
#    `low_confidence`, instead of a size-only cut.
# 5. **New Section 6h**: drops `low_confidence` cells and recomputes
#    neighbors + UMAP before the final save, so the saved embedding isn't
#    still shaped by cells that were ultimately excluded from annotation.
# 6. **Removed the old Sections 8/9** (score_genes panels + t-test DE,
#    inline here). Superseded by the separate `cell_type_annotation.py`,
#    which does the same job with z-scored panel heatmaps, a candidate-
#    identity table, filtered rank_genes_groups, and CSV exports -- no
#    reason to duplicate a weaker version of it here.
#
# **Tried and reverted**: a `sc.pp.regress_out(["total_counts",
# "pct_counts_mt"])` step before PCA was added and then removed. Two
# reasons: (a) it silently corrupted `adata.X` for every downstream use
# that assumes log1p-normalized values (rank_genes_groups' logfoldchanges
# came back NaN for ~30/32 clusters -- regress_out produces residuals,
# which can be negative, and expm1() on a negative residual breaks the
# fold-change math); (b) even before that was caught, comparing the
# silhouette sweep with vs. without it showed no real difference
# (-0.033/-0.010/0.001/0.012/0.027/0.026/0.019 vs. -0.030/-0.007/-0.002/
# 0.008/0.012/0.020/0.020 across the same 7 resolutions) -- the "rays" in
# the UMAP were a real depth effect, but not one that was hurting the
# actual clustering quality in the space Leiden runs on, just how it
# looks projected to 2D. Not worth the corrupted-X risk for a purely
# cosmetic fix. If this gets revisited, don't overwrite `adata.X` in
# place -- write to a separate layer and point PCA's `layer=` argument at
# it instead, leaving `adata.X` untouched for DE/dotplots.
#
# **IMPORTANT**: because of the filter changes (#3), every downstream
# number from a previous session (n_pcs, resolution, cluster IDs,
# marker-less cluster IDs) is stale and gets re-derived by actually running
# the notebook again -- nothing below is hardcoded from a prior run except
# where explicitly noted as "empirically confirmed on THIS filter" (n_pcs=15
# gave 1 connected component, confirmed twice independently -- with and
# without the reverted regress_out step, same result both times).

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
import harmonypy as hm
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

# %% [markdown]
# ## 1. Load merged object
# No checkpoint/resume mechanism -- runs end to end every time (QC filter
# -> HVG -> PCA -> Harmony -> Leiden -> annotation), always from the raw
# unclustered merged object. This filter is not optional, so there is no
# correct default that skips it.

# %%
adata = sc.read_h5ad("/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_normalized_merged.h5ad")
print("Loaded from the original (unclustered) merged object:", adata)

samples = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs",
           "Injured-12hrs", "Injured-24hrs", "ST0001", "ST0002"]

# condition_label was never actually saved as a column in the merged
# object (only existed as a plotting-label dict in qc_stardist_cell.py) --
# rebuild it here from `sample`, since it's 1:1 in this design.
# Correct mapping, per explicit confirmation: ST0001=Injured-6hrs,
# ST0002=Control-young (matches qc_stardist_cell.py's own sample_labels
# dict). Previously swapped here -- fixed.
sample_to_condition = {
    "Control-GER": "Control-GER",
    "Control-Old": "Control-Old",
    "Injured-1hrs": "Injured-1hrs",
    "Injured-3hrs": "Injured-3hrs",
    "Injured-12hrs": "Injured-12hrs",
    "Injured-24hrs": "Injured-24hrs",
    "ST0001": "Injured-6hrs",
    "ST0002": "Control-young",
}
adata.obs["condition_label"] = adata.obs["sample"].map(sample_to_condition).astype("category")
print(adata.obs["condition_label"].value_counts())

print(adata.obs["sample"].value_counts())
# X is already normalize_total(1e4) + log1p, computed per sample (Section 7
# of the QC notebook). layers['counts'] = raw.

# %% [markdown]
# ## 1a. Additional QC filter — per-(sample, labels_joint_source) lower-tail threshold
# The original qc_stardist_cell.py filter (Section 6d) only applied a floor
# of 10 raw UMIs, no minimum on genes detected. Too permissive: caused
# disconnected KNN-graph components downstream (isolated cells that no
# Leiden resolution could merge into the main graph).
#
# A single fixed threshold (first attempt: MIN_COUNTS_FLOOR_EXTRA=100,
# MIN_GENES=50) doesn't work here for two compounding reasons, confirmed
# by the 1a-dist violin plots below:
#   1. Sample-level depth confound: within primary cells alone, at a fixed
#      cutoff, Control-GER lost 16.9% vs. Injured-1hrs 54.2% -- a ~3.2x gap
#      not explained by primary/secondary composition (nearly identical,
#      0.71-0.85 primary fraction across all 8 samples). Injured-1h/3h/12h
#      are all substantially lower-depth than controls.
#   2. primary/secondary scale mismatch WITHIN the same sample: e.g.
#      Control-GER primary median ~190 vs. secondary median ~2100
#      total_counts -- any shared cutoff always penalizes primary
#      disproportionately (primary is individual interstitial nuclei,
#      secondary is whole myofibers, structurally different scales).
#
# Fix: compute the percentile threshold independently within each
# (sample, labels_joint_source) group, cutting only the LOW tail (no upper
# cap -- we don't want to remove genuinely large/high-signal cells).
#
# Filters on BOTH n_genes_by_counts AND total_counts, each with its own
# quantile-5 threshold computed separately per (sample, source) group, kept
# only if a cell passes both -- a cell can be low-UMI/high-gene-diversity
# or high-UMI/low-gene-diversity (e.g. one dominant transcript, see Section
# 4a/4b), so the two metrics aren't fully redundant.
#
# Plus a hard absolute floor (MIN_GENES_FLOOR=20) on top: the per-group 5th
# percentile alone still let very-low-complexity cells through in low-depth
# groups (e.g. Injured-1h/3h, where even the 5th percentile of
# n_genes_by_counts sits at 10-16). This absolute floor is what actually
# resolved the KNN-graph fragmentation (Section 5a: 23-32 connected
# components -> 1, empirically confirmed).
#
# This is a deliberate trade-off (relative per-group cutoff vs. a single
# "objective" number) -- document in PROJECT_CONTEXT.md.

# %%
QUALITY_PERCENTILE = 5  # drop the bottom 5% of EACH metric, within each (sample, source) group
FILTER_METRICS = ["n_genes_by_counts", "total_counts"]
MIN_GENES_FLOOR = 20  # hard absolute floor, on top of the per-group quantile cuts below

keep_1a = pd.Series(True, index=adata.obs_names, dtype=bool)
threshold_log = []
for s in samples:
    for src in adata.obs["labels_joint_source"].unique():
        mask_group = (adata.obs["sample"] == s) & (adata.obs["labels_joint_source"] == src)
        if mask_group.sum() == 0:
            continue
        row = {"sample": s, "source": src, "n_cells": mask_group.sum()}
        group_keep = pd.Series(True, index=adata.obs_names[mask_group])
        for metric in FILTER_METRICS:
            thresh = adata.obs.loc[mask_group, metric].quantile(QUALITY_PERCENTILE / 100)
            group_keep &= (adata.obs.loc[mask_group, metric] >= thresh).values
            row[f"{metric}_threshold"] = thresh
        group_keep &= (adata.obs.loc[mask_group, "n_genes_by_counts"] >= MIN_GENES_FLOOR).values
        keep_1a.loc[mask_group] = group_keep.values
        row["n_kept"] = int(group_keep.sum())
        threshold_log.append(row)

# keep_1a must stay boolean -- pandas can silently coerce it to
# object/int during mixed .loc assignment, which turns ~keep_1a into
# bitwise complement (-1/-2) instead of boolean negation and breaks
# every downstream .loc[~keep_1a, ...] call with a cryptic KeyError.
assert keep_1a.dtype == bool, f"keep_1a is not bool (got {keep_1a.dtype}) -- fix before proceeding"

threshold_df = pd.DataFrame(threshold_log)
print(threshold_df.to_string(index=False))
print(f"\nGroups where the {MIN_GENES_FLOOR}-gene floor is STRICTER than the group's own "
      f"5th percentile (i.e. the floor is doing extra work beyond the quantile cut):")
print(threshold_df[threshold_df["n_genes_by_counts_threshold"] < MIN_GENES_FLOOR]
      [["sample", "source", "n_genes_by_counts_threshold"]].to_string(index=False))

n_before = adata.n_obs
print("\nCells lost per sample:")
print(adata.obs.loc[~keep_1a, "sample"].value_counts())
print("\n% lost per sample (won't be exactly ~5% -- AND of two quantile cuts + "
      "the absolute floor drops more than 5% per group, since the tails don't "
      "fully overlap):")
print((adata.obs.loc[~keep_1a, "sample"].value_counts() / adata.obs["sample"].value_counts() * 100).round(1))

# %% [markdown]
# ## 1a-check. Does the filter disproportionately remove primary
# (StarDist, immune/stromal) cells vs. secondary (Cellpose, muscle)?
# Both populations are already merged into this object (qc_stardist_cell.py
# Section 14) -- nothing here removes secondary, this only checks whether a
# single fixed threshold filters the two populations at very different
# rates. Primary cells are naturally smaller/fewer counts by construction
# (individual interstitial nuclei vs. whole myofibers) -- a single fixed
# threshold could disproportionately remove primary cells (the actual
# target compartment, per PROJECT_CONTEXT.md Section 1) while barely
# touching the muscle compartment.

# %%
print("QC distribution by labels_joint_source, BEFORE filtering:")
print(adata.obs.groupby("labels_joint_source")[["total_counts", "n_genes_by_counts"]].median())

print("\n% of each source category that would be LOST by the 1a filter:")
loss_by_source = (~keep_1a).groupby(adata.obs["labels_joint_source"]).mean() * 100
print(loss_by_source.round(1))

print("\nCounts lost/kept by source:")
print(pd.crosstab(adata.obs["labels_joint_source"], keep_1a, margins=True))

# %% [markdown]
# ## 1a-dist. Distributions by labels_joint_source — UMIs, genes, mito %

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    ["Total counts per cell", "Genes detected per cell", "% mitochondrial counts per cell"],
):
    sns.violinplot(data=adata.obs, x="labels_joint_source", y=metric, ax=ax, cut=0, inner="quartile")
    ax.set_title(title)
    if metric != "pct_counts_mt":
        ax.set_yscale("log")
plt.tight_layout()
plt.show()

print(adata.obs.groupby("labels_joint_source")[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe().T)

fig, axes = plt.subplots(3, 1, figsize=(18, 16))
for ax, metric, title in zip(
    axes,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    ["Total counts per cell", "Genes detected per cell", "% mitochondrial counts per cell"],
):
    sns.violinplot(data=adata.obs, x="sample", y=metric, hue="labels_joint_source", ax=ax, cut=0, inner="quartile", split=True)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    if metric != "pct_counts_mt":
        ax.set_yscale("log")
plt.tight_layout()
plt.show()

# %%
adata_before_1a = adata  # keep a reference for the spatial plot in 1b, before subsetting
adata = adata[keep_1a].copy()
print(f"\nFiltered {n_before - adata.n_obs} cells ({(n_before - adata.n_obs)/n_before*100:.1f}%), {adata.n_obs} remain")

# %% [markdown]
# ## 1b. Spatial plot — which cells were lost to the 1a filter, per sample
# Random scatter across the tissue = likely genuinely low-signal cells
# (expected, safe to drop). Spatial clustering in one region = worth a
# closer look before trusting the filter blindly.

# %%
for s in samples:
    a_s = adata_before_1a[adata_before_1a.obs["sample"] == s]
    coords = a_s.obsm["spatial"]
    lost = ~keep_1a[adata_before_1a.obs["sample"] == s].values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[~lost, 0], coords[~lost, 1], c="lightgray", s=6, alpha=0.4, label="retained")
    ax.scatter(coords[lost, 0], coords[lost, 1], c="#C0392B", s=10, alpha=0.85, label="lost (1a filter)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    n_lost = lost.sum()
    n_total = len(lost)
    ax.set_title(f"{s} ({n_lost}/{n_total} lost, {n_lost/n_total*100:.1f}%)")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()

del adata_before_1a  # free memory, not needed past this point

# %% [markdown]
# ## 2. Separate gene vs. TE features (SoloTE)
# Cell-type clustering is done on genes only; TE burden per cluster/
# condition is a downstream analysis on this same annotated object (TEs are
# excluded from HVG/PCA to avoid biasing the embedding with TE dropout,
# which is much noisier than gene dropout).

# %%
adata.var["is_te"] = adata.var_names.str.contains("SoloTE")
print("genes:", (~adata.var["is_te"]).sum(), "| TE features:", adata.var["is_te"].sum())

# %% [markdown]
# ## 3. Batch-aware HVG (seurat_v3, genes only, batch_key='sample')
# flavor='seurat_v3' only uses n_top_genes (min_mean/max_mean/min_disp do
# not apply with this flavor). batch_key='sample' computes HVGs per sample
# and combines the rankings -- important with 8 very different
# samples/conditions (Control vs. Injured 1h...24h) so technical variance
# between samples isn't mistaken for real biological variance.

# %%
adata_genes = adata[:, ~adata.var["is_te"]].copy()

sc.pp.highly_variable_genes(
    adata_genes,
    layer="counts",
    n_top_genes=2000,
    flavor="seurat_v3",
    batch_key="sample",
)
sc.pl.highly_variable_genes(adata_genes)

# Propagate the result to the full object (TEs are never HVGs)
for col in ["highly_variable", "highly_variable_rank", "means", "variances", "variances_norm"]:
    adata.var[col] = np.nan if col != "highly_variable" else False
    adata.var.loc[adata_genes.var_names, col] = adata_genes.var[col].values
adata.var["highly_variable"] = adata.var["highly_variable"].fillna(False).astype(bool)

print("HVGs selected:", adata.var["highly_variable"].sum())

# %% [markdown]
# ## 4. PCA

# %%
sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
sc.pl.pca(adata, color=["sample", "condition_label"])

# %% [markdown]
# ## 4a-npcs. How many PCs to use for neighbors/Harmony — variance check
# n_pcs controls how many of the 50 Harmony PCs go into the neighbor graph.
# Too few loses real biological signal; too many adds noise from
# low-variance PCs, which can worsen graph fragmentation. This is a
# first-pass heuristic only -- cross-checked empirically against
# connected_components in Section 5a, since that's the actual problem
# being debugged.

# %%
var_ratio = adata.uns["pca"]["variance_ratio"]
cum_var = np.cumsum(var_ratio)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
ax.axhline(0.8, color="gray", linestyle="--", label="80%")
ax.axhline(0.9, color="gray", linestyle=":", label="90%")
ax.axvline(15, color="red", linestyle="--", label="candidate n_pcs=15")
ax.axvline(30, color="orange", linestyle="--", label="candidate n_pcs=30")
ax.set_xlabel("number of PCs")
ax.set_ylabel("cumulative variance explained")
ax.legend()
plt.show()

print(f"Variance explained by first 15 PCs: {cum_var[14]*100:.1f}%")
print(f"Variance explained by first 30 PCs: {cum_var[29]*100:.1f}%")
print(f"PCs needed for 80%: {np.searchsorted(cum_var, 0.8) + 1}")
print(f"PCs needed for 90%: {np.searchsorted(cum_var, 0.9) + 1}")

# %% [markdown]
# ## 4a. PCA outlier diagnostic — investigate the "ray" pattern
# Historically traced to top-loading genes (mt-Nd1, Myl1, Myh4, Actn3,
# Myh1 -- muscle/mitochondrial) and low-complexity cells where
# normalize_total() inflates whichever single transcript happens to
# dominate the (very few) raw counts. Among these, a subset have a TE as
# their single dominant transcript (L1/Alu/B2 families), strongly skewed
# toward Injured-12h/24h -- see Finding #10 investigation below (Section
# 4b).

# %%
loadings = pd.DataFrame(
    adata.varm["PCs"][:, :2], index=adata.var_names, columns=["PC1", "PC2"]
)
print(loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(15))
print(loadings.reindex(loadings["PC2"].abs().sort_values(ascending=False).index).head(15))

sc.pl.pca(adata, color=["total_counts", "pct_counts_mt", "n_genes_by_counts"], components=["1,2"])
sc.pl.pca(adata, color=["Myh1", "Myh4", "mt-Nd1", "Ldha"], components=["1,2"])

# %% [markdown]
# ## 4b. TE-dominant outlier cells — raw-counts confirmation (Finding #10)
# Identify the extreme-tail cells, find each one's single most-abundant
# raw transcript, and flag the ones dominated by a TE. Then confirm on RAW
# (pre-normalization) TE_fraction -- if the effect persists there, it is
# not a normalize_total() artifact.

# %%
pc = adata.obsm["X_pca"][:, :2]
extreme_mask = pc[:, 0] < np.percentile(pc[:, 0], 1)  # adjust percentile to match where the rays sit
sub = adata[extreme_mask]

raw = sub.layers["counts"]
top_gene_idx = np.asarray(raw.argmax(axis=1)).flatten()
top_genes = sub.var_names[top_gene_idx]
print(pd.Series(top_genes).value_counts().head(10))

te_dominant_mask = pd.Series(top_genes, index=sub.obs_names).str.contains("SoloTE")
te_dominant_cells = sub[te_dominant_mask.values]
print(te_dominant_cells.n_obs, "TE-dominant cells")
print(te_dominant_cells.obs["sample"].value_counts())

adata.obs["te_dominant_outlier"] = False
adata.obs.loc[te_dominant_cells.obs_names, "te_dominant_outlier"] = True

# Raw TE_fraction check (Section 11 of the QC notebook: raw TE-UMIs /
# raw total-UMIs, computed pre-normalization -- not affected by the
# normalize_total() inflation on low-count cells)
sub_12h = adata[adata.obs["sample"] == "Injured-12hrs"]
is_te_dom_12h = sub_12h.obs_names.isin(te_dominant_cells.obs_names)
print("TE-dominant (raw TE_fraction):")
print(sub_12h.obs.loc[is_te_dom_12h, "TE_fraction"].describe())
print("\nRest of Injured-12hrs (raw TE_fraction):")
print(sub_12h.obs.loc[~is_te_dom_12h, "TE_fraction"].describe())
# Still open: raw-counts specificity check (is a TE dominant among
# low-complexity cells more often than chance, vs. a normal gene being
# dominant) to distinguish "real localized TE reactivation" from "generic
# RNA degradation in damaged/necrotic cells that happens to hit TEs". H&E
# cross-check for spatial correlation with the lesion site also pending.

# Spatial visualization (raw TE_fraction, top decile highlighted for contrast)
coords = sub_12h.obsm["spatial"]
te_frac = sub_12h.obs["TE_fraction"].values * 100
threshold = np.percentile(te_frac, 90)
is_high = te_frac >= threshold

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(coords[~is_high, 0], coords[~is_high, 1], c="lightgray", s=6, alpha=0.4)
sca = ax.scatter(coords[is_high, 0], coords[is_high, 1], c=te_frac[is_high],
                  cmap="viridis_r", s=20, alpha=1.0, vmin=threshold, vmax=te_frac.max())
plt.colorbar(sca, ax=ax, shrink=0.6, label="% UMIs from TE (top 10%, raw)")
ax.scatter(coords[is_te_dom_12h, 0], coords[is_te_dom_12h, 1],
           facecolors="none", edgecolors="red", s=60, linewidths=1.5, label="TE-dominant outlier")
ax.invert_yaxis()
ax.set_aspect("equal")
ax.axis("off")
ax.legend(markerscale=1, fontsize=8)
ax.set_title("Injured-12hrs — top 10% raw TE fraction + TE-dominant outliers (red)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Batch integration — Harmony
# batch_key = 'sample' (8 samples = 8 conditions, one replicate each, so
# 'sample' and 'condition' are equivalent here). **Important caveat**: with
# 1 sample per condition, Harmony cannot distinguish technical batch effect
# from real condition-driven biology -- see Finding #10 / Section 6d below
# for why this matters when a small, condition-specific population (like
# the TE-dominant cells above) needs to survive integration rather than
# being dissolved by it.
#
# NOTE: `sc.external.pp.harmony_integrate()` is NOT used here -- it has a
# known bug with harmonypy>=0.1.0 (confirmed on 2.0.0, the version
# installed here): the scanpy wrapper still applies `Z_corr.T`, but
# harmonypy now already returns Z_corr in (n_obs, n_pcs) format, so the
# extra transpose corrupts the shape (see scanpy issues #3940, #3962).
# Calling harmonypy directly avoids the bug.

# %%
data_mat = adata.obsm["X_pca"]
print(data_mat.shape)  # should be (n_cells, 50)

ho = hm.run_harmony(data_mat, adata.obs, "sample", verbose=True)

# harmonypy>=0.1.0 (incl. 2.0.0 here) returns Z_corr already as
# (n_obs, n_pcs) -- do NOT transpose. Verify shape before trusting it.
print(ho.Z_corr.shape)  # should be (n_cells, 50)
adata.obsm["X_pca_harmony"] = ho.Z_corr

# %% [markdown]
# ## 5a. Empirical check — which n_pcs gives the healthiest neighbor graph?
# More directly relevant than the variance-explained heuristic in 4a-npcs.
# Recomputes neighbors at several n_pcs values and counts connected
# components for each -- expensive (recomputes neighbors N times), but
# gives a direct empirical answer instead of a variance-based guess.
#
# Empirically confirmed on the two-metric-quantile + MIN_GENES_FLOOR=20
# filter (Section 1a): n_pcs=15/20/30/50 all gave exactly 1 connected
# component (down from 23-32 pre-floor) -- the graph fragmentation problem
# is resolved. Confirmed twice, with and without the (since-reverted)
# regress_out step -- same result either way, so this number doesn't
# depend on that decision.

# %%
from scipy.sparse.csgraph import connected_components  # local import, in case the top imports cell wasn't re-run this session

npcs_component_counts = {}
for n_pcs_test in [15, 20, 30, 50]:
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, n_pcs=n_pcs_test, key_added=f"test_pcs{n_pcs_test}")
    n_comp, _ = connected_components(adata.obsp[f"test_pcs{n_pcs_test}_connectivities"], directed=False)
    npcs_component_counts[n_pcs_test] = n_comp
    print(f"n_pcs={n_pcs_test}: {n_comp} connected components")
# Pick the n_pcs value with the fewest connected components (ideally 1,
# or close to it) and use it in Section 6 below.
FINAL_N_PCS = min(npcs_component_counts, key=npcs_component_counts.get)
print(f"\nChosen n_pcs based on connectivity: {FINAL_N_PCS}")

# %% [markdown]
# ## 6. Neighbors, UMAP and Leiden resolution sweep (on Harmony-corrected space)

# %%
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, n_pcs=FINAL_N_PCS)
sc.tl.umap(adata)

# %% [markdown]
# ## 6a. Resolution sweep — how many clusters "make sense"?
# There's no single correct number a priori; sweep several resolutions and
# compare against the checks in 6b-6f before committing to one for
# annotation.

# %%
resolutions = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]

# Clear any stale cluster columns from a previous run with different
# neighbors/objective_function settings -- otherwise the `if key not in
# adata.obs.columns` guard below silently skips recomputation and prints
# old results.
for res in resolutions:
    key = f"clusters_r{res}"
    if key in adata.obs.columns:
        del adata.obs[key]

for res in resolutions:
    key = f"clusters_r{res}"
    sc.tl.leiden(
        adata, key_added=key, resolution=res,
        flavor="igraph", n_iterations=2, directed=False,
        objective_function="modularity",  # default is CPM with flavor="igraph",
        # which needs resolution ~0.0001-0.01 to behave sensibly -- with
        # typical resolution values (0.1-2) it produced 160-234 clusters
        # regardless of resolution. modularity matches legacy leidenalg
        # behavior and responds correctly to this resolution range.
    )
    n_clusters = adata.obs[key].nunique()
    sizes = adata.obs[key].value_counts()
    print(f"res={res}: {n_clusters} clusters | smallest={sizes.min()} cells | "
          f"largest={sizes.max()} cells | clusters <50 cells={sum(sizes < 50)}")

# %% [markdown]
# ## 6b. Stability across resolutions (clustree-style)
# If a cluster cleanly splits in two as resolution increases, that's real
# substructure. If cells reshuffle chaotically between neighboring
# resolutions (not nested splits), the clustering isn't robust there.

# %%
for i in range(len(resolutions) - 1):
    r1, r2 = resolutions[i], resolutions[i + 1]
    ct = pd.crosstab(adata.obs[f"clusters_r{r1}"], adata.obs[f"clusters_r{r2}"])
    frac = ct.div(ct.sum(axis=1), axis=0)
    n_splits = (frac > 0.1).sum(axis=1)  # r2 clusters receiving >10% of an r1 cluster
    print(f"{r1} -> {r2}: each cluster splits into {n_splits.mean():.1f} clusters on average "
          f"(1 = stable/no split; >2 = ambiguous/unstable split)")

# %% [markdown]
# ## 6c. Silhouette score per resolution (Harmony-space separation quality)
# Subsampled -- silhouette is O(n^2) and won't finish on 147k cells.

# %%
N_SUBSAMPLE = 15000
rng = np.random.default_rng(0)
sub_idx = rng.choice(adata.n_obs, size=min(N_SUBSAMPLE, adata.n_obs), replace=False)

sil_scores = {}
for res in resolutions:
    labels_sub = adata.obs[f"clusters_r{res}"].values[sub_idx]
    if adata.obs[f"clusters_r{res}"].nunique() < 2:
        continue
    score = silhouette_score(adata.obsm["X_pca_harmony"][sub_idx], labels_sub)
    sil_scores[res] = score
    print(f"res={res}: silhouette={score:.3f}")
# Higher = more compact, better-separated clusters -- not the only metric
# that matters; a low silhouette on a genuine biological continuum (e.g. a
# differentiation trajectory, cf. Finding #20 -- fiber-type continuum) is
# expected and not necessarily a problem. Confirmed twice (post
# 2-metric-quantile filter), with and without the since-reverted
# regress_out step: silhouette stayed low across the board both times
# (-0.03 to 0.02, no sharp peak, nearly identical numbers either way) --
# consistent with continuum biology, and confirms the regress_out
# question doesn't affect this metric one way or the other.

# %% [markdown]
# ## 6d. Do samples mix well within each cluster? (post-Harmony)
# A cluster dominated almost entirely by one sample is suspicious -- either
# insufficient integration (Harmony didn't correct that batch effect), or a
# genuinely condition-specific cell type/state (like the TE-dominant
# population from Injured-12hrs, Finding #10 -- there we WANT it to show up
# this way, not be forced to mix).

# %%
RES_TO_CHECK = 0.2  # last chosen value: gave a large drop in tiny/
# single-sample clusters vs. higher resolutions (only 3-6 clusters <50
# cells, down from ~42 pre-floor) with almost the same structure as
# res=0.3 at 7 fewer clusters, AND the worst (most negative) silhouette of
# the whole sweep at res=0.1 -- re-confirm against 6a-6c on this run before
# trusting it again.

sample_props = pd.crosstab(adata.obs[f"clusters_r{RES_TO_CHECK}"], adata.obs["sample"], normalize="index")
max_sample_frac = sample_props.max(axis=1)
cluster_entropy = sample_props.apply(lambda row: entropy(row + 1e-12), axis=1)
max_entropy = np.log(adata.obs["sample"].nunique())

summary = pd.DataFrame({
    "n_cells": adata.obs[f"clusters_r{RES_TO_CHECK}"].value_counts(),
    "max_sample_frac": max_sample_frac,
    "dominant_sample": sample_props.idxmax(axis=1),
    "entropy_norm": cluster_entropy / max_entropy,  # 0 = single sample, 1 = perfectly spread across all 8
}).sort_values("entropy_norm")

print(summary)
print("\nClusters with >70% of a single sample (inspect carefully):")
print(summary[summary["max_sample_frac"] > 0.7])

# %% [markdown]
# ## 6e. Visual — UMAP and PCA (Harmony vs. raw) colored by sample/cluster

# %%
sc.pl.umap(adata, color=[f"clusters_r{RES_TO_CHECK}", "sample", "condition_label"], ncols=1)

# Depth/QC diagnostic -- check this every time the filter changes. Shows a
# dense high-depth blob plus uniformly-low-depth "rays" radiating
# outward -- a real sequencing-depth effect on the embedding, confirmed
# via a regress_out experiment (since reverted, see the change-log note
# above): it didn't move silhouette or any downstream clustering metric,
# so it's read as mostly a 2D-projection/visual effect, not something
# distorting the actual cluster structure Leiden works on. Worth
# rechecking here each run in case that changes.
sc.pl.umap(adata, color=["n_genes_by_counts", "total_counts", "pct_counts_mt"], cmap="viridis", ncols=3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, rep, title in zip(axes, ["X_pca", "X_pca_harmony"], ["PCA (uncorrected)", "PCA + Harmony"]):
    coords = adata.obsm[rep][:, :2]
    for s in adata.obs["sample"].unique():
        mask = adata.obs["sample"] == s
        ax.scatter(coords[mask, 0], coords[mask, 1], s=2, alpha=0.3, label=s)
    ax.set_title(title)
axes[0].legend(markerscale=4, fontsize=7, loc="upper right")
plt.tight_layout()
plt.show()
# If samples are still as separated in X_pca_harmony as in X_pca, Harmony
# isn't doing its job -- revisit batch_key/theta.

# %% [markdown]
# ## 6f. Marker separation — does each cluster have a clear identity?
# Well-defined clusters should have marker genes with high logFC, expressed
# in a high % of their own cells and low % elsewhere. Use this output to
# fill NO_MARKER_CLUSTERS in Section 6g below -- cluster IDs are re-derived
# every run, don't reuse IDs from a previous session's notes.

# %%
sc.tl.rank_genes_groups(adata, f"clusters_r{RES_TO_CHECK}", method="t-test")

marker_quality = []
for cl in adata.obs[f"clusters_r{RES_TO_CHECK}"].cat.categories:
    names = adata.uns["rank_genes_groups"]["names"][cl][:5]
    lfcs = adata.uns["rank_genes_groups"]["logfoldchanges"][cl][:5]
    pvals = adata.uns["rank_genes_groups"]["pvals_adj"][cl][:5]
    marker_quality.append({
        "cluster": cl,
        "top_gene": names[0],
        "top_lfc": round(float(lfcs[0]), 2),
        "top_padj": pvals[0],
        "n_sig_top5": int((pvals < 0.05).sum()),
    })
marker_df = pd.DataFrame(marker_quality)
print(marker_df)
# Clusters with low top_lfc (<0.5-1) or non-significant top_padj are
# candidates for "no clear identity" -- likely over-split at this
# resolution, or an artifact/doublet population worth a closer look.

sc.pl.rank_genes_groups_dotplot(adata, n_genes=3, groupby=f"clusters_r{RES_TO_CHECK}")

# %% [markdown]
# ## 6g. Finalize clusters
# Cross-reference 6a-6f: prefer the resolution where silhouette is
# reasonable, clusters are stable across neighboring resolutions, and each
# has real marker-driven identity (not just one-sample noise).
#
# Two independent reasons to fold a cluster into `low_confidence`, not just
# one:
#   1. Too few cells (MIN_CLUSTER_SIZE) -- statistically unreliable
#      regardless of marker quality.
#   2. No clean marker in the 6f dotplot (NO_MARKER_CLUSTERS) -- even if
#      large enough by cell count, a cluster with no gene reaching a
#      decent fraction/expression anywhere isn't a real identity to name.
#      Fill this in BY HAND after reading 6f's dotplot on this run -- IDs
#      are not stable across reruns of the resolution sweep.
#
# Check where the Finding #10 TE-dominant population lands after this --
# it should survive as its own reasonably-sized cluster, not fall into
# low_confidence, confirming it's real signal and not part of this noise.

# %%
FINAL_RESOLUTION = 0.2  # <- set based on the checks above
MIN_CLUSTER_SIZE = 20   # lowered from 50: post-filter, far fewer tiny clusters
                        # remain, and the ones that do are worth a marker-based
                        # look before assuming they're noise (cf. the TE-dominant
                        # cluster, which stayed small but stable across res=0.1-0.3)
NO_MARKER_CLUSTERS = []  # <- fill in by hand from Section 6f's dotplot on THIS run,
                          # e.g. clusters with an all-blank row (no gene with
                          # decent fraction/expression anywhere)

sizes = adata.obs[f"clusters_r{FINAL_RESOLUTION}"].value_counts()
tiny_clusters = sizes[sizes < MIN_CLUSTER_SIZE].index

adata.obs["clusters"] = adata.obs[f"clusters_r{FINAL_RESOLUTION}"].astype(str)
adata.obs.loc[adata.obs["clusters"].isin(tiny_clusters), "clusters"] = "low_confidence"
adata.obs.loc[adata.obs["clusters"].isin(NO_MARKER_CLUSTERS), "clusters"] = "low_confidence"
adata.obs["clusters"] = adata.obs["clusters"].astype("category")

print(adata.obs["clusters"].value_counts())
print(f"\n{(adata.obs['clusters'] == 'low_confidence').sum()} cells in low_confidence total "
      f"({len(tiny_clusters)} tiny clusters by size, {len(NO_MARKER_CLUSTERS)} by no-marker)")

print("\nWhere did the Finding #10 TE-dominant cells land?")
print(adata.obs.loc[adata.obs["te_dominant_outlier"], "clusters"].value_counts())

# %% [markdown]
# ## 6h. Drop low_confidence + recompute neighbors/UMAP
# The saved embedding shouldn't still be shaped by cells that are excluded
# from annotation. Neighbors must be recomputed (not just UMAP) -- removing
# cells invalidates parts of the existing KNN graph, cells that had one of
# the dropped cells as a neighbor need new graph edges.

# %%
n_before_drop = adata.n_obs
adata = adata[adata.obs["clusters"] != "low_confidence"].copy()
adata.obs["clusters"] = adata.obs["clusters"].cat.remove_unused_categories()
print(f"Dropped low_confidence: {n_before_drop} -> {adata.n_obs} cells")

sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, n_pcs=FINAL_N_PCS)
sc.tl.umap(adata)

sc.pl.umap(adata, color=["clusters", "sample", "condition_label"], ncols=1)
# Sanity check: dropping low_confidence and recomputing shouldn't visibly
# distort the rest of the structure -- if it does, MIN_CLUSTER_SIZE/
# NO_MARKER_CLUSTERS above may have been too aggressive.

# %% [markdown]
# ## 7. Spatial plot per sample, colored by cluster

# %%
for s in samples:
    a_s = adata[adata.obs["sample"] == s]
    coords = a_s.obsm["spatial"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for cl in a_s.obs["clusters"].cat.categories:
        mask = a_s.obs["clusters"] == cl
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, label=cl)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(s)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Save final clustered object
# Annotation (marker panels, DE, cluster naming) happens separately in
# `cell_type_annotation.py` -- not duplicated here.

# %%
adata.write("/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_clustered.h5ad")
print("Saved:", adata)