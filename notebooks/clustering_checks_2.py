# %% [markdown]
# # Clustering checks, part 2 — fiber type, mast cells, high-mito markers,
# # TE-dominant fix, TE_L1/L2 split, TE-only exploratory clustering
#
# Continuation of `clustering_checks.py`. Reads the same checkpoint (now
# with `clusters` + the Section 8 `score_*` columns already in `.obs`) and
# works through the 5 items flagged in the last review round:
#
#   1. rank_genes_groups on the 6 high-mito clusters — fiber-type genes?
#   2. Mast cell panel (score_genes) — absent from the original 29-panel set
#   3. Fiber-type panel (I/IIA/IIX/IIB) — applied broadly + to cluster 7
#   4. Fix the 96-vs-11,108 TE-dominant-cell mismatch (config fix below)
#   5. TE_L1 / TE_L2 fractions separately, not just the combined TE_fraction
#   6. TE-only exploratory clustering (no genes, no Harmony) — hidden
#      TE-driven substructure the gene-based clustering wouldn't catch
#
# Same caveat as before: untested against the real object. Config below is
# my best guess from what's been confirmed in chat; adjust and re-run.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 110
warnings.filterwarnings("ignore", category=FutureWarning)

OUTDIR = "clustering_checks_figs"
import os
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    plt.savefig(f"{OUTDIR}/{name}.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Config

# %%
CHECKPOINT = "/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_clustered.h5ad"

CLUSTER_COL = "clusters"
MT_COL = "pct_counts_mt"
COUNTS_COL = "total_counts"
GENES_COL = "n_genes_by_counts"
TE_FRAC_COL = "TE_fraction"
SAMPLE_COL = "sample"
SPATIAL_KEY = "spatial"
UMAP_KEY = "X_umap"

# Confirmed from the last Load section output — this fixes the 96-vs-11,108
# mismatch IF this column really is Finding #10's original flag.
TE_DOMINANT_COL = "te_dominant_outlier"

# TE features are identified as `"SoloTE" in var_name` (qc_stardist_cell.py
# Section 2). Family-level names live inside that string, e.g. something
# like "SoloTE_L1" / "SoloTE_L2" / "SoloTE_Alu" — CONFIRM the exact pattern
# by checking the printed examples in the first cell below before trusting
# the L1/L2 split.
TE_VAR_FLAG_COL = "is_te"
L1_PATTERN = "L1"
L2_PATTERN = "L2"

HIGH_MITO_CLUSTERS = ["35", "36", "24", "28", "2", "25"]
RESIDUAL_CLUSTERS = ["57", "63", "61", "60", "58", "64", "48", "65", "59", "62"]
TE_DOMINANT_CLUSTERS = ["12", "46"]
CLUSTER_7 = ["7"]

# %% [markdown]
# ## Load

# %%
adata = sc.read_h5ad(CHECKPOINT)
adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype(str)
print(adata)

for col in [MT_COL, COUNTS_COL, GENES_COL, TE_FRAC_COL, SAMPLE_COL]:
    assert col in adata.obs.columns, f"'{col}' not in adata.obs — check CONFIG."

# Check what the TE feature names actually look like before trusting the
# L1/L2 pattern match below.
te_examples = adata.var_names[adata.var[TE_VAR_FLAG_COL]][:15].tolist()
print(f"\nExample TE feature names ({adata.var[TE_VAR_FLAG_COL].sum()} total):")
print(te_examples)
print("\n^ if these don't contain recognizable 'L1'/'L2' substrings, fix "
      "L1_PATTERN/L2_PATTERN in the CONFIG cell before running Section 5 below.")

# %% [markdown]
# ## Helpers (same as clustering_checks.py — repeated here so this file runs standalone)

# %%
def highlight_col(adata, groups, label):
    col = f"hl_{label}"
    adata.obs[col] = np.where(adata.obs[CLUSTER_COL].isin(groups), adata.obs[CLUSTER_COL], "other")
    cats = [g for g in groups if g in adata.obs[col].unique()] + ["other"]
    adata.obs[col] = pd.Categorical(adata.obs[col], categories=cats)
    return col


def palette_for(groups):
    colors = sns.color_palette("tab10", len(groups))
    pal = {g: colors[i] for i, g in enumerate(groups)}
    pal["other"] = "#d3d3d3"
    return pal


def umap_highlight(adata, groups, label, title=None):
    col = highlight_col(adata, groups, label)
    pal = palette_for(groups)
    sc.pl.umap(adata, color=col, palette=pal, size=8, title=title or f"UMAP — {label}", show=False)
    savefig(f"umap_{label}")
    plt.show()


def spatial_highlight(adata, groups, label, sample_col=SAMPLE_COL, title=None):
    col = highlight_col(adata, groups, label)
    pal = palette_for(groups)
    samples = sorted(adata.obs[sample_col].unique())
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, samp in zip(axes, samples):
        sub = adata[adata.obs[sample_col] == samp]
        coords = sub.obsm[SPATIAL_KEY]
        is_other = sub.obs[col] == "other"
        ax.scatter(coords[is_other, 0], coords[is_other, 1], s=2, c=pal["other"], rasterized=True)
        for g in groups:
            m = (sub.obs[col] == g).values
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=8, c=[pal[g]], label=f"cl {g}", rasterized=True)
        ax.set_title(samp, fontsize=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", bbox_to_anchor=(1.08, 1.0), fontsize=9)
    fig.suptitle(title or f"Spatial — {label}", y=1.02)
    plt.tight_layout()
    savefig(f"spatial_{label}")
    plt.show()


def best_marker_table(adata_sub, groupby, n_genes=20):
    """rank_genes_groups, but report the best-padj gene within the top-N
    by score, not just the top-1 — same logic we used for Scenario C."""
    sc.tl.rank_genes_groups(adata_sub, groupby=groupby, method="wilcoxon", n_genes=n_genes)
    rg = adata_sub.uns["rank_genes_groups"]
    rows = []
    for cl in rg["names"].dtype.names:
        names = rg["names"][cl]
        padjs = rg["pvals_adj"][cl]
        lfcs = rg["logfoldchanges"][cl]
        best_idx = int(np.argmin(padjs))
        rows.append({
            "cluster": cl,
            "top_gene_by_score": names[0],
            "best_gene_by_padj": names[best_idx],
            "logfc": lfcs[best_idx],
            "padj": padjs[best_idx],
            "rank_of_best": best_idx,
        })
    return pd.DataFrame(rows).sort_values("cluster")


# %% [markdown]
# ## 1. High-mito clusters — marker genes. Fiber-oxidative or something else?

# %%
adata_mito = adata[adata.obs[CLUSTER_COL].isin(HIGH_MITO_CLUSTERS)].copy()
mito_markers = best_marker_table(adata_mito, CLUSTER_COL, n_genes=20)
print(mito_markers.to_string(index=False))
mito_markers.to_csv(f"{OUTDIR}/high_mito_markers.csv", index=False)

# %% [markdown]
# ## 2 & 3. New marker panels — mast cells + fiber types
#
# Mast cells were absent from the original 29-panel set. Fiber types let us
# test two hypotheses at once: are the high-mito clusters oxidative
# (type I) fibers, and does cluster 7 sit in the middle of the fiber-type
# continuum (shares markers with neighbors, no exclusive one of its own)?

# %%
new_panels = {
    "mast_cell": ["Cma1", "Mcpt4", "Cpa3", "Kit", "Fcer1a"],
    "fiber_type_I": ["Myh7", "Tnnt1", "Tnnc1", "Atp2a2", "Mb"],       # slow/oxidative/red
    "fiber_type_IIA": ["Myh2", "Tnnt3", "Atp2a1"],                    # fast-oxidative-glycolytic, intermediate
    "fiber_type_IIX": ["Myh1"],                                       # fast/glycolytic
    "fiber_type_IIB": ["Myh4", "Actn3", "Ldha"],                      # fast/glycolytic, white
}

for name, genes in new_panels.items():
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        print(f"  [{name}] missing from var_names, skipped: {missing}")
    if present:
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}")
    else:
        print(f"  [{name}] NO genes found in adata.var_names — skipped entirely")

new_score_cols = [f"score_{n}" for n in new_panels if f"score_{n}" in adata.obs.columns]
print(f"\nComputed: {new_score_cols}")

# %%
# Cluster-level view of the new panels, focused on high-mito + cluster 7 + core
focus_clusters = HIGH_MITO_CLUSTERS + CLUSTER_7 + RESIDUAL_CLUSTERS + TE_DOMINANT_CLUSTERS
score_by_cluster_new = adata.obs.groupby(CLUSTER_COL, observed=True)[new_score_cols].mean()
print(score_by_cluster_new.loc[score_by_cluster_new.index.isin(focus_clusters)]
      .sort_values("score_fiber_type_I", ascending=False))

plt.figure(figsize=(6, 0.3 * len(score_by_cluster_new) + 2))
sns.heatmap(score_by_cluster_new, cmap="RdBu_r", center=0, cbar_kws={"label": "mean score_genes"})
plt.title("New panels (mast cell, fiber types) x cluster")
plt.tight_layout()
savefig("heatmap_new_panels_by_cluster")
plt.show()

# %%
# Direct fiber-type check for cluster 7: does it sit between two fiber types
# (shares both, dominated by neither) rather than having its own?
fiber_cols = [c for c in new_score_cols if c.startswith("score_fiber_type")]
print("\nCluster 7 fiber-type profile (continuum check):")
print(adata.obs.loc[adata.obs[CLUSTER_COL] == "7", fiber_cols].mean())
print("\nFor comparison, its likely UMAP neighbors (adjust based on the UMAP if these aren't right):")
for cl in ["0", "1", "3", "23"]:
    if cl in adata.obs[CLUSTER_COL].unique():
        print(f"  cluster {cl}:", adata.obs.loc[adata.obs[CLUSTER_COL] == cl, fiber_cols].mean().to_dict())

# %% [markdown]
# ## 4. Fix: the real 96 TE-dominant cells (not 11,108)

# %%
if TE_DOMINANT_COL in adata.obs.columns:
    te_dominant = adata.obs[TE_DOMINANT_COL].astype(bool)
    n = te_dominant.sum()
    print(f"Using '{TE_DOMINANT_COL}': {n} TE-dominant cells")
    if abs(n - 96) > 20:
        print(f"  ⚠ still doesn't match the ~96 reference (Finding #10) — this column may not be "
              f"the one, or the definition has changed since. Don't trust the plots below yet if so.")
    else:
        print("  ✓ matches the ~96 reference reasonably well.")
else:
    print(f"'{TE_DOMINANT_COL}' still not found in adata.obs — available columns:")
    print([c for c in adata.obs.columns if "te" in c.lower() or "dom" in c.lower()])
    te_dominant = pd.Series(False, index=adata.obs_names)

adata.obs["te_dom_group"] = np.where(te_dominant, "TE_dominant_cell", "other")

# %%
if te_dominant.sum() > 0:
    sc.pl.umap(adata, color="te_dom_group", palette={"TE_dominant_cell": "crimson", "other": "#d3d3d3"},
               size=10, title=f"TE-dominant cells (n={te_dominant.sum()}) — UMAP", show=False)
    savefig("umap_TE_dominant_cells_fixed")
    plt.show()

    samples = sorted(adata.obs[SAMPLE_COL].unique())
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, samp in zip(axes, samples):
        sub = adata[adata.obs[SAMPLE_COL] == samp]
        coords = sub.obsm[SPATIAL_KEY]
        is_te = sub.obs["te_dom_group"].values == "TE_dominant_cell"
        ax.scatter(coords[~is_te, 0], coords[~is_te, 1], s=2, c="#d3d3d3", rasterized=True)
        ax.scatter(coords[is_te, 0], coords[is_te, 1], s=14, c="crimson", rasterized=True)
        ax.set_title(f"{samp} (n={is_te.sum()})", fontsize=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(f"TE-dominant cells (n={te_dominant.sum()}) — spatial, per sample", y=1.02)
    plt.tight_layout()
    savefig("spatial_TE_dominant_cells_fixed")
    plt.show()

# %% [markdown]
# ## 5. TE_L1 and TE_L2 fractions, separately

# %%
te_names = adata.var_names[adata.var[TE_VAR_FLAG_COL]]
l1_features = te_names[te_names.str.contains(L1_PATTERN, case=False, regex=False)]
l2_features = te_names[te_names.str.contains(L2_PATTERN, case=False, regex=False)]
print(f"L1-matching TE features ({len(l1_features)}): {l1_features[:10].tolist()}")
print(f"L2-matching TE features ({len(l2_features)}): {l2_features[:10].tolist()}")

assert len(l1_features) > 0, "No TE features matched L1_PATTERN — check the example names printed in Load, fix L1_PATTERN"
assert len(l2_features) > 0, "No TE features matched L2_PATTERN — check the example names printed in Load, fix L2_PATTERN"

# Use raw counts if available (mirrors how TE_fraction itself was computed
# — pre-normalization, so it's not distorted by log1p/normalize_total)
raw_X = adata.layers["counts"] if "counts" in adata.layers else adata.X
import scipy.sparse as sp
raw_X = raw_X.tocsr() if sp.issparse(raw_X) else sp.csr_matrix(raw_X)

def raw_fraction(feature_names):
    idx = [adata.var_names.get_loc(f) for f in feature_names]
    sub_sum = np.asarray(raw_X[:, idx].sum(axis=1)).ravel()
    total = np.asarray(raw_X.sum(axis=1)).ravel()
    return np.divide(sub_sum, total, out=np.zeros_like(sub_sum, dtype=float), where=total > 0)

adata.obs["TE_L1_fraction"] = raw_fraction(l1_features)
adata.obs["TE_L2_fraction"] = raw_fraction(l2_features)

print("\nMedian TE_L1_fraction / TE_L2_fraction by cluster (flagged clusters only):")
print(adata.obs.groupby(CLUSTER_COL, observed=True)[["TE_L1_fraction", "TE_L2_fraction", TE_FRAC_COL]]
      .median().loc[lambda d: d.index.isin(HIGH_MITO_CLUSTERS + RESIDUAL_CLUSTERS + TE_DOMINANT_CLUSTERS)]
      .sort_values(TE_FRAC_COL, ascending=False))

# %%
# Spatial + UMAP for L1 and L2 separately, continuous colorscale (not highlight-style,
# since these are continuous per-cell values, not a discrete cluster membership)
for metric in ["TE_L1_fraction", "TE_L2_fraction"]:
    sc.pl.umap(adata, color=metric, cmap="Reds", size=8, title=f"{metric} — UMAP", show=False)
    savefig(f"umap_{metric}")
    plt.show()

    samples = sorted(adata.obs[SAMPLE_COL].unique())
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    vmax = adata.obs[metric].quantile(0.99)
    for ax, samp in zip(axes, samples):
        sub = adata[adata.obs[SAMPLE_COL] == samp]
        coords = sub.obsm[SPATIAL_KEY]
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=sub.obs[metric], cmap="Reds",
                          s=4, vmin=0, vmax=vmax, rasterized=True)
        ax.set_title(samp, fontsize=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.colorbar(sca, ax=axes.tolist(), shrink=0.5, label=metric)
    fig.suptitle(f"{metric} — spatial, per sample", y=1.02)
    savefig(f"spatial_{metric}")
    plt.show()

# %% [markdown]
# ## 6. TE-only exploratory clustering (no genes, no Harmony)
#
# Tests whether TE expression alone reveals structure the gene-based
# clustering doesn't capture (per the atlas-paper precedent discussed —
# HERV-only HVFs found subclusters within known PBMC types). No batch
# correction here on purpose: with 1 sample per condition, Harmony can't
# tell technical batch effect from real condition-driven biology (same
# reasoning as why TE is kept out of the main pipeline's Harmony step) —
# so this is deliberately exploratory/uncorrected, read with that in mind.

# %%
adata_te = adata[:, adata.var[TE_VAR_FLAG_COL]].copy()
print(f"TE-only object: {adata_te.n_obs} cells x {adata_te.n_vars} TE features")

# Raw counts -> normalize/log for this exploratory embedding only
adata_te.layers["counts"] = adata_te.X.copy()
sc.pp.normalize_total(adata_te, target_sum=1e4)
sc.pp.log1p(adata_te)

n_pcs_te = min(30, adata_te.n_vars - 1)
sc.pp.pca(adata_te, n_comps=n_pcs_te)
sc.pp.neighbors(adata_te, n_neighbors=30, n_pcs=n_pcs_te)
sc.tl.umap(adata_te)
sc.tl.leiden(adata_te, resolution=0.5, objective_function="modularity", flavor="igraph")

print(f"TE-only clustering: {adata_te.obs['leiden'].nunique()} clusters")
print(adata_te.obs["leiden"].value_counts())

# %%
sc.pl.umap(adata_te, color="leiden", size=8, title="TE-only UMAP, colored by TE-only Leiden", show=False)
savefig("umap_TE_only_clusters")
plt.show()

# Does this TE-only clustering line up with the existing gene-based clusters,
# or does it cut across them (= real independent structure)?
adata.obs["te_only_leiden"] = adata_te.obs["leiden"].reindex(adata.obs_names)
crosstab = pd.crosstab(adata.obs[CLUSTER_COL], adata.obs["te_only_leiden"], normalize="index")
plt.figure(figsize=(0.4 * crosstab.shape[1] + 3, 0.25 * crosstab.shape[0] + 2))
sns.heatmap(crosstab, cmap="viridis", cbar_kws={"label": "fraction of gene-cluster"})
plt.title("Gene-based clusters (rows) vs. TE-only clusters (cols)")
plt.xlabel("TE-only Leiden cluster"); plt.ylabel("Gene-based cluster")
plt.tight_layout()
savefig("crosstab_gene_vs_TE_only_clusters")
plt.show()

print("\nReading this: if each gene-cluster row lights up in mostly ONE TE-only column, "
      "TE structure just re-derives the existing clusters (nothing new). If rows spread "
      "across several TE-only columns unevenly, that's independent TE-driven structure "
      "worth a closer look — check which gene-clusters split that way.")

# %% [markdown]
# ## Done
# New figures in `./clustering_checks_figs/`. `high_mito_markers.csv` saved
# for pulling into the next round of slides.