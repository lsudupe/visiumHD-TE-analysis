# %% [markdown]
# # Clustering checks, part 2 — fiber type, mast cells, high-mito markers,
# # TE-dominant cells, TE_L1/L2 split, TE-only exploratory clustering
#
# Continuation of `clustering_checks.py`. Reads the same checkpoint and
# works through: high-mito cluster markers, mast-cell + fiber-type panels,
# the TE-dominant cells (Finding #10), TE_L1/L2 fraction split, and TE-only
# exploratory clustering.
#
# ## Updated against the reconstructed `clustering.py` (Finding #24)
# - **Removed the 44-cluster-era hardcoded lists** (`HIGH_MITO_CLUSTERS`,
#   `RESIDUAL_CLUSTERS`, `TE_DOMINANT_CLUSTERS=["12","46"]`, `CLUSTER_7`).
#   Section 0 below derives high-mito/TE-dominant clusters automatically
#   from the object, same approach as `clustering_checks.py`. `CLUSTER_7`
#   (the old fiber-type-continuum candidate) is now a manual fill-in --
#   there's no automatic way to detect "sits between two fiber types",
#   that's a visual call from the UMAP/dotplot.
# - **`RESIDUAL_CLUSTERS` retired as a concept** -- `clustering.py`
#   Section 6g now handles that tier directly (`MIN_CLUSTER_SIZE` +
#   `NO_MARKER_CLUSTERS`) before this script ever runs. Nothing here
#   should reference it.
# - **Fixed the Finding #23 bug in Section 6** (TE-only clustering):
#   `sc.pp.pca(adata_te, ...)` was missing `use_highly_variable=False` --
#   `adata_te` inherits `highly_variable=False` on every TE feature from
#   the parent object (TEs are never HVGs, clustering.py Section 2), so
#   PCA's default `use_highly_variable=True` filtered the matrix to 0
#   columns and crashed with `ValueError: Found array with 0 feature(s)`.
#   This is exactly the bug PROJECT_CONTEXT.md Finding #23 documented as
#   "fix drafted, not yet confirmed run" -- it was never actually applied
#   in this script until now.
# - **Section 4's "~96 cells" reference check removed.** That number was
#   from the pre-reconstruction 44-cluster object; the reconstructed
#   pipeline gives a different (and not yet fixed) count each run --
#   see PROJECT_CONTEXT.md Finding #24 (212/241 cells in one test run).
#   Section 4 now just reports whatever count it finds, no hardcoded
#   expectation to compare against.
#
# Same caveat as before: untested against the real object. CONFIG below is
# best-guess from what's been confirmed in chat; adjust and re-run.

# %%
import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 110
warnings.filterwarnings("ignore", category=FutureWarning)

OUTDIR = "clustering_checks_figs"
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

TE_DOMINANT_COL = "te_dominant_outlier"  # from clustering.py Section 4b (Finding #10)

# TE features are identified as `"SoloTE" in var_name` (qc_stardist_cell.py
# Section 2 / clustering.py Section 2). Family-level names live inside that
# string, e.g. "SoloTE|L1" / "SoloTE|L2" / "SoloTE|Alu" (per Finding #24's
# 6f dotplot, the real separator is "|", not "_" -- confirm against the
# printed examples below before trusting the L1/L2 split).
TE_VAR_FLAG_COL = "is_te"
L1_PATTERN = "L1"
L2_PATTERN = "L2"

# Automatic derivation thresholds, same as clustering_checks.py Section 0.
HIGH_MITO_PCT_THRESHOLD = 20
TE_DOMINANT_MIN_FRACTION = 0.05

# Manual fill-ins -- no automatic way to derive these, they're visual/
# biological judgment calls from the dotplot and UMAP neighbors.
MANUAL_HIGH_MITO_CLUSTERS = []
MANUAL_TE_DOMINANT_CLUSTERS = []
CLUSTER_7_EQUIVALENT = []  # fill in the cluster(s) that look like they sit
# in the middle of the fiber-type continuum (no significant marker despite
# reasonable size) -- was hardcoded "7" in the 44-cluster era, not stable
# across reruns. Check clustering.py Section 6f's padj column for
# candidates (non-significant top_padj, decent n_cells) before filling.
UMAP_NEIGHBOR_CANDIDATES = []  # fill in a few clusters near
# CLUSTER_7_EQUIVALENT on the UMAP, for the continuum comparison in
# Section 3 below -- look at the UMAP plot, not derivable automatically.

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

if "low_confidence" in adata.obs[CLUSTER_COL].unique():
    print("\n[NOTE] 'low_confidence' present in this checkpoint -- clustering.py "
          "Section 6h should have dropped it before saving. Confirm this checkpoint "
          "is current before trusting the rest of this script.")

# %% [markdown]
# ## 0. Derive high-mito / TE-dominant cluster groups automatically
# Same logic as `clustering_checks.py` Section 0 -- repeated here so this
# script runs standalone. Recomputed fresh every run.

# %%
mito_by_cluster = adata.obs.groupby(CLUSTER_COL, observed=True)[MT_COL].median().sort_values(ascending=False)
print("Median pct_counts_mt by cluster (top 10):")
print(mito_by_cluster.head(10))

HIGH_MITO_CLUSTERS = sorted(set(
    mito_by_cluster[mito_by_cluster > HIGH_MITO_PCT_THRESHOLD].index.tolist()
) | set(MANUAL_HIGH_MITO_CLUSTERS))
print(f"\nHIGH_MITO_CLUSTERS (median %mt > {HIGH_MITO_PCT_THRESHOLD}): {HIGH_MITO_CLUSTERS}")

if TE_DOMINANT_COL in adata.obs.columns:
    n_te_dom_total = adata.obs[TE_DOMINANT_COL].astype(bool).sum()
    te_dom_by_cluster = adata.obs.loc[adata.obs[TE_DOMINANT_COL].astype(bool), CLUSTER_COL].value_counts()
    print(f"\n{n_te_dom_total} te_dominant_outlier cells total, by cluster:")
    print(te_dom_by_cluster)
    TE_DOMINANT_CLUSTERS = sorted(set(
        te_dom_by_cluster[te_dom_by_cluster / max(n_te_dom_total, 1) >= TE_DOMINANT_MIN_FRACTION].index.tolist()
    ) | set(MANUAL_TE_DOMINANT_CLUSTERS))
    print(f"TE_DOMINANT_CLUSTERS (>= {TE_DOMINANT_MIN_FRACTION*100:.0f}% of flagged cells): {TE_DOMINANT_CLUSTERS}")
else:
    print(f"\n'{TE_DOMINANT_COL}' not in adata.obs -- fill MANUAL_TE_DOMINANT_CLUSTERS by hand.")
    TE_DOMINANT_CLUSTERS = sorted(set(MANUAL_TE_DOMINANT_CLUSTERS))

CLUSTER_7 = sorted(set(CLUSTER_7_EQUIVALENT))
if not CLUSTER_7:
    print("\nCLUSTER_7_EQUIVALENT is empty -- Sections 2-3's continuum check will be skipped. "
          "Fill it in from clustering.py Section 6f's dotplot (non-significant top_padj, "
          "decent cluster size).")

focus_clusters = sorted(set(HIGH_MITO_CLUSTERS + CLUSTER_7 + TE_DOMINANT_CLUSTERS))

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
    if not groups:
        print(f"[{label}] empty group list -- skipping.")
        return
    col = highlight_col(adata, groups, label)
    pal = palette_for(groups)
    sc.pl.umap(adata, color=col, palette=pal, size=8, title=title or f"UMAP — {label}", show=False)
    savefig(f"umap_{label}")
    plt.show()


def spatial_highlight(adata, groups, label, sample_col=SAMPLE_COL, title=None):
    if not groups:
        print(f"[{label}] empty group list -- skipping.")
        return
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
    by score, not just the top-1."""
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
if HIGH_MITO_CLUSTERS:
    adata_mito = adata[adata.obs[CLUSTER_COL].isin(HIGH_MITO_CLUSTERS)].copy()
    mito_markers = best_marker_table(adata_mito, CLUSTER_COL, n_genes=20)
    print(mito_markers.to_string(index=False))
    mito_markers.to_csv(f"{OUTDIR}/high_mito_markers.csv", index=False)
else:
    print("HIGH_MITO_CLUSTERS is empty -- nothing to check.")

# %% [markdown]
# ## 2 & 3. New marker panels — mast cells + fiber types
#
# Mast cells absent from the original 29-panel set. Fiber types let us test
# two hypotheses at once: are the high-mito clusters oxidative (type I)
# fibers, and does CLUSTER_7 sit in the middle of the fiber-type continuum
# (shares markers with neighbors, no exclusive one of its own)?

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
score_by_cluster_new = adata.obs.groupby(CLUSTER_COL, observed=True)[new_score_cols].mean()
if focus_clusters:
    print(score_by_cluster_new.loc[score_by_cluster_new.index.isin(focus_clusters)]
          .sort_values("score_fiber_type_I", ascending=False))
else:
    print("focus_clusters is empty -- showing full table instead:")
    print(score_by_cluster_new.sort_values("score_fiber_type_I", ascending=False))

plt.figure(figsize=(6, 0.3 * len(score_by_cluster_new) + 2))
sns.heatmap(score_by_cluster_new, cmap="RdBu_r", center=0, cbar_kws={"label": "mean score_genes"})
plt.title("New panels (mast cell, fiber types) x cluster")
plt.tight_layout()
savefig("heatmap_new_panels_by_cluster")
plt.show()

# %%
# Direct fiber-type check: does CLUSTER_7_EQUIVALENT sit between two fiber
# types (shares both, dominated by neither) rather than having its own?
fiber_cols = [c for c in new_score_cols if c.startswith("score_fiber_type")]
if CLUSTER_7:
    print(f"\n{CLUSTER_7} fiber-type profile (continuum check):")
    print(adata.obs.loc[adata.obs[CLUSTER_COL].isin(CLUSTER_7), fiber_cols].mean())
    if UMAP_NEIGHBOR_CANDIDATES:
        print("\nFor comparison, its UMAP-neighboring clusters:")
        for cl in UMAP_NEIGHBOR_CANDIDATES:
            if cl in adata.obs[CLUSTER_COL].unique():
                print(f"  cluster {cl}:", adata.obs.loc[adata.obs[CLUSTER_COL] == cl, fiber_cols].mean().to_dict())
    else:
        print("\nUMAP_NEIGHBOR_CANDIDATES is empty -- fill in a few clusters near "
              f"{CLUSTER_7} on the UMAP to compare against.")
else:
    print("CLUSTER_7_EQUIVALENT is empty -- skipping the continuum check. Fill it in "
          "from clustering.py Section 6f's dotplot first.")

# %% [markdown]
# ## 4. TE-dominant cells (Finding #10) — where do they land?
# No hardcoded reference count anymore -- the reconstructed pipeline (Finding
# #24) gives a different number each run depending on filter/resolution.
# Just reports what's actually in the object.

# %%
if TE_DOMINANT_COL in adata.obs.columns:
    te_dominant = adata.obs[TE_DOMINANT_COL].astype(bool)
    n = te_dominant.sum()
    print(f"Using '{TE_DOMINANT_COL}': {n} TE-dominant cells")
    print(adata.obs.loc[te_dominant, SAMPLE_COL].value_counts())
    print("\nBy cluster:")
    print(adata.obs.loc[te_dominant, CLUSTER_COL].value_counts())
else:
    print(f"'{TE_DOMINANT_COL}' not found in adata.obs -- available columns with 'te'/'dom':")
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
if focus_clusters:
    print(adata.obs.groupby(CLUSTER_COL, observed=True)[["TE_L1_fraction", "TE_L2_fraction", TE_FRAC_COL]]
          .median().loc[lambda d: d.index.isin(focus_clusters)]
          .sort_values(TE_FRAC_COL, ascending=False))
else:
    print(adata.obs.groupby(CLUSTER_COL, observed=True)[["TE_L1_fraction", "TE_L2_fraction", TE_FRAC_COL]]
          .median().sort_values(TE_FRAC_COL, ascending=False).head(10))

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
# clustering doesn't capture. No batch correction here on purpose: with 1
# sample per condition, Harmony can't tell technical batch effect from real
# condition-driven biology -- deliberately exploratory/uncorrected, read
# with that in mind.
#
# **Bug fix applied here (Finding #23)**: `adata_te` inherits
# `highly_variable=False` on every TE feature from the parent object (TEs
# are never HVGs) -- `sc.pp.pca`'s default `use_highly_variable=True`
# would filter the matrix to 0 columns and crash. Must pass
# `use_highly_variable=False` explicitly. This was documented as the known
# fix in PROJECT_CONTEXT.md but had never actually been applied in this
# script until now.

# %%
adata_te = adata[:, adata.var[TE_VAR_FLAG_COL]].copy()
print(f"TE-only object: {adata_te.n_obs} cells x {adata_te.n_vars} TE features")

# Check for all-zero TE vectors before running -- a large fraction can
# cause pathological behavior in the approximate-nearest-neighbor step
# (per PROJECT_CONTEXT.md Finding #23's troubleshooting note).
raw_te = adata_te.X
frac_all_zero = np.asarray((raw_te.sum(axis=1) == 0)).ravel().mean() if sp.issparse(raw_te) else (raw_te.sum(axis=1) == 0).mean()
print(f"Fraction of cells with an all-zero TE vector: {frac_all_zero:.3f}")
if frac_all_zero > 0.5:
    print("[WARNING] Over half the cells have zero TE signal -- neighbors/UMAP may behave "
          "poorly or hang. Consider filtering to cells with at least 1 TE count first.")

# Raw counts -> normalize/log for this exploratory embedding only
adata_te.layers["counts"] = adata_te.X.copy()
sc.pp.normalize_total(adata_te, target_sum=1e4)
sc.pp.log1p(adata_te)

n_pcs_te = min(30, adata_te.n_vars - 1)
sc.pp.pca(adata_te, n_comps=n_pcs_te, use_highly_variable=False)  # Finding #23 fix
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
# (if HIGH_MITO_CLUSTERS was non-empty) for pulling into the next round of slides.