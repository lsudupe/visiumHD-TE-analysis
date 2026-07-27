# %% [markdown]
# # Clustering checks — post score_genes / tier decisions
#
# Reads the finalized clustered checkpoint directly (no need to re-run
# HVG/PCA/Harmony/Leiden) and walks through per-cell correlations, where
# flagged clusters sit (UMAP + spatial), where the TE-dominant cells are,
# where immune/vascular/myogenic markers land, and an optional
# drop-selected-clusters cut.
#
# ## Rewritten against the reconstructed `clustering.py` (Finding #24)
# The old version of this script hardcoded cluster IDs from the 44-cluster
# era (`HIGH_MITO_CLUSTERS`, `RESIDUAL_CLUSTERS`, `TE_DOMINANT_CLUSTERS =
# ["12", "46"]`, the slide-8 `IMMUNE_CLUSTERS`/`VASCULAR_FAP_NEURAL_CLUSTERS`
# lists, and "Scenario C"). None of that carries over — cluster IDs are not
# stable across Leiden reruns, and the pipeline itself changed:
#   - **`low_confidence` no longer exists in the saved checkpoint.**
#     `clustering.py` Section 6h now drops `low_confidence` cells and
#     recomputes neighbors/UMAP BEFORE saving — this script used to expect
#     to find and analyze that category; it won't be there anymore.
#   - **The old "residual clusters, likely merge into low_confidence"
#     tier doesn't apply.** `clustering.py` Section 6g now handles that
#     directly (`MIN_CLUSTER_SIZE` + `NO_MARKER_CLUSTERS`) before this
#     script ever runs — there's no separate "residual tier" left to
#     review here.
#   - **High-mito / immune / vascular groupings are now derived, not
#     hardcoded** — see Section 0 below. Fill in `NO_MARKER_CLUSTERS`-style
#     manual lists yourself if the automatic derivation doesn't match what
#     you see in the dotplot, but don't reuse old IDs (12, 46, 35, 36...) —
#     they don't correspond to anything in the new object.
#   - **"Scenario C" (drop high-mito + residual, keep TE-dominant) is
#     retired as a concept** — Section 6 below is now a generic
#     "drop selected clusters" utility with an empty list, not a specific
#     preset scenario.
#
# **Still untested against your real object** — column names are my best
# guess from `clustering.py`/`cell_type_annotation.py`. CONFIG cell is the
# one place to fix names; the rest has defensive asserts.

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
# ## Config — adjust these to match your actual `adata.obs` / `adata.var` columns

# %%
CHECKPOINT = "/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_clustered.h5ad"

CLUSTER_COL = "clusters"
MT_COL = "pct_counts_mt"
COUNTS_COL = "total_counts"
GENES_COL = "n_genes_by_counts"
TE_FRAC_COL = "TE_fraction"          # per-cell TE fraction, already in obs
SAMPLE_COL = "sample"                # raw sample id: ST0001, ST0002, or Injured-Xhrs / Control-...
SPATIAL_KEY = "spatial"              # adata.obsm[SPATIAL_KEY]
UMAP_KEY = "X_umap"                  # adata.obsm[UMAP_KEY]

RAW_COUNTS_LAYER = "counts"          # matches clustering.py's layers["counts"]
TE_DOMINANT_COL = "te_dominant_outlier"  # boolean obs column from clustering.py Section 4b (Finding #10)
TE_VAR_FLAG_COL = "is_te"            # boolean adata.var column, set in clustering.py Section 2

# Fallback only used if TE_DOMINANT_COL is missing AND TE_VAR_FLAG_COL is
# also missing (shouldn't happen if this checkpoint came from the current
# clustering.py, kept here just as a safety net).
TE_NAME_PATTERNS = ["SoloTE"]

# Automatic derivation thresholds (Section 0) -- replace the old hardcoded
# tier lists. Adjust after seeing what Section 0 prints.
HIGH_MITO_PCT_THRESHOLD = 20   # a cluster's MEDIAN pct_counts_mt above this -> flagged high-mito
TE_DOMINANT_MIN_FRACTION = 0.05  # a cluster holding >= this fraction of ALL te_dominant_outlier cells -> flagged TE-dominant

# Manual override / addition -- fill in by hand from Section 6f's dotplot
# in clustering.py if the automatic derivation above misses something you
# can see visually. Empty by default, same philosophy as
# NO_MARKER_CLUSTERS in clustering.py Section 6g.
MANUAL_HIGH_MITO_CLUSTERS = []
MANUAL_TE_DOMINANT_CLUSTERS = []

# Clusters to drop for the optional Section 6 cut -- empty by default.
# Fill in only after deciding, from Sections 0-5 below, which clusters (if
# any) you actually want excluded. Not a preset "Scenario" anymore.
DROP_CLUSTERS = []

# %% [markdown]
# ## Load

# %%
adata = sc.read_h5ad(CHECKPOINT)
print(adata)

adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype(str)
print(adata.obs[CLUSTER_COL].value_counts())

for col in [MT_COL, COUNTS_COL, GENES_COL, TE_FRAC_COL, SAMPLE_COL]:
    assert col in adata.obs.columns, (
        f"'{col}' not found in adata.obs — fix the CONFIG cell. "
        f"Available columns: {list(adata.obs.columns)}"
    )
assert SPATIAL_KEY in adata.obsm, f"'{SPATIAL_KEY}' not in adata.obsm — available: {list(adata.obsm.keys())}"
assert UMAP_KEY in adata.obsm, f"'{UMAP_KEY}' not in adata.obsm — available: {list(adata.obsm.keys())}"

if "low_confidence" in adata.obs[CLUSTER_COL].unique():
    print("\n[NOTE] 'low_confidence' is present in this checkpoint -- if this came from "
          "the reconstructed clustering.py, Section 6h should have already dropped it "
          "before saving. Either this checkpoint predates that fix, or MIN_CLUSTER_SIZE/"
          "NO_MARKER_CLUSTERS produced a fresh low_confidence bucket you haven't reviewed "
          "yet -- confirm before proceeding.")
else:
    print("\nNo 'low_confidence' category in this checkpoint -- expected if it came from "
          "clustering.py Section 6h (dropped before save).")

# %% [markdown]
# ## 0. Derive high-mito / TE-dominant cluster groups automatically
# Replaces the old hardcoded 44-cluster-era lists. Recomputed fresh every
# time this script runs, against whatever `clusters` actually contains now.

# %%
mito_by_cluster = adata.obs.groupby(CLUSTER_COL, observed=True)[MT_COL].median().sort_values(ascending=False)
print("Median pct_counts_mt by cluster (top 10):")
print(mito_by_cluster.head(10))

HIGH_MITO_CLUSTERS = sorted(set(
    mito_by_cluster[mito_by_cluster > HIGH_MITO_PCT_THRESHOLD].index.tolist()
) | set(MANUAL_HIGH_MITO_CLUSTERS))
print(f"\nHIGH_MITO_CLUSTERS (median %mt > {HIGH_MITO_PCT_THRESHOLD}): {HIGH_MITO_CLUSTERS}")

if TE_DOMINANT_COL in adata.obs.columns:
    te_dom_by_cluster = adata.obs.loc[adata.obs[TE_DOMINANT_COL].astype(bool), CLUSTER_COL].value_counts()
    n_te_dom_total = adata.obs[TE_DOMINANT_COL].astype(bool).sum()
    print(f"\n{n_te_dom_total} te_dominant_outlier cells, by cluster:")
    print(te_dom_by_cluster)
    TE_DOMINANT_CLUSTERS = sorted(set(
        te_dom_by_cluster[te_dom_by_cluster / max(n_te_dom_total, 1) >= TE_DOMINANT_MIN_FRACTION].index.tolist()
    ) | set(MANUAL_TE_DOMINANT_CLUSTERS))
    print(f"TE_DOMINANT_CLUSTERS (>= {TE_DOMINANT_MIN_FRACTION*100:.0f}% of flagged cells): {TE_DOMINANT_CLUSTERS}")
else:
    print(f"\n'{TE_DOMINANT_COL}' not in adata.obs -- can't derive TE_DOMINANT_CLUSTERS automatically. "
          f"Fill MANUAL_TE_DOMINANT_CLUSTERS by hand, or re-run clustering.py Section 4b first.")
    TE_DOMINANT_CLUSTERS = sorted(set(MANUAL_TE_DOMINANT_CLUSTERS))

# %% [markdown]
# ## Helpers

# %%
def highlight_col(adata, groups, label):
    """New categorical obs column: cells in `groups` keep their cluster id,
    everyone else becomes 'other'. Returns the column name."""
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
        print(f"[{label}] empty group list -- skipping UMAP highlight.")
        return
    col = highlight_col(adata, groups, label)
    pal = palette_for(groups)
    sc.pl.umap(
        adata, color=col, palette=pal, size=8,
        title=title or f"UMAP — {label}", show=False,
    )
    savefig(f"umap_{label}")
    plt.show()


def spatial_highlight(adata, groups, label, sample_col=SAMPLE_COL, title=None):
    """One spatial panel per sample; highlighted clusters in color, rest in light gray."""
    if not groups:
        print(f"[{label}] empty group list -- skipping spatial highlight.")
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
        ax.scatter(coords[is_other, 0], coords[is_other, 1], s=2, c=pal["other"], label="other", rasterized=True)
        for g in groups:
            m = (sub.obs[col] == g).values
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=8, c=[pal[g]], label=f"cl {g}", rasterized=True)
        ax.set_title(samp, fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", bbox_to_anchor=(1.08, 1.0), fontsize=9)
    fig.suptitle(title or f"Spatial — {label}", y=1.02)
    plt.tight_layout()
    savefig(f"spatial_{label}")
    plt.show()


def corr_scatter(adata, x, y, hue_groups=None, hue_label="group", title=None, sample=None):
    """Per-cell scatter of two obs columns, optionally colored by cluster membership.
    Prints Pearson + Spearman r for (a) all cells and (b) just the highlighted group."""
    df = adata.obs[[x, y, CLUSTER_COL]].copy()
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=0)

    if hue_groups:
        df["group"] = np.where(df[CLUSTER_COL].isin(hue_groups), hue_label, "other")
        pal = {hue_label: "crimson", "other": "#c9c9c9"}
        order = ["other", hue_label]
    else:
        df["group"] = "all cells"
        pal = {"all cells": "steelblue"}
        order = ["all cells"]

    plt.figure(figsize=(5, 4.2))
    for g in order:
        sub = df[df["group"] == g]
        plt.scatter(sub[x], sub[y], s=4, alpha=0.4, c=pal[g], label=g, rasterized=True)
    plt.xlabel(x); plt.ylabel(y)
    plt.legend(markerscale=3)
    plt.title(title or f"{y} vs {x}")
    plt.tight_layout()
    fname = f"{x}_vs_{y}" + (f"_{hue_label}" if hue_groups else "")
    savefig(fname)
    plt.show()

    for label_, sub in [("all cells", df), *([(hue_label, df[df["group"] == hue_label])] if hue_groups else [])]:
        if len(sub) < 3:
            continue
        pear = sub[x].corr(sub[y], method="pearson")
        spear = sub[x].corr(sub[y], method="spearman")
        print(f"  [{label_:>10}] n={len(sub):>7}  pearson r={pear:+.3f}  spearman r={spear:+.3f}")


# %% [markdown]
# ## 1. High-mito clusters — correlation with TE / UMIs / genes (per-cell, not medians)

# %%
print("=== %mito vs TE_fraction ===")
corr_scatter(adata, MT_COL, TE_FRAC_COL, hue_groups=HIGH_MITO_CLUSTERS, hue_label="high_mito",
             title="%mito vs TE_fraction, per cell")

print("\n=== %mito vs total_counts ===")
corr_scatter(adata, MT_COL, COUNTS_COL, hue_groups=HIGH_MITO_CLUSTERS, hue_label="high_mito",
             title="%mito vs total_counts, per cell")

print("\n=== %mito vs n_genes ===")
corr_scatter(adata, MT_COL, GENES_COL, hue_groups=HIGH_MITO_CLUSTERS, hue_label="high_mito",
             title="%mito vs n_genes, per cell")

# %% [markdown]
# ## 2. Where are the high-mito clusters? UMAP + spatial, highlighted

# %%
umap_highlight(adata, HIGH_MITO_CLUSTERS, "high_mito", title=f"High-mito clusters {HIGH_MITO_CLUSTERS}")
spatial_highlight(adata, HIGH_MITO_CLUSTERS, "high_mito", title="High-mito clusters — spatial, per sample")

# %% [markdown]
# ## 3. TE-dominant clusters — same treatment, plus a per-sample barplot

# %%
print("=== TE_fraction vs %mito, TE-dominant clusters only ===")
corr_scatter(adata, TE_FRAC_COL, MT_COL, hue_groups=TE_DOMINANT_CLUSTERS, hue_label="TE_dominant_cl",
             title=f"TE_fraction vs %mito (clusters {TE_DOMINANT_CLUSTERS} vs rest)")

print("\n=== TE_fraction vs total_counts, TE-dominant clusters only ===")
corr_scatter(adata, TE_FRAC_COL, COUNTS_COL, hue_groups=TE_DOMINANT_CLUSTERS, hue_label="TE_dominant_cl",
             title=f"TE_fraction vs total_counts (clusters {TE_DOMINANT_CLUSTERS} vs rest)")

umap_highlight(adata, TE_DOMINANT_CLUSTERS, "TE_clusters", title=f"TE-dominant clusters {TE_DOMINANT_CLUSTERS}")
spatial_highlight(adata, TE_DOMINANT_CLUSTERS, "TE_clusters", title="TE-dominant clusters — spatial, per sample")

# %%
# Which samples have more cells in the TE-dominant clusters?
if TE_DOMINANT_CLUSTERS:
    sub = adata.obs[adata.obs[CLUSTER_COL].isin(TE_DOMINANT_CLUSTERS)]
    counts = sub.groupby([SAMPLE_COL, CLUSTER_COL], observed=True).size().unstack(fill_value=0)
    print(counts)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("tab10", len(TE_DOMINANT_CLUSTERS)))
    ax.set_ylabel("n cells")
    ax.set_title("TE-dominant clusters — cell counts by sample")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("barplot_TE_clusters_by_sample")
    plt.show()
else:
    print("TE_DOMINANT_CLUSTERS is empty -- nothing to plot.")

# %% [markdown]
# ## 4. The Finding #10 TE-dominant *cells* — not the whole clusters, the flagged cells
#
# Reuses `te_dominant_outlier` from clustering.py Section 4b if present.
# Only recomputes as a fallback (shouldn't be needed if this checkpoint
# came from the current clustering.py, which always sets this column).

# %%
if TE_DOMINANT_COL in adata.obs.columns:
    te_dominant = adata.obs[TE_DOMINANT_COL].astype(bool)
    print(f"Using existing '{TE_DOMINANT_COL}' column: {te_dominant.sum()} TE-dominant cells")
else:
    print(f"'{TE_DOMINANT_COL}' not found — recomputing from raw counts (mirrors clustering.py Section 4b). "
          f"This should not normally be needed.")

    raw_X = adata.layers[RAW_COUNTS_LAYER] if RAW_COUNTS_LAYER else (adata.raw.X if adata.raw is not None else adata.X)
    var_names = adata.raw.var_names if (RAW_COUNTS_LAYER is None and adata.raw is not None) else adata.var_names

    if TE_VAR_FLAG_COL in adata.var.columns:
        is_te_gene = adata.var[TE_VAR_FLAG_COL].reindex(var_names).fillna(False).values
    else:
        pattern = "|".join(TE_NAME_PATTERNS)
        is_te_gene = var_names.str.contains(pattern, case=False, regex=True).values
        print(f"  '{TE_VAR_FLAG_COL}' not in adata.var — using name-pattern fallback "
              f"({is_te_gene.sum()} of {len(var_names)} genes matched)")

    import scipy.sparse as sp
    X = raw_X.tocsr() if sp.issparse(raw_X) else sp.csr_matrix(raw_X)
    top_gene_idx = np.asarray(X.argmax(axis=1)).ravel()
    has_counts = np.asarray(X.sum(axis=1)).ravel() > 0
    te_dominant = pd.Series(is_te_gene[top_gene_idx] & has_counts, index=adata.obs_names)
    adata.obs[TE_DOMINANT_COL] = te_dominant
    print(f"  Recomputed: {te_dominant.sum()} TE-dominant cells")

adata.obs["te_dom_group"] = np.where(te_dominant, "TE_dominant_cell", "other")

# %%
# UMAP — just the flagged cells, highlighted
sc.pl.umap(adata, color="te_dom_group", palette={"TE_dominant_cell": "crimson", "other": "#d3d3d3"},
           size=10, title="Where are the TE-dominant cells? (UMAP)", show=False)
savefig("umap_TE_dominant_cells")
plt.show()

# %%
# Spatial — just the flagged cells, highlighted, per sample
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
    ax.scatter(coords[is_te, 0], coords[is_te, 1], s=14, c="crimson", label="TE-dominant", rasterized=True)
    ax.set_title(f"{samp} (n={is_te.sum()})", fontsize=10)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
for ax in axes[len(samples):]:
    ax.axis("off")
fig.suptitle("Where are the TE-dominant cells? (spatial, per sample)", y=1.02)
plt.tight_layout()
savefig("spatial_TE_dominant_cells")
plt.show()

# %% [markdown]
# ## 5. score_* panels — where do immune / vascular / myogenic markers land?
#
# Depends on `score_*` columns existing in `adata.obs`. `clustering.py`
# no longer computes these itself (the old Section 8 was removed --
# superseded by `cell_type_annotation.py`, see Finding #24). Run
# `cell_type_annotation.py` first and use ITS output object if you want
# this section populated -- otherwise it prints a note and skips.

# %%
score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
print(f"Found {len(score_cols)} score_* columns: {score_cols}")

if score_cols:
    score_by_cluster = adata.obs.groupby(CLUSTER_COL, observed=True)[score_cols].mean()
    plt.figure(figsize=(0.5 * len(score_cols) + 2, 0.3 * len(score_by_cluster) + 2))
    sns.heatmap(score_by_cluster, cmap="RdBu_r", center=0, cbar_kws={"label": "mean score_genes"})
    plt.title("score_genes panels x cluster")
    plt.tight_layout()
    savefig("heatmap_scores_by_cluster")
    plt.show()

    # Candidate identity per cluster = panel with the highest RAW mean score.
    # For a properly z-scored version (recommended for cross-panel
    # comparison), use cell_type_annotation.py's PASO 1c instead -- this is
    # just a quick raw-score derivation to group clusters for the
    # UMAP/spatial highlights below.
    candidate_panel = score_by_cluster.idxmax(axis=1).str.replace("score_", "", regex=False)
    print("\nCandidate top panel per cluster (raw mean score, not z-scored -- "
          "myonuclei/large-range panels can dominate here, see "
          "cell_type_annotation.py PASO 1b/1c for the z-scored version):")
    print(candidate_panel)

    IMMUNE_PANELS = ["neutrophil", "monocyte_patrol", "macro_Mrc1", "macro_Cx3cr1", "macro_Cxcl10",
                      "macro_general", "dendritic", "T_cell", "T_cell_cycling", "NK_cell", "B_cell", "mast_cell"]
    VASCULAR_FAP_NEURAL_PANELS = ["FAP_general", "FAP_adipogenic", "FAP_pro_remodeling", "FAP_stem",
                                   "tenocyte", "schwann_neural", "pericyte_SMC", "endothelial_general",
                                   "endo_arterial", "endo_capillary", "endo_venous"]

    IMMUNE_CLUSTERS = candidate_panel[candidate_panel.isin(IMMUNE_PANELS)].index.tolist()
    VASCULAR_FAP_NEURAL_CLUSTERS = candidate_panel[candidate_panel.isin(VASCULAR_FAP_NEURAL_PANELS)].index.tolist()
    print(f"\nDerived IMMUNE_CLUSTERS: {IMMUNE_CLUSTERS}")
    print(f"Derived VASCULAR_FAP_NEURAL_CLUSTERS: {VASCULAR_FAP_NEURAL_CLUSTERS}")

    umap_highlight(adata, IMMUNE_CLUSTERS, "immune", title="Immune-leaning clusters (by top score_ panel)")
    spatial_highlight(adata, IMMUNE_CLUSTERS, "immune", title="Immune-leaning clusters — spatial")

    umap_highlight(adata, VASCULAR_FAP_NEURAL_CLUSTERS, "vascular_fap_neural",
                    title="Vascular / FAP / neural clusters (by top score_ panel)")
    spatial_highlight(adata, VASCULAR_FAP_NEURAL_CLUSTERS, "vascular_fap_neural",
                       title="Vascular / FAP / neural — spatial")

    flagged = set(HIGH_MITO_CLUSTERS + TE_DOMINANT_CLUSTERS + IMMUNE_CLUSTERS + VASCULAR_FAP_NEURAL_CLUSTERS)
    myogenic_clusters = sorted(set(adata.obs[CLUSTER_COL].unique()) - flagged)
    print(f"\nRemaining (myogenic/core) clusters ({len(myogenic_clusters)}): {myogenic_clusters}")
    umap_highlight(adata, myogenic_clusters, "myogenic", title="Myogenic / core clusters")
    spatial_highlight(adata, myogenic_clusters, "myogenic", title="Myogenic / core clusters — spatial")
else:
    print("No score_* columns found -- run cell_type_annotation.py first if you want this "
          "section populated. Skipping Section 5's immune/vascular/myogenic breakdown.")
    IMMUNE_CLUSTERS, VASCULAR_FAP_NEURAL_CLUSTERS = [], []

# %% [markdown]
# ## 6. Optional cut — drop selected clusters
#
# Generic utility, not a preset "Scenario" anymore. `DROP_CLUSTERS` is
# empty by default (CONFIG cell) -- fill it in only after deciding, from
# Sections 0-5 above, which clusters (if any) you actually want excluded
# for a specific downstream analysis. TE-dominant clusters are excluded
# from the default recommendation (kept in `adata`) unless you explicitly
# add them to `DROP_CLUSTERS` yourself.

# %%
if DROP_CLUSTERS:
    print(f"Dropping {len(DROP_CLUSTERS)} clusters: {DROP_CLUSTERS}")
    if set(DROP_CLUSTERS) & set(TE_DOMINANT_CLUSTERS):
        print(f"[WARNING] DROP_CLUSTERS includes TE-dominant cluster(s): "
              f"{set(DROP_CLUSTERS) & set(TE_DOMINANT_CLUSTERS)} -- double-check this is intentional.")

    adata_cut = adata[~adata.obs[CLUSTER_COL].isin(DROP_CLUSTERS)].copy()
    print(f"Cells: {adata.n_obs} -> {adata_cut.n_obs}  ({adata.n_obs - adata_cut.n_obs} removed, "
          f"{100 * (adata.n_obs - adata_cut.n_obs) / adata.n_obs:.1f}%)")
    print(f"Clusters: {adata.obs[CLUSTER_COL].nunique()} -> {adata_cut.obs[CLUSTER_COL].nunique()}")

    sc.pl.umap(adata_cut, color=CLUSTER_COL, size=8, title="After cut — UMAP", show=False, legend_loc="right margin")
    savefig("cut_umap")
    plt.show()

    samples = sorted(adata_cut.obs[SAMPLE_COL].unique())
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    clusters_cut = sorted(adata_cut.obs[CLUSTER_COL].unique())
    pal_cut = {c: col for c, col in zip(clusters_cut, sns.color_palette("tab20", len(clusters_cut)))}
    for ax, samp in zip(axes, samples):
        sub = adata_cut[adata_cut.obs[SAMPLE_COL] == samp]
        coords = sub.obsm[SPATIAL_KEY]
        for c in clusters_cut:
            m = (sub.obs[CLUSTER_COL] == c).values
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=3, c=[pal_cut[c]], rasterized=True)
        ax.set_title(samp, fontsize=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle("After cut — spatial", y=1.02)
    plt.tight_layout()
    savefig("cut_spatial")
    plt.show()

    sc.tl.rank_genes_groups(adata_cut, groupby=CLUSTER_COL, method="wilcoxon", n_genes=5)
    rg = adata_cut.uns["rank_genes_groups"]
    top_markers = pd.DataFrame({
        "cluster": list(rg["names"].dtype.names),
        "top_gene": [rg["names"][cl][0] for cl in rg["names"].dtype.names],
        "logfc": [rg["logfoldchanges"][cl][0] for cl in rg["names"].dtype.names],
        "padj": [rg["pvals_adj"][cl][0] for cl in rg["names"].dtype.names],
    })
    print(top_markers.sort_values("cluster"))
    top_markers.to_csv(f"{OUTDIR}/cut_top_markers.csv", index=False)
else:
    print("DROP_CLUSTERS is empty -- nothing cut. Fill it in by hand in the CONFIG cell "
          "if you decide, from the sections above, that specific clusters should be excluded.")

# %% [markdown]
# ## Done
# Figures saved to `./clustering_checks_figs/`. `top_markers` for the
# optional cut also saved as CSV if `DROP_CLUSTERS` was non-empty.