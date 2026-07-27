# %% [markdown]
# # Clustering checks — post score_genes / tier decisions
#
# Reads the finalized clustered checkpoint directly (no need to re-run
# HVG/PCA/Harmony/Leiden) and walks through the checks discussed for the
# team review: mito/TE/UMI/gene correlations at the per-cell level, where
# the flagged clusters actually sit (UMAP + spatial), where the 96
# TE-dominant cells are, where the immune/vascular/myogenic markers land,
# and what Scenario C (the aggressive cut) looks like.
#
# **This is untested against your real object** — I don't have access to
# it, only the numbers/column names implied by the notebook exports and
# tables you've shared. Everything in the CONFIG cell below is my best
# guess; fix names there and the rest of the script shouldn't need touching.
# Each section also does a defensive `assert col in adata.obs` so it fails
# fast with a clear message instead of silently plotting garbage.

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
# ## Config — adjust these to match your actual `adata.obs` / `adata.var` columns

# %%
CHECKPOINT = "/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_clustered.h5ad"

CLUSTER_COL = "clusters"
MT_COL = "pct_counts_mt"
COUNTS_COL = "total_counts"
GENES_COL = "n_genes_by_counts"
TE_FRAC_COL = "TE_fraction"          # per-cell TE fraction, already in obs per the tier tables
SAMPLE_COL = "sample"                # raw sample id: ST0001, ST0002, or Injured-Xhrs / Control-...
SPATIAL_KEY = "spatial"              # adata.obsm[SPATIAL_KEY]
UMAP_KEY = "X_umap"                  # adata.obsm[UMAP_KEY]

RAW_COUNTS_LAYER = None              # e.g. "counts" if raw counts live in a layer, else None -> uses adata.raw.X
TE_DOMINANT_COL = "te_dominant"      # boolean obs column, if it already exists from Section 4b/Finding #10
TE_VAR_FLAG_COL = "is_te"            # boolean adata.var column flagging TE features, if it exists

# Fallback: if TE_VAR_FLAG_COL doesn't exist, genes matching any of these
# (case-insensitive) prefixes/substrings are treated as TE features.
# Adjust to match your actual TE nomenclature.
TE_NAME_PATTERNS = ["L1", "Alu", "SINE", "LINE", "ERV", "LTR", "SVA", "B1_", "B2_", "MIR"]

# Tier definitions from the clustering review (slides 3-6)
HIGH_MITO_CLUSTERS = ["35", "36", "24", "28", "2", "25"]
RESIDUAL_CLUSTERS = ["57", "63", "61", "60", "58", "64", "48", "65", "59", "62"]
TE_DOMINANT_CLUSTERS = ["12", "46"]

# Slide 8 marker-based groupings (union of clusters that stood out per identity)
IMMUNE_CLUSTERS = ["60", "62", "64", "58", "63", "65", "5", "46", "12", "61"]
VASCULAR_FAP_NEURAL_CLUSTERS = ["28", "35", "36", "10", "29", "34"]

# Scenario C: drop high-mito + residual, TE-dominant clusters are NEVER dropped
SCENARIO_C_DROP = HIGH_MITO_CLUSTERS + RESIDUAL_CLUSTERS

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

    if hue_groups is not None:
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
    fname = f"{x}_vs_{y}" + (f"_{hue_label}" if hue_groups is not None else "")
    savefig(fname)
    plt.show()

    for label_, sub in [("all cells", df), *([(hue_label, df[df["group"] == hue_label])] if hue_groups is not None else [])]:
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
umap_highlight(adata, HIGH_MITO_CLUSTERS, "high_mito", title="High-mito clusters (35,36,24,28,2,25)")
spatial_highlight(adata, HIGH_MITO_CLUSTERS, "high_mito", title="High-mito clusters — spatial, per sample")

# %% [markdown]
# ## 3. TE-dominant clusters (12, 46) — same treatment, plus a per-sample barplot

# %%
print("=== TE_fraction vs %mito, clusters 12 & 46 only ===")
corr_scatter(adata, TE_FRAC_COL, MT_COL, hue_groups=TE_DOMINANT_CLUSTERS, hue_label="TE_dominant_cl",
             title="TE_fraction vs %mito (clusters 12 & 46 vs rest)")

print("\n=== TE_fraction vs total_counts, clusters 12 & 46 only ===")
corr_scatter(adata, TE_FRAC_COL, COUNTS_COL, hue_groups=TE_DOMINANT_CLUSTERS, hue_label="TE_dominant_cl",
             title="TE_fraction vs total_counts (clusters 12 & 46 vs rest)")

umap_highlight(adata, TE_DOMINANT_CLUSTERS, "TE_clusters", title="TE-dominant clusters 12 & 46")
spatial_highlight(adata, TE_DOMINANT_CLUSTERS, "TE_clusters", title="Clusters 12 & 46 — spatial, per sample")

# %%
# Which samples have more cells in clusters 12/46?
sub = adata.obs[adata.obs[CLUSTER_COL].isin(TE_DOMINANT_CLUSTERS)]
counts = sub.groupby([SAMPLE_COL, CLUSTER_COL], observed=True).size().unstack(fill_value=0)
print(counts)

fig, ax = plt.subplots(figsize=(7, 4))
counts.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("tab10", len(TE_DOMINANT_CLUSTERS)))
ax.set_ylabel("n cells")
ax.set_title("Clusters 12 & 46 — cell counts by sample")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
savefig("barplot_TE_clusters_by_sample")
plt.show()

# %% [markdown]
# ## 4. The 96 TE-dominant *cells* (Finding #10) — not the whole clusters, the flagged cells
#
# If `TE_DOMINANT_COL` already exists in `adata.obs` (saved from Section 4b),
# this reuses it. Otherwise it recomputes "single most-abundant raw transcript
# is a TE" from raw counts — **check this reproduces ~96 cells**; if it
# doesn't, your TE gene identification differs from Section 4b's and you
# should adjust `TE_VAR_FLAG_COL` / `TE_NAME_PATTERNS` or just load the
# original boolean column instead of recomputing.

# %%
if TE_DOMINANT_COL in adata.obs.columns:
    te_dominant = adata.obs[TE_DOMINANT_COL].astype(bool)
    print(f"Using existing '{TE_DOMINANT_COL}' column: {te_dominant.sum()} TE-dominant cells")
else:
    print(f"'{TE_DOMINANT_COL}' not found — recomputing from raw counts (mirrors Finding #10 / Section 4b)")

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
    print(f"  Recomputed: {te_dominant.sum()} TE-dominant cells (Finding #10 reference: 96)")

adata.obs["te_dom_group"] = np.where(te_dominant, "TE_dominant_cell", "other")

# %%
# UMAP — just the 96 cells, highlighted
sc.pl.umap(adata, color="te_dom_group", palette={"TE_dominant_cell": "crimson", "other": "#d3d3d3"},
           size=10, title="Where are the TE-dominant cells? (UMAP)", show=False)
savefig("umap_TE_dominant_cells")
plt.show()

# %%
# Spatial — just the 96 cells, highlighted, per sample
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
fig.suptitle("Where are the 96 TE-dominant cells? (spatial, per sample)", y=1.02)
plt.tight_layout()
savefig("spatial_TE_dominant_cells")
plt.show()

# %% [markdown]
# ## 5. Slide 8 markers — where do immune / vascular / myogenic actually show up?
#
# Spatial + UMAP highlight for each group, plus a matrixplot of the score_*
# columns across all clusters as the "something else" view — the numeric
# summary the UMAP/spatial plots are illustrating.

# %%
umap_highlight(adata, IMMUNE_CLUSTERS, "immune", title="Immune-leaning clusters")
spatial_highlight(adata, IMMUNE_CLUSTERS, "immune", title="Immune-leaning clusters — spatial")

umap_highlight(adata, VASCULAR_FAP_NEURAL_CLUSTERS, "vascular_fap_neural", title="Vascular / FAP / neural clusters")
spatial_highlight(adata, VASCULAR_FAP_NEURAL_CLUSTERS, "vascular_fap_neural", title="Vascular / FAP / neural — spatial")

# Myogenic = everything not flagged elsewhere (the "core" tier)
flagged = set(HIGH_MITO_CLUSTERS + RESIDUAL_CLUSTERS + TE_DOMINANT_CLUSTERS
              + IMMUNE_CLUSTERS + VASCULAR_FAP_NEURAL_CLUSTERS)
myogenic_clusters = sorted(set(adata.obs[CLUSTER_COL].unique()) - flagged - {"low_confidence"})
print(f"Myogenic/core clusters ({len(myogenic_clusters)}): {myogenic_clusters}")
umap_highlight(adata, myogenic_clusters, "myogenic", title="Myogenic / core clusters")
spatial_highlight(adata, myogenic_clusters, "myogenic", title="Myogenic / core clusters — spatial")

# %%
# Matrixplot of score_* columns across clusters — the numeric version of "where do markers land"
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
else:
    print("No score_* columns found in adata.obs — re-run Section 8's score_genes first, "
          "or check the column prefix.")

# %% [markdown]
# ## 6. Scenario C — the aggressive cut (drop high-mito + residual, keep TE-dominant)

# %%
print(f"Dropping {len(SCENARIO_C_DROP)} clusters: {SCENARIO_C_DROP}")
print(f"(TE-dominant clusters {TE_DOMINANT_CLUSTERS} are never dropped)")

adata_c = adata[~adata.obs[CLUSTER_COL].isin(SCENARIO_C_DROP)].copy()
print(f"Cells: {adata.n_obs} -> {adata_c.n_obs}  ({adata.n_obs - adata_c.n_obs} removed, "
      f"{100 * (adata.n_obs - adata_c.n_obs) / adata.n_obs:.1f}%)")
print(f"Clusters: {adata.obs[CLUSTER_COL].nunique()} -> {adata_c.obs[CLUSTER_COL].nunique()}")

# %%
# Spatial + UMAP after the cut (reuses the existing embedding, no recomputation)
sc.pl.umap(adata_c, color=CLUSTER_COL, size=8, title="Scenario C — UMAP after cut", show=False, legend_loc="right margin")
savefig("scenarioC_umap")
plt.show()

samples = sorted(adata_c.obs[SAMPLE_COL].unique())
ncols = min(4, len(samples))
nrows = int(np.ceil(len(samples) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.atleast_1d(axes).flatten()
clusters_c = sorted(adata_c.obs[CLUSTER_COL].unique())
pal_c = {c: col for c, col in zip(clusters_c, sns.color_palette("tab20", len(clusters_c)))}
for ax, samp in zip(axes, samples):
    sub = adata_c[adata_c.obs[SAMPLE_COL] == samp]
    coords = sub.obsm[SPATIAL_KEY]
    for c in clusters_c:
        m = (sub.obs[CLUSTER_COL] == c).values
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=3, c=[pal_c[c]], rasterized=True)
    ax.set_title(samp, fontsize=10)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
for ax in axes[len(samples):]:
    ax.axis("off")
fig.suptitle("Scenario C — spatial after cut", y=1.02)
plt.tight_layout()
savefig("scenarioC_spatial")
plt.show()

# %%
# Marker genes after the cut — does removing the noisy clusters clean up the DE results?
# (slow-ish; comment out if you just want the plots above)
sc.tl.rank_genes_groups(adata_c, groupby=CLUSTER_COL, method="wilcoxon", n_genes=5)
rg = adata_c.uns["rank_genes_groups"]
top_markers = pd.DataFrame(
    {
        "cluster": np.repeat(rg["names"].dtype.names, 1),
        "top_gene": [rg["names"][cl][0] for cl in rg["names"].dtype.names],
        "logfc": [rg["logfoldchanges"][cl][0] for cl in rg["names"].dtype.names],
        "padj": [rg["pvals_adj"][cl][0] for cl in rg["names"].dtype.names],
    }
)
print(top_markers.sort_values("cluster"))
top_markers.to_csv(f"{OUTDIR}/scenarioC_top_markers.csv", index=False)

# %% [markdown]
# ## Done
# Figures saved to `./clustering_checks_figs/`. `top_markers` for Scenario C
# also saved as CSV for pulling into the next round of slides.