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
# # Cell type annotation — marker scores, DE, cluster identity
#
# Reads the final clustered checkpoint and works through cell-type identity:
# marker-gene dotplots + score_* heatmaps (z-scored per panel), a
# per-cluster "best panel" candidate table, rank_genes_groups DE (top
# 10/cluster + full CSV), TE L1/L2 + depth summary per cluster, and a few
# extra standard annotation checks (dendrogram, final_call crosstab if
# present).
#
# ## Updated against the reconstructed `clustering.py` (Finding #24)
# - **No more Scenario B merge.** The old version of this script merged 10
#   hardcoded "residual" cluster IDs (from the 44-cluster era) into
#   `low_confidence` itself. That's no longer this script's job --
#   `clustering.py` Section 6g now builds `low_confidence`
#   (`MIN_CLUSTER_SIZE` + `NO_MARKER_CLUSTERS`) AND Section 6h drops it
#   entirely before saving. By the time this script loads the checkpoint,
#   there should be no `low_confidence` category left to handle.
# - **`PRIOR_HIGH_MITO_CLUSTERS`/`PRIOR_RESIDUAL_CLUSTERS`/
#   `PRIOR_TE_DOMINANT_CLUSTERS` hardcoded lists removed.** Cluster IDs
#   aren't stable across Leiden reruns -- Section 0 below derives
#   high-mito/TE-dominant clusters automatically instead, same approach as
#   `clustering_checks.py`/`clustering_checks_2.py`. The `RESIDUAL` concept
#   is retired entirely (see above).
# - **New Section 4**: TE_L1/TE_L2 fraction per cluster (which clusters
#   carry more L1 vs. L2 signal) + a reads/cells depth summary per cluster
#   (total UMIs, median UMIs/cell, n_cells) -- requested this session,
#   wasn't in the original version of this script.
#
# **Untested against the real object** — column names/values are my best
# guess from `clustering.py`/`clustering_checks*.py`. CONFIG cell is the
# single place to fix names if something doesn't match.
#
# ## Explicit assumptions (fix in CONFIG if wrong)
# - Checkpoint already has `score_*` columns for the 25-panel set built in
#   the old clustering.py Section 8 AND the 5 from clustering_checks_2.py
#   (mast_cell, fiber_type_I/IIA/IIX/IIB) — 30 panels total. Note: the
#   reconstructed clustering.py no longer computes these itself (its old
#   Section 8 was removed, Finding #24) -- if `score_*` columns are
#   missing, this script recomputes them from MARKER_PANELS below rather
#   than failing, same as before.
# - `clusters` is a string/categorical obs column, already final (post
#   clustering.py Section 6g/6h) — no `low_confidence` category expected.
# - `sample` holds raw sample ids (ST0001, ST0002, Control-GER, ...).
# - `condition_label` should already be correct (ST0001=Injured-6hrs,
#   ST0002=Control-young, fixed at the source in clustering.py per
#   Finding #24) — PASO 0 below is now a confirming sanity check rather
#   than a suspected-bug check, but still stops rather than silently
#   overwriting if something doesn't match.
# - `final_call` (Level A) may or may not have survived the merge
#   (Finding #15) — guarded, skipped with a warning if absent.
# - PCA/Harmony embedding for the dendrogram lives in
#   `adata.obsm["X_pca_harmony"]` (Finding #16 naming). Adjust
#   DENDROGRAM_REP if it's actually under a different key.
# - `is_te` in `adata.var` flags TE features; TE family names use "|" as
#   separator (e.g. `TE_SoloTE|Alu`), confirmed from the 6f dotplot output
#   this session — used for the new Section 4 L1/L2 split.

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp

sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 110
warnings.filterwarnings("ignore", category=FutureWarning)

OUTDIR = "cell_type_annotation_figs"
os.makedirs(OUTDIR, exist_ok=True)


def savefig(name):
    plt.savefig(f"{OUTDIR}/{name}.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## CONFIG — single cell to adjust everything

# %%
CHECKPOINT = "/ibex/user/medinils/data/objects/all_samples_family_stardist_cell_clustered.h5ad"

CLUSTER_COL = "clusters"
SAMPLE_COL = "sample"
CONDITION_COL = "condition_label"
FINAL_CALL_COL = "final_call"          # Level A, may be absent (Finding #15)
TE_VAR_FLAG_COL = "is_te"
COUNTS_COL = "total_counts"
TE_FRAC_COL = "TE_fraction"
SPATIAL_KEY = "spatial"                # adata.obsm[SPATIAL_KEY]
UMAP_KEY = "X_umap"                    # adata.obsm[UMAP_KEY]

DENDROGRAM_REP = "X_pca_harmony"       # Finding #16 naming

# TE family pattern for the L1/L2 split (Section 4) -- separator confirmed
# "|" this session (e.g. TE_SoloTE|Alu), not "_". Adjust if your feature
# names look different -- check the printed examples in Load below.
L1_PATTERN = "L1"
L2_PATTERN = "L2"

# Confirmed correct at the source (clustering.py, Finding #24): ST0001=
# Injured-6hrs, ST0002=Control-young. Kept here as a sanity check, not
# because a mismatch is expected anymore.
SAMPLE_TO_CONDITION = {
    "Control-GER": "Control-GER",
    "Control-Old": "Control-Old",
    "Injured-1hrs": "Injured-1hrs",
    "Injured-3hrs": "Injured-3hrs",
    "Injured-12hrs": "Injured-12hrs",
    "Injured-24hrs": "Injured-24hrs",
    "ST0001": "Injured-6hrs",
    "ST0002": "Control-young",
}

# Marker panels — same gene lists as the old clustering.py Section 8 (25
# panels) + clustering_checks_2.py new_panels (mast cell + 4 fiber types).
# score_<name> is expected to already exist in adata.obs for each of
# these; recomputed on the fly if missing (clustering.py no longer
# computes these itself, Finding #24).
MARKER_PANELS = {
    # --- Immune ---
    "neutrophil":         ["S100a8", "S100a9", "Ly6g", "Mpo"],
    "monocyte_patrol":    ["Ccr2"],
    "macro_Mrc1":         ["Mrc1", "Cd163"],
    "macro_Cx3cr1":       ["Cx3cr1"],
    "macro_Cxcl10":       ["Cxcl10"],
    "macro_general":      ["Cd68", "Csf1r", "Adgre1", "Itgam"],
    "dendritic":          ["Cd209a", "Xcr1", "Fscn1", "Cd72", "H2-Ab1"],
    "T_cell":             ["Cd3e", "Cd8a", "Cd8b1", "Cd4"],
    "T_cell_cycling":     ["Cdk1", "Hmgb2"],
    "NK_cell":            ["Nkg7", "Gzma", "Klra4", "Klre1"],
    "B_cell":             ["Cd19", "Cd22", "Ms4a1"],
    "erythrocyte":        ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt"],  # not a native tissue cell type
    "mast_cell":          ["Cma1", "Mcpt4", "Cpa3", "Kit", "Fcer1a"],

    # --- FAPs / tenocytes / neural ---
    "FAP_general":        ["Pdgfra", "Col3a1", "Ly6a", "Dcn"],
    "FAP_adipogenic":     ["Adam12", "Bmp5", "Myoc", "Col1a1", "Mmp2", "Apod"],
    "FAP_pro_remodeling": ["Tnfaip6", "Il33", "Bgn"],
    "FAP_stem":           ["Igfbp5", "Dpp4", "Cd34", "Gsn"],
    "tenocyte":           ["Tnmd", "Scx", "Col1a1", "Dcn"],
    "schwann_neural":     ["Ptn", "Mpz"],

    # --- Myogenic / vascular ---
    "satellite":          ["Pax7", "Myf5", "Myod1", "Cdh15"],
    "myonuclei":          ["Myh1", "Myh2", "Acta1", "Ttn", "Des", "Myh4"],
    "pericyte_SMC":       ["Rgs5", "Acta2", "Myl9", "Myh11"],
    "endothelial_general":["Cdh5", "Pecam1"],
    "endo_arterial":      ["Alpl", "Hey1"],
    "endo_capillary":     ["Lpl"],
    "endo_venous":        ["Vwf", "Icam1", "Lrg1", "Aplnr"],

    # --- Fiber type ---
    "fiber_type_I":       ["Myh7", "Tnnt1", "Tnnc1", "Atp2a2", "Mb"],
    "fiber_type_IIA":     ["Myh2", "Tnnt3", "Atp2a1"],
    "fiber_type_IIX":     ["Myh1"],
    "fiber_type_IIB":     ["Myh4", "Actn3", "Ldha"],
}

# Automatic derivation thresholds (Section 0) -- same logic as
# clustering_checks.py/clustering_checks_2.py, kept consistent across all
# three scripts.
HIGH_MITO_PCT_THRESHOLD = 20
TE_DOMINANT_MIN_FRACTION = 0.05
MT_COL = "pct_counts_mt"
TE_DOMINANT_COL = "te_dominant_outlier"

N_TOP_GENES = 10  # DE genes per cluster for heatmap/dotplot

# %% [markdown]
# ## Load + defensive checks

# %%
adata = sc.read_h5ad(CHECKPOINT)
adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype(str)
print(adata)

for col in [CLUSTER_COL, SAMPLE_COL, MT_COL, COUNTS_COL]:
    assert col in adata.obs.columns, f"'{col}' not in adata.obs — check CONFIG."

n_clusters = adata.obs[CLUSTER_COL].nunique()
print(f"\n{n_clusters} clusters found: {sorted(adata.obs[CLUSTER_COL].unique())}")

if "low_confidence" in adata.obs[CLUSTER_COL].unique():
    print("\n[NOTE] 'low_confidence' present in this checkpoint -- clustering.py "
          "Section 6h should have dropped it before saving (Finding #24). Either "
          "this checkpoint predates that fix, or something upstream changed -- "
          "confirm before treating the annotation below as final.")

te_examples = adata.var_names[adata.var[TE_VAR_FLAG_COL]][:10].tolist() if TE_VAR_FLAG_COL in adata.var.columns else []
print(f"\nExample TE feature names: {te_examples}")
print("^ confirm these contain 'L1'/'L2' substrings before trusting Section 4's split.")

# %% [markdown]
# ## PASO 0 — Confirm sample / condition_label naming
#
# ST0001 = Injured-6hrs, ST0002 = Control-young — fixed at the source in
# clustering.py (Finding #24), so this should already match. Kept as a
# sanity check rather than removed: if a stale/pre-fix checkpoint somehow
# gets loaded here, better to stop than silently annotate on backwards
# labels.

# %%
print("Cells per sample:")
print(adata.obs[SAMPLE_COL].value_counts())

if CONDITION_COL in adata.obs.columns:
    print(f"\nCells per {CONDITION_COL} (as currently in the object):")
    print(adata.obs[CONDITION_COL].value_counts())

    mismatches = {}
    for s in ["ST0001", "ST0002"]:
        n_sample = (adata.obs[SAMPLE_COL] == s).sum()
        expected_label = SAMPLE_TO_CONDITION[s]
        n_expected_label = (
            (adata.obs[SAMPLE_COL] == s) & (adata.obs[CONDITION_COL] == expected_label)
        ).sum()
        if n_expected_label != n_sample:
            mismatches[s] = {
                "n_sample": n_sample,
                "expected_label": expected_label,
                "n_matching_expected": n_expected_label,
                "actual_labels_present": adata.obs.loc[adata.obs[SAMPLE_COL] == s, CONDITION_COL].unique().tolist(),
            }

    if mismatches:
        raise AssertionError(
            "condition_label for ST0001/ST0002 does NOT match the expected "
            f"mapping (ST0001=Injured-6hrs, ST0002=Control-young), despite this "
            f"being fixed at the source in clustering.py. Details: {mismatches}. "
            "This checkpoint may predate the Finding #24 fix -- confirm before "
            "proceeding, don't force the rename blindly."
        )
    else:
        print("\ncondition_label matches the expected mapping -- confirmed.")
else:
    print(f"\n'{CONDITION_COL}' not present -- creating it fresh from SAMPLE_TO_CONDITION.")
    adata.obs[CONDITION_COL] = adata.obs[SAMPLE_COL].map(SAMPLE_TO_CONDITION).astype("category")
    print(adata.obs[CONDITION_COL].value_counts())

# %% [markdown]
# ## Section 0 — Derive high-mito / TE-dominant cluster groups automatically
# Replaces the old hardcoded 44-cluster-era `PRIOR_*` lists. Recomputed
# fresh every run, same logic as clustering_checks.py/clustering_checks_2.py.

# %%
mito_by_cluster = adata.obs.groupby(CLUSTER_COL, observed=True)[MT_COL].median().sort_values(ascending=False)
HIGH_MITO_CLUSTERS = mito_by_cluster[mito_by_cluster > HIGH_MITO_PCT_THRESHOLD].index.tolist()
print(f"HIGH_MITO_CLUSTERS (median %mt > {HIGH_MITO_PCT_THRESHOLD}): {HIGH_MITO_CLUSTERS}")

if TE_DOMINANT_COL in adata.obs.columns:
    n_te_dom_total = adata.obs[TE_DOMINANT_COL].astype(bool).sum()
    te_dom_by_cluster = adata.obs.loc[adata.obs[TE_DOMINANT_COL].astype(bool), CLUSTER_COL].value_counts()
    TE_DOMINANT_CLUSTERS = te_dom_by_cluster[
        te_dom_by_cluster / max(n_te_dom_total, 1) >= TE_DOMINANT_MIN_FRACTION
    ].index.tolist()
    print(f"TE_DOMINANT_CLUSTERS (>= {TE_DOMINANT_MIN_FRACTION*100:.0f}% of "
          f"{n_te_dom_total} flagged cells): {TE_DOMINANT_CLUSTERS}")
else:
    print(f"'{TE_DOMINANT_COL}' not found -- TE_DOMINANT_CLUSTERS left empty.")
    TE_DOMINANT_CLUSTERS = []

# %% [markdown]
# ## PASO 1a — Marker gene dotplot (individual genes, grouped by panel)
#
# Mean expression + % expressing per cluster, genes grouped by panel.

# %%
panels_present = {}
for name, genes in MARKER_PANELS.items():
    genes_present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        print(f"  [{name}] missing from var_names, skipped: {missing}")
    if genes_present:
        panels_present[name] = genes_present
    else:
        print(f"  [{name}] NO genes found in adata.var_names -- panel skipped entirely")

sc.pl.dotplot(
    adata,
    var_names=panels_present,
    groupby=CLUSTER_COL,
    standard_scale="var",
    dendrogram=False,
    figsize=(0.35 * sum(len(v) for v in panels_present.values()) + 3, 0.25 * n_clusters + 3),
)
savefig("dotplot_marker_genes_by_cluster")
plt.show()

# %% [markdown]
# ## PASO 1b — score_* heatmap, z-scored per panel (per column)
#
# Each panel standardized to its own mean/std across clusters before
# comparing across panels -- a raw heatmap lets wide-range panels (e.g.
# myonuclei) drown out the rest.

# %%
expected_score_cols = [f"score_{name}" for name in panels_present]
missing_scores = [c for c in expected_score_cols if c not in adata.obs.columns]

if missing_scores:
    print(f"Recomputing {len(missing_scores)} missing score_* columns: {missing_scores}")
    for c in missing_scores:
        name = c[len("score_"):]
        sc.tl.score_genes(adata, panels_present[name], score_name=c)

score_cols = [c for c in expected_score_cols if c in adata.obs.columns]
print(f"\nUsing {len(score_cols)} score_* columns.")

score_by_cluster = adata.obs.groupby(CLUSTER_COL, observed=True)[score_cols].mean()
score_by_cluster_z = (score_by_cluster - score_by_cluster.mean(axis=0)) / score_by_cluster.std(axis=0)

plt.figure(figsize=(0.5 * len(score_cols) + 3, 0.3 * len(score_by_cluster_z) + 3))
sns.heatmap(score_by_cluster_z, cmap="RdBu_r", center=0, cbar_kws={"label": "z-score (across clusters, per panel)"})
plt.title("score_genes panels x cluster (column-wise z-score)")
plt.tight_layout()
savefig("heatmap_scores_zscored_by_cluster")
plt.show()

# %% [markdown]
# ## PASO 1c — Candidate identity table: best panel per cluster + margin to 2nd best
#
# Margin close to 0 flags ambiguous clusters (e.g. a cluster sitting in a
# continuum, no exclusive marker).

# %%
candidate_rows = []
for cl in score_by_cluster_z.index:
    row = score_by_cluster_z.loc[cl].sort_values(ascending=False)
    candidate_rows.append({
        "cluster": cl,
        "n_cells": (adata.obs[CLUSTER_COL] == cl).sum(),
        "top_panel": row.index[0].replace("score_", ""),
        "top_zscore": row.iloc[0],
        "second_panel": row.index[1].replace("score_", ""),
        "second_zscore": row.iloc[1],
        "margin": row.iloc[0] - row.iloc[1],
        "flag": (
            "high_mito" if cl in HIGH_MITO_CLUSTERS else
            "TE_dominant" if cl in TE_DOMINANT_CLUSTERS else
            ""
        ),
    })

candidate_table = pd.DataFrame(candidate_rows).sort_values("margin")
print("Candidate identity table (sorted by margin, most ambiguous first):")
print(candidate_table.to_string(index=False))
candidate_table.to_csv(f"{OUTDIR}/candidate_identity_by_cluster.csv", index=False)

AMBIGUOUS_MARGIN_THRESHOLD = 0.5  # z-score units, adjust after eyeballing the table
ambiguous = candidate_table[candidate_table["margin"] < AMBIGUOUS_MARGIN_THRESHOLD]
print(f"\n{len(ambiguous)} clusters with margin < {AMBIGUOUS_MARGIN_THRESHOLD} "
      f"(candidates for manual review, mixed identity, or continuum position):")
print(ambiguous[["cluster", "n_cells", "top_panel", "second_panel", "margin", "flag"]].to_string(index=False))

# %% [markdown]
# ## PASO 2a — Differential expression, Wilcoxon, all clusters

# %%
sc.tl.rank_genes_groups(adata, groupby=CLUSTER_COL, method="wilcoxon")

# %% [markdown]
# ## PASO 2b — Top 10 genes per cluster, heatmap

# %%
_dendrogram_ok = DENDROGRAM_REP in adata.obsm
if _dendrogram_ok:
    sc.tl.dendrogram(adata, groupby=CLUSTER_COL, use_rep=DENDROGRAM_REP)

sc.pl.rank_genes_groups_heatmap(
    adata,
    n_genes=N_TOP_GENES,
    show_gene_labels=True,
    groupby=CLUSTER_COL,
    dendrogram=_dendrogram_ok,
    figsize=(0.3 * N_TOP_GENES * n_clusters + 3, 0.25 * n_clusters + 3),
)
savefig("de_top10_heatmap")
plt.show()

# Dotplot equivalent -- easier to read exact effect size/expressing % per gene
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=N_TOP_GENES,
    groupby=CLUSTER_COL,
    dendrogram=_dendrogram_ok,
    figsize=(0.3 * N_TOP_GENES * n_clusters + 3, 0.25 * n_clusters + 3),
)
savefig("de_top10_dotplot")
plt.show()

# %% [markdown]
# ## PASO 2c — Full DE table to CSV (gene, logFC, padj, cluster)

# %%
de_full = sc.get.rank_genes_groups_df(adata, group=None)
de_full = de_full.rename(columns={
    "group": "cluster",
    "names": "gene",
    "logfoldchanges": "logFC",
    "pvals_adj": "padj",
})
de_full = de_full[["cluster", "gene", "logFC", "pvals", "padj", "scores"]]
de_full.to_csv(f"{OUTDIR}/de_full_wilcoxon_all_clusters.csv", index=False)
print(f"Full DE table: {de_full.shape[0]} rows (genes x clusters), saved to "
      f"{OUTDIR}/de_full_wilcoxon_all_clusters.csv")
print(de_full.head(10).to_string(index=False))

de_top10 = de_full.sort_values(["cluster", "padj"]).groupby("cluster").head(N_TOP_GENES)
de_top10.to_csv(f"{OUTDIR}/de_top10_per_cluster.csv", index=False)

# %% [markdown]
# ## PASO 3 — Additional standard annotation checks

# %%
if DENDROGRAM_REP in adata.obsm:
    sc.pl.dendrogram(adata, groupby=CLUSTER_COL)
    savefig("cluster_dendrogram")
    plt.show()
else:
    print(f"'{DENDROGRAM_REP}' not in adata.obsm -- skipping dendrogram. "
          f"Available obsm keys: {list(adata.obsm.keys())}")

# %%
if FINAL_CALL_COL in adata.obs.columns:
    crosstab = pd.crosstab(adata.obs[CLUSTER_COL], adata.obs[FINAL_CALL_COL], normalize="index")
    print(f"Cluster x {FINAL_CALL_COL} (Level A) crosstab, row-normalized:")
    print(crosstab.round(2).to_string())
    crosstab.to_csv(f"{OUTDIR}/crosstab_cluster_vs_final_call.csv")
else:
    print(f"'{FINAL_CALL_COL}' not in adata.obs (Finding #15 -- didn't survive "
          "the ad.concat merge). Level B / Level A cross-validation skipped, "
          "annotation proceeds on marker-score + DE evidence only.")

# %% [markdown]
# ## PASO 3b — Cluster size barplot

# %%
sizes = adata.obs[CLUSTER_COL].value_counts().sort_values(ascending=False)
plt.figure(figsize=(0.25 * len(sizes) + 3, 5))
sizes.plot(kind="bar", color="#4c72b0")
plt.yscale("log")
plt.ylabel("n cells (log scale)")
plt.title("Cluster sizes")
plt.tight_layout()
savefig("cluster_sizes_barplot")
plt.show()

# %% [markdown]
# ## PASO 3c — Correlation matrix between clusters on the score_* profile

# %%
cluster_corr = score_by_cluster_z.T.corr()
plt.figure(figsize=(0.3 * len(cluster_corr) + 3, 0.3 * len(cluster_corr) + 3))
sns.heatmap(cluster_corr, cmap="RdBu_r", center=0, square=True,
            cbar_kws={"label": "correlation (score_* profile)"})
plt.title("Cluster x cluster correlation on score_* panel profile")
plt.tight_layout()
savefig("cluster_score_correlation_matrix")
plt.show()

HIGH_CORR_THRESHOLD = 0.9
high_corr_pairs = (
    cluster_corr.where(np.triu(np.ones(cluster_corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={"level_0": "cluster_a", "level_1": "cluster_b", 0: "corr"})
)
high_corr_pairs = high_corr_pairs[high_corr_pairs["corr"] > HIGH_CORR_THRESHOLD].sort_values("corr", ascending=False)
print(f"Cluster pairs with score_* correlation > {HIGH_CORR_THRESHOLD} "
      f"(candidates for a shared identity):")
print(high_corr_pairs.to_string(index=False) if not high_corr_pairs.empty else "  none")

# %% [markdown]
# ## PASO 3d — UMAP + spatial colored by candidate identity (top_panel)

# %%
top_panel_map = candidate_table.set_index("cluster")["top_panel"].to_dict()
adata.obs["candidate_identity"] = adata.obs[CLUSTER_COL].map(top_panel_map).astype("category")

if UMAP_KEY in adata.obsm:
    sc.pl.umap(adata, color="candidate_identity", size=8, legend_loc="right margin",
               title="UMAP -- candidate identity (top_panel per cluster)", show=False)
    savefig("umap_candidate_identity")
    plt.show()
else:
    print(f"'{UMAP_KEY}' not in adata.obsm -- skipping UMAP by candidate identity.")

if SPATIAL_KEY in adata.obsm:
    samples = sorted(adata.obs[SAMPLE_COL].unique())
    ncols = min(4, len(samples))
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    identities = sorted(adata.obs["candidate_identity"].dropna().unique())
    pal = {ident: col for ident, col in zip(identities, sns.color_palette("tab20", len(identities)))}
    for ax, samp in zip(axes, samples):
        sub = adata[adata.obs[SAMPLE_COL] == samp]
        coords = sub.obsm[SPATIAL_KEY]
        for ident in identities:
            m = (sub.obs["candidate_identity"] == ident).values
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=3, c=[pal[ident]], rasterized=True)
        ax.set_title(samp, fontsize=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle("Spatial -- candidate identity (top_panel per cluster), all samples", y=1.02)
    plt.tight_layout()
    savefig("spatial_candidate_identity")
    plt.show()
else:
    print(f"'{SPATIAL_KEY}' not in adata.obsm -- skipping spatial plot by candidate identity.")

# %% [markdown]
# ## PASO 3e — Margin check restricted to the flagged tiers (high-mito, TE-dominant)

# %%
flagged_tiers = HIGH_MITO_CLUSTERS + TE_DOMINANT_CLUSTERS
flagged_present = [c for c in flagged_tiers if c in candidate_table["cluster"].values]
print(f"Candidate identity for flagged tiers (high-mito: {HIGH_MITO_CLUSTERS}, "
      f"TE-dominant: {TE_DOMINANT_CLUSTERS}):")
print(candidate_table[candidate_table["cluster"].isin(flagged_present)]
      [["cluster", "n_cells", "top_panel", "second_panel", "margin", "flag"]]
      .to_string(index=False))

# %% [markdown]
# ## PASO 4 — TE L1/L2 + depth summary per cluster
#
# Requested this session: which clusters carry more L1 (vs. L2) signal,
# how many reads (UMIs) per cluster, how many cells per cluster. All from
# RAW counts (pre-normalization), same convention as `TE_fraction` itself
# (Finding #10) -- not distorted by normalize_total()'s inflation on
# low-count cells.

# %%
if TE_VAR_FLAG_COL in adata.var.columns:
    te_names = adata.var_names[adata.var[TE_VAR_FLAG_COL]]
    l1_features = te_names[te_names.str.contains(L1_PATTERN, case=False, regex=False)]
    l2_features = te_names[te_names.str.contains(L2_PATTERN, case=False, regex=False)]
    print(f"L1-matching TE features ({len(l1_features)}): {l1_features[:10].tolist()}")
    print(f"L2-matching TE features ({len(l2_features)}): {l2_features[:10].tolist()}")

    if len(l1_features) == 0 or len(l2_features) == 0:
        print("[WARNING] L1_PATTERN or L2_PATTERN matched 0 features -- fix the pattern "
              "in CONFIG (check the TE feature name examples printed in Load) before "
              "trusting the numbers below.")

    raw_X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    raw_X = raw_X.tocsr() if sp.issparse(raw_X) else sp.csr_matrix(raw_X)

    def raw_fraction(feature_names):
        if len(feature_names) == 0:
            return np.zeros(adata.n_obs)
        idx = [adata.var_names.get_loc(f) for f in feature_names]
        sub_sum = np.asarray(raw_X[:, idx].sum(axis=1)).ravel()
        total = np.asarray(raw_X.sum(axis=1)).ravel()
        return np.divide(sub_sum, total, out=np.zeros_like(sub_sum, dtype=float), where=total > 0)

    if "TE_L1_fraction" not in adata.obs.columns:
        adata.obs["TE_L1_fraction"] = raw_fraction(l1_features)
    if "TE_L2_fraction" not in adata.obs.columns:
        adata.obs["TE_L2_fraction"] = raw_fraction(l2_features)

    depth_summary = adata.obs.groupby(CLUSTER_COL, observed=True).agg(
        n_cells=(CLUSTER_COL, "size"),
        total_reads=(COUNTS_COL, "sum"),
        median_reads_per_cell=(COUNTS_COL, "median"),
        median_TE_L1_fraction=("TE_L1_fraction", "median"),
        median_TE_L2_fraction=("TE_L2_fraction", "median"),
    )
    if TE_FRAC_COL in adata.obs.columns:
        depth_summary["median_TE_fraction"] = adata.obs.groupby(CLUSTER_COL, observed=True)[TE_FRAC_COL].median()

    depth_summary = depth_summary.sort_values("median_TE_L1_fraction", ascending=False)
    print("\nDepth + TE L1/L2 summary per cluster (sorted by L1 fraction, highest first):")
    print(depth_summary.to_string())
    depth_summary.to_csv(f"{OUTDIR}/depth_and_TE_L1_L2_by_cluster.csv")

    print("\nTop 5 clusters by median TE_L1_fraction:")
    print(depth_summary.head(5)[["n_cells", "total_reads", "median_reads_per_cell", "median_TE_L1_fraction"]])
    print("\nTop 5 clusters by median TE_L2_fraction:")
    print(depth_summary.sort_values("median_TE_L2_fraction", ascending=False)
          .head(5)[["n_cells", "total_reads", "median_reads_per_cell", "median_TE_L2_fraction"]])

    # Visual — L1 vs L2 per cluster, side by side, sorted by L1
    fig, axes = plt.subplots(1, 2, figsize=(0.3 * len(depth_summary) + 4, 5), sharey=True)
    axes[0].bar(depth_summary.index.astype(str), depth_summary["median_TE_L1_fraction"], color="#c0392b")
    axes[0].set_title("Median TE_L1_fraction by cluster")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar(depth_summary.index.astype(str), depth_summary["median_TE_L2_fraction"], color="#2980b9")
    axes[1].set_title("Median TE_L2_fraction by cluster")
    axes[1].tick_params(axis="x", rotation=90)
    plt.tight_layout()
    savefig("barplot_TE_L1_L2_by_cluster")
    plt.show()

    # Reads-per-cluster, two views: total (pseudobulk depth contributed)
    # and median-per-cell (avg depth) -- these tell different stories,
    # a cluster can be high in one and low in the other just from n_cells.
    fig, axes = plt.subplots(1, 2, figsize=(0.3 * len(depth_summary) + 4, 5))
    order = depth_summary.sort_values("total_reads", ascending=False)
    axes[0].bar(order.index.astype(str), order["total_reads"], color="#4c72b0")
    axes[0].set_yscale("log")
    axes[0].set_title("Total reads (UMIs) per cluster")
    axes[0].tick_params(axis="x", rotation=90)
    order2 = depth_summary.sort_values("median_reads_per_cell", ascending=False)
    axes[1].bar(order2.index.astype(str), order2["median_reads_per_cell"], color="#55a868")
    axes[1].set_yscale("log")
    axes[1].set_title("Median reads/cell per cluster")
    axes[1].tick_params(axis="x", rotation=90)
    plt.tight_layout()
    savefig("barplot_reads_per_cluster")
    plt.show()
else:
    print(f"'{TE_VAR_FLAG_COL}' not in adata.var -- skipping PASO 4 entirely.")

# %% [markdown]
# ## Save annotated checkpoint (only after cluster names are actually assigned)
#
# Leaving this as a manual step -- fill CLUSTER_NAME_MAP after reviewing
# the outputs above, then run this cell.

# %%
CLUSTER_NAME_MAP = {
    # "0": "myonuclei_IIB", "1": "FAP_general", ...  # fill in after review
}

if CLUSTER_NAME_MAP:
    unmapped = set(adata.obs[CLUSTER_COL].unique()) - set(CLUSTER_NAME_MAP.keys())
    assert not unmapped, f"CLUSTER_NAME_MAP missing entries for: {sorted(unmapped)}"
    adata.obs["cell_type"] = adata.obs[CLUSTER_COL].map(CLUSTER_NAME_MAP).astype("category")
    adata.write(CHECKPOINT)
    print("Checkpoint saved with 'cell_type' column:", CHECKPOINT)
else:
    print("CLUSTER_NAME_MAP is empty -- nothing saved. Fill it in after reviewing "
          "PASO 1/2/3/4 outputs, then re-run this cell.")