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
# # Visium HD TE Analysis — QC by Cell (StarDist + Cellpose combined segmentation)
#
# Segmentation uses the real H&E microscope image (~8900x10600px), not the
# lower-resolution CytAssist fallback (~3200x3000px). Object construction up
# to segmentation (destripe, StarDist, both label expansions) lives in
# `scripts/build_stardist_cell_object.py`, run per sample via SLURM. This
# notebook picks up from the bin-level object: compares the two expansion
# algorithms visually, combines StarDist with Cellpose (StarDist "pure" +
# "pure muscle" = Cellpose minus StarDist), aggregates bins to cells,
# validates the segmentation, and runs the standard QC/normalization
# pipeline. Marker-based classification and mitochondrial filtering are
# left for after clustering, and are not part of this notebook.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bin2cell as b2c
from scipy.stats import median_abs_deviation
import math

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
samples = ["Control-GER", "Control-Old", "Injured-1hrs", "Injured-3hrs", "Injured-12hrs", "Injured-24hrs", "ST0001", "ST0002"]

sample_labels = {
    "Control-GER": "Control (geriatric)",
    "Control-Old": "Control (old)",
    "ST0002": "Control (young)",
    "ST0001": "Injured, 6h",
    "Injured-1hrs": "Injured, 1h",
    "Injured-3hrs": "Injured, 3h",
    "Injured-12hrs": "Injured, 12h",
    "Injured-24hrs": "Injured, 24h",
}

N_COLS = 4
N_ROWS = math.ceil(len(samples) / N_COLS)

bin_adatas = {}
for s in samples:
    a = sc.read_h5ad(f"/ibex/user/medinils/data/objects/{s}_family_stardist_bins.h5ad")
    a.obs["condition_label"] = sample_labels[s]
    bin_adatas[s] = a
    n_nuclei = (a.obs["labels_he"] > 0).sum()
    print(s, "->", sample_labels[s], "|", a.shape, "| bins with a nucleus label:", n_nuclei)

# %% [markdown]
# ## 1a. Compare raw image vs. the two expansion algorithms, all 8 samples
# Columns: raw H&E crop -> raw nuclei (StarDist) -> expanded (distance-based)
# -> expanded (volume_ratio).

# %%
fig, axes = plt.subplots(len(samples), 4, figsize=(24, 5 * len(samples)), dpi=70)

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
# `labels_expanded_volume`: biologically motivated per-cell expansion,
# rather than a fixed distance applied equally to every cell. **Adjust
# EXPANSION_KEY below if the comparison above suggests otherwise.**

# %%
EXPANSION_KEY = "labels_expanded_volume"

# %% [markdown]
# ## 1c. Combine segmentations: pure StarDist + pure muscle
# A true bin-by-bin subtraction, not object-level: wherever StarDist
# (expanded) has a nucleus, StarDist wins (`labels_joint_source =
# "primary"`, pure StarDist). Everywhere else, if Cellpose claims that
# bin, it keeps its ORIGINAL Cellpose object id (`"secondary"`, pure
# muscle = bin_cellpose − bin_stardist), preserving Cellpose's own
# per-cell perimeter. This is bin-level, so it must run before
# `bin_to_cell` (Section 1e).

# %%
OFFSET = 100_000  # keeps Cellpose ids from colliding with StarDist ids in labels_joint

for s, a in bin_adatas.items():
    mapping = pd.read_parquet(f"/ibex/user/medinils/data/samples/{s}/barcode_mappings.parquet")
    mapping = mapping[["square_002um", "cell_id"]].dropna(subset=["cell_id"])
    mapping = mapping.set_index("square_002um")["cell_id"]

    raw = mapping.reindex(a.obs_names)
    codes, _ = pd.factorize(raw)          # -1 for NaN, 0,1,2... for each unique cell_id
    labels_cellpose = codes + 1           # 0 = unassigned, 1..N = cells
    labels_cellpose[codes == -1] = 0
    a.obs["labels_cellpose"] = labels_cellpose

    is_stardist = a.obs[EXPANSION_KEY] > 0
    is_musculo = (~is_stardist) & (a.obs["labels_cellpose"] > 0)

    labels_joint = np.zeros(a.n_obs, dtype=np.int64)
    labels_joint[is_stardist.values] = a.obs.loc[is_stardist, EXPANSION_KEY].astype(np.int64).values
    labels_joint[is_musculo.values] = a.obs.loc[is_musculo, "labels_cellpose"].astype(np.int64).values + OFFSET

    a.obs["labels_joint"] = labels_joint
    a.obs["labels_joint_source"] = np.select(
        [is_stardist, is_musculo], ["primary", "secondary"], default="none"
    )
    print(s, "| sources:", a.obs["labels_joint_source"].value_counts().to_dict())

# %% [markdown]
# ## 1c-i. Visual check: StarDist vs Cellpose vs Joint, same crop as 1a

# %%
fig, axes = plt.subplots(len(samples), 4, figsize=(24, 5 * len(samples)), dpi=70)

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

    bdata_sd = a[mask].copy()
    bdata_sd = bdata_sd[bdata_sd.obs[EXPANSION_KEY] > 0]
    bdata_sd.obs[EXPANSION_KEY] = bdata_sd.obs[EXPANSION_KEY].astype(str)
    sc.pl.spatial(bdata_sd, color=[EXPANSION_KEY], show=False, ax=axes[row, 1],
                  img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer", legend_loc=None)
    axes[row, 1].set_title("StarDist", fontsize=9)

    bdata_cp = a[mask].copy()
    bdata_cp = bdata_cp[bdata_cp.obs["labels_cellpose"] > 0]
    bdata_cp.obs["labels_cellpose"] = bdata_cp.obs["labels_cellpose"].astype(str)
    sc.pl.spatial(bdata_cp, color=["labels_cellpose"], show=False, ax=axes[row, 2],
                  img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer", legend_loc=None)
    axes[row, 2].set_title("Cellpose", fontsize=9)

    bdata_j = a[mask].copy()
    bdata_j = bdata_j[bdata_j.obs["labels_joint"] > 0]
    sc.pl.spatial(bdata_j, color=["labels_joint_source"], show=False, ax=axes[row, 3],
                  img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer",
                  palette={"primary": "#028090", "secondary": "#C97B2E"}, legend_loc=None)
    axes[row, 3].set_title("Joint (orange = pure muscle)", fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1c-ii. Validate: do the "secondary" (pure muscle) bins make sense?
# Quantitative, spatial (spread across the whole tissue vs. clumped in one
# area, which would suggest StarDist failed there), and expression-based
# (does it still look like muscle, or is there real immune signal inside?).

# %%
# a. quantitative summary
summary_source = []
for s, a in bin_adatas.items():
    counts = a.obs["labels_joint_source"].value_counts()
    total = counts.get("primary", 0) + counts.get("secondary", 0)
    summary_source.append({"sample": s, "primary": counts.get("primary", 0),
                            "secondary": counts.get("secondary", 0),
                            "%_secondary": round(counts.get("secondary", 0) / total * 100, 2) if total else 0})
print(pd.DataFrame(summary_source).set_index("sample"))

# %%
# b. spatial, full tissue — StarDist (the minority) should be spread across
# the whole tissue, not missing from one large region
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = bin_adatas[s]
    coords = a.obsm["spatial"]
    is_p = a.obs["labels_joint_source"] == "primary"
    is_s = a.obs["labels_joint_source"] == "secondary"
    ax.scatter(coords[is_s, 0], coords[is_s, 1], c="lightgray", s=2, alpha=0.3)
    ax.scatter(coords[is_p, 0], coords[is_p, 1], c="#028090", s=4, alpha=0.9)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{s} ({is_p.sum()} StarDist bins)")
plt.suptitle("Pure StarDist (teal) over pure muscle (gray)", y=1.02)
plt.tight_layout()
plt.show()

# %%
# c. marker profile: primary vs secondary
mt_genes = [g for g in bin_adatas[samples[0]].var_names if g.lower().startswith("mt-")]
muscle_markers = [g for g in ["Myh1", "Myh2", "Acta1", "Ttn", "Des"] if g in bin_adatas[samples[0]].var_names]
immune_markers = [g for g in ["S100a8", "S100a9", "Ly6g", "Mpo", "Nos2", "Cd68", "Cd86", "Il1b", "Tnf",
                               "Cd163", "Mrc1", "Arg1", "Pdgfra", "Ly6a", "Dcn"] if g in bin_adatas[samples[0]].var_names]

rows = []
for s, a in bin_adatas.items():
    for source in ["primary", "secondary"]:
        sub = a[a.obs["labels_joint_source"] == source]
        if sub.n_obs == 0:
            continue
        total = np.asarray(sub.X.sum(axis=1)).flatten().sum()
        muscle_sum = np.asarray(sub[:, muscle_markers].X.sum()).flatten()[0]
        immune_sum = np.asarray(sub[:, immune_markers].X.sum()).flatten()[0]
        rows.append({"sample": s, "source": source, "n_bins": sub.n_obs,
                      "muscle_frac": round(muscle_sum / total, 4) if total else 0,
                      "immune_frac": round(immune_sum / total, 4) if total else 0})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# **Pending re-confirmation** with the current proportions (secondary is
# now ~60-80% of bins, not <1% as it was before the Section 1c fix) —
# review the table above before assuming pure muscle carries no signal
# distinct from pure StarDist.

# %% [markdown]
# ## 1d. Spatial containment prior (dominant_cellpose_size)
# For each StarDist cell: which Cellpose object dominates its territory,
# and how large is it (in bins)? A large dominant Cellpose territory means
# the nucleus sits inside a big fused myofiber -> myonuclear prior.
# Small/zero -> candidate for a distinct (immune/stromal) cell type. Uses
# raw `labels_cellpose` (not `labels_joint`) — this is about which
# territory a nucleus falls in, not about gap-filling. Bin-level, computed
# before aggregation (Section 1e).

# %%
dominant_cellpose_size = {}
for s, a in bin_adatas.items():
    bins_df = a.obs[a.obs[EXPANSION_KEY] > 0][[EXPANSION_KEY, "labels_cellpose"]]
    cellpose_sizes = a.obs.loc[a.obs["labels_cellpose"] > 0, "labels_cellpose"].value_counts()

    def dominant(sub):
        vals = sub[sub > 0]
        return vals.value_counts().idxmax() if len(vals) else 0

    dom_label = bins_df.groupby(EXPANSION_KEY)["labels_cellpose"].apply(dominant)
    dom_size = dom_label.map(cellpose_sizes).fillna(0)

    dominant_cellpose_size[s] = dom_size  # indexed by the StarDist (EXPANSION_KEY) label id
    print(s, "| StarDist cells with a large Cellpose territory (>p90):",
          (dom_size > cellpose_sizes.quantile(0.90)).sum(), "of", len(dom_size))

# %% [markdown]
# ## 1e. Aggregate bins -> cells (bin_to_cell on labels_joint), the
# minimum bin_count filter, and merge in the Section 1d prior.

# %%
cell_adatas = {}
for s, a in bin_adatas.items():
    print(f"[{s}] aggregating bins -> cells (labels_joint) ...")
    pseudo_sc = b2c.bin_to_cell(a, labels_key="labels_joint", spatial_keys=["spatial"])
    print(f"[{s}] pseudo-cells before bin_count filter: {pseudo_sc.shape}")

    keep = pseudo_sc.obs["bin_count"] >= 3  # drop cells built from <=2 bins
    pseudo_sc = pseudo_sc[keep].copy()
    print(f"[{s}] pseudo-cells after bin_count filter: {pseudo_sc.shape} ({keep.sum()}/{len(keep)} kept)")

    cell_adatas[s] = pseudo_sc

# %%
# bin_to_cell stores the cell id under "object_id", not under the
# labels_key name that was passed in ("labels_joint") — confirmed against
# the printed column list.
print(cell_adatas[samples[0]].obs.columns.tolist())

CELL_ID_COL = "object_id"
assert CELL_ID_COL in cell_adatas[samples[0]].obs.columns, (
    f"'{CELL_ID_COL}' not found -- check the printed column list above, "
    "this bin2cell version may name the column differently."
)

# %%
for s, a in cell_adatas.items():
    a.obs["dominant_cellpose_size"] = (
        a.obs[CELL_ID_COL].map(dominant_cellpose_size[s]).fillna(0)
    )
    # "secondary" cells (pure muscle, no StarDist nucleus of their own) have
    # no StarDist label id to match against dominant_cellpose_size -- 0 is
    # the correct "not applicable" value for them, not a real "no
    # containment" signal.
    a.write(f"/ibex/user/medinils/data/objects/{s}_family_stardist_cell.h5ad")

# %% [markdown]
# ## 1f. Cell size sanity check (bin_count distribution)
# Are there cells with an unusually high bin_count (possible fusion of two
# neighbouring cells into one label)?

# %%
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    ax.hist(a.obs["bin_count"], bins=50, color="#02C39A", alpha=0.8)
    ax.axvline(3, color="#C97B2E", linestyle="--", linewidth=1.2, label="min=3")
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
# ## 2. QC metrics: counts, genes, mitochondrial %, and raw TE burden per
# cell. Zero-count cells are dropped before pct_counts_mt (avoids a
# NaN-propagation issue).

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
# Defines MIN_COUNTS_FLOOR (canonical definition, reused throughout the
# notebook) and bin_count upper-bound candidates (p99/p99.5), and checks
# whether a large bin_count correlates with high total_counts/n_genes/
# pct_counts_mt as expected for genuinely larger cells, or looks decoupled
# (possible segmentation error).

# %%
MIN_COUNTS_FLOOR = 10

BIN_COUNT_P99 = qc_df["bin_count"].quantile(0.99)
BIN_COUNT_P995 = qc_df["bin_count"].quantile(0.995)

TOTAL_COUNTS_P99 = qc_df["total_counts"].quantile(0.99)
N_GENES_P99 = qc_df["n_genes_by_counts"].quantile(0.99)
PCT_MT_P99 = qc_df["pct_counts_mt"].quantile(0.99)

# "large but low-signal" -- suspicious of fusion / segmentation artifact
LOW_SIGNAL_THRESHOLD = qc_df["total_counts"].quantile(0.25)
suspicious = (qc_df["bin_count"] > BIN_COUNT_P99) & (qc_df["total_counts"] < LOW_SIGNAL_THRESHOLD)
print(f"Large cells (bin_count>p99) with low signal (total_counts<p25): {suspicious.sum()} of {len(qc_df)}")

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

    sub = qc_df[suspicious]
    ax.scatter(sub["bin_count"], sub[metric], s=18, color="black", marker="x",
               linewidths=1, label="large + low signal", zorder=5)

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
print(f"Large + low-signal cells: {suspicious.sum()}")

high_mt_small = (qc_df["pct_counts_mt"] > 90) & (qc_df["bin_count"] < BIN_COUNT_P99)
print(f"Small cells with >90% mitochondrial: {high_mt_small.sum()} of {len(qc_df)} ({high_mt_small.sum()/len(qc_df)*100:.1f}%)")

# %% [markdown]
# ## 4. Violin plots — raw QC metrics, samples side by side
# `MAX_PCT_MT` here is only a visual reference line — it is not applied
# as a filter (decision: no mitochondrial cap at this stage, see 6d).

# %%
MAX_PCT_MT = 40  # visual reference only

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
        ax.axhline(MAX_PCT_MT, color="#8B0000", linestyle="--", linewidth=1.2, label=f"ref={MAX_PCT_MT}% (not applied)")
        ax.legend(fontsize=7)
    elif metric == "bin_count":
        ax.axhline(3, color="#C97B2E", linestyle="--", linewidth=1.2, label="min=3")
        ax.axhline(BIN_COUNT_P99, color="#8B0000", linestyle="--", linewidth=1.2, label="p99")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Spatial plots — raw TE burden and mitochondrial % side by side

# %%
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], np.log1p(a.obs["TE_burden"].values), s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="log1p(TE burden)")
plt.suptitle("TE burden (raw, log1p) — by cell", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], a.obs["pct_counts_mt"].values, s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% mt counts")
plt.suptitle("Mitochondrial % per cell", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6a. Distributions and where the floor falls, per sample

# %%
fig, axes = plt.subplots(2, N_COLS, figsize=(6 * N_COLS, 8))
for col, s in enumerate(samples):
    a = cell_adatas[s]
    for row, (metric, label) in enumerate([
        ("total_counts", "Total counts"),
        ("n_genes_by_counts", "Genes detected"),
    ]):
        ax = axes[row, col]
        ax.hist(a.obs[metric], bins=60, color="#028090", alpha=0.75)
        if metric == "total_counts":
            ax.axvline(MIN_COUNTS_FLOOR, color="#C97B2E", linestyle="--", linewidth=1.2, label="floor")
        if row == 0:
            ax.set_title(s, fontsize=10)
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6b. Comparing mitochondrial thresholds — 25% vs 40%
# Exploration only -- no filter applied here. Team decision: no
# mitochondrial filter at this stage (revisited after clustering) — kept
# as a reference in case this is picked back up later.

# %%
MT_THRESHOLDS = [25, 40]

comparison_rows = []
for s in samples:
    a = cell_adatas[s]
    for mt_thresh in MT_THRESHOLDS:
        keep = (a.obs["total_counts"] >= MIN_COUNTS_FLOOR) & (a.obs["pct_counts_mt"] <= mt_thresh)
        lost = ~keep
        comparison_rows.append({
            "sample": s,
            "mt_threshold": mt_thresh,
            "cells_kept": keep.sum(),
            "cells_lost": lost.sum(),
            "%_lost": round(lost.sum() / len(keep) * 100, 1),
            "median_total_counts_lost": a.obs.loc[lost, "total_counts"].median() if lost.sum() > 0 else None,
            "median_total_counts_kept": a.obs.loc[keep, "total_counts"].median(),
        })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## 6c. Spatial — which cells would be lost at each threshold (exploratory)

# %%
fig, axes = plt.subplots(len(samples), 2, figsize=(12, 5 * len(samples)))
for row, s in enumerate(samples):
    a = cell_adatas[s]
    coords = a.obsm["spatial"]
    for col, mt_thresh in enumerate(MT_THRESHOLDS):
        keep = (a.obs["total_counts"] >= MIN_COUNTS_FLOOR) & (a.obs["pct_counts_mt"] <= mt_thresh)
        ax = axes[row, col]
        ax.scatter(coords[keep, 0], coords[keep, 1], c="lightgray", s=6, alpha=0.5)
        ax.scatter(coords[~keep, 0], coords[~keep, 1], c="#8B0000", s=8, alpha=0.85)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{s} — mt<={mt_thresh}% ({(~keep).sum()} lost)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6d. Apply the chosen filter: counts floor only
# No mitochondrial cap (team decision, see 6b). The bin_count (>=3) filter
# was already applied in 1e.

# %%
filtered_cell_adatas = {}
for s, a in cell_adatas.items():
    keep = a.obs["total_counts"] >= MIN_COUNTS_FLOOR
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
# ## 10. Spatial plots — which cells survive the counts filter, per sample

# %%
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = cell_adatas[s]  # object before filtering, all cells
    keep_mask = a.obs["total_counts"] >= MIN_COUNTS_FLOOR
    coords = a.obsm["spatial"]

    ax.scatter(coords[keep_mask, 0], coords[keep_mask, 1], c="lightgray", s=8, alpha=0.5, label="retained")
    ax.scatter(coords[~keep_mask, 0], coords[~keep_mask, 1], c="#8B0000", s=10, alpha=0.85, label="discarded")

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    n_discarded = (~keep_mask).sum()
    n_total = len(keep_mask)
    ax.set_title(f"{s} ({n_discarded}/{n_total} discarded, {n_discarded/n_total*100:.1f}%)")

axes[0].legend(markerscale=2, fontsize=8, loc="upper right")
plt.suptitle("Cells retained vs. discarded (counts floor only)", y=1.02)
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
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()
for ax, s in zip(axes, samples):
    a = normalized_cell_adatas[s]
    sca = plot_spatial_te(ax, a.obsm["spatial"], a.obs["TE_fraction"].values * 100, s)
    plt.colorbar(sca, ax=ax, shrink=0.6, label="% UMIs from TE")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Merge all samples (outer join) + single gene filter
# Last step before integration/clustering (Harmony + Leiden, still pending).

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
    labels_col = bin_adatas[s].obs[EXPANSION_KEY]  # pure StarDist, bins with a nucleus (before combining with Cellpose)
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