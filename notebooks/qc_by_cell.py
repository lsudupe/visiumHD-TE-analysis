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
# # Visium HD TE Analysis — QC by Cell (Cellpose segmentation)
# Using Space Ranger's native cell segmentation (segmented_outputs/) instead
# of raw 8um bins.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# %% [markdown]
# ## 1. Cargar el mapeo bin(2um) -> célula, y la matriz de TE nativa (2um)

# %%
SAMPLE = "Control-GER"

mapping = pd.read_parquet(f"/ibex/user/medinils/data/samples/{SAMPLE}/barcode_mappings.parquet")
mapping = mapping[["square_002um", "cell_id"]].dropna(subset=["cell_id"])
print("bins con celula asignada:", mapping.shape)

# %%
import sys
sys.path.insert(0, "/ibex/user/medinils/visiumHD-TE-analysis/scripts")
from build_spatial_object import load_te_adata_2um  # ya escrita, reutilizamos

te_2um = load_te_adata_2um(SAMPLE, "family")
print("TE (2um, crudo):", te_2um.shape)

# %% [markdown]
# ## 2. Agregar los bins de TE por célula (sumando todos los bins de la misma cell_id)

# %%
te_barcodes = pd.Series(te_2um.obs_names, name="square_002um")
te_map = te_barcodes.to_frame().merge(mapping, on="square_002um", how="inner")
print("bins de TE con celula asignada:", te_map.shape)

# reindexar te_2um a solo los bins que sí tienen célula, en el mismo orden que te_map
te_2um_sub = te_2um[te_map["square_002um"].values].copy()

# matriz de agrupación: (n_celulas x n_bins) para sumar por cell_id
cell_ids = te_map["cell_id"].values
unique_cells = pd.unique(cell_ids)
cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
row_idx = [cell_to_idx[c] for c in cell_ids]

n_cells = len(unique_cells)
n_bins = te_2um_sub.n_obs
grouping = sp.coo_matrix(
    (np.ones(n_bins, dtype="float32"), (row_idx, range(n_bins))),
    shape=(n_cells, n_bins),
).tocsr()

te_cell_X = (grouping @ te_2um_sub.X).tocsr()
te_cell = ad.AnnData(X=te_cell_X, obs=pd.DataFrame(index=unique_cells), var=te_2um_sub.var.copy())
print("TE a nivel celula:", te_cell.shape)

# %% [markdown]
# ## 3. Cargar genes a nivel célula (segmented_outputs, Cellpose nativo)

# %%
cell_matrix_dir = Path(f"/ibex/project/c2344/20260402_LL00134_0018_A23J2Y7LT4/{SAMPLE}/outs/segmented_outputs/filtered_feature_cell_matrix")
genes_cell = sc.read_10x_mtx(cell_matrix_dir, var_names="gene_symbols", cache=True)
genes_cell.var_names_make_unique()
print("Genes a nivel celula:", genes_cell.shape)
print(genes_cell.obs_names[:5])

# %% [markdown]
# ## 3b. Verificar que los IDs de célula coinciden entre genes y el mapeo

# %%
print("genes_cell.obs_names (formato Space Ranger):")
print(genes_cell.obs_names[:5].tolist())

print("\nmapping['cell_id'] (formato barcode_mappings.parquet):")
print(mapping["cell_id"].unique()[:5])

# %% [markdown]
# ## 4. Fusionar genes + TE por cell_id (intersect + subset, mismo patrón de siempre)

# %%
common_cells = genes_cell.obs_names.intersection(te_cell.obs_names)
print(f"{len(common_cells)} celulas compartidas de {genes_cell.n_obs} (genes) / {te_cell.n_obs} (TE)")

genes_sub = genes_cell[common_cells].copy()
te_sub = te_cell[common_cells].copy()

merged_X = sp.hstack([genes_sub.X, te_sub.X], format="csr")
merged_var = pd.concat([genes_sub.var, te_sub.var])
adata_cell = ad.AnnData(X=merged_X, obs=genes_sub.obs.copy(), var=merged_var)
print("Objeto final (por celula):", adata_cell.shape)

# %% [markdown]
# ## 5. Añadir coordenadas espaciales (centroides de célula)

# %%
spatial_dir = Path(f"/ibex/project/c2344/20260402_LL00134_0018_A23J2Y7LT4/{SAMPLE}/outs/segmented_outputs/spatial")
pos = pd.read_parquet(spatial_dir / "tissue_positions.parquet")
print(pos.columns.tolist())
print(pos.head(3))
