# Visium HD — Transposable Element Analysis

Analysis pipeline for detecting and quantifying transposable element (TE)
expression in Visium HD spatial transcriptomics data (mouse skeletal muscle
injury model), using SoloTE merged with Space Ranger gene expression.

## Research question

Does skeletal muscle injury reactivate transposable elements, which TE
types are involved, where in the tissue does this happen, and does it
resolve as the tissue recovers? (CTX injury model, Tibialis Anterior,
timepoints: control, 1h, 3h, 12h, 24h post-injury.)

## Two parallel pipelines

Two ways of defining the spatial unit of analysis are maintained side by
side, to compare results:

| | Bin-level (8µm) | Cell-level (Cellpose) |
|---|---|---|
| Unit | Fixed 8x8µm square | Real segmented cell |
| Units per sample | ~70,000–150,000 | ~2,900–5,700 |
| Build script | `scripts/build_spatial_object.py` | `scripts/build_cell_object.py` |
| QC notebook | `notebooks/young_preprocessing.py` | `notebooks/qc_by_cell.py` |
| Pros | Fast, simple, no segmentation dependency | Real biological unit, much more signal per unit |
| Cons | Arbitrary square may mix cell types | Fewer spatial points, more complex pipeline |

### Bin-level pipeline

```
Space Ranger count -> possorted_genome_bam.bam
                              |
                              v
                          SoloTE -> TE matrices (class/family/subfamily/locus),
                                    native 2um barcodes
                              |
Space Ranger binned_outputs/  |
  square_008um (genes) -------+
                              v
                build_spatial_object.py
        (aggregates TE 2um->8um bins, merges with genes)
                              |
                              v
                young_preprocessing.py (QC, normalization, merge)
```

### Cell-level pipeline

```
SoloTE (native 2um TE matrix)          Space Ranger segmented_outputs/
       |                                (Cellpose cell segmentation)
       |  aggregate via                        |
       |  barcode_mappings.parquet              |  filtered_feature_cell_matrix
       |  (square_002um -> cell_id)             |  (genes, per cell)
       v                                        v
              build_cell_object.py
   (TE aggregated to cell_id, merged with genes,
    cell centroid from cell_segmentations.geojson)
                              |
                              v
                  qc_by_cell.py (QC, normalization, merge)
```

Both pipelines currently use **family-level** TE (39 categories, e.g. L1,
Alu, ERVK) as the working default -- a middle ground in SoloTE's hierarchy
(Class -> Family -> Subfamily -> Locus). Class-level objects also exist
(`*_class_8um.h5ad`); subfamily-level was tested for L1 specifically but
found too sparse per unit to trust directly (see notes).

## Samples

| Sample | Condition | Notes |
|---|---|---|
| Control-GER | Control | |
| Control-Old | Control | |
| Injured-1hrs | Injured, 1h | lower sequencing efficiency (depth/saturation, not tissue) |
| Injured-3hrs | Injured, 3h | |
| Injured-12hrs | Injured, 12h | |
| Injured-24hrs | Injured, 24h | |
| ST0001 | pending confirmation | pilot sample, different processing route (Core Labs) |
| ST0002 | pending confirmation | pilot sample, different processing route (Core Labs) |

## Repo structure

```
scripts/
  run_solote_laura.sh          SoloTE SLURM submission (all 6 samples)
  build_spatial_object.py      Bin-level (8um) genes+TE object builder
  run_build_spatial_object.sh  SLURM wrapper for the above
  build_cell_object.py         Cell-level (Cellpose) genes+TE object builder
  run_build_cell_object.sh     SLURM wrapper for the above

notebooks/
  young_preprocessing.py       Bin-level QC pipeline (jupytext-paired .py)
  qc_by_cell.py                Cell-level QC pipeline (jupytext-paired .py)

docs/
  SoloTE_notas_referencia.md   Pipeline background, UMI/TE concepts, QC findings

slides/
  Various explanatory decks (SoloTE mechanics, stats summaries, timeline)
```

## Notebooks (jupytext)

Notebooks are paired with a `.py:percent` mirror via jupytext.
Only the `.py` file is versioned in git (`.ipynb` and exported `.html` are
gitignored).

To regenerate the `.ipynb` after cloning:
```bash
jupytext --sync notebooks/<name>.py
```

Before committing changes made in the `.ipynb`:
```bash
jupytext --sync notebooks/<name>.ipynb
```

## QC pipeline (both notebooks follow this structure)

1. Load samples
2. QC metrics: total_counts, n_genes, pct_counts_mt, TE_burden
   (zero-count units dropped before mt%, to avoid a NaN-propagation bug)
3. Combine across samples for comparison
4. Violin plots (4 panels)
5. Spatial plots (TE burden, mitochondrial %)
6. MAD-based outlier filtering (per sample, 5 MADs from median) + threshold
   visualization + spatial check of discarded units
7. Normalization (normalize_total target_sum=1e4 + log1p, per sample)
8-10. TE burden, normalized (table, violin, spatial)
11. Merge (outer join across samples) + single gene/TE filter
12-13. TE_fraction (raw TE UMIs / raw total UMIs) -- direct, depth-independent
   metric requested by PI, plus spatial plots

## Key findings so far (bin-level, family)

- Raw TE burden shows a "V" pattern across timepoints: high in controls,
  drops in early injury (1-12h), partially recovers by 24h.
- The gap narrows after normalization (~3.75x -> ~2.6x) but doesn't
  disappear -- some of it is sequencing depth, not all.
- Comparing against a published bulk qPCR reference (same injury model):
  our family-level TE categories (ERVK, ERVL, B2, L1) all move in lockstep
  with total counts/genes, unlike the bulk data (where e.g. L1_T stays flat
  while L1_A/L1_G rise) -- suggests the pattern may partly reflect shifting
  cell-type composition after injury (fewer myofibers, more infiltrating
  immune cells) and/or depth confounds, rather than confirmed TE-specific
  regulation. This is the main open question the cell-level pipeline is
  meant to help resolve.
- Subfamily-level L1 (L1_A/T/G) is too sparse per bin/cell to trust directly
  (mean UMIs per unit well below 1) -- would need pseudo-bulk aggregation per
  sample to compare fairly against qPCR.

## Data location (not versioned here)

All raw and intermediate data lives on IBEX, not in this repo:
- Space Ranger outputs: `/ibex/project/c2344/20260402_LL00134_0018_A23J2Y7LT4/`
- SoloTE outputs: `/ibex/project/c2344/Spatial/STE_results/`
- Working copies for analysis: `/ibex/user/medinils/data/`
- Built objects (both pipelines): `/ibex/user/medinils/data/objects/`

## Key references

- SoloTE: https://github.com/bvaldebenitom/SoloTE
- Reference workshop notebook (Ana): `VisiumHD_workshop.ipynb` -- followed
  for QC/normalization conventions (`normalize_total` + `log1p`,
  `sc.pp.filter_genes`/`filter_cells`)
- See `docs/SoloTE_notas_referencia.md` for detailed notes on the pipeline,
  UMI/read concepts, and earlier QC findings.
