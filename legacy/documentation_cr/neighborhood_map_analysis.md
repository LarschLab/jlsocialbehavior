# Neighborhood Map Analysis

This note documents an archived neighborhood-map workflow and how to use `legacy/Analyses/old/ShoalingNeighborhoodMaps_2026.ipynb`.

This notebook is inactive. Use the repository root for shared imports and consult `legacy/README.md` before running it.

## What A Neighborhood Map Is

A neighborhood map is a two-dimensional summary of where the partner animal was relative to a focal animal. Each frame is transformed into the focal animal's local coordinate system, so the map is not simply arena position. Instead, it asks where the neighbor tended to appear around the focal animal.

The current writer is `models/experiment.py`. When `SaveNeighborhoodMaps` is true, `experiment.saveExpData()` writes one `*MapData.npy` file per trajectory file into `ProcessingDir`.

The map bins are defined in `models/experiment.py` as `np.linspace(-30, 30, 63)`, producing 62 by 62 maps. Values in the neighbor-density map are normalized by map area, so values near 1 are approximately chance-level occupancy, values above 1 are enriched occupancy, and values below 1 are depleted occupancy.

## Required Inputs

The neighborhood-map notebook follows the same metadata conventions as `ShoalingSelectionAnalysis_2025_slim_clean.ipynb`.

Required metadata:

- `MetaData_CR.xlsx`
- `AllExp` sheet with experiment paths, folders, animal numbers, date labels, and setup labels
- `AllAn` sheet with animal number, line, genotype, birth date, and experiment date

Required experiment-folder files:

- `PositionTxt*` trajectory file
- `PL*` pair-list file
- AVI and ROI/scale files when animal size must be recomputed

Required processed outputs:

- `*_siSummary_epi<duration>.csv`
- `*MapData.npy`

The slim selection notebook normally sets `SaveNeighborhoodMaps = 0`, so it usually creates summary CSV files but not map files. The neighborhood-map notebook therefore uses its own processing cache and does not read the slim notebook's summary CSVs.

## `MapData.npy` Contents

Each `*MapData.npy` file is a 5D array with this conceptual layout:

```text
episode-or-animal-row x map type x condition x map x map
```

The map-type axis is:

```text
0 = neighbor density
1 = speed force map
2 = turn force map
```

The condition axis is:

```text
0 = real data
1 = shifted-control data
```

The new notebook focuses on map type `0`, the neighbor-density map. It also loads the shifted-control version, so plots can compare real occupancy to the time-shifted baseline.

## Relation To Vector-Field Analysis

The larger `legacy/functions/vector_field_analysis.py` workflow combines several related analyses:

- neighborhood maps
- bout-triggered vector fields
- bout-probability maps
- thigmotaxis exclusion
- attraction plots

The vector-field part is bout-triggered. It detects swim bouts from smoothed speed, finds a pre-bout and post-bout position, and converts that movement into a displacement vector. Those bout vectors are binned by the focal animal's starting position relative to the neighbor. The resulting vector field asks: given where the neighbor was, what bout displacement did the focal animal make?

Neighborhood maps are simpler. They ask only where the neighbor was relative to the focal animal, independent of bout-triggered displacement.

Important intermediate outputs from the older vector-field workflow include:

- `df_<expset_name>.p`
- `exp_set_<expset_name>.p`
- `all_bouts_all_bout_idx_<expset_name><tag>.p`
- `all_bout_xys_<expset_name><tag>.p`
- `bout_df_<expset_name><tag>.p`
- `mapdict_<expset_name><tag>.p`
- `histograms_<expset_name><tag>.p`

Important figure outputs include:

- `<groupset>_plot0_<tag>.png`: neighbor density, bout probability, and bout vector field
- `<groupset>_plot1_<tag>.png`: optional difference plots
- `thighmohist.png`: thigmotaxis exclusion diagnostic
- `attraction_animalsbygroup_<expset_name>.png`: attraction/social-index summary

The new notebook does not use this whole legacy vector-field class. It uses the same map files but gives a focused, easier-to-edit workflow for neighborhood-density maps.

## How To Use The Notebook

Open:

```text
legacy/Analyses/old/ShoalingNeighborhoodMaps_2026.ipynb
```

Start in the `Settings` cell.

Set the repository and NAS paths if needed:

```python
CODE_DIR
META_FOLDER
META_FILE
PROCESSING_ROOT_DIR
PROCESSING_DIR
OUTPUT_DIR
```

`PROCESSING_DIR` is a notebook-specific folder under `PROCESSING_ROOT_DIR`, currently `neighborhood_maps`. It stores the summary CSVs and `MapData.npy` files generated for this workflow.

Choose the data subset:

```python
INCLUDE_YEARS = [2026]
INCLUDE_LINES = None
INCLUDE_GENOTYPES = None
INCLUDE_LINESETS = None
INCLUDE_EXPERIMENT_FOLDERS = None
EXCLUDE_EXPERIMENT_FOLDERS = []
```

Use `None` to keep all values for a filter. For example, to use every available year:

```python
INCLUDE_YEARS = None
```

To focus on one genotype:

```python
INCLUDE_GENOTYPES = ['Hi']
```

The notebook writes a map-specific processing settings file:

```text
processingSettings_neighborhood_maps.csv
```

This file is written inside the notebook-specific processing cache. This is deliberate: it avoids overwriting the main slim notebook's `processingSettings.csv` and keeps neighborhood-map summaries separate from slim-notebook summaries.

## Checking Existing Processed Data

Run the discovery cells before loading maps. They write:

```text
neighborhood_map_file_index.csv
```

This table tells you, per selected experiment:

- whether the summary CSV exists
- whether the `MapData.npy` file exists
- whether the experiment is ready to load
- whether the map or summary is missing

If maps are missing, you can either stop and regenerate them separately or use the notebook's controlled processing rerun.

## Regenerating Maps

Map generation can be slow because it reads trajectory data and computes map arrays. The notebook therefore defaults to:

```python
RUN_PROCESSING = False
```

To regenerate maps for the selected subset:

```python
RUN_PROCESSING = True
PROCESSING_MISSING_ONLY = True
```

Use a small experiment subset first. With `PROCESSING_MISSING_ONLY = True`, the core pipeline only regenerates selected experiments whose notebook-specific summary CSVs are missing. Because this notebook sets `SaveNeighborhoodMaps = 1`, new summary CSVs and `MapData.npy` files are written together into the notebook cache.

## Notebook Outputs

The notebook writes outputs under:

```text
OUTPUT_DIR
```

Main CSV outputs:

- `neighborhood_map_file_index.csv`: one row per selected experiment and discovered processed file
- `neighborhood_map_row_index.csv`: one row per loaded map row after filtering
- `neighborhood_map_group_summary.csv`: group-level counts and scalar map summaries
- `neighborhood_map_group_summary_by_<grouping>.csv`: optional alternative groupings

Main figure outputs:

- `neighborhood_density_by_genotype_episode.pdf`
- `neighborhood_density_by_line_episode.pdf`
- `neighborhood_density_by_lineSet_episode.pdf`
- `neighborhood_density_by_year_genotype_episode.pdf`

Each figure shows:

- real neighbor-density map
- shifted-control neighbor-density map
- real minus shifted-control map

The default grouping is:

```python
GROUP_COLUMNS = ['genotype', 'episode']
```

Change `GROUP_COLUMNS` in the settings cell if you want a different primary grouping.

## Interpreting The Plots

The axes are neighbor position relative to the focal animal in millimeters. The origin is the focal animal. Because maps are heading-aligned, positions should be interpreted as local positions around the focal animal rather than absolute arena positions.

The real map shows observed relative neighbor occupancy. The shifted-control map shows the same kind of map after applying the pipeline's time-shifted control. The difference map highlights occupancy structure that is stronger or weaker than the shifted baseline.

## Adding F2 Data Later

The notebook is designed so F2 data can be added without rewriting the analysis. The expected route is:

1. Add or load the F2 metadata.
2. Make sure it can produce the same columns used by the notebook: experiment folder, animal numbers, line or genotype labels, and experiment dates.
3. Point `PROCESSING_DIR` to the F2 processed outputs or concatenate a second file index.
4. Add a dataset label such as `DATASET_LABEL = 'F2'`.
5. Include `dataset` in `GROUP_COLUMNS` when comparing selection and F2 data.

The important contract is that every dataset should eventually provide summary CSVs and `MapData.npy` files with the same schema.

## Known Caveats

- The active slim selection notebook does not normally write `MapData.npy`.
- Generating maps may require rerunning selected raw trajectory processing with `SaveNeighborhoodMaps = 1` in the notebook-specific processing cache.
- `PROCESSING_MISSING_ONLY=True` checks for notebook-specific summary CSVs; it does not inspect or reuse slim-notebook summaries.
- The legacy vector-field code contains historical y-flip comments around map orientation. Interpret left/right and front/back with care until validated against a known control plot.
- The notebook assumes summary rows align with `MapData.npy` rows. It warns if row counts differ and uses the shared prefix of both arrays.
