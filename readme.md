# jlsocialbehavior README

This guide documents the repository as it is organized now, with special focus on the active notebooks in `Analyses/`. The active notebooks are the current user-facing analysis layer. Other folders contain reusable runtime code, video preprocessing utilities, historical notebooks, exploratory code, or an embedded environment; those areas are marked separately below.

## Repository Map

Primary active workflow:

- `Analyses/`: seven active notebooks for selection, selection history, QTL mapping and quality control, loom responses, and dish use.
- `models/`: core behavioral analysis objects. These are used when the notebooks process raw tracking files into per-animal/per-episode summary CSVs.
- `functions/`: notebook helpers, plotting defaults, matrix utilities, video/ROI/animal-size helpers, and other reusable functions.

Secondary or legacy areas:

- `legacy/Analyses/old/` and `legacy/Analyses/others/`: the user-designated inactive notebook collections.
- `legacy/`: inactive workflows and their exclusive helpers; see [archive instructions](legacy/README.md) and [relocation manifest](legacy/move-manifest.csv).
- `legacy/DeprecatedAnalyses/`: historical notebooks, including the 2018 paper reproduction notebooks and older example analyses. These are not part of the active `Analyses/` workflow unless you open them deliberately.
- `scripts/`: repository maintenance tooling (`sync_coding_doc.py`). Archived operational helpers are in `legacy/scripts/`.
- `legacy/playground/` and `legacy/obsolete/`: exploratory or retired code. Do not treat these as authoritative for current analyses.
- Archived GUI/video scripts such as `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`, `legacy/root-scripts/vidGui.py`, `legacy/root-scripts/VidGui2ArenaShoal.py`, `legacy/root-scripts/rotationWrapper.py`, and `legacy/root-scripts/processingVideo.py`: useful for preprocessing or inspecting videos, but not part of the current `Analyses/` notebook execution path.
- `conda-meta/`, `DLLs/`, `Lib/`, `Library/`, `libs/`, `include/`, `Tools/`, `Menu/`, `share/`, and `etc/`: embedded environment files, not repository source code.

## Environment And Run Assumptions

The active notebooks assume a Python/Jupyter environment with common scientific packages:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scipy`, `statsmodels`
- `PythonMeta` for meta-analysis helpers
- OpenCV/video dependencies if animal size is recomputed from AVI files

The repository includes `environment.yml` that can be used to regenerate the conda environment used for the repository. 

Selection and QTL notebooks configure NAS and local code paths near the top. Dish-use and loom notebooks also support `JLSOCIALBEHAVIOR_CODE_DIR`; QTL QC discovers the repository through parent directories. Before running on another computer, check the applicable settings:

- `metaFolder`: folder containing the metadata workbook.
- `metaFile`: Excel workbook name.
- `codeDir`: local path to this repository. The notebooks call `os.chdir(codeDir)`.
- `ProcessingDir`: folder where generated processing settings and `*_siSummary_epi*.csv` files are written.
- `outputDir`: folder where notebook figures and exported tables are written.
- `nasPath`: only used by the selection-history notebook, where `AllExp.path` is relative to the NAS root.

The processing notebooks use `MissingOnly=True` through `models.experiment_set.experiment_set`, so existing `*_siSummary*.csv` files in `ProcessingDir` are reused. To force a complete recomputation, remove or move those generated summary files, or change the notebook processing call deliberately.

## Active Notebooks In `Analyses/`

The active entrypoints remain at their existing paths:

| Notebook | Purpose |
| --- | --- |
| [Selection](Analyses/Selection/ShoalingSelectionAnalysis_2025_slim_clean.ipynb) | Current selection analysis and `dfEpiAn_selection.csv` export |
| [Selection history](Analyses/Selection/ShoalingSelectionAnalysis_2025History.ipynb) | Historical selection cohorts and progression |
| [Loom responses (`Selection/`)](Analyses/Selection/LoomAnalysisCR.ipynb) | Metadata-selected loom responses; currently configured for QTL metadata |
| [QTL mapping](Analyses/QTLmapping/ShoalingQTLMappingAnalysis_2026.ipynb) | QTL/F2 shoaling analysis and selection comparison |
| [QTL quality control](Analyses/QTLmapping/QTLQualityControl.ipynb) | Animal/experiment QC, dish evenness, and metric/family plots |
| [QTL loom responses](Analyses/QTLmapping/LoomAnalysis_QTL.ipynb) | Metadata-selected QTL recordings and loom responses |
| [Dish use](Analyses/Dish-useAnalysis.ipynb) | Trajectory occupancy, evenness, and distribution figures |

### `ShoalingSelectionAnalysis_2025_slim_clean.ipynb`

Purpose:

- Main current selection analysis for ongoing shoaling-selection experiments.
- Converts experiment metadata into `processingSettings.csv`.
- Runs the core social-behavior pipeline when summary CSVs are missing.
- Aggregates per-frame/per-episode outputs into animal-level episode summaries.
- Plots shoaling index, attraction to linear versus bout stimuli, setup effects, escapee fish, fish size, speed, thigmotaxis, and selection-candidate rankings.
- Writes `dfEpiAn_selection.csv`, which is later consumed by the QTL notebook.

Important configured inputs:

- Metadata workbook: `MetaData_CR.xlsx` in `metaFolder`.
- Required workbook sheets: `AllExp` and `AllAn`.
- Experiment folders are taken from `AllExp.path` plus `AllExp.folder`.
- Each experiment folder must contain a `PositionTxt*` trajectory file and a `PL*` pair-list file.
- Because `aviPath` is set to `default` and `recomputeAnimalSize` is set to `1`, each experiment folder should also contain an AVI file and an ROI/scale file if animal-size recomputation is needed.

Important generated outputs:

- `ProcessingDir/processingSettings.csv`
- `ProcessingDir/<PositionTxt basename>_siSummary_epi5.0.csv`
- `<trajectory folder>/<PositionTxt basename>_anSize.csv` when animal size is recomputed
- `outputDir/dfEpiAn_selection.csv`
- Multiple PDF figures in `outputDir`, including selection, size, speed, thigmotaxis, and final selected-fish overview plots.

Key analysis choices:

- Uses 5 minute episodes.
- Processes 24 episodes.
- Uses `readLim = 24 * 5 * 60 * 30 + 11` frames.
- Uses `arenaDiameter_mm = 70`.
- Uses `minShift = 60` seconds for shifted-control inter-animal distance.
- Uses `SaveNeighborhoodMaps = 0`, so `*MapData.npy` files are not normally generated by this notebook.
- Uses `ComputeBouts = 1`.
- Uses a downstream analysis window of `80 < inDishTime < 240`.
- Assumes 35 animal slots per experiment during notebook-level remapping:
  `tmp.animalIndex = tmp.animalIndex + (i * 35)`, then maps those indices through `anIDsAll`.

Main intermediate tables:

- `info`: experiment metadata loaded from `AllExp`, then augmented with processing columns.
- `infoAn`: animal metadata loaded from `AllAn`.
- `df`: concatenated per-episode summary rows from all `*_siSummary*.csv` files, merged with line/genotype/date/setup metadata.
- `dfDR`: `df` filtered to the main in-dish time window.
- `dfEpiAn`: one row per `episode`, `animalIndex`, `line`, `setup`, `genotype`, `date`, and `lineSet`, with numeric values averaged.
- `dfPlot`: time-course summary table grouped by `inDishTime`, `episode`, `genotype`, and `lineSet`.

### `ShoalingSelectionAnalysis_2025History.ipynb`

Purpose:

- Selection-history analysis across older selection experiments.
- Focuses on progression across backgrounds, cohorts, generations, selected animals, and RNA-seq subsets.
- Builds historical `dfEpiAn` summaries and provides a reusable `plot_selection_history(...)` plotting helper.

Important configured inputs:

- Metadata workbook: `MetaData_JL_2019_consolidated_v3.xlsx`.
- `metaFolder` points to a metadata folder under the NAS.
- `nasPath` is prepended to `AllExp.path + AllExp.folder`.
- `AllExp` is filtered to rows where `stimulusProtocol == 'selection'`.
- Each experiment folder must contain `PositionTxt*` and `PL*`.

Important generated outputs:

- `ProcessingDir/processingSettings.csv`
- `ProcessingDir/<PositionTxt basename>_siSummary_epi5.0.csv`
- Notebook figures for historical selection trajectories.

Unlike the current selection notebook, this notebook does not write `dfEpiAn_selection.csv` in its active cells.

Key analysis choices:

- Uses the same 5 minute, 24 episode, 70 mm arena, 60 second shift, and `readLim` pattern as the current selection notebook.
- Sets `recomputeAnimalSize = 0`, so animal size is not recomputed during the standard history run.
- Groups `dfEpiAn` by `episode`, `animalIndex`, `line`, `setup`, `genotype`, `date`, `lineSet`, `bg`, `cohort`, `generation`, `repeat`, `set`, and `rnaSet`.
- Uses `80 < inDishTime < 240` for the episode-level animal summaries.

### `ShoalingQTLMappingAnalysis_2026.ipynb`

Purpose:

- QTL/F2 shoaling analysis adapted from the current selection notebook.
- Processes QTL experiment metadata and raw tracking outputs.
- Builds episode-level animal summaries for F2/QTL experiments.
- Compares F2 metrics against selected-line metrics from `dfEpiAn_selection.csv`.
- Plots shoaling index, delta SI, size, speed, thigmotaxis, escapees, and Sel1/Sel3 comparisons.

Important configured inputs:

- Metadata workbook: `MetaData_QTL.xlsx` in the QTL `metaFolder`.
- Required workbook sheets: `AllExp` and `AllAn`.
- Experiment folders are taken from `AllExp.path` plus `AllExp.folder`.
- Each experiment folder must contain `PositionTxt*` and `PL*`.
- Animal-size recomputation is enabled, so AVI and ROI/scale files are expected when no cached size file is available.
- Comparison input: `ShoalSelec_temp_output/dfEpiAn_selection.csv`, produced by `ShoalingSelectionAnalysis_2025_slim_clean.ipynb`.

Important generated outputs:

- `ProcessingDir/processingSettings.csv`
- `ProcessingDir/<PositionTxt basename>_siSummary_epi5.0.csv`
- QTL and comparison PDF figures in `outputDir`.
- `shoalindelta_hivslo.pdf` and `shoalindelta_sel1_sel3.pdf` for delta SI, including separate Sel1/Sel3 panels.

Key analysis choices:

- Uses 5 minute episodes, 24 episodes, `arenaDiameter_mm = 70`, `minShift = 60`, `ComputeBouts = 1`, and `SaveNeighborhoodMaps = 0`.
- Sets `QTL_INCLUDE_FROM_DATE = '2026-06-01'`: after loading saved summaries, retains experiments dated on or after June 1, 2026 for downstream analysis and plotting. Set this to `None` to include all dates. This filter does not limit raw experiment processing or remove cached summaries.
- Removes animals with missing genotype or genotype equal to `na` before downstream merges and plotting.
- Builds `dfEpiAn` with the same grouping pattern as the current selection notebook.
- Imports selected-line `dfEpiAn_selection.csv`, labels the QTL data as `dataset = "F2"` and the selection data as `dataset = "Selection"`, concatenates them into `comboEpiAn`, and appends dataset labels to `animalIndex` to avoid collisions.
- Calculates comparison-table `DeltaSI` as bout SI (`02k20f`) minus linear SI (`01k01f`), matching episode summaries by `animalIndex`, `dataset`, `genotype`, and `lineSet`. Duplicate fish/episode rows raise an error; missing matched values leave `DeltaSI` missing and those fish are omitted from the DeltaSI panels with a warning.
- Several later analyses filter to `episode == '02k20f'` and use empirical quality thresholds such as speed, bout duration, and thigmotaxis windows.

### `QTLQualityControl.ipynb`

This notebook reads an existing QTL `processingSettings.csv` through `PROCESSING_SETTINGS_CSV`, animal metadata from the `AllAn` sheet of `META_FILE`, and matching saved `*_siSummary*.csv` files. Generate the settings and summaries with the shoaling-QTL notebook before running QC. Behavioral metrics are aggregated from the saved summary blocks; raw trajectories are used for occupancy/evenness.

Current defaults:

- `EXPERIMENT_START_DATE = '2026-06-01'` and `EXPERIMENT_END_DATE = None`: include experiments from June 1, 2026 onward.
- `QC_WINDOW_METHOD = 'continuous'` and `CONTINUOUS_WINDOW_MIN = (60, 120)`: use a continuous 60-120 minute raw-recording window across episode types. Episode mode instead uses `EPISODE_WINDOW_MIN = (80, 240)` on the in-dish clock (minutes), retaining complete `QC_EPISODE = '02k20f'` blocks whose starts fall strictly inside that window.
- `QC_GENOTYPE = 'F2'`, `QC_THRESHOLD_MODE = 'selection_family'`, and `ZMAD_THRESHOLD = 4`: use separate Sel1 and Sel3 median/MAD references, with a pooled reference for other lines. An absolute zMAD score of at least 4 flags an outlier. Set the threshold mode to `'global'` for pooled references across all retained samples.
- `USE_TRAJECTORY_CACHE = True`: reuse the notebook's trajectory caches when applicable.

QC uses `functions.dish_use.analyze_dish_use` for sampled occupancy and normalized Shannon evenness, and retains its own trajectory-cache schema, animal completeness/QC flags, and family/metric displays.

### `Dish-useAnalysis.ipynb`

This notebook reads an existing `processingSettings.csv` through `PROCESSING_SETTINGS_CSV` to select a recording. Its current paths point to QTL processing settings and `MetaData_QTL.xlsx`; it reads `AllAn` for genotype labels. Generate the processing settings with the corresponding selection or QTL notebook first. Dish-use analysis reads raw trajectories and does not require saved `*_siSummary*.csv` files.

The current defaults select experiment index 10, use `TIME_WINDOW_MIN = (60, 120)` minutes, sample every 30th frame, and use `DISH_GRID_SIDE_LENGTH = 40` cells per axis. `USE_TRAJECTORY_CACHE = False` means raw trajectories are loaded on each run. Raw loading requires an `ROIdef*` file; with caching enabled, a readable trajectory cache can supply centered coordinates and dish geometry without an ROI file, with a warning.

The notebook uses `functions.dish_use.analyze_dish_use` and writes occupancy and evenness-distribution figures and CSV tables. Output names and radius-field compatibility are documented in `.agents/references/canonical-outputs.md`.

### `LoomAnalysisCR.ipynb` and `LoomAnalysis_QTL.ipynb`

These notebooks use `functions.joh_loom_helpers` for metadata selection, stimulus trial parsing, velocity and displacement extraction, and per-experiment caches. Both are currently configured to read `MetaData_QTL.xlsx` from the QTL metadata folder, including `LoomAnalysisCR.ipynb` under `Analyses/Selection/`. To analyze selection recordings, change `META_FOLDER`, `META_FILE`, and the processing-cache location together; the directory containing the notebook does not determine its cohort.

Loom notebooks select recordings directly from `AllExp` and `AllAn` and read `PositionTxt*` trajectories with embedded stimulus columns. They do not require `processingSettings.csv`, pair-list files, or shoaling summary CSVs. Review their date and optional line/genotype/folder filters before running: `LoomAnalysisCR.ipynb` currently uses `'exclude_date_range'`, while `LoomAnalysis_QTL.ipynb` uses `'include_from_date'` with `INCLUDE_FROM_DATE = '2026-06-01'`.

Both write seven notebook-specific compressed cache tables with JSON settings sidecars. Cache reuse checks extraction settings and raw-file signatures; `FORCE_REPROCESS_RAW_DATA = True` requests fresh extraction. See `.agents/references/notebook-stage-map.md` for trial alignment and aggregation semantics.

## Common Active Notebook Data Flow

The current selection, selection-history, and shoaling-QTL notebooks follow this shared pattern:

1. Configure folders and metadata workbook names.
2. Load `AllExp` into `info`.
3. Load `AllAn` into `infoAn`.
4. For each experiment row, resolve the raw experiment folder, find `PositionTxt*` and `PL*`, parse animal IDs from `anNr`, and collect birth dates from `infoAn`.
5. Add processing columns to `info`.
6. Write `ProcessingDir/processingSettings.csv`.
7. Call `models.experiment_set.experiment_set(csvFile=csvFile, MissingOnly=True)`.
8. Read generated `*_siSummary*.csv` files from `ProcessingDir`.
9. Merge summaries with animal metadata and experiment metadata.
10. Build `dfEpiAn` by averaging numeric columns within the selected in-dish time window.
11. Plot and optionally save notebook-specific figures/tables.

## Expected Input File Contracts

### Metadata Workbook

The metadata workbook must contain at least two sheets.

`AllExp` describes experiments. Common required columns:

- `path`: base path to an experiment folder. In current selection and QTL notebooks this is used directly. In the history notebook this is relative to `nasPath`.
- `folder`: experiment subfolder appended to `path`.
- `anNr`: animal numbers for the experiment. The notebooks accept either a range like `1:35` or a whitespace-separated list like `1 2 3`.
- `date`: date string used to label `lineSet`.
- `setup`: setup identifier used in plots and summaries.

Additional `AllExp` columns used by the history notebook:

- `stimulusProtocol`: filtered to `selection`.
- `selected`: selection flag used by history plots.
- `set`: historical set label.
- `rnaSet`: RNA-seq subset label.

`AllAn` describes individual animals. Common required columns:

- `anNr`: animal number. This must match values parsed from `AllExp.anNr`.
- `line`: selection/QTL line label.
- `genotype`: genotype/category label.
- `bd`: birth date.
- `expDate`: experiment date. Used in current selection and QTL notebooks.

Additional `AllAn` columns used by the history notebook:

- `bg`: genetic background label.
- `cohort`: cohort label.
- `generation`: generation label.
- `repeat`: repeat label.

Date parsing differs by notebook:

- Current selection and QTL parse `bd` and `expDate` as day-first strings with format `%d-%m-%Y`.
- History parses `bd` with format `%Y%m%d`.

### Experiment Folder

Each experiment folder referenced by the metadata should contain:

- `PositionTxt*`: raw trajectory file.
- `PL*`: pair-list matrix.
- `*.avi`: video file, required when `aviPath = 'default'` and animal size is recomputed.
- One ROI/scale file if pixel scaling or animal size is needed:
  - `ROIdef*`
  - `bgMed_scale*`
  - `*bgMed.csv`

The current selection and QTL notebooks set `recomputeAnimalSize = 1`, so missing AVI/ROI files can block full processing. The history notebook sets `recomputeAnimalSize = 0`.

### `processingSettings.csv`

The notebooks generate this CSV from `info`; users normally do not hand-edit it. It is the input to `models.experiment_set.experiment_set`.

Important columns written by the notebooks:

- `txtPath`: full path to the `PositionTxt*` file.
- `pairList`: full path to the `PL*` file.
- `aviPath`: either `default` or an explicit AVI path.
- `birthDayAll`: space-separated birth dates for animals in the experiment.
- `camHeight`: camera height used for size correction.
- `epiDur`: episode duration in minutes. Active notebooks use `5`.
- `episodes`: number of episodes. Active notebooks use `24`.
- `inDish`: minutes in dish before experiment start. Active notebooks use `10`.
- `arenaDiameter_mm`: dish/arena diameter. Active notebooks use `70`.
- `minShift`: minimum shifted-control offset in seconds. Active notebooks use `60`.
- `episodePLcode`: whether episode labels encode pair-list blocks. Active notebooks use `0`.
- `recomputeAnimalSize`: whether to recompute size from AVI.
- `SaveNeighborhoodMaps`: whether to save `*MapData.npy`.
- `computeLeadership`: whether to compute leadership.
- `ComputeBouts`: whether to compute bout duration.
- `ProcessingDir`: output directory for summary CSVs.
- `outputDir`: notebook output directory.
- `expTime`: experiment start time, currently set to `dummy` in active notebooks.
- `readLim`: maximum number of trajectory rows to read.

### Raw Trajectory Files: `PositionTxt*`

The loader in `models.experiment.experiment.loadData()` supports several formats:

- NPY idtracker format: path ending in `.npy`; expected shape is frames by animals by 2 coordinates. The loader adds a zero heading channel.
- Old Bonsai format: first character of the first parsed value is `(`. The loader extracts selected coordinate columns and inserts zero heading columns.
- idTracker CSV format: first character is `X`. The loader skips the header row and reads whitespace-delimited numeric columns.
- Default VR/Bonsai format: whitespace-delimited file with no header. The last column is used as the episode label.

Downstream time-series code assumes per-animal data are arranged in triples:

- column `animal * 3`: x position
- column `animal * 3 + 1`: y position
- column `animal * 3 + 2`: tracked heading

The time-series layer also reads the second-to-last column as stimulus size and the last column as episode label in the default VR/Bonsai flow. Position values are converted from pixels to mm using ROI-derived `pxPmm`.

### Pair List Files: `PL*`

Pair lists are loaded with `numpy.loadtxt(..., dtype=int)`.

For the active notebooks, `episodePLcode = 0`, so a single matrix applies to all episodes:

- Columns represent focal animal slots.
- Rows represent possible partners or stimulus entries.
- Nonzero entries indicate which partner is paired with each focal animal.
- If a focal column has no nonzero partner, the code falls back to the last row as the stimulus partner.

The active notebooks also assume 35 animal slots at the notebook aggregation stage. If a future dataset has a different number of animals per experiment, update the hard-coded `35` offset in the notebook aggregation cells before trusting merged animal IDs.

### ROI And Scale Files

`models.experiment.ExperimentMeta.PxPmmFromRois()` searches the trajectory folder for ROI/scale files in this order:

1. `ROIdef*`
2. `bgMed_scale*`
3. `*bgMed.csv`

For `ROIdef*`, the file is loaded with flexible delimiter handling, and the last column is treated as the ROI radius. For `bgMed_scale*` and `*bgMed.csv`, the file is loaded as comma-delimited with one header row skipped, and column index `3` is treated as radius. Pixel scaling is computed as:

```text
pxPmm = 2 * mean_radius_px / arenaDiameter_mm
```

### Generated Summary CSVs: `*_siSummary_epi<episodeDur>.csv`

These files are written by `models.experiment.experiment.saveExpData()` into `ProcessingDir`.

For active notebooks, filenames usually look like:

```text
<PositionTxt basename>_siSummary_epi5.0.csv
```

Important columns:

- `animalSet`: experiment index from the processing run.
- `animalIndex`: focal animal index before notebook-level remapping.
- `animalID`: animal ID from metadata when available.
- `CurrentPartner`: partner/stimulus index for this episode.
- `si`: shoaling index.
- `episode`: episode/stimulus label.
- `epStart`: start frame of the episode.
- `inDishTime`: time since animal was placed in dish.
- `epiNr`: episode number.
- `time`: experiment time if available, otherwise in-dish time.
- `birthDay`: birth date if available.
- `stimulusProtocol`: stimulus protocol value.
- `age`: computed age if dates are available.
- `avgSpeed`: mean speed.
- `avgSpeed_smooth`: mean smoothed speed.
- `anSize`: animal size.
- `thigmoIndex`: mean radial position/thigmotaxis metric.
- `boutDur`: median inter-bout interval in seconds when bout computation is enabled.
- `leadershipIndex`: leadership/frontness value when enabled.
- `sync_amp`, `sync_t`: synchronization outputs when enabled.

### Current Selection Export: `dfEpiAn_selection.csv`

Written by `ShoalingSelectionAnalysis_2025_slim_clean.ipynb` to `outputDir`.

It contains the notebook-level `dfEpiAn` table and is required by `ShoalingQTLMappingAnalysis_2026.ipynb` for selected-line versus F2 comparisons.

Core grouping columns:

- `episode`
- `animalIndex`
- `line`
- `setup`
- `genotype`
- `date`
- `lineSet`

Numeric columns are means over the analysis window, usually `80 < inDishTime < 240`.

### Archived F2 Size CSV

Used only by the archived `legacy/Analyses/others/size_data_analysis.ipynb`.

Required columns:

- `Tank`
- `Size`

Other columns may be present but are ignored by the active plotting cell.

## Repository Files Imported By The Active Notebooks

### `functions/matrixUtilities_joh.py`

Directly imported as `mu`.

Used directly by notebooks for:

- `splitall(path)`: extracts the `PositionTxt` basename when finding generated `*_siSummary*.csv` files.

Other helpers:

- `cart2pol`, `pol2cart`: coordinate transforms.
- `runningMean`, `distance`: simple numeric utilities.
- `equalizePath`: resamples a 2D path by distance.
- `smooth`: 1D/2D window smoothing used by time-series calculations.

### `functions/notebookHelper.py`

Directly imported as `nh`, and also imported by `models.experiment`.

Notebook-facing utilities:

- `cohend`, `groupCohen`, `groupPower`: effect size and power helpers.
- `readExperiment(csvFile, keepData=False, MissingOnly=True)`: wrapper around `experiment_set`.
- `savedCsvToDf(...)`: loads multiple saved `*_siSummary*.csv` files into one dataframe.
- `computeExpTimeOfDay`: converts time strings into time-of-day features.
- `shiftedColorMap`: creates shifted matplotlib colormaps.
- `speedFft`, `powerToSample`: miscellaneous analysis helpers.
- `file_len`: line-count caching helper.
- `load_data_flexDelim`: reads numeric files with either whitespace or comma delimiters. `ExperimentMeta.PxPmmFromRois()` uses this for `ROIdef*` files.

The active notebooks currently duplicate some helper behavior inline, especially experiment processing and CSV loading. Treat `notebookHelper.py` as reusable support, not as the owner of the active notebook-specific grouping choices.

### `functions/metaTree.py`

Directly imported as `mt`.

Provides `pyMeta`, a small wrapper around the external `PythonMeta` package:

- Formats included studies.
- Runs continuous or categorical meta-analysis according to settings.
- Prints results and optionally shows a forest plot.

The active notebook cells import this module but do not appear to rely on it for the main selection/QTL processing path.

### `models/experiment_set.py`

Directly imported as `es`.

Main role:

- Reads `processingSettings.csv`.
- Iterates over each row.
- Checks whether a matching `*_siSummary*.csv` already exists in `ProcessingDir`.
- If `MissingOnly` is true, only instantiates `models.experiment.experiment` for rows whose summary is missing.

Important class:

- `experiment_set(csvFile=[], MissingOnly=False)`

Important behavior:

- If no `csvFile` is provided, opens a file dialog.
- Stores loaded processing settings in `self.df`.
- Stores processed experiments in `self.experiments`.
- `getAnimalPerExp()` reports per-experiment animal indexing.
- `saveExperimentOverviewPDF(...)` can save overview PDFs outside the notebook path.

### `models/experiment.py`

Directly imported as `xp`, and used by `experiment_set`.

Main classes:

- `ExperimentMeta`: file paths, defaults, ROI/scale lookup, video metadata, episode settings, and processing flags.
- `experiment`: loads raw trajectories, links animals and pairs, splits data into episodes, computes metrics, and writes outputs.

Core outputs:

- `*_siSummary_epi<episodeDur>.csv`
- `*MapData.npy` when `SaveNeighborhoodMaps` is true.
- `*_anSize.csv` when animal size is recomputed.

Important computations:

- Reads trajectory data with `loadData()`.
- Builds full animals and pairs with `linkFullAnimals()` and `linkFullPairs()`.
- Splits each experiment into animal-pair-episodes with `splitToPairs()`.
- Computes shoaling index through `computeSocialIndex()`.
- Collects average speed, smoothed speed, thigmotaxis, bout duration, leadership, and synchronization in `saveExpData()`.

### `functions/paperFigureProps.py`

Directly imported as `pfp`.

Provides paper-style plotting defaults:

- `figSq`, `figLs`, `figPt`: standard figure-size tuples.
- `paper()`: applies seaborn/matplotlib settings for paper-context plots.

The active notebooks import this module, but most plot styling is still done inline with seaborn and matplotlib calls.

## Runtime Dependencies Used After Notebook Processing Starts

These files are not imported directly by the notebooks, but they are involved when `experiment_set` creates experiments.

- `models/pair.py`: defines one animal-partner episode and computes shoaling index, inter-animal distance, average speed, smoothed speed, thigmotaxis, bout duration, leadership, and synchronization.
- `models/animal.py`: lightweight animal wrapper that links an animal to an experiment or pair and attaches time-series collections.
- `models/AnimalTimeSeriesCollection.py`: converts raw trajectory columns into positions, headings, speed, thigmotaxis/radial-position histograms, neighbor maps, bout starts, force maps, and synchronization maps.
- `models/geometry.py`: trajectory, vector, circle, region, angle, and distance helpers.
- `functions/getAnimalSizeFromVideo.py`: samples video frames and estimates animal sizes when size recomputation is enabled.
- `functions/video_functions.py`: pixel scaling, frame extraction, animal-length, and animal-size helpers.
- `functions/getVideoProperties.py`: video metadata lookup.
- `functions/plotFunctions_joh.py`: map and group comparison plotting helpers, mainly used by overview or vector-field paths.
- `functions/randomDotsOnCircle.py`: random-spacing comparison helper used in overview plotting.

## Metrics In The Active Summary Tables

The most important behavioral metric is `si`, the shoaling index:

```text
si = (mean shifted-control IAD - mean real IAD) / mean shifted-control IAD
```

where `IAD` is inter-animal distance. Positive values indicate that the real animal-stimulus distance is smaller than the shifted-control expectation.

Other commonly plotted metrics:

- `avgSpeed`: mean instantaneous speed from frame-to-frame travel times FPS.
- `avgSpeed_smooth`: mean speed after trajectory smoothing.
- `anSize`: estimated animal size in mm.
- `thigmoIndex`: mean radial distance from arena center, derived from polar position.
- `boutDur`: median interval between detected speed peaks, in seconds.
- `leadershipIndex`: front-versus-back neighbor-position metric when leadership is enabled.
- `sync_amp`, `sync_t`: synchronization/cross-correlation outputs when sync is enabled.

## Outputs And Cache Behavior

Generated files are intentionally reused.

- `experiment_set(..., MissingOnly=True)` checks `ProcessingDir` for matching `*_siSummary*.csv`.
- If a summary file exists, that experiment is skipped.
- The notebooks then load the saved summaries with a `glob` pattern based on each `PositionTxt` basename.
- Animal-size files may be cached as `<PositionTxt basename>_anSize.csv`.
- `SaveNeighborhoodMaps = 0` in active notebooks, so `*MapData.npy` is not normally written from these notebooks.

If results look stale, check `ProcessingDir` first. Existing summary CSVs can be older than the notebook settings you are testing.

The archived `legacy/Analyses/old/ShoalingNeighborhoodMaps_2026.ipynb` uses a separate map-generation workflow: it uses a notebook-specific processing cache under the selection temp-processing folder, with `SaveNeighborhoodMaps = 1`. Its summary CSVs are generated for that workflow and are not read from the slim selection notebook cache.

## Other Repository Areas, More Briefly

### Historical analysis notebooks

`legacy/DeprecatedAnalyses/LarschAndBaier2018/` contains the historical paper-reproduction workflow described in `readme.md`. It is separate from the current `Analyses/` notebooks. Use it only when reproducing or inspecting the old 2018 workflow.

### Archived F2 size notebook

`legacy/Analyses/others/size_data_analysis.ipynb` is inactive.

Purpose:

- Small notebook for F2 size measurement summaries.
- Reads one CSV and plots size by tank.

Expected input:

- CSV file configured by `datfolder` and `filename`, currently `Sel1_F2_size.csv`.
- Required columns:
  - `Tank`: tank/category identifier used on the x-axis.
  - `Size`: numeric size measurement in mm, used on the y-axis.

Output:

- Interactive pointplot plus swarmplot of `Size` by `Tank`.
- The notebook does not save a figure in its active cells.

### Vector-field analysis

`legacy/functions/vector_field_analysis.py` is a larger helper-backed analysis pipeline for vector fields, neighborhood maps, bout maps, thigmotaxis, and attraction plots. It expects `*MapData.npy` and `*_siSummary*.csv` files. Since the main selection notebooks set `SaveNeighborhoodMaps = 0`, this is not part of their standard run.

The archived `legacy/Analyses/old/ShoalingNeighborhoodMaps_2026.ipynb` provides a focused neighborhood-map workflow. It writes its own summary CSVs and map arrays into a notebook-specific processing cache. The human-facing walkthrough is `documentation_cr/neighborhood_map_analysis.md`.

### Video preprocessing and GUI utilities

Archived scripts and active shared functions support video splitting, background/median generation, ROI drawing, video metadata, and visual inspection:

- `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`
- `functions/getMedVideo.py`
- `functions/gui_circle.py`
- `functions/video_functions.py`
- `legacy/root-scripts/vidGui.py`
- `legacy/root-scripts/VidGui2ArenaShoal.py`
- `legacy/root-scripts/processingVideo.py`
- `legacy/root-scripts/rotationWrapper.py`

The standalone scripts are archived preprocessing/inspection entrypoints. The shared `functions/` helpers remain active dependencies of experiment loading and animal-size computation.

### Shape and image processing

Animal shape and image helpers live in:

- `legacy/models/AnimalShapeParameters.py`
- `legacy/models/AnimalShapeParameters_MP.py`
- `functions/ImageProcessor.py`
- `legacy/functions/tailfit.py`

The shape and tail modules are archived. `functions/ImageProcessor.py` remains active through the video/size helper dependency tree; it supports contour, skeleton, morphology, and crop operations.

### Operational scripts

`legacy/scripts/` contains archived helpers:

- `pairLists.py`: creates common pair-list matrices.
- `calibration.py`: calibration exploration.
- `splitMultiPairTxt.py`: raw text splitting utility.
- `tiffProcessing.py`: TIFF stack processing.

`scripts/sync_coding_doc.py` remains active repository maintenance tooling and keeps `coding.md` synchronized with `setupAgents.md`.

The archived operational scripts have no callers in the active notebook run paths.

### Exploratory and obsolete code

`legacy/playground/` and `legacy/obsolete/` contain experiments, old scripts, and retired helper copies. They should not be used as current behavioral-analysis authority.

## Validation

From the repository root, run `python -B -m unittest discover -s tests -v` for active dish-use and loom tests. Archived SI slowdown tests run separately with `python -B -m unittest discover -s legacy/tests -v`. Default pytest discovery is restricted to `tests/` by `pytest.ini`.

## Practical Checklist Before Running Active Notebooks

1. Confirm repository discovery or `codeDir`/`CODE_DIR` resolves to this repository. Dish-use and loom notebooks also accept `JLSOCIALBEHAVIOR_CODE_DIR`.
2. Confirm the configured metadata workbook and required sheets exist. Check the actual configuration, especially in `LoomAnalysisCR.ipynb`, which currently points to QTL metadata.
3. Review date/cohort filters, time windows, genotype choices, and cache controls near the top of the selected notebook.
4. Confirm the workflow-specific inputs below are available.
5. Confirm the configured processing/cache and output folders are writable. For shoaling processing, check whether existing `*_siSummary*.csv` files will be reused by `MissingOnly=True`.
6. Run the notebook from top to bottom.
7. Inspect the notebook's summary/QC tables and plots before using results for downstream interpretation.

| Workflow | Required inputs and run order |
| --- | --- |
| Current selection, selection history, and shoaling-QTL | `AllExp` and `AllAn`, with experiment folders containing `PositionTxt*` and `PL*`. Confirm AVI and ROI/scale inputs when recomputing size. These notebooks generate `processingSettings.csv` and missing shoaling summaries. |
| QTL selection comparisons | Run the current selection notebook first to create `dfEpiAn_selection.csv` in the configured comparison-input folder. |
| QTL QC | Run shoaling-QTL processing first to create `processingSettings.csv` and matching summary CSVs. Confirm `AllAn` and the trajectories/geometry needed by the QC loader are available. |
| Dish use | Supply an existing `processingSettings.csv` and the selected raw trajectory with `ROIdef*` geometry, or an applicable trajectory cache with caching enabled. Configure `AllAn` for genotype labels. Saved shoaling summaries are not required. |
| Loom responses | Supply `AllExp`, `AllAn`, and selected `PositionTxt*` trajectories with supported embedded loom stimulus blocks. Processing settings, pair lists, and shoaling summaries are not required. |
