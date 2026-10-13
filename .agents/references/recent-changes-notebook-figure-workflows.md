# Recent Changes - Notebook Figure Workflows

### 2026-10-12 - Correct QTL QC explanations and comments

- What changed: Updated QTL QC Markdown, comments, and docstrings to match current window selection/sampling, summary aggregation and provenance, cache-or-ROI loading, threshold modes and zero-MAD handling, coverage/missingness, robust family plots, SI correlations, overlays, and experiment-animal counts. Executable code, settings, plot labels, schemas, cell IDs, metadata, execution counts, and saved outputs are preserved.
- Rerun implications: Documentation-only change; no analysis rerun is required. Existing local edits and historical log entries are preserved; NAS data, caches, and exports were not accessed or rewritten.
- Validation performed: Parsed notebook JSON, compiled all 14 code cells with the repository's jlsocial interpreter, and confirmed before/after AST equality after removing docstrings. Checked unchanged notebook/cell metadata and saved outputs, unique IDs for all 27 cells, source-only byte replacements, and preserved log contents. No analysis cells were executed.

### 2026-10-09 - Simplify QTL diameter naming and show full plot ranges

- What changed: QTL QC now reads only `dishDiameter_mm` (70-mm default) and uses `dish_diameter_mm` locally and in its cache key. The renamed hashed key produces fresh cache filenames for a one-time ROI rebuild; old files are left untouched and are not searched as a fallback. Cache array fields and existing collector handling are preserved. QC metric axes now default to automatic ranges; `FINAL_SI_Y_LIMITS = None` replaces the fixed SI range, with settings validation and hidden-point/review-flag counts for manual metric, family-panel, and SI ranges. Updated explanations and cleared affected outputs/counts. No helper module, alias resolver, or legacy-cache checker was added; completeness and QC-flag calculations are unchanged.
- Rerun implications: Rerun from Settings. Section 4 requires ROI files on the first rerun to create the new matching caches; subsequent runs can reuse them without ROI. Explicit physical diameters in processing rows must use `dishDiameter_mm`. NAS caches, exports, and processing tables were not modified during this change.
- Validation performed: All 15 code cells compiled. In-memory checks confirmed the common diameter field/default, unchanged equivalent ROI scaling/centering, changed cache digest with other inputs preserved, old-cache preservation, new-cache creation with the existing schema, subsequent cache reuse without ROI, and missing-ROI collector errors retaining fish. Rendered all five metrics in both plotting layouts plus the final SI view; verified automatic bounds contain finite points, intervals, and displayed cutoffs, and manual bounds/counts handle endpoints and each family correctly. Visually inspected six representative automatic/manual previews under repository `temp/`; plotting emitted no warnings. Confirmed preserved point-5 sources, notebook metadata, unrelated cells, and unchanged dish-use notebook/helper. NAS/real-recording execution was not performed.

### 2026-10-09 - Align QTL QC dish grid with dish-use analysis

- What changed: Renamed QTL QC `SPACE_EVENNESS_BINS` to `DISH_GRID_SIDE_LENGTH` and set it to 40 cells per axis, matching `Analyses/Dish-useAnalysis.ipynb`; updated settings validation and the helper call, and cleared edited-cell outputs/execution counts. Other settings, trajectory-cache keys/schema, and metric columns are preserved.
- Rerun implications: Rerun Settings and validation, then Section 4 onward to recompute evenness and dependent QC references/flags at 40x40 resolution. Existing trajectory caches remain usable; external exports were not rewritten.
- Validation performed: Parsed the notebook, compiled all 15 code cells, confirmed matching settings and no old-name source references, and checked QTL caller agreement with the shared helper on a synthetic uniform 1,264-bin circular grid. Verified unchanged output columns, unrelated cells, and notebook metadata. NAS/recording execution was not performed.

### 2026-10-09 - Require cache or ROI for QTL QC trajectories

- Slice goal: Remove background-scale and `pxPmm` fallbacks from `Analyses/QTLmapping/QTLQualityControl.ipynb` while preserving matching trajectory caches.
- What changed: Raw trajectory loading now requires a regular `ROIdef*` file beside the trajectory, choosing the first sorted match; missing geometry raises an error identifying the directory and trajectory. ROI parsing, centering, scale conversion, cache-first loading, keys/schema, and collector missing-data handling are preserved. Section 4 documents cache-or-ROI loading and the shared dish-use helper; loader/downstream outputs and execution counts are cleared.
- Rerun implications: Rerun from Section 4 with upstream summary tables available, or restart and rerun from Settings. Existing matching caches remain eligible without ROI files and regardless of their original scale source; rebuilding raw trajectories requires an ROI. No external caches or exports were rewritten.
- Validation performed: Parsed the notebook and compiled all 15 code cells. In-memory fixtures covered cache hits without ROI, missing-ROI failures with cache absent/disabled despite alternate scale sources, whitespace/comma ROI parsing, directory exclusion, sorted file selection, original centering/scaling and sampled segments, fish-count validation, unchanged cache filename/key/writer fields, and downstream evenness agreement. Collector fixtures retained all 35 fish with warning/error details and verified QC missingness/review flags. Compared against the pre-edit working notebook to confirm preserved unrelated sources, metadata, cache logic, and existing local edits. NAS execution and real recording/export writes were not performed.

### 2026-10-08 - Correct grid naming and migrate dish-use callers

- Slice goal: Distinguish grid side length in cells from bin counts and make the dish-use notebook, QTL caller, tests, and documentation coherent.
- What changed: `analyze_dish_use` now takes `grid_side_length` and returns `dish_bin_count`/`visited_bin_count`; occupancy and entropy calculations are unchanged. The dish notebook restores `reachable_bin_count`/`visited_reachable_bin_count` CSV columns, fixes `DISH_EVENNESS_ZMAD_THRESHOLD` references, and clears loader/downstream outputs and execution counts. Its cache reader prefers `dish_radius_mm`, accepts legacy `arena_radius_mm`, and reports geometry-free caches; new caches still write only `dish_radius_mm`. QTL QC and the six tests migrate to the dish API without an arena compatibility module. Updated current routing, symbol/output references, function documentation, and notebook explanations; preserved historical change entries.
- Rerun implications: Restart the notebook kernel to load the changed helper API and rerun dish-use analysis from Settings, using existing matching caches if enabled. Rerunning exports restores the original bin-count column names; existing external caches and exports were not rewritten. Restart QTL QC from setup before rerunning dependent cells; its numerical metrics, output columns, loader settings, and cache schema are unchanged. The former arena-use API and mistaken length-named count fields are removed rather than aliased.
- Validation performed: Both notebooks parsed and all code cells compiled; all six migrated tests and seven doctest statements passed. Verified helper executable AST equivalence after name substitutions and equality of every result field against the previous arena implementation. Fresh synthetic 35-fish execution covered original identity order/CSV columns, nonzero/zero MAD, missing scores/flags, and duplicate-free reruns; 40x40 geometry retained 1,264 eligible bins. QTL synthetic results matched previous values and columns. In-memory fixtures covered current/legacy/both/missing cache radius fields, missing-ROI warnings/errors, raw centering, unchanged cache keys, and writer fields. Rendered and visually inspected occupancy and genotype-evenness plots, including fish labels and both threshold lines, with NAS exports disabled and previews confined to the repository. Confirmed unchanged notebook IDs/metadata and unrelated source cells, no obsolete active API/name references, and a clean whitespace check. Real recording/NAS execution was not performed.

### 2026-10-08 - Document dish-use helper inputs and outputs

- Slice goal: Explain how to call `analyze_dish_use` and interpret its results without reading the implementation.
- What changed: Expanded the function's NumPy-style docstring with input shape/units, validation, return semantics, boundary masking, missing-data behavior, and a runnable concentrated/uniform two-fish example. The result-class docstring documents all 13 fields with shapes, units, and y-row/x-column orientation. The notebook's existing occupancy/metrics Markdown sections now show the actual call, full result-field table, evenness/coverage formulas, exported-column mappings, finite/in-dish/binned count distinctions, and notebook-added zMAD screening.
- Rerun implications: None. Executable helper behavior, APIs, notebook code cells, outputs, metadata, cache logic, and export schemas are unchanged.
- Validation performed: Confirmed identical helper AST after removing docstrings and unchanged notebook cells except the two Markdown sources. Compiled the helper and all eight notebook code cells. All seven doctest statements passed; checked coverage of all result fields, shapes/dtypes, x/y orientation, count/probability maps, entropy endpoints, boundary-count differences, zero/NaN behavior, empty/missing/out-of-dish inputs, documented input-validation errors, the notebook call example, and metric-mapping coverage. Validation ran in memory without analysis exports or temporary files.

### 2026-10-08 - Require ROI geometry for raw dish-use trajectories

- Slice goal: Require `ROIdef*` geometry when reading raw trajectories in `Analyses/Dish-useAnalysis.ipynb`, while allowing existing matching caches without ROI files.
- What changed: ROI discovery now precedes the cache/raw branch. Cache hits without an ROI emit a warning identifying the searched directory and cache used; raw loading without an ROI raises `FileNotFoundError` with the directory and trajectory path. Removed `bgMed` and uncentered `pxPmm` fallbacks, corrected the two-pixel crop explanation, and cleared loader/downstream saved outputs and execution counts. The helper, remaining cell sources, cache key/schema, centering formula, and export names are unchanged.
- Rerun implications: Rerun from the trajectory-loading cell. A missing ROI is acceptable only when caching is enabled and a matching readable cache exists; rebuilding trajectories requires an ROI. Legacy cache-field compatibility remains separate work.
- Validation performed: Parsed the notebook and compiled all eight code cells. In-memory NumPy file fixtures confirmed whitespace/comma ROI parsing and centered-coordinate equivalence, unchanged cache names/schema, warning-free cache hits with ROI, the exact missing-ROI warning on cache hits, and raw-load errors despite `pxPmm`/`bgMed` alternatives when the cache is absent or disabled. Verified matching directories are excluded, cached arrays/IDs/radius are preserved, and cached coordinates feed the unchanged 35-fish occupancy/evenness cells correctly with exports disabled. No synthetic files were written; real recording/network-data execution was not performed.

### 2026-10-07 - Dish-use-only notebook

- Slice goal: Remove speed and movement screening from `Analyses/Dish-useAnalysis.ipynb` while retaining dish-use occupancy, coverage, evenness, and low-evenness screening.
- What changed: Removed movement calculations, the speed-distribution section and exports, movement labels, and movement-outlier tables. The notebook builds `dish_metrics` directly from the unchanged `functions/dish_use.py` helper, labels heatmaps with identity/genotype/evenness/zMAD, and retains the genotype evenness plot. Existing dish-use export filenames are preserved; their CSVs omit movement fields. Affected saved outputs and execution counts were cleared.
- Rerun implications: Rerun from the occupancy calculation after loading trajectories to refresh figures and tables. Trajectory loading, cache handling, helper semantics, and previously exported files are unchanged; old-cache radius-key compatibility remains separate work.
- Validation performed: Parsed the notebook and compiled all eight code cells; checked source, cell IDs, and saved outputs for remaining movement/speed analysis. A 35-fish synthetic smoke run covered concentrated/uniform occupancy, stationary fish, partial/all-missing positions, and out-of-dish coordinates; checked helper agreement, missing-score/flag behavior, zero-MAD handling, screening order, stable identities, and duplicate-free reruns. Rendered and visually inspected the occupancy and genotype-evenness figures with notebook exports disabled. Expected headless `plt.show()` warnings did not prevent rendering. Real recording/network-data execution was not performed.

### 2026-10-06 - Sampled-frame arena-use occupancy

- Slice goal: Express arena occupancy as retained frame counts without estimating elapsed time in the evenness helper.
- What changed: `functions/arena_use.py` now returns `occupancy_frame_counts` and `total_binned_frame_count` from its three-argument API. The trajectory-grid notebook uses frame-count heatmap labels and exports `total_binned_frame_count` in place of `total_reachable_seconds`; QTL QC calls the same helper without a sampling interval. Movement speed and duration metrics remain time based.
- Rerun implications: Rerun the trajectory-grid analysis cells to refresh embedded outputs and saved figures/CSVs. QTL evenness values are unchanged for the same selected frames, so raw processing need not be rerun.
- Validation performed: All six arena-helper tests passed; both notebooks parsed and their code cells compiled. A synthetic 35-fish run confirmed frame-count totals, the renamed metrics column, heatmap title and colorbar labels, and agreement with the QTL evenness caller.

### 2026-10-02 - QTL QC missing animal-size review markers

- Slice goal: Include fish with missing animal size in the final QTL QC figure and line summary without treating missingness as an outlier.
- What changed: `QTLQualityControl.ipynb` now marks complete-SI fish with missing animal size using a distinct X overlay, reports per-experiment missing-size counts, and adds `Animal size missing` plus `Unique flagged fish (outlier or size missing)` columns to the line summary; the latter counts overlapping flags once.
- Rerun implications: No data-pipeline rerun is required; rerun Sections 11 and 12 after `qc_table` is available.
- Validation performed: Parsed notebook JSON, syntax-checked all 15 code cells, and smoke-tested ordinary, animal-size-outlier, missing-size-only, and overlapping-flag rows; verified the marker, per-experiment count, line count, all-lines total, unchanged outlier total, and the non-double-counting flagged-fish union.

### 2026-09-27 - DeltaSI panels in selection/QTL metric comparisons

- Slice goal: Add bout-minus-linear DeltaSI to the final Sel1 and Sel3 comparison figures without changing the existing panels' samples.
- What changed: `ShoalingQTLMappingAnalysis_2026.ipynb` now pairs episode summaries by dataset-qualified fish identity, fills the sixth panel with DeltaSI, draws a zero reference, and warns separately when eligible Sel1 or Sel3 fish lack a valid pair.
- Rerun implications: No pipeline rerun is required; rerun the combined-data and final comparison cells against the existing episode summaries.
- Validation performed: Parsed and compiled the notebook and smoke-tested paired and missing-episode behavior plus both six-panel figures with synthetic data.

### 2026-09-27 - QTL QC outlier summary by line

- Slice goal: Summarize the final QTL QC outlier counts by metric category and selection line.
- What changed: `QTLQualityControl.ipynb` now ends with a line-by-category table containing unique outlier-fish counts, all five metric-specific counts, and an all-lines total row.
- Rerun implications: No data-pipeline rerun is required; rerun the final summary cell after `qc_table` is available.
- Validation performed: Parsed and compiled the notebook, then exercised the aggregation with overlapping flags, multiple lines, and a zero-outlier line in synthetic data.

### 2026-09-24 - External QTL QC correlation legends

- Slice goal: Keep correlation legends from covering data or statistics in the pooled and Sel1/Sel3 QC figures.
- What changed: Every correlation panel now places its two-entry legend in one centered row below the x-axis title; both grids are taller and use additional vertical spacing to accommodate the external legends.
- Rerun implications: No data-pipeline rerun is required; rerun the correlation plotting cell after `qc_table` is available to refresh both figures.
- Validation performed: Parsed and compiled the notebook, rendered both grids, measured every legend below its x-axis label and outside its axes, checked row and figure-boundary clearance, and confirmed shared family x-limits, annotations, and output stems.

### 2026-09-24 - QTL QC family correlation axis alignment

- Slice goal: Make each Sel1/Sel3 QC-correlation pair directly comparable and keep Space-use evenness legends clear of the statistics annotations.
- What changed: `QTLQualityControl.ipynb` now shares x-axis limits within each family-grid metric row and places Space-use evenness legends in the upper-right in both pooled and family-specific correlation figures.
- Recovery note: A concurrent notebook save corrupted embedded outputs in the adjacent line/date and correlation cells during editing; the stale outputs were cleared, the affected line/date helper header was restored, and all five recovered metric plots were smoke-rendered.
- Rerun implications: No data-pipeline rerun is required; rerun Section 9 after `qc_table` is available to refresh both correlation figures.
- Validation performed: Parsed and compiled the notebook, smoke-rendered both correlation figures, verified matching family-panel x-limits per metric, independent ranges across metrics, upper-right Space-use legends, and unchanged annotations and output stems.

### 2026-09-23 - QTL QC SI correlations by selection family

- Slice goal: Compare QC-metric relationships with individual shoaling index independently for Sel1 and Sel3 fish.
- What changed: `QTLQualityControl.ipynb` retains its pooled correlation figure and adds a five-row, two-column Sel1/Sel3 grid with family-specific Pearson correlations, p-values, regression lines, complete-case counts, and metric-outlier overlays.
- Rerun implications: No data-pipeline rerun is required; rerun Section 9 after `qc_table` is available to refresh both correlation figures and save `qtl_qc_si_correlations_by_selection_family` as PNG and PDF.
- Validation performed: Parsed and compiled the notebook and smoke-rendered pooled and family-specific figures with synthetic Sel1/TLSel1/Sel3/TLSel3 data, including missing, outlier, insufficient-sample, and constant-variable cases.

### 2026-09-17 - Restore missing QTL QC plotting sections

- Slice goal: Restore plotting cells accidentally removed during the shared arena-evenness refactor without reverting the expected-arena metric changes.
- What changed: Restored the complete per-experiment QC plotting cell, the Sel1/Sel3 QC-metric boxplot section, and the SI-correlation section from their authoritative notebook sources; removed temporary duplicate plotting helpers and cleared restored/downstream plot outputs.
- Rerun implications: Rerun Sections 7-11 to regenerate the figures; trajectory loading, expected-arena evenness, and QC calculations do not need to be repeated when their in-memory tables are current.
- Validation performed: Parsed all 27 notebook cells, matched the restored Section 7-9 sources to their committed versions, compiled all 14 code cells, rendered Sections 7-11 with synthetic data for all five QC metrics, and passed all six arena-helper unit tests with `unittest`.

### 2026-09-17 - Shared expected-arena evenness helper

- Slice goal: Give the trajectory-grid and QTL QC notebooks one complete arena-use evenness implementation without fish-specific occupancy radii.
- What changed: Added `functions/arena_use.py` to build the fixed expected-arena grid, occupancy seconds, probability maps, normalized Shannon evenness, and coverage diagnostics directly from centered trajectories. Both notebooks now call the helper; the trajectory heatmap consumes its occupancy output, and personal-radius settings, flags, columns, and plotting code were removed.
- Rerun implications: Rerun both notebooks from their trajectory-loading sections. QTL `spaceEvenness`, derived zMAD thresholds, and flagged fish will change because all fish now use the full expected-arena denominator and a fixed-diameter grid.
- Validation performed: Added unit tests for entropy endpoints, fixed-grid/full-arena normalization, out-of-arena exclusion, sample-time invariance, empty data, and input validation. Both notebooks parse as JSON and their rewritten sources were inspected; Python execution remains pending because no accessible interpreter was available in this session.

### 2026-09-16 - Explain per-fish arena-use evenness

- Slice goal: Make the evenness calculation in `ShoalingTrajectoryGrid_2026.ipynb` readable without changing its behavior.
- What changed: Expanded comments in the per-fish evenness cell to explain sampled occupancy time, the percentile radius and approximate bin mask, Shannon normalization including unvisited bins, coverage versus evenness, pooled zMAD flags, edge cases, and exports.
- Rerun implications: None; executable code, saved outputs, and notebook metadata are unchanged.
- Validation performed: Parsed notebook JSON, verified identical executable AST for the edited cell, and checked that all other cells, outputs, and metadata match the pre-edit notebook.

### 2026-09-16 - Switchable QTL QC selection-family thresholds

- Slice goal: Support global pooled QC cutoffs or separate Sel1 and Sel3 cutoffs for each QC metric.
- What changed: `QTLQualityControl.ipynb` now exposes `QC_THRESHOLD_MODE`; `'global'` preserves the pooled zMAD reference, while `'selection_family'` calculates independent Sel1/Sel3 references and uses the pooled reference only for other lines. Section 7 draws both families' lower and upper threshold lines in the latter mode.
- Rerun implications: After changing the threshold mode, rerun from the QC-flag calculation through the dependent tables and figures.
- Validation performed: Parsed notebook JSON and all code cells; synthetic global and selection-family flag checks plus a Section 7 threshold-line plot smoke test.

### 2026-09-16 - QTL QC distributions by selection line

- Slice goal: Compare all F2 QC-metric distributions across the individual Sel1 and Sel3 lines.
- What changed: Added a five-panel boxplot grid to `QTLQualityControl.ipynb`, with Sel1 lines first, Sel3 lines second, and PNG/PDF export through the notebook's existing figure helper.
- Rerun implications: Rerun the notebook through the QC-table and new selection-line plotting sections to refresh `qtl_qc_metrics_by_selection_line`.
- Validation performed: Notebook JSON and code-cell syntax checks plus a synthetic selection-line plotting smoke test.

### 2026-09-16 - QTL QC deterministic scatter overlays

- Slice goal: Keep QTL QC outlier markers aligned with their individual fish points.
- What changed: Restored deterministic Matplotlib scatter layers in the per-metric and final SI experiment plots, so colored QC overlays use the same x position as their gray base point and legends remain concise.
- Rerun implications: Rerun the QTL QC plotting sections to refresh the displayed and saved figures; no data-processing rerun is required.
- Validation performed: Parsed and compiled the notebook and smoke-rendered both affected plotting functions with synthetic multi-experiment data.

### 2026-09-15 - Editable QTL QC pointplot limits

- Slice goal: Allow manual y-axis ranges when reviewing QC pointplot outliers.
- What changed: `QTLQualityControl.ipynb` now exposes `QC_POINTPLOT_Y_LIMITS`, with independent `(lower, upper)` overrides for every QC metric and automatic scaling retained by default.
- Rerun implications: Edit the desired limits in settings and rerun the pointplot section to regenerate figures.
- Validation performed: Compiled notebook code and smoke-rendered automatic and explicit-limit metric plots.

### 2026-09-15 - Unified QTL QC time-window modes

- Slice goal: Apply one user-selected time-window method to every QTL QC metric.
- What changed: `QTLQualityControl.ipynb` now switches between an exact continuous raw-recording window and episode-based in-dish selection; the shared selected frames drive both canonical summary aggregation and raw-trajectory evenness.
- Rerun implications: Changing method or either window requires rerunning from settings; versioned cache keys prevent reuse across incompatible frame selections.
- Validation performed: Tested switch validation, continuous boundary alignment, both real-data modes, summary-block completeness, trajectory-grid evenness agreement, and notebook plotting.

### 2026-09-15 - QTL QC pointplot readability

- Slice goal: Make the experiment-level QC plots easier to read without changing their data or statistics.
- What changed: `QTLQualityControl.ipynb` now labels experiments with unique `lineSet_s<setup>` names and uses taller, editable heights for the five metric pointplots and final annotated SI plot.
- Rerun implications: Rerun from summary loading to rebuild the concise labels, then rerun the plot sections to refresh saved figures.
- Validation performed: Checked label uniqueness and smoke-rendered the experiment-level figures with the full 595-fish dataset.

### 2026-09-15 - QTL QC global sample identifiers

- Slice goal: Correct the QTL QC sample count and metadata labels by using `AllAn.QTLID` as the global F2 fish identifier.
- What changed: `QTLQualityControl.ipynb` now maps summary `animalIndex` through each experiment's `anNr` roster before joining `QTLID`, and uses `QTLID` throughout QC tables, trajectory-evenness joins, exports, and plot identities.
- Rerun implications: Rerun from the metadata-loading section; the versioned trajectory cache key prevents reuse of caches with the old identifier payload.
- Validation performed: Checked F2 identifier uniqueness, repeated-local-index mapping, notebook syntax, and NAS-backed experiment coverage.

### 2026-09-14 - Self-contained QTL quality-control notebook

- Slice goal: Review F2 QTL fish for pooled-distribution outliers in speed, thigmotaxis, space-use evenness, animal size, and bout duration while retaining individual shoaling index.
- What changed: Added `Analyses/QTLmapping/QTLQualityControl.ipynb` with editable date/window settings, notebook-local loading and trajectory functions, global two-sided zMAD flags, coverage and flagged tables, per-experiment metric plots, SI correlations, a ranked zMAD heatmap, and a final SI plot with colored multi-metric overlays and outlier counts.
- Rerun implications: Run from the settings cell with NAS access; changing cohort, time-window, trajectory, binning, or threshold settings requires rerunning downstream cells, while compatible trajectory blocks can reuse notebook-specific caches.
- Validation performed: Parsed the notebook JSON, compiled all code cells, and smoke-ran entropy, pooled zMAD, all metric plots, correlations, the zMAD heatmap, and multi-metric SI overlays on synthetic multi-experiment data. Real-data NAS execution remains a documented acceptance step.

### 2026-09-11 - QTL fish counts by selection line

- Slice goal: Show both the available metadata fish and the date-filtered QTL fish used for each selection line in the QTL mapping notebook.
- What changed: Added an unfiltered `line`/`n_fish` metadata summary and a second post-filter summary that counts each retained `animalIndex` once after applying `QTL_INCLUDE_FROM_DATE`.
- Rerun implications: No analysis pipeline rerun or new output file is required; rerun the metadata-loading and summary cells against the accessible QTL metadata workbook.
- Validation performed: Parsed the edited notebook JSON, checked code-cell syntax, exercised filtering, unique-fish counting, sorting, and total-row behavior with synthetic data, and verified the grouped total against the live `AllAn` sheet.

### 2026-08-07 - Fish-level legacy response grid by QTL line set

- Slice goal: Compare legacy response distance and response/baseline across every selected QTL metadata line set and all available sided loom sizes.
- What changed: Added a dynamic row-per-line-set, two-metric grid to `LoomAnalysis_QTL.ipynb`; repeated L/R trials are averaged per fish before plotting individual fish values and genotype mean +/- SEM.
- Rerun implications: No cache rebuild is required; rerun the notebook through response-metric loading, then run the new line-set grid cell.
- Validation performed: Parsed and syntax-checked the notebook and smoke-rendered multi-line-set synthetic data in genotype-split and pooled modes, including unequal trials and a missing loom size.

### 2026-08-07 - Inclusive QTL plot date filter

- Slice goal: Let users restrict every QTL-derived plot to experiments on or after one editable date without changing processing inputs or imported selection comparison data.
- What changed: Added `QTL_INCLUDE_FROM_DATE` to `ShoalingQTLMappingAnalysis_2026.ipynb` and applied it to the merged QTL summary before all plot tables are derived; `None` disables the filter, and invalid dates or empty results fail clearly.
- Rerun implications: No processing rerun is required; rerun the settings and summary-loading cells, then rerun the desired plots.
- Validation performed: Parsed and syntax-checked the notebook, exercised inclusive, disabled, invalid-date, and empty-result cases with synthetic data, and checked that imported selection rows remain unfiltered.

### 2026-08-07 - Switchable Loom QTL experiment or line-set comparisons

- Slice goal: Let all fish-level across-experiment Loom QTL plots, including unsided `CLsemi030` summaries, pool experiments from the same metadata line set.
- What changed: Added an editable comparison-group selector at the top of each affected `LoomAnalysis_QTL.ipynb` plot cell; `experiment` preserves the existing display and `line_set` pools fish by one validated metadata line-set label while retaining genotype splits and fish-level SD bars.
- Rerun implications: No cache rebuild is required; rerun the desired comparison plot cell after changing that cell's selector. The unsided maximum-velocity and pre/post cells still require the preceding unsided response cell to define the shared summary plotting helper.
- Validation performed: Parsed and syntax-checked notebook cells, exercised both grouping modes with synthetic fish-level data, and smoke-rendered the two comparison plots.

### 2026-08-06 - Unsided pre/post loom velocity plots

- Slice goal: Show individual `CLsemi030` trial maxima in one selectable experiment and compare fish-level pre-loom mean speed with post-loom maximum speed across experiments.
- What changed: Added an editable pre-loom velocity window, stored fish-by-trial pre-loom mean speed in cache schema version 7, and extended QTL Section 5 with a five-trial point plot, an overlaid pre/post plot, and a post/pre ratio plot.
- Rerun implications: Rerun the QTL collection cell once to rebuild schema-version-7 caches before running the pre/post and ratio cells; the selected-experiment post-maximum plot can use the existing fish-trial table.
- Validation performed: Verified exact pre-window speeds and invalid-window handling with synthetic contracts, passed the full test suite, parsed and compiled the notebook, and smoke-rendered all three plots in split and pooled modes including incomplete-trial and invalid-ratio handling.

### 2026-08-05 - Separate unsided CLsemi030 loom analysis

- Slice goal: Keep the last two QTL experiments' unsided `CLsemi030` pulses distinct from size-30 L/R looms and analyze them at fish level.
- What changed: The cache schema is version 6 with two dedicated unsided tables; the long unsided block is split at its five positive-size pulse onsets, existing tables remain L/R-only, and `LoomAnalysis_QTL.ipynb` adds fish-level response and maximum-velocity plots for the unsided events.
- Rerun implications: Rerun the QTL collection cell once to rebuild all seven per-experiment cache tables before running the new Section 5 cells.
- Validation performed: Inspected both experiments' identical `stimulusProtocol` CSVs, passed pulse-splitting and mixed sided/unsided helper contracts, parsed and compiled the notebook, and smoke-rendered the new plots with synthetic fish data.

### 2026-08-05 - Fish-level post-loom maximum velocity across experiments

- Slice goal: Compare fish-level post-loom maximum velocity across QTL experiments at one editable loom maximum size.
- What changed: Added a fifth fish-by-trial maximum-velocity cache table and a Section 4 seaborn swarm/point plot that merges L/R and repeated trials per fish downstream before showing experiment means with fish-level SD bars.
- Rerun implications: The loom cache schema is version 5, so rerun the QTL collection cell once before running the new Section 4 plot.
- Validation performed: Extended synthetic cache contracts, passed the loom helper tests, compiled the notebook, and smoke-rendered split and pooled multi-experiment plots with exact trial merging and invalid-size validation.

### 2026-08-05 - Across-experiment fish-level loom-response point plots

- Slice goal: Compare legacy response distance and response/baseline across all selected QTL experiments at one editable loom maximum size.
- What changed: Added a Section 2 seaborn point-plot cell to `Analyses/LoomAnalysis_QTL.ipynb`; it averages L/R trials per fish downstream, overlays the resulting fish averages as swarm points, then plots experiment means with fish-level SD bars, genotype hues, and metadata-ordered connecting lines.
- Rerun implications: Rerun the QTL notebook through the response-metric cache-loading cell, then edit `EXPERIMENT_COMPARISON_LOOM_MAX_SIZE` and run the new plot cell.
- Validation performed: Parsed and compiled the notebook, smoke-rendered split and pooled synthetic multi-experiment data, and checked trial merging, swarm-point counts, metadata ordering, and invalid loom-size handling.

### 2026-08-04 - Fish-level legacy loom-response plots

- Slice goal: Plot legacy response distance and response/baseline for individual fish in one editable QTL experiment after averaging that fish's repeated L/R trials by loom size.
- What changed: The loom response cache now retains fish-by-trial rows without merging sides, and `Analyses/LoomAnalysis_QTL.ipynb` adds seaborn strip/point plots with fish points and mean +/- SEM; L/R are merged only in the plotting cell.
- Rerun implications: The loom cache schema is version 4, so rerun the collection cell once to rebuild caches before running the new Section 2 cell.
- Validation performed: Extended the synthetic processing contract, passed the loom helper tests, parsed and syntax-checked the notebook, and smoke-rendered split and pooled plots with unequal L/R trial counts.

### 2026-08-04 - Support CLsemi loom episodes

- Slice goal: Process `CLsemi###L/R` closed-loop loom events with the same semantics as existing `CLfull###L/R` events.
- What changed: Shared loom-label parsing now recognizes both prefixes in trial extraction and multi-experiment filtering; the loom cache schema is version 3 so mixed recordings rebuild with the newly supported events.
- Rerun implications: Rerun the loom data-collection cell once; existing loom caches rebuild automatically.
- Validation performed: Added `CLfull`/`CLsemi` parsing and `CLsemi`-only processing tests, then ran the targeted helper test suite.

### 2026-07-24 - Sel1 versus Sel3 hi/lo loom-response plots

- Slice goal: Compare hi and lo loom responses separately within Sel1 and Sel3 line families while excluding Sel1_GC.
- What changed: Appended family-specific legacy center-distance, merged-response, maximum-velocity time/fit, and both side-layout plots to `Analyses/LoomAnalysisCR.ipynb`; Sel1/TLSel1 and Sel3/TLSel3 reuse the existing experiment-level caches with shared plot scales.
- Rerun implications: No cache rebuild is required; rerun the notebook through the new final section after the compact loom tables have loaded.
- Validation performed: Notebook JSON, executable-cell syntax, line-family classification, genotype filtering, and synthetic plot rendering were checked.

### 2026-07-24 - Loom cohort date filtering

- Slice goal: Replace the fixed year selector in `Analyses/LoomAnalysisCR.ipynb` with editable date-based cohort selection.
- What changed: Added inclusive start-date and exclusion-range modes, removed the fixed expected-experiment-count warning, and report the number of selected experiments after metadata/date filtering.
- Rerun implications: Rerun the notebook from its settings and metadata-selection cells after changing the date mode or date values; matching per-experiment loom caches remain reusable.
- Validation performed: Added helper tests for inclusive boundaries, missing dates, and invalid date settings; notebook JSON and executable-cell syntax checks will be run after the edit.

### 2026-07-22 - Gaussian peak fits for max velocity over time

- Slice goal: Capture the initial max-velocity rise at loom sizes 1 and 5 before comparing the subsequent genotype-specific decline.
- What changed: Replaced the estimated-floor exponential curves in `Analyses/LoomAnalysisCR.ipynb` with bounded estimated-floor Gaussian peak fits, expanded the time axis to include the complete protocol, and report peak timing plus the maximum post-peak velocity-loss rate and related width/half-decay summaries.
- Rerun implications: No raw-data or cache rebuild is required; rerun the time-versus-maximum-velocity plot cell after loading `trial_max_velocity`.
- Validation performed: Parsed and syntax-checked the notebook, recovered known Gaussian parameters in split and pooled synthetic tests, checked fit failure modes and derived-rate formulas, and smoke-rendered the eight cached experiments for visual inspection.

### 2026-07-22 - Genotype-specific velocity decay curves

- Slice goal: Compare how quickly post-loom maximum velocity decreases over experiment time for each displayed genotype.
- What changed: Replaced the pooled Pearson annotation in `Analyses/LoomAnalysisCR.ipynb` with independent nonnegative estimated-floor exponential fits, genotype-coloured curve overlays, decay constants in the legend, and a descriptive table of floors, amplitudes, initial decrease rates, half-lives, fit quality, and point counts.
- Rerun implications: No raw-data or cache rebuild is required; rerun the time-versus-maximum-velocity plot cell after loading `trial_max_velocity`.
- Validation performed: Parsed and syntax-checked the notebook, smoke-rendered split and pooled modes with synthetic exponential data, checked failure handling, and visually inspected the split plot.

### 2026-07-22 - Loom-size shapes in maximum-velocity side plots

- Slice goal: Make loom maximum size shape-coded in both maximum-velocity-by-side figures while retaining genotype colour coding.
- What changed: Added deterministic `SIZE_MARKERS`, removed loom-size colour encoding, applied size markers to individual points, and split the legends into genotype-colour and loom-size-shape entries. Pooled mode uses neutral box/point colours.
- Validation performed: Parsed the notebook and smoke-rendered both side plots in split and pooled modes; visually inspected the split figure for consistent genotype colours, size shapes, and readable legends.

### 2026-07-22 - Genotype-aware loom-response plots

- Slice goal: Compare user-selected fish genotypes throughout the multi-experiment loom notebook while preserving pooled views.
- What changed: Added normalized `AllAn` genotype mapping, pooled plus per-genotype loom cache rows, dynamic genotype catalog/selection controls defaulting to hi/lo/F2, genotype-aware versions of all seven existing figure outputs, and descriptive maximum-speed and legacy-response summaries with experiment and fish counts. The cache schema is now version 2, so old pooled-only caches reprocess automatically.
- Rerun implications: Rerun `Analyses/LoomAnalysisCR.ipynb` from metadata selection through processing; the first run rebuilds the four per-experiment loom caches, while later plot-switch changes reuse them.
- Validation performed: Compiled the helper, parsed every notebook code cell, passed targeted metadata/extraction/cache-contract tests, and smoke-rendered all plot cells with synthetic pooled and three-genotype data in split and pooled modes. Live metadata and raw-data rendering were unavailable because the network share denied access in the agent environment.

Append meaningful completed changes here.

### 2026-07-21 - Detail SI slowdown cluster bootstrap

- Slice goal: Make every sampling, filtering, and inferential step of the SI slowdown bootstrap auditable from the notebook and helper source.
- What changed: Added a step-by-step description of per-stimulus whole-experiment resampling, duplicate-cluster relabeling, retained fish/time dependence, model refitting and genotype reweighting, failed and nonpositive draw handling, requested-draw rate denominators, the positive-total-increase gate, the conditional crossing-time interval, and the maximum-deviation simultaneous upper band. Clarified that the default performs 500 refits per stimulus rather than one paired set of 500 two-stimulus refits, and documented cluster-count and late-window interpretation limits.
- Rerun implications: Documentation/comments only; no processing, bootstrap, table, or figure rerun is required.
- Validation performed: Parsed the notebook JSON and checked the expanded bootstrap source against `functions/si_slowdown.py`; targeted contracts verified the sampling unit, duplicate relabeling, draw accounting, positive-increase gate, conditional interval, simultaneous-band formula, and per-stimulus refit count. Python execution remained unavailable because no working interpreter is installed in the checkout or on PATH.

### 2026-07-21 - Move 2h SI slowdown inference after progression-by-experiment

- Slice goal: Keep the 2h SI slowdown inference at the end of the 2h progression sections.
- What changed: Moved the slowdown explanation, fit, and plot cells together to immediately follow the 2h progression-by-experiment grid; no analysis code or outputs changed.
- Rerun implications: No processing rerun is required; rerun the moved slowdown cells only if refreshed inference tables or figures are needed.
- Validation performed: Confirmed notebook cell ordering and preserved slowdown cell contents, parsed/syntax-checked the notebook, and ran the SI slowdown unittest suite.

### 2026-07-21 - Explain SI slowdown inference and vertical markers

- Slice goal: Make the SI slowdown estimand, uncertainty criterion, outputs, and figure markers understandable from the notebook and helper source.
- What changed: Expanded the notebook markdown with the forward-increment equation, persistence rule, mixed-model hierarchy, experiment-cluster bootstrap, simultaneous upper-band interpretation, result/diagnostic guidance, and an explicit explanation of the solid cyan 100-minute line. Documented every slowdown setting and the analysis/plot cells, expanded `functions/si_slowdown.py` docstrings and intent comments, added dashed/solid vertical-line legend keys, and clarified that lower-panel x-values start forward same-stimulus intervals.
- Rerun implications: No processing or inference rerun is needed for the documentation changes. Rerun only the slowdown plot cell after the existing slowdown tables are in memory to refresh the saved PDF with the new legend keys and x-axis wording.
- Validation performed: Parsed the notebook JSON, inspected the three slowdown cell sources, and checked targeted documentation/plot-label contracts. No runnable Python interpreter was available in this checkout or on PATH, so Python compilation and rendered figure inspection were unavailable.

### 2026-07-21 - Restrict SI slowdown inference to the 2h window

- Slice goal: Run SI slowdown-point inference only on the first 24 five-minute episodes and remove stale full-window slowdown artifacts.
- What changed: Moved the slowdown analysis under the 2h progression summary, restricted its raw input to `episode_number <= 24`, renamed notebook outputs with the `_2h` suffix, and deleted the seven stale `_4h_all` CSV/PDF artifacts from the configured output directory.
- Rerun implications: Rerun the slowdown cells after loading the summaries to regenerate the seven `_2h` outputs; the full-window progression plot remains unchanged.
- Validation performed: Notebook JSON parsed; cell order, episode-limit assertion, input source, and output-name contracts passed; all seven stale external artifacts were verified absent after deletion. Python compilation and unit tests were unavailable because the installed Python command is a nonfunctional Windows Store alias.

### 2026-07-21 - Switchable SEM and confidence-interval plot uncertainty

- Slice goal: Allow `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` plots to show 95% confidence intervals instead of SEM while preserving the current default.
- What changed: Added the editable `PLOT_CONFIDENCE_INTERVALS` setting, retained parallel SEM and 95% CI summary columns, and routed progression, normalized-SI, per-experiment, and slowdown uncertainty displays through the selected column.
- Rerun implications: No processing rerun is required; rerun the settings and affected summary/plot cells after changing the setting.
- Validation performed: Parsed the notebook JSON and passed source-contract checks for both uncertainty modes, including singleton guards and selector coverage; Python AST compilation and rendered smoke tests were unavailable because no working project Python interpreter is present in this session.

### 2026-07-21 - Shoaling-index slowdown inference

- Slice goal: Estimate when each stimulus-specific SI increase becomes smaller than an editable fraction of the total fitted experiment increase.
- What changed: Added `functions/si_slowdown.py` with spline mixed models, experiment-clustered bootstrap inference, persistent crossings, simultaneous one-sided upper bounds, genotype standardization, residual/coverage checks, spline and leave-one-experiment-out sensitivity tables, and an asymptotic mixed-model comparison. Extended `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` with editable settings, result exports, and fitted-SI/fraction plots while preserving the canonical summary writer and existing progression plot.
- Rerun implications: No SI processing rerun is required. Rerun the notebook from its imports/settings and loaded-summary cells; the default requests 500 mixed-model bootstrap refits per stimulus (1,000 primary bootstrap refits across two stimuli) and therefore takes materially longer than a descriptive plot cell.
- Validation performed: Compiled the helper and tests, parsed and syntax-checked every notebook code cell, ran three unit tests, and smoke-tested both stimulus-specific clustered bootstraps plus the profiled asymptotic and leave-one-experiment-out paths on synthetic repeated-measures data.

### 2026-07-20 - Mortality bars by line or lineage

- Slice goal: Show how selection lines or editable ancestry lineages contribute to both mortality record histograms and tank-normalized mortality proportions without changing either plot's statistic.
- What changed: Added a shared `MORTALITY_COLOR_BY` setting to `temp/fish_analyzes.ipynb`; `none` preserves the three original category colours, while `line` and `lineage` draw additive stacks and expose record-count and mean-proportion contribution tables. Lineage mode warns and excludes unmapped records and tanks before calculating bins and denominators; `MORTALITY_LINE_FILTER` remains limited to the tank-normalized section.
- Rerun implications: Rerun the settings and both mortality sections after changing the colour mode, lineage mapping, age-bin width, mortality line filter, or source workbook.
- Validation performed: Parsed and compiled all 12 code cells; ran all six plots in all three modes against the real workbook; confirmed `none` and `line` retain 320 records/284 usable tanks, lineage mode warns once per section and retains 311 records/273 tanks after excluding four unmapped lines, all record stacks equal histogram counts, all normalized stacks equal the original means, category colours and legends are correct, one- and two-month bins work, single/multiple line filters work, duplicate mapping validation is lineage-only, invalid modes fail, and representative plots in every mode are readable.

### 2026-07-20 - Uncoloured incross-failure age mode

- Slice goal: Provide a pooled, neutral version of the incross-failure-by-age plot alongside lineage- and line-coloured views.
- What changed: `temp/fish_analyzes.ipynb` now accepts `INCROSS_FAILURE_COLOR_BY = \"none\"`, which includes all eligible selection lines, draws one gray failure-proportion bar per populated age class, and suppresses the grouping legend without consulting `LINEAGE_GROUPS`.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after changing `INCROSS_FAILURE_COLOR_BY`.
- Validation performed: Parsed and compiled all 12 code cells; ran all three modes against the real workbooks; confirmed `none` retains 126 crosses/27 failures with one gray series, exact failure-proportion heights, no lineage warnings, and no legend; reconfirmed lineage mode excludes the current unmapped cross with one warning, line mode retains all crosses, invalid values fail clearly, and visually inspected the uncoloured render.

### 2026-07-20 - Optional lineage exclusions for incross failures

- Slice goal: Allow users to omit selection lines from the editable ancestry mapping without blocking the lineage-coloured age analysis.
- What changed: Lineage mode in `temp/fish_analyzes.ipynb` now warns once about unmapped lines and excludes their crosses before calculating age-bin totals, failure proportions, stacks, and sample-size labels; line mode remains independent of the lineage mapping.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after adding or removing lines from `LINEAGE_GROUPS`.
- Validation performed: Parsed and compiled all 12 code cells; ran complete and user-edited mappings against the real workbooks; confirmed the complete result remains 126 crosses/27 failures, omitting `TL sel1 lo9xTLN inx` warns once and leaves 125 lineage-mode crosses with updated denominators, line mode retains all 126 crosses without warning, stacks match failure proportions, and multiple omissions, duplicate assignments, invalid modes, and an all-unmapped cohort behave as intended; visually inspected the partial-lineage render.

### 2026-07-20 - Switchable lineage coloring for incross failures

- Slice goal: Let the fish-age incross-failure stacks be grouped and coloured by editable ancestry lineages while retaining the original per-line view.
- What changed: Added a top-level `INCROSS_FAILURE_COLOR_BY` switch and editable nine-lineage mapping to `temp/fish_analyzes.ipynb`; the age-bin failure cell now validates lineage assignments and summarizes, colours, and labels stacks by either lineage or selection line without changing denominators.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after changing the display switch, lineage mapping, filters, bin width, or source workbook.
- Validation performed: Parsed and compiled all 12 code cells; ran both lineage and line modes against the real workbooks; confirmed 126 eligible crosses, 27 failures, unchanged stacks across 12 one-month bins, the original 11 line segments, six represented lineage segments, correct legends, readable rendered plots, six two-month bins, and clear rejection of invalid switches and missing or duplicate lineage assignments.

### 2026-07-20 - Tank-normalized mortality proportions by age

- Slice goal: Compare the proportion of each tank's initial fish dying within age classes, with an editable selection-line filter.
- What changed: Extended `temp/fish_analyzes.ipynb` with `Plate`-normalized, non-cumulative mortality summaries and three plots for combined qualifying mortality, overall-bad-shape sacrifices, and found-dead events; only those two causes are included, tanks are equally weighted, and the setting accepts `all`, one line, or multiple lines.
- Rerun implications: Rerun the loading/settings, mortality, and new mortality-proportion cells after changing `MORTALITY_LINE_FILTER`, `AGE_BIN_MONTHS`, or `fish_database.xlsx`.
- Validation performed: Parsed and compiled all 12 code cells; ran the new section against the real workbook; confirmed 37 valid lines, 284 usable tanks, 1,059 found-dead fish, 184 bad-shape fish, exclusion of 2,044 old-age/no-use fish, bounded/additive proportions, three rendered plots, single/multiple/case-insensitive/zero-death filters, two-month bins, and rejection of invalid filters, widths, tank metadata, and mortality amounts.

### 2026-07-19 - Cumulative near-maximum SI probability progression

- Slice goal: Show the cumulative probability that pooled fish have reached at least 95% of their own stimulus-specific maximum SI during the first 2 hours.
- What changed: Added persistent per-fish/type reached states and a pooled continuous-versus-bout-like cumulative probability trajectory with observed-fish SEM and counts, bounded probability bands, and a dedicated PDF output to `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`; fish with missing SI are omitted only from the current episode's denominator.
- Rerun implications: No processing rerun is required; rerun the maximum-normalized calculation and new probability cells to create `si_near_max_probability_progression_2026_2h.pdf`.
- Validation performed: Parsed and syntax-checked every notebook code cell, then smoke-tested exact-threshold inclusion, persistent reached states after SI declines and across missing episodes, independent stimulus histories, category-specific exclusions, observed-only denominators, pooled probabilities/SEM/counts, first-episode inclusion, bounded bands, plot labels, axis limits, legend, and PDF target with synthetic data.

### 2026-07-19 - Skip recordings without loom episodes

- Slice goal: Let multi-experiment loom analysis continue past recordings that do not contain supported `CLfull###L/R` stimulus episodes.
- What changed: Loom collection now warns once per skipped experiment, writes no cache for it, preserves valid experiments, and raises a clear final error when every experiment is skipped.
- Rerun implications: Rerun the loom data-collection cell; existing valid caches remain compatible.
- Validation performed: Compiled the helper and smoke-tested mixed, all-invalid, valid cached, no-cache-on-skip, and unrelated-error behavior with synthetic data.

### 2026-07-19 - Maximum-normalized SI progression

- Slice goal: Replace episode-to-episode SI changes with SI levels expressed relative to each fish's own stimulus-specific maximum.
- Passes completed: Simplified the final section of `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` to pool normalized fish observations directly for the cohort plot and within experiments for the grid.
- What changed: Differencing and the pooling-mode switch were removed; first episodes are included, category-specific invalid-SI exclusion remains active, SEM is calculated across fish, and two newly named normalized-progression PDFs replace the old increase outputs in notebook code.
- Rerun implications: No processing rerun is required; rerun the normalized calculation and plotting cells to create `si_normalized_progression_2026_2h.pdf` and `si_normalized_progression_2026_2h_by_experiment.pdf`. Previously generated increase PDFs are not deleted.
- Validation performed: Parsed and compiled all notebook cells and smoke-tested per-fish 100% maxima, category-specific exclusions, retained negative normalized observations, first-episode inclusion, pooled means and SEM, nine experiment panels, labels, and new PDF targets with synthetic data.

### 2026-07-19 - Category-specific invalid-SI exclusion

- Slice goal: Allow SI-increase plots to run when individual fish have no positive SI maximum for one or both stimulus categories.
- Passes completed: Replaced the fatal maximum-SI validation in `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` with an auditable fish/type exclusion table and category-specific filtering.
- What changed: Fish/type combinations with nonfinite or nonpositive maximum SI are reported and removed before normalization; a fish remains eligible in its other stimulus category when that maximum is valid.
- Rerun implications: No processing rerun is required; rerun the SI-increase calculation and plot cells to regenerate the existing PDFs.
- Validation performed: Parsed and compiled the notebook and smoke-tested one-category exclusion, two-category exclusion, retained valid maxima, both pooling modes, pooled sample counts, and aggregate/grid rendering with synthetic data.

### 2026-07-19 - Fish-pooled SI increase modes

- Slice goal: Match the SI progression hierarchy by pooling fish in the episode-to-episode SI increase plots while retaining an alternate pooled-means calculation.
- Passes completed: Revised `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` with a `SI_INCREASE_POOLING_MODE` switch, per-fish/per-stimulus maxima, pooled fish-change summaries, and pooled-normalized-mean summaries.
- What changed: The default `fish_changes` mode pools normalized fish-level changes and shows SEM across fish; `pooled_means` differences pooled normalized SI and omits uncertainty bands. Experiment-balanced aggregation and experiment/type denominators were removed from this analysis.
- Rerun implications: No processing rerun is required; rerun the revised episode-to-episode cells after `df_plot` is loaded to overwrite the existing PDFs.
- Validation performed: Parsed and compiled all notebook cells and smoke-tested unequal experiment sizes, distinct fish/type maxima, pooled weighting, both switch modes, SEM-band behavior, invalid settings and maxima, first-occurrence omission, time alignment, nine panels, labels, and PDF targets with synthetic data.

### 2026-07-18 - Stimulus-specific SI maximum normalization

- Slice goal: Express episode-to-episode SI changes relative to the maximum SI reached separately for continuous and bout-like stimuli within each experiment.
- Passes completed: Revised the episode-to-episode section of `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` to derive experiment/type maxima from animal-averaged episode SI and use them as distinct denominators.
- What changed: Each animal-level same-type SI difference is now reported as a percentage of its experiment and stimulus type's maximum mean SI; the overall and experiment-grid plots retain their existing aggregation and output filenames.
- Rerun implications: No processing rerun is required; rerun the revised episode-to-episode cells after `df_plot` is loaded to overwrite the two PDFs.
- Validation performed: Parsed and compiled the notebook and smoke-tested type-specific denominators, normalized changes, invalid-maximum rejection, unequal experiment sizes, equal experiment weighting, time alignment, labels, nine panels, and PDF targets with synthetic data.

### 2026-07-17 - Episode-to-episode SI increase plots

- Slice goal: Show how SI changes between successive same-type 5-minute episodes during the first 2 hours, both across the cohort and within each experiment.
- Passes completed: Extended `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb` with animal-level percentage-point changes, equal-experiment aggregation, an overall plot, and a nine-panel experiment grid.
- What changed: Continuous (`01k01f`) and bout-like (`02k20f`) changes are aligned to the current episode start, with the first occurrence of each type omitted; bands show SEM across experiments in the overall plot and across animals in each experiment panel.
- Rerun implications: No processing rerun is required; rerun the new cells after `df_plot` is loaded to create the two new PDFs in the existing output directory.
- Validation performed: Parsed the notebook JSON, compiled all 14 code cells, and smoke-tested unequal experiment sizes, same-animal/same-type differencing, equal experiment weighting, first-occurrence omission, time alignment, both rendered figures, nine grid panels, labels, axis limits, and PDF targets with synthetic data.

### 2026-07-16 - Selection-line mortality histograms by age

- Slice goal: Compare mortality-record counts across one-month age classes for selection-line fish.
- Passes completed: Extended `temp/fish_analyzes.ipynb` with fish-move filtering and three age histograms for all matching mortality records, overall-bad-shape sacrifices, and found-dead records.
- What changed: The analysis reads the row-5 worksheet header, keeps lines containing `sel`, treats each retained row as one observation, and uses the crossing analysis `AGE_BIN_MONTHS` setting with shared age-bin edges across all three plots.
- Rerun implications: Rerun the loading/settings and mortality cells after changing `AGE_BIN_MONTHS`, `SELECTION_LINE_TOKEN`, or `fish_database.xlsx`.
- Validation performed: Parsed and compiled all 10 notebook code cells; executed the new section against the real workbook; confirmed 320 usable records (112 overall-bad-shape sacrifices and 208 found-dead records), zero invalid ages, shared one-month bins from 0 through 18 months, histogram totals matching the filtered rows, and readable rendered titles, axes, and bin labels for all three plots.

### 2026-07-16 - Age-bin failure stacks by selection line

- Slice goal: Refocus the stacked age-bin histogram from successful crossings to failed crossings.
- Passes completed: Updated `temp/fish_analyzes.ipynb` to summarize failure proportions, stack each line's contribution to failures, and retain total-attempt sample-size labels.
- What changed: Each segment is the line's failed crossings divided by all crossings in that age bin, so the full stack equals the bin's overall failure proportion; bins with attempts but no failures remain labeled empty slots.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after changing the bin width, crossing filters, or source workbook.
- Validation performed: Parsed and compiled all notebook cells; executed the real workbook data with 126 eligible crosses, 27 failures across 11 lines, 12 age bins, and zero age/line ambiguities; confirmed stacks equal overall failure proportions, failure and success proportions are complementary, total-attempt labels are correct, three zero-failure bins remain visible, 30-day and 60-day widths work, ambiguous multi-line crosses are excluded, invalid widths fail, and the rendered plot and legend are readable.

### 2026-07-16 - Age-bin success stacks by selection line

- Slice goal: Keep age-bin bar heights equal to overall incross success while showing each selection line's contribution within the bars.
- Passes completed: Updated the age-bin plot in `temp/fish_analyzes.ipynb` to retain one unambiguous line per cross, calculate line-specific success contributions, draw stacked colors, and label total cross attempts.
- What changed: Each segment is the line's successful crossings divided by all crossings in that age bin, so the stack still equals the bin's overall success proportion; ambiguous multi-age or multi-line crosses are excluded and reported.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after changing the bin width, crossing filters, or source workbook.
- Validation performed: Parsed and compiled all notebook cells; executed the real workbook data with 126 eligible crosses across 26 lines and zero age/line ambiguities; confirmed 23 lines with successes appear in the stack, stack totals equal overall success proportions, labels equal total attempts, 30-day and 60-day widths work, zero-success bins retain labeled empty slots, multi-line crosses are excluded, invalid widths fail, and the rendered plot and legend are readable.

### 2026-07-16 - Age-binned incross success plot

- Slice goal: Summarize crossing success as a function of fish age without splitting the result by selection line.
- Passes completed: Extended `temp/fish_analyzes.ipynb` with a user-editable 30-day month bin width, cross-level age reduction, a bin summary table, and a histogram-style proportion plot.
- What changed: The new plot pools the existing filtered selection-line cohort, counts each cross once, excludes crosses with multiple distinct known parent ages, and omits empty age classes rather than treating them as unsuccessful.
- Rerun implications: Rerun the settings, crossing-history, and age-bin cells after changing `AGE_BIN_MONTHS`, another crossing filter, or the source workbook.
- Validation performed: Parsed and compiled all notebook cells; executed the real workbook data through the new plot; confirmed 126 eligible unique crosses, zero multi-age exclusions, correct per-bin proportions, 30-day and 60-day widths, rejection of a zero-width setting, and readable rendered axes without line-based encoding.

### 2026-07-16 - Incross success barplot by selection line

- Slice goal: Plot crossing success percentages per selection line while excluding outcrosses.
- Passes completed: Extended `temp/fish_analyzes.ipynb` with a seaborn barplot and line-level success summary.
- What changed: The plot uses the editable `CROSSING_TYPE_FILTER` and `SELECTION_LINE_TOKEN` settings, maps `sel1`/`sel3` families including `TLSel GC` and `TL sel mid`, orders lines by family, and treats `0 (only males)` as an unsuccessful crossing while excluding missing success values.
- Rerun implications: Rerun the settings, filtering, and plotting cells after changing either filter setting or the source workbook.
- Validation performed: Compiled all notebook code cells, rendered the plot with a noninteractive Matplotlib backend, confirmed the default incross run yields 143 filtered records and 28 line summaries, and confirmed changing the type setting to `outcross` changes the result to 6 records and 4 summaries.

### 2026-07-16 - Fish database loading notebook

- Slice goal: Add a temporary notebook entrypoint for inspecting the two multi-sheet fish-information workbooks.
- Passes completed: Added `temp/fish_analyzes.ipynb` with explicit workbook paths, all-sheet loading, missing-file validation, a worksheet shape summary, editable selection-line and crossing-type filters, and a filtered crossing-efficiency table.
- What changed: The notebook loads `crosses.xlsx` and `fish_database.xlsx` into nested dictionaries of pandas DataFrames, reads the four-row-header `Crosses efficiency` sheet, filters rows whose male or female strain contains the configured selection token and whose crossing type matches the configured value, and summarizes selection lines with their tank IDs.
- Rerun implications: Rerun the loading cell whenever either source workbook changes.
- Validation performed: Parsed and syntax-checked the notebook, then executed the loading cell against both current workbooks.

### 2026-07-15 - Multi-experiment LoomAnalysisCR notebook

- Slice goal: Extend the legacy loom-response analyses from one recording to the 2026 shoaling-selection cohort while keeping experiments as equally weighted replicates.
- Passes completed: Added `Analyses/LoomAnalysisCR.ipynb`; extended `functions/joh_loom_helpers.py` with metadata selection, condition-trial alignment, compact extraction, settings-validated per-experiment caches, and cross-experiment summaries.
- What changed: The notebook dynamically selects the current eight 2026 experiments and presents legacy center-distance traces, legacy baseline/response metrics, trial-level velocity traces, trial maximum velocity versus experiment time, and trial maximum velocity by loom side. Dot-plot cells switch between experiment points and aligned condition/trial averages; trace plots always average experiments.
- Rerun implications: The first run reads each selected raw `PositionTxt` recording and creates notebook-specific `csv.gz` caches plus JSON sidecars. Later runs reuse matching caches; set `FORCE_REPROCESS_RAW_DATA=True` to rebuild them deliberately.
- Validation performed: Parsed and compiled all notebook cells, confirmed the live metadata filter selects eight 35-animal experiments with no missing raw paths, and smoke-tested unequal experiment weighting, trial alignment, cache reuse/invalidation, all five plots, both point modes, and averaged point counts with synthetic data. Full raw-recording processing was not run because continued NAS cache access was not authorized.

### 2026-07-15 - Looming animal-response workflow registration

- Slice goal: Register the new `CLfull...` loom-stimulus notebook and reusable helper module in the agentic documentation.
- Passes completed: Added `Analyses/LoomingAnimalResponseAnalysis.ipynb` and `functions/joh_loom_helpers.py` to the notebook stage map and symbol index.
- What changed: Documented the raw embedded-stimulus input, fixed epFrame-150 legacy alignment, no-L/R-mirroring convention, notebook analysis scope, helper ownership, and absence of canonical output files.
- Rerun implications: No analysis rerun required; future agents should inspect the stage map and `functions/joh_loom_helpers.py` before changing loom workflow behavior.
- Validation performed: Parsed the notebook structure and reviewed its headings/imports, inspected all helper entrypoints, and confirmed the documentation references the repository paths and symbols.

### 2026-07-09 - Trajectory heatmap time occupancy

- Slice goal: Change the trajectory-grid heatmap colors from distance traveled per bin to time spent per bin.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: The 5x7 heatmap now bins finite sampled x/y positions with `sample_seconds` weights, so colors show seconds spent per spatial bin while subplot labels and movement metrics still report distance-based speed and zMAD.
- Rerun implications: No pipeline rerun required; rerun the movement heatmap/metrics cell to refresh the displayed and saved PNG/PDF/metrics outputs.
- Validation performed: Parsed the notebook JSON, compiled notebook code cells with Python `ast`, and smoke-tested the heatmap cell on synthetic 35-fish trajectories to confirm stationary occupancy, seconds/bin labels, speed labels, and speed-metric columns.

### 2026-07-09 - Speed distribution zMAD guides

- Slice goal: Replace percentile guides in the trajectory-grid speed distribution plot with robust outlier thresholds.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: The final speed-distribution cell now computes the mean speeds corresponding to movement `zMAD = +3` and `zMAD = -3` from `distance_per_min`, converts them to mm/s, and draws those values as horizontal dotted guide lines instead of 95th-percentile markers.
- Rerun implications: No pipeline rerun required; rerun the movement heatmap/metrics cell and final speed-distribution cell to refresh the displayed and saved figure.
- Validation performed: Parsed the notebook JSON, confirmed stale speed-cell percentile references are absent, and syntax/smoke-checked the edited speed plot cell.

### 2026-07-09 - Seaborn speed distribution cleanup

- Slice goal: Use seaborn to simplify the individual speed distribution plot in the trajectory-grid notebook.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: Added `seaborn` to the notebook imports, replaced the manual jittered scatter/color loop with `sns.stripplot`, retained Matplotlib percentile annotations, and moved genotype sample counts onto the x-axis labels.
- Rerun implications: No pipeline rerun required; rerun the import/settings cell and final speed-distribution cell after `movement_metrics` exists.
- Validation performed: Parsed the notebook JSON, compiled all code cells with Python `ast`, and smoke-executed the speed plot cell on synthetic `movement_metrics` with saving disabled.

### 2026-07-09 - Trajectory-grid readability refactor

- Slice goal: Make the trajectory-grid notebook easier to read by keeping plot logic inside the plot cells.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: Reduced the `## Helpers` code cell to the reused filename helper, inlined one-use trajectory loading, metadata mapping, trajectory-grid plotting, movement-heatmap plotting, and speed-plot logic into their owning notebook cells, and added step-by-step comments for human readers.
- Rerun implications: No data rerun required; rerun the notebook cells to refresh displayed and saved figures.
- Validation performed: Parsed the notebook JSON, compiled all code cells with Python `ast`, confirmed the removed one-use helper names are absent, and smoke-rendered the trajectory grid, movement heatmap grid, and speed plot on synthetic 35-fish data with saving disabled.

### 2026-07-09 - Movement heatmap colorbar spacing

- Slice goal: Move the movement heatmap grid colorbar outside the subplot grid.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: The heatmap grid now reserves a right margin and draws the shared colorbar in a dedicated outer figure axis instead of letting it consume or overlap subplot space.
- Rerun implications: No data rerun required; rerun the movement heatmap cell to refresh the displayed and saved PNG/PDF.
- Validation performed: Parsed the notebook JSON, compiled all code cells with Python `ast`, and smoke-rendered the heatmap helper to confirm the colorbar axis sits outside the grid axes.

### 2026-07-09 - Individual speed distribution plot

- Slice goal: Add a notebook-local plot of individual speed distribution colored by fish genotype.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: Appended a final inline code cell that uses the previous cell's `movement_metrics` table, plots one dot per fish grouped and colored by genotype, draws genotype-specific and overall 95th-percentile reference lines, and optionally saves PNG/PDF plus a speed table when `SAVE_FIGURES=True`.
- Rerun implications: No pipeline rerun required; rerun the movement heatmap/metrics cell and the final speed-distribution cell to refresh the figure for a selected experiment/window.
- Validation performed: Parsed the notebook JSON, compiled all code cells with Python `ast`, and smoke-rendered the new cell on dummy 35-fish trajectory data with a noninteractive Matplotlib backend.

### 2026-07-09 - Movement heatmap comments

- Slice goal: Explain the trajectory movement heatmap analysis and its outlier metrics in-place.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: Added notebook markdown and code comments defining the movement-density binning, shared color scale, `movement_z_mad` formula, `movement_z_mad_abs` ranking, and all displayed metrics-table columns.
- Rerun implications: Documentation/comment-only change; rerun the movement heatmap cell only to view the clarified notebook text alongside refreshed outputs.
- Validation performed: Parsed the notebook JSON and compiled all code cells with Python `ast`.

### 2026-07-09 - Trajectory movement heatmap grid

- Slice goal: Add an absolute movement-density heatmap version of the 5x7 trajectory grid for outlier individual screening.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: The notebook now bins frame-to-frame path length into shared-scale per-fish arena heatmaps, annotates each subplot with distance-per-minute and robust MAD z-score, displays a movement outlier table, and saves PNG/PDF heatmaps plus a metrics CSV.
- Rerun implications: No raw-data or summary rerun required; rerun the trajectory-grid notebook cells after loading a trajectory window to refresh the displayed heatmap and saved movement metrics.
- Validation performed: Parsed the notebook JSON, checked all notebook code cells with Python `ast`, and smoke-rendered the movement heatmap helper on dummy 35-fish trajectories with a deliberately high-movement outlier.

### 2026-07-07 - Trajectory-grid genotype labels

- Slice goal: Add each fish's genotype to the 5x7 raw trajectory grid.
- Passes completed: Updated `Analyses/ShoalingTrajectoryGrid_2026.ipynb`.
- What changed: The notebook now loads the selection animal metadata workbook, maps plotted animal IDs to `AllAn.genotype`, prints the genotype vector, and adds a genotype line to each fish subplot title.
- Rerun implications: No raw-data or summary rerun required; rerun the trajectory-grid notebook cells to refresh the displayed and saved PNG/PDF.
- Validation performed: Parsed the notebook JSON, checked all notebook code cells with Python `ast`, and smoke-rendered the grid helper on dummy trajectories with genotype labels.

### 2026-07-07 - Fast trajectory-grid notebook

- Slice goal: Add an independent, frequently runnable 5x7 raw trajectory grid for selected 35-fish experiments.
- Passes completed: Added `Analyses/ShoalingTrajectoryGrid_2026.ipynb` and registered it in the notebook stage map.
- What changed: The notebook uses the slim selection `processingSettings.csv` as an experiment index, reads only requested raw x/y columns for a selected time window, caches parsed windows as compressed `.npz`, and saves 5x7 trajectory grids as PNG/PDF.
- Rerun implications: No summary or neighborhood-map rerun required; rerun the new notebook after editing experiment selection, time window, stride, or cache controls.
- Validation performed: Parsed the notebook JSON and checked all notebook code cells with Python `ast`.

### 2026-07-07 - Neighborhood-map alternative grid row spacing

- Slice goal: Prevent overlapping y-axis labels in the optional alternative-grouping neighborhood-map grids.
- Passes completed: Updated `Analyses/ShoalingNeighborhoodMaps_2026.ipynb`.
- What changed: `plot_group_summary` now accepts optional row padding for tight layout, and the final optional alternative-grouping cell uses larger row padding for the 24-row grid plots.
- Rerun implications: No data rerun required; rerun the grouped-map plotting cells to refresh the displayed and saved PDFs.
- Validation performed: Parsed the notebook JSON, checked all notebook code cells with Python `ast` after skipping IPython magic lines, and smoke-rendered `plot_group_summary` on dummy map data with `row_h_pad=2.8` while intercepting PDF writes.

### 2026-07-06 - Neighborhood-map notebook cache isolation

- Slice goal: Keep neighborhood-map summaries and maps in a notebook-specific processing cache instead of reusing slim-selection summary CSVs.
- Passes completed: Updated `Analyses/ShoalingNeighborhoodMaps_2026.ipynb`, `documentation_cr/neighborhood_map_analysis.md`, and `USER_GUIDE.md`.
- What changed: The notebook now points `ProcessingDir` and `processingSettings_neighborhood_maps.csv` to a `neighborhood_maps` subfolder, keeps `PROCESSING_MISSING_ONLY=True`, and removes the stale warning about slim summaries causing map skips.
- Rerun implications: First neighborhood-map processing in the new cache regenerates summary CSVs and `MapData.npy` files for selected experiments; later runs with `MissingOnly=True` skip experiments whose notebook-specific summaries already exist.
- Validation performed: Parsed notebook JSON, checked code-cell syntax, and statically checked cache paths, `PROCESSING_MISSING_ONLY`, and stale warning removal.

### 2026-07-06 - Neighborhood-map notebook

- Slice goal: Add a focused notebook and human guide for neighborhood-density map analysis using processed selection data.
- Passes completed: Added `Analyses/ShoalingNeighborhoodMaps_2026.ipynb`, documented the workflow in `documentation_cr/neighborhood_map_analysis.md`, and linked the new entrypoint from `USER_GUIDE.md` and the notebook stage map.
- What changed: The notebook can filter selected experiments by year, line, genotype, lineSet, or folder; discovers notebook-specific `*_siSummary*.csv` and `*MapData.npy` files; writes a separate map-specific processing settings CSV; loads neighbor-density maps; exports file/row/group summaries; and saves real, shifted-control, and difference map figures.
- Rerun implications: Neighborhood-map summaries and `MapData.npy` files live in the notebook-specific processing cache; selected experiments can be rerun with `SaveNeighborhoodMaps = 1`.
- Validation performed: Parsed the notebook JSON and checked all notebook code cells with Python `ast` after skipping IPython magic lines.

### 2026-06-30 - Mixed/separate start-date filter

- Slice goal: Move the mixed-vs-separate notebook experiment date filter into the top settings cell.
- Passes completed: Added `FILTER_START_DATE = '2026-01-01'` and changed metadata filtering to include experiments dated on or after that cutoff.
- What changed: `Analyses/ShoalingMixedvsSeparateDotEpisodes.ipynb` now uses `info_filtered` for processing-table construction instead of the hard-coded 2026 year filter.
- Rerun implications: Rerun the notebook from the settings and metadata cells after changing `FILTER_START_DATE`; no pipeline rerun is required just to validate the source change.
- Validation performed: Parsed the notebook JSON, checked all code-cell syntax, confirmed stale 2026 filter source patterns are gone, and confirmed `prepare_processing_table` receives `info_filtered`.

### 2026-06-29 - QTL SI correlation annotation overlap

- Slice goal: Remove the hidden/stacked label under the `R^2` annotation in the lineSet-split SI correlation grid.
- Passes completed: Added a local genotype list for the grid and offset each per-genotype `R^2` label vertically inside `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb`.
- What changed: Multiple genotype annotations no longer draw at the exact same axes position; when more than one genotype is plotted, each `R^2` label is prefixed by genotype.
- Rerun implications: No data rerun required; rerun the lineSet correlation cell to refresh the displayed output.
- Validation performed: Parsed the notebook JSON, checked all notebook code cells with Python `ast` after skipping IPython magic lines, and smoke-rendered the lineSet plotting cell on dummy two-genotype data to confirm each panel has separated `R^2` label positions.

### 2026-06-29 - QTL SI correlations by lineSet thresholds

- Slice goal: Make the lineSet-split SI correlation plot run without depending on the preceding plot cell and draw red dotted threshold guides.
- Passes completed: Removed the `previous_corr_axis_limits` dependency from `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb` and hard-coded the grid thresholds locally.
- What changed: The lineSet grid now plots shoaling index against average speed, animal size, and thigmotaxis distance; average-speed panels draw red dotted guides at `1` and `10`, and thigmotaxis panels draw red dotted guides at `10` and `30`.
- Rerun implications: No data rerun required; rerun the lineSet correlation cell to refresh the displayed output.
- Validation performed: Parsed the notebook JSON, checked all notebook code cells with Python `ast` after skipping IPython magic lines, and smoke-ran the lineSet plotting cell alone on dummy data with a noninteractive Matplotlib backend.

### 2026-06-29 - QTL lineSet correlation x-axis range

- Slice goal: Set the lineSet-split correlation plot x-axis range to `-0.2` through `1`.
- Passes completed: Added a shared `ax.set_xlim(-0.2, 1)` inside the lineSet subplot loop in `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb`.
- What changed: Each panel in the lineSet grid created after the original AvgSpeed correlation plot now uses the same fixed x-axis limits; the mistaken limit on the later genotype correlation grid was removed.
- Rerun implications: No data rerun required; rerun the affected plotting cell to refresh the display.
- Validation performed: Parsed the notebook JSON and checked all notebook code cells with Python `ast` after skipping IPython magic lines.

### 2026-06-29 - QTL speed correlations by lineSet

- Slice goal: Add a lineSet-split version of the three-panel speed correlation plot in `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb`.
- Passes completed: Inserted a new markdown/code cell immediately after the original speed-correlation plot.
- What changed: The notebook now plots one row per `lineSet` and one column for each original comparison: speed versus shoaling index, animal size, and thigmotaxis distance.
- Rerun implications: No data rerun required; rerun the original correlation cell and the new cell to refresh the displayed grid.
- Validation performed: Parsed the notebook JSON and checked all notebook code cells with Python `ast` after skipping IPython magic lines.

### 2026-06-29 - QTL single-genotype plot palette

- Slice goal: Remove the fake genotype color from cell 33 of `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb` while preserving the F2 plot color.
- Passes completed: Replaced the two-entry palette with a single F2 palette and explicit `genotype_hue_order`.
- What changed: The point and swarm plots now both use `hue_order=['F2']` and `palette={'F2': "#1F77B4"}`.
- Rerun implications: No data rerun required; rerun cell 33 to refresh the displayed plot.
- Validation performed: Parsed the notebook JSON and smoke-tested the one-genotype seaborn point/swarm palette pattern on dummy data.

### 2026-06-26 - 2h vs 4h plot color settings

- Slice goal: Add editable plot color settings to `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`.
- Passes completed: Inserted a plot color block at the top of the Settings cell and routed episode, genotype, reference-line, cutoff-line, baseline, and SEM-band colors through it.
- What changed: Progression plots now use named settings for episode palettes, one-hour markers, 24-episode cutoffs, zero baselines, and shaded error-band alpha values.
- Rerun implications: No data rerun required; rerun the setup/settings and plotting cells to refresh figures with edited colors.
- Validation performed: Parsed the notebook JSON, checked all code-cell syntax with IPython magic lines skipped, and scanned for remaining hard-coded plot color literals outside the new settings block.

### 2026-06-26 - Carryover plot color settings

- Slice goal: Add editable plot color settings to `Analyses/ShoalingCarryoverRawASD_2026.ipynb`.
- Passes completed: Inserted a plot color block at the top of the Settings cell and routed notebook-local line, SEM band, episode band, genotype, and reference-line colors through it.
- What changed: `draw_lineplot_with_sem` now resolves colors from split-specific palettes for episode and genotype plots, while unsplit traces and zero reference lines use named settings.
- Rerun implications: No data rerun required; rerun the setup/settings and plotting cells to refresh figures with edited colors.
- Validation performed: Parsed the notebook JSON, checked all code-cell syntax with IPython magic lines skipped, and smoke-tested the plotting helper on dummy unsplit, episode-split, and genotype-split summaries.

### 2026-06-25 - 2h aggregate progression plot

- Slice goal: Add an aggregate 2h-only shoaling-index progression plot to `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`.
- Passes completed: Inserted a new notebook section immediately before the existing 2h by-experiment progression grid.
- What changed: Summarized `df_plot` rows with `condition_key == '2h'` by episode type and 5-minute episode number; plotted bout and linear trajectories overlaid with SEM bands; saved `attraction_progression_2026_2h.pdf`.
- Rerun implications: No pipeline rerun required; rerun the new plot cell after summaries are loaded to generate the displayed figure and PDF.
- Validation performed: Parsed the edited notebook as JSON and checked all notebook code cells with Python `ast` after skipping IPython magic lines.

### 2026-06-24 - Carryover stale-cache warnings

- Slice goal: Make stale processed carryover data obvious when cached tables are loaded after analysis settings change.
- Passes completed: Added per-experiment cache settings sidecars, loud mismatch warnings during cached loads, and passed processed-data settings from `Analyses/ShoalingCarryoverRawASD_2026.ipynb`.
- What changed: `functions/carryover_effects.py` now writes `*_settings.json` next to each processed ASD/SI cache and warns when cached settings are missing or differ from the current processing settings.
- Rerun implications: Existing caches created before this change will warn until rebuilt once with `FORCE_REPROCESS_RAW_DATA = True`; future setting changes will warn when stale cached tables are loaded.
- Validation performed: Compiled `functions/carryover_effects.py`, parsed all carryover notebook code cells, and smoke-tested matching versus changed cache settings with monkeypatched extraction.

### 2026-06-24 - Carryover per-experiment cache loading

- Slice goal: Avoid repeating raw carryover ASD/SI extraction when processed tables already exist for an experiment.
- Passes completed: Added cache-aware carryover extraction helpers and routed `Analyses/ShoalingCarryoverRawASD_2026.ipynb` through them with a `FORCE_REPROCESS_RAW_DATA` switch.
- What changed: `functions/carryover_effects.py` now writes and loads per-experiment `asd_frame` and `si_1min` caches; the notebook also saves combined cohort-level copies for manual loading.
- Rerun implications: Existing combined `asd_frame.csv.gz` alone is not enough for skipping raw processing; after one rerun, per-experiment ASD and SI cache files allow future runs to load processed data.
- Validation performed: Compiled `functions/carryover_effects.py` and parsed all carryover notebook code cells with IPython magic lines skipped.

### 2026-06-24 - Carryover notebook-local plotting

- Slice goal: Move raw-ASD carryover plotting code into `Analyses/ShoalingCarryoverRawASD_2026.ipynb` and add a 10-minute-window 1-minute SI grid.
- Passes completed: Replaced all `functions.carryover_effects` plot calls with explicit notebook plotting code, added notebook-local summary/SEM and episode-band utilities, and added the requested Analysis 2 grid grouped by episode start time window.
- What changed: Removed carryover plot functions from `functions/carryover_effects.py`; the helper now owns data extraction/preparation only.
- Rerun implications: Rerun the notebook plotting cells after `asd_frame` and `si_1min` are loaded or regenerated.
- Validation performed: Parsed the notebook JSON, checked all notebook code-cell syntax with IPython magic lines skipped, compiled `functions/carryover_effects.py`, and searched for stale `ce.plot_*` references.

### 2026-06-24 - Carryover notebook cache and progress

- Slice goal: Add visible progress reporting and temporary output caching to the raw-ASD carryover notebook.
- Passes completed: Set the notebook to the first 24 raw episode blocks, added an `INCLUDE_ESCAPEE_FISH` switch defaulting to `False`, wrote `processingSettings.csv`, `carryover_analysis_settings.json`, and `asd_frame.csv.gz` to a carryover temp-processing folder.
- What changed: Updated `Analyses/ShoalingCarryoverRawASD_2026.ipynb`; plots still consume the in-memory `asd_frame` and `si_1min` tables.
- Rerun implications: Rerun the extraction cell to refresh the cached ASD table and settings file.
- Validation performed: Parsed the notebook JSON and checked code-cell syntax with IPython magic lines skipped.

### 2026-06-24 - Raw ASD carryover notebook

- Slice goal: Create a clean notebook for frame-level ASD and 1-minute shoaling-index carryover plots.
- Passes completed: Added `Analyses/ShoalingCarryoverRawASD_2026.ipynb` and routed reusable work through `functions/carryover_effects.py`.
- What changed: Notebook loads current selection metadata, extracts raw ASD for `01k01f` and `02k20f`, plots frame-by-frame ASD over the experiment, per-episode ASD segments, genotype splits, 1-minute SI over time, and 1-minute SI averaged by episode.
- Rerun implications: Run the new notebook against accessible raw position files; no existing summary pipeline rerun is required.
- Validation performed: Parsed the notebook JSON and checked code-cell syntax with IPython magic lines skipped.

### 2026-06-23 - 2h experiment progression plot

- Slice goal: Add a per-experiment 2h shoaling-index progression plot to `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`.
- Passes completed: Inserted a new notebook section immediately after the existing full-window progression plot.
- What changed: Summarized `df_plot` rows with `condition_key == '2h'` by experiment, episode type, and 5-minute episode number; plotted one subplot per experiment with bout and linear trajectories overlaid; saved `attraction_progression_2026_2h_by_experiment.pdf`.
- Rerun implications: No pipeline rerun required; rerun the notebook plot cells after summaries are loaded to generate the new PDF.
- Validation performed: Parsed the edited notebook as JSON and reviewed the targeted notebook diff.

### 2026-06-23 - 2h progression marker and comments

- Slice goal: Make the per-experiment 2h progression cell easier to understand and edit.
- Passes completed: Added explanatory comments throughout the plotting cell and a single editable `REFERENCE_MINUTE` setting.
- What changed: Added a red dotted vertical reference line at minute 60 to every experiment subplot.
- Rerun implications: No pipeline rerun required; rerun the notebook plot cell to refresh the displayed figure and saved PDF.
- Validation performed: Parsed the edited notebook as JSON and inspected the updated cell source.

### 2026-06-23 - Progression SD bands

- Slice goal: Use standard deviation bands instead of SEM bands for shoaling-index progression plots.
- Passes completed: Updated the full-window progression plot and the 2h per-experiment progression grid.
- What changed: Replaced `sem_si` summary columns with `sd_si=('si', 'std')` and changed shaded bands to `mean_si +/- sd_si`.
- Rerun implications: No pipeline rerun required; rerun the affected plot cells to refresh displayed outputs and saved PDFs.
- Validation performed: Parsed the edited notebook as JSON and checked executable source for remaining `sem_si` references.

### 2026-06-23 - Revert progression bands to SEM

- Slice goal: Restore standard-error bands for shoaling-index progression plots.
- Passes completed: Updated the full-window progression plot and the 2h per-experiment progression grid.
- What changed: Replaced `sd_si=('si', 'std')` with `sem_si=('si', sem)` and changed shaded bands back to `mean_si +/- sem_si`.
- Rerun implications: No pipeline rerun required; rerun the affected plot cells to refresh displayed outputs and saved PDFs.
- Validation performed: Parsed the edited notebook as JSON and checked the progression source for `sem_si` aggregation and band references.

### 2026-06-19 - Active notebook path routing

- Slice goal: Align notebook workflow routing with the current `Analyses/` notebook layout.
- Passes completed: Updated notebook router, notebook stage map, and current-state notes from stale `exampleAnalysis/` references to current `Analyses/` and `DeprecatedAnalyses/LarschAndBaier2018/` paths.
- What changed: Documentation-only route corrections for active selection, history, QTL, size, and historical figure notebooks.
- Rerun implications: No analysis rerun required.
- Validation performed: Checked notebook file locations and searched docs for remaining stale path references.

### 2026-06-19 - Selection notebook heading structure

- Slice goal: Add informative hierarchical markdown titles to the current selection and selection-history example notebooks.
- Passes completed: Inserted and refined `#`, `##`, and `###` headings in `ShoalingSelectionAnalysis_2025_slim_clean.ipynb` and `ShoalingSelectionAnalysis_2025History.ipynb`.
- What changed: Organized setup, shoaling-index analyses, metric checks, filtering, escapee analysis, selection grid, and history/cohort sections without changing notebook code cells.
- Rerun implications: Markdown-only change; no analysis rerun required.
- Validation performed: Parsed all three relevant notebooks and printed heading outlines; confirmed code-cell counts stayed stable for the edited notebooks.

### 2026-06-19 - Shared notebook loading headers

- Slice goal: Mirror the early loading/settings markdown headers from the selection-history notebook into the slim selection and QTL notebooks.
- Passes completed: Added matching headers for metadata loading, processing settings, processed summaries, episode-level summaries, and statistic helpers.
- What changed: Updated `Analyses/ShoalingSelectionAnalysis_2025_slim_clean.ipynb` and `Analyses/ShoalingQTLMappingAnalysis_2026.ipynb` without changing notebook code cells.
- Rerun implications: Markdown-only change; no analysis rerun required.
- Validation performed: Parsed the three relevant notebooks and checked early heading outlines and code-cell counts.

### 2026-06-22 - 2h vs 4h single-pass processing

- Slice goal: Avoid running the long processing stage twice in `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`.
- Passes completed: Changed the notebook to write one full-experiment processing settings CSV, run `experiment_set` only for `4h_all`, load the full summaries once, and derive the 2h window from `episode_number <= 24`.
- What changed: Removed condition-specific processing tables and downstream duplicated summary loading while preserving the existing 2h vs 4h/all plotting interface.
- Rerun implications: Full summaries may need one rerun in the `4h_all` processing folder; the 2h window is now an in-memory subset and does not need its own processing output.
- Validation performed: Parsed the edited notebook as JSON, searched for stale condition-loop references, and compiled all 10 transformed notebook code cells with the `jlsocial` Python environment.

### 2026-09-23 - QTL selection-family plot panels

- Slice goal: Separate the F2 cohort plot into directly comparable Sel1 and Sel3 panels.
- Passes completed: Reworked the targeted `ShoalingQTLMappingAnalysis_2026.ipynb` plotting cell into a shared-y two-panel figure.
- What changed: Each panel now orders cohorts by line and date, colors individual lines within family palettes, shows cohort mean +/- SD, annotates F2 counts, and overlays an equally cohort-weighted family mean; one combined line legend and both family-mean values appear at the right.
- Rerun implications: No pipeline rerun required; rerun the edited notebook cell to refresh `shoalindex_hivslo.pdf` and its displayed output.
- Validation performed: Parsed the notebook JSON, compiled the edited cell, and rendered representative synthetic Sel1/Sel3 data to verify two axes with identical `[-0.1, 0.6]` limits and the intended legend and average overlays.

### 2026-09-23 - Delta SI selection-family plot panels

- Slice goal: Add a separate F2 Delta SI comparison for Sel1 and Sel3 without replacing the existing Delta SI plot.
- Passes completed: Added a heading and a new shared-y two-panel plotting cell immediately after the original Delta SI cell in `ShoalingQTLMappingAnalysis_2026.ipynb`.
- What changed: The new figure orders cohorts by line and date, colors individual lines by family palette, shows cohort mean +/- SD and F2 counts, overlays equally cohort-weighted family means, and saves `shoalindelta_sel1_sel3.pdf`; the original plot and `shoalindelta_hivslo.pdf` remain unchanged.
- Rerun implications: No pipeline rerun required; run the existing Delta SI cell to create `delta_data`, then run the new panel cell.
- Validation performed: Parsed and compiled the new cell, confirmed adjacency and distinct output filenames, and rendered representative data with matching `[-0.15, 0.35]` y-limits.

### 2026-09-23 - Line- and date-ordered QTL QC plots

- Slice goal: Add line-grouped, chronological variants of every section 7 QC metric plot without changing the originals.
- Passes completed: Added a new subsection and five-figure generation block to `Analyses/QTLmapping/QTLQualityControl.ipynb` before section 8, then split each added figure into shared-y Sel1 and Sel3 panels.
- What changed: New figures sort experiments case-insensitively by line and then date, use the QTL mapping notebook's custom line colors, retain deterministic jitter and mean +/- SD, show metric outliers as hollow colored rings, and restrict zMAD lines to the matching family panel; outputs use `qtl_qc_<metric>_by_line_and_date` stems.
- Rerun implications: No data-pipeline rerun required; rerun through section 7 to populate `qc_table` and `qc_reference`, then run the added cell.
- Validation performed: Parsed the notebook, checked unique cell IDs and distinct output stems, compiled the added cell, rendered all five metrics from synthetic experiments, and visually inspected the custom palette, shared axes, outlier rings, and family-specific thresholds.

### 2026-09-25 - QTL post-loom family median panels

- Slice goal: Add a robust Sel1/Sel3 comparison for fish-level post-loom maximum velocity without replacing the existing mean +/- SD plot.
- Passes completed: Added a heading and shared-y two-panel cell immediately after the section 4 plot in `Analyses/QTLmapping/LoomAnalysis_QTL.ipynb`.
- What changed: The new F2 view pools setup recordings by line set, orders line sets by line and date, uses the custom QTL line colors, summarizes fish with the median +/- normal-scaled MAD, annotates counts, and overlays the median of line-set medians for each family.
- Rerun implications: No cache or extraction rerun required when `experiment_fish_max_velocity` is already available; run the existing section 4 cell before the new panel cell.
- Validation performed: Parsed and compiled the notebook cell, preserved the original mean/SD cell and outputs, and rendered representative data to verify F2 filtering, setup pooling, order, custom colors, shared axes, MAD intervals, counts, and family medians.

### 2026-09-25 - Unsided loom response family grid

- Slice goal: Add a robust Sel1/Sel3 comparison for the first two unsided `CLsemi030` response metrics without replacing their existing plots.
- Passes completed: Added a heading and a 2-by-2 family grid immediately after the original Section 5 response figure in `Analyses/QTLmapping/LoomAnalysis_QTL.ipynb`.
- What changed: The F2-only grid orders line sets by line and date, uses the custom QTL line colors, summarizes fish with the median +/- normal-scaled MAD, annotates metric-specific counts, and overlays the median of line-set medians for each family and response metric.
- Rerun implications: No cache or extraction rerun is required; run through the Section 4 family-median cell to define the shared metadata, palettes, and MAD helper, then run the original Section 5 response cell before the new grid.
- Validation performed: Parsed and compiled the notebook, verified unique cell IDs and cell placement, preserved the original response cell, and smoke-rendered representative Sel1/Sel3 data to verify row-shared axes, ordering, colors, robust summaries, counts, family medians, and the external legend.

### 2026-09-27 - Unsided loom velocity family grid

- Slice goal: Compare post-loom maximum velocity and valid post/pre velocity ratios between Sel1 and Sel3 without replacing the existing Section 5 figures.
- Passes completed: Added a heading and 2-by-2 family grid after the existing unsided pre/post analysis in `Analyses/QTLmapping/LoomAnalysis_QTL.ipynb`.
- What changed: The F2-only grid orders line sets by line and date, uses custom QTL line colors, shows median +/- normal-scaled MAD, annotates metric-specific counts, and overlays equally line-set-weighted family medians for both velocity metrics.
- Rerun implications: No cache rebuild is required when the Section 5 source tables already exist; run the existing unsided maximum-velocity and pre/post cells before the new grid.
- Validation performed: Parsed and compiled the notebook, checked unique cell IDs and placement, preserved existing plotting cells, and smoke-rendered representative data to verify row-shared axes, ordering, filtering, colors, counts, robust summaries, family medians, and the external legend.

## Template

### YYYY-MM-DD - Short label

- Slice goal:
- Passes completed:
- What changed:
- Rerun implications:
- Validation performed:

### 2026-10-12 - Consolidate inactive workflows under legacy

- Slice goal: retain seven active notebook entrypoints and consolidate inactive analyses and their exclusive dependencies in one archive.
- What changed: relocated 149 files (35 notebooks, 112 Python files, and two assets) into `legacy/`, preserving original filenames and collection hierarchy. The original `Analyses/old/`, `Analyses/others/`, `DeprecatedAnalyses/`, `playground/`, `obsolete/`, and `temp/` collections are archived. Ten archive-exclusive helpers/models and standalone/operational scripts moved with their consumers. Seven archived files received only import-path updates, including qualification of the shared geometry import in the shape modules. Added archive package initializers, README, and an original/destination hash manifest. Archived SI slowdown tests run separately; `pytest.ini` restricts default discovery to active tests. Updated the user guide, historical README paths, neighborhood-map walkthrough, workflow routes, stage/symbol/output references, and repository-layout entrypoint.
- Preservation: all seven active notebooks and 22 reachable implementation modules, their package initializers, and `scripts/sync_coding_doc.py` are byte-identical to the pre-move inventory. Notebook outputs, metadata, execution counts, source ordering, canonical output names/schemas, and prior QTL QC corrections are preserved. The other 142 relocated files are byte-identical; both edited archived notebooks retain all non-code content and only their helper imports change. Existing historical change-log entries are preserved.
- Validation performed: inventoried imports transitively with ASTs; checked 149 source/destination hashes and every relocation; parsed all 42 notebook JSON documents; compiled 193 code cells across active and edited archived notebooks; ran isolated original-order local-import smoke checks for all seven active entrypoints and imported their full 22-module closure. All 19 active dish-use/loom tests and all three archived slowdown tests passed. Five archived reusable modules imported successfully; historical shape/spline/SVG imports require unavailable dependencies or Python 2 syntax, recorded in remaining-work-video-preprocessing-gui.md. Reviewed current documentation paths and staged protected-document diffs.
- Rerun implications: active analysis reruns require no migration. Use `legacy/README.md` and `legacy/move-manifest.csv` for archived entrypoints and keep the repository root on the import path. Historical pickles may retain original module names. Real recording/NAS processing, GUI/ffmpeg execution, and historical figure reproduction were not run; no external data or caches were modified.

### 2026-10-12 - Refresh user guide after source review

- What changed: corrected the current QTL metadata configuration of `LoomAnalysisCR.ipynb`, documented the shoaling-QTL date cutoff and matched-fish `DeltaSI`, replaced the stale pass-only PDF entry with current delta figure names, expanded QC/dish-use defaults, and separated workflow prerequisites and cache controls in `USER_GUIDE.md`.
- Validation performed: checked ten documented settings/output names against notebook code, verified both loom workbook configurations, confirmed removal of the stale output name, checked all local Markdown links, and reviewed the documentation diff for whitespace errors.
- Rerun implications: documentation only; no analysis code, notebooks, generated outputs, or external data were changed. No analysis rerun is required.
