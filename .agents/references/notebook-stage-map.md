# Notebook Stage Map

Purpose: route notebook and figure work without reading entire notebooks first.

Use this file for tasks involving `Analyses/*.ipynb`, `legacy/DeprecatedAnalyses/LarschAndBaier2018/*.ipynb`, figure composition, or notebook-specific reruns.

## Workflow groups

1. Historical paper reproduction
   - Entry: `legacy/DeprecatedAnalyses/LarschAndBaier2018/2018_00_GenerateAllFigures.ipynb`.
   - Related notebooks: `2018_BM_*` and `composeFigure*.ipynb`.
   - Purpose: load analysis outputs, run panel-specific statistics/plots, compose final figures.

2. Active analyses
   - Entries: `Analyses/Selection/ShoalingSelectionAnalysis_2025_slim_clean.ipynb`, `Analyses/Selection/ShoalingSelectionAnalysis_2025History.ipynb`, `Analyses/Selection/LoomAnalysisCR.ipynb`, `Analyses/QTLmapping/ShoalingQTLMappingAnalysis_2026.ipynb`, `Analyses/QTLmapping/QTLQualityControl.ipynb`, `Analyses/QTLmapping/LoomAnalysis_QTL.ipynb`, and `Analyses/Dish-useAnalysis.ipynb`.
   - Purpose: current selection and selection history, QTL mapping and quality control, multi-experiment loom responses, and dish-use summaries.
   - Inactive notebooks formerly in `Analyses/old/` and `Analyses/others/` are now in `legacy/Analyses/old/` and `legacy/Analyses/others/`. Use `legacy/README.md` and `legacy/move-manifest.csv` to find archived workflows and helpers.

3. Helper-backed notebook work
   - Owners: `functions/notebookHelper.py`, `functions/dish_use.py`, `functions/joh_loom_helpers.py`, `functions/plotFunctions_joh.py`, and `functions/paperFigureProps.py`. Archive-only carryover, slowdown, and vector-field helpers live in `legacy/functions/`.
   - Use active helpers for repeated loading, statistics, and plotting idioms; archived helpers belong to their historical workflows.
   - Dish-use analysis: `Analyses/Dish-useAnalysis.ipynb` and QTL QC use `functions.dish_use.analyze_dish_use` for sampled-frame occupancy and evenness. `grid_side_length` is cells per axis; `dish_bin_count` and `visited_bin_count` count eligible and visited bins. The dish notebook requires ROI geometry for raw loading and reads current or legacy radius fields on cache hits; QTL retains its separate trajectory-loader and cache conventions.

4. Looming animal-response analysis
   - Active entries: `Analyses/Selection/LoomAnalysisCR.ipynb`/`Analyses/QTLmapping/LoomAnalysis_QTL.ipynb` for metadata-selected multi-experiment cohorts. The single-recording `legacy/Analyses/old/LoomingAnimalResponseAnalysis.ipynb` is inactive.
   - Input: a raw primary-2026 animal trajectory file with embedded stimulus columns; the notebook derives sided closed-loop trials from `CLfull###L/R` and `CLsemi###L/R`, while unsided `CLsemi###` blocks are split into trials at each positive-size pulse.
   - Purpose: inspect trial-specific orientation, raw linear/angular velocity, trial timing, fish-level pre-loom mean and post-loom maximum velocity, actual rotated trajectories, legacy center-distance traces, and baseline/response escape metrics.
   - Legacy alignment: reusable extraction uses a fixed `epFrame=150` onset within each CL block, the legacy rotation convention, no L/R mirroring, and raw-coordinate baseline/response displacement metrics. The notebook owns the analysis settings and display composition; `functions/joh_loom_helpers.py` owns loading, trial parsing, extraction, and legacy metric semantics.
   - Multi-experiment semantics: `LoomAnalysisCR.ipynb` selects experiments dynamically from `MetaData_CR.xlsx`, reduces each recording before combining experiments, aligns repeats with a one-based condition-trial index, and treats experiments as equally weighted replicates. Its global plot switch selects either exact pooled-fish cache rows or experiment-level genotype summaries derived from `AllAn` animal IDs.

## Key outputs

- Figure files in configured `output` locations.
- Archived vector-field plots from `legacy/functions/vector_field_analysis.py`, including vector-field and attraction/thigmotaxis images.
- Downstream tables loaded from `*_siSummary_epi*.csv`, `*MapData.npy`, and size files.
- Loom notebook outputs are in-memory figures/tables; no canonical output file or summary-writer stage is registered for `LoomingAnimalResponseAnalysis.ipynb`.
- `LoomAnalysisCR.ipynb` and `LoomAnalysis_QTL.ipynb` write seven notebook-specific per-experiment `csv.gz` cache tables: five sided tables plus unsided fish-level response and maximum-velocity tables. JSON settings sidecars validate each derived cache set; these are not canonical pipeline outputs.

## Navigation notes

- Read `canonical-outputs.md` before changing notebook assumptions about generated file names or columns.
- If a notebook contains reusable transformations, move them to the owning module only when the task requires a refactor.
- For plot changes, render or inspect the resulting artifact before claiming success.
