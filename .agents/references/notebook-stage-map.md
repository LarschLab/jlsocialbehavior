# Notebook Stage Map

Purpose: route notebook and figure work without reading entire notebooks first.

Use this file for tasks involving `Analyses/*.ipynb`, `DeprecatedAnalyses/LarschAndBaier2018/*.ipynb`, figure composition, or notebook-specific reruns.

## Workflow groups

1. Historical paper reproduction
   - Entry: `DeprecatedAnalyses/LarschAndBaier2018/2018_00_GenerateAllFigures.ipynb`.
   - Related notebooks: `2018_BM_*` and `composeFigure*.ipynb`.
   - Purpose: load analysis outputs, run panel-specific statistics/plots, compose final figures.

2. Current example analyses
   - Entries: `Analyses/ShoalingSelectionAnalysis_2025_slim_clean.ipynb`, `Analyses/ShoalingSelectionAnalysis_2025History.ipynb`, `Analyses/ShoalingSelection_2h_vs_4h_2026.ipynb`, `Analyses/ShoalingNeighborhoodMaps_2026.ipynb`, `Analyses/ShoalingTrajectoryGrid_2026.ipynb`, `Analyses/ShoalingCarryoverRawASD_2026.ipynb`, `Analyses/QTLmapping/ShoalingQTLMappingAnalysis_2026.ipynb`, `Analyses/QTLmapping/QTLQualityControl.ipynb`, `Analyses/LoomingAnimalResponseAnalysis.ipynb`, `Analyses/LoomAnalysisCR.ipynb`, `Analyses/LoomAnalysis_QTL.ipynb`, and `Analyses/size_data_analysis.ipynb`.
   - Purpose: selection-shoaling analysis, current data exploration, neighborhood-map summaries, raw ASD/carryover checks, QTL mapping and quality control, cFos examples, loom-response analysis, and size summaries.

3. Helper-backed notebook work
   - Owners: `functions/notebookHelper.py`, `functions/carryover_effects.py`, `functions/joh_loom_helpers.py`, `functions/plotFunctions_joh.py`, `functions/paperFigureProps.py`, `functions/vector_field_analysis.py`.
   - Use helpers for repeated loading, statistics, vector fields, and plotting idioms.

4. Looming animal-response analysis
   - Entries: `Analyses/LoomingAnimalResponseAnalysis.ipynb` for one recording and `Analyses/LoomAnalysisCR.ipynb`/`Analyses/LoomAnalysis_QTL.ipynb` for metadata-selected multi-experiment cohorts.
   - Input: a raw primary-2026 animal trajectory file with embedded stimulus columns; the notebook derives sided closed-loop trials from `CLfull###L/R` and `CLsemi###L/R`, while unsided `CLsemi###` blocks are split into trials at each positive-size pulse.
   - Purpose: inspect trial-specific orientation, raw linear/angular velocity, trial timing, fish-level pre-loom mean and post-loom maximum velocity, actual rotated trajectories, legacy center-distance traces, and baseline/response escape metrics.
   - Legacy alignment: reusable extraction uses a fixed `epFrame=150` onset within each CL block, the legacy rotation convention, no L/R mirroring, and raw-coordinate baseline/response displacement metrics. The notebook owns the analysis settings and display composition; `functions/joh_loom_helpers.py` owns loading, trial parsing, extraction, and legacy metric semantics.
   - Multi-experiment semantics: `LoomAnalysisCR.ipynb` selects experiments dynamically from `MetaData_CR.xlsx`, reduces each recording before combining experiments, aligns repeats with a one-based condition-trial index, and treats experiments as equally weighted replicates. Its global plot switch selects either exact pooled-fish cache rows or experiment-level genotype summaries derived from `AllAn` animal IDs.

## Key outputs

- Figure files in configured `output` locations.
- Saved plots from `functions/vector_field_analysis.py`, including vector-field and attraction/thigmotaxis images.
- Downstream tables loaded from `*_siSummary_epi*.csv`, `*MapData.npy`, and size files.
- Loom notebook outputs are in-memory figures/tables; no canonical output file or summary-writer stage is registered for `LoomingAnimalResponseAnalysis.ipynb`.
- `LoomAnalysisCR.ipynb` and `LoomAnalysis_QTL.ipynb` write seven notebook-specific per-experiment `csv.gz` cache tables: five sided tables plus unsided fish-level response and maximum-velocity tables. JSON settings sidecars validate each derived cache set; these are not canonical pipeline outputs.

## Navigation notes

- Read `canonical-outputs.md` before changing notebook assumptions about generated file names or columns.
- If a notebook contains reusable transformations, move them to the owning module only when the task requires a refactor.
- For plot changes, render or inspect the resulting artifact before claiming success.
