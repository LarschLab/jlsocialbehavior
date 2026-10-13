# Symbol Index

Purpose: compact ownership index for public or workflow-called symbols.

Use this file when choosing the narrowest code owner before editing.

## Trusted implementation patterns

- `models/` owns active experiment, animal, pair, time-series, and geometry semantics. Archived shape/bout models live in `legacy/models/`.
- `functions/` owns active notebook, plotting, image/video, and statistics helpers. Archive-only helpers live in `legacy/functions/`.
- Archived standalone scripts and GUIs own historical orchestration, command construction, and user interaction.
- Notebooks own parameter setup, stage order, and display/composition only.

## Models

- `models.experiment_set.experiment_set`: batch wrapper over experiment-definition CSV rows; owns `MissingOnly` and per-row experiment instantiation.
- `models.experiment.ExperimentMeta`: experiment metadata, file paths, defaults, ROI/scale discovery, and output directories.
- `models.experiment.experiment`: raw trajectory loading, animal/pair linking, summary/map/size output writing, overview plotting.
- `models.pair.Pair`: pair-level behavior such as inter-animal distance, shifted controls, and shoaling index.
- `models.animal.Animal`: animal-level wrapper around trajectories and derived time series.
- `models.AnimalTimeSeriesCollection.AnimalTimeSeriesCollection`: position, speed, curvature, thigmotaxis, neighborhood maps, force maps.
- `models.geometry`: `Trajectory`, `Circle`, `Region`, `Vector`, angle and distance geometry helpers.

## Functions

- `functions.dish_use.analyze_dish_use`: fixed expected-dish sampled-frame count grids, probability distributions, normalized Shannon evenness, and coverage diagnostics from centered trajectories; takes `xy_mm`, `dish_radius_mm`, and `grid_side_length` (cells per axis). Returns `dishUseResult` with `dish_bin_count` and per-fish `visited_bin_count`; used by dish-use analysis and QTL QC. Replaces the removed arena-use API.
- `functions.notebookHelper.readExperiment`: notebook-facing experiment loading helper.
- `functions.notebookHelper.savedCsvToDf`: aggregation of saved summary CSVs.
- `functions.notebookHelper.computeExpTimeOfDay`, `cohend`, `groupCohen`, `groupPower`: notebook statistics/time helpers.
- `functions.joh_loom_helpers.load_animal_file`, `embedded_stimulus_from_animal_file`, `stimulus_blocks`, `loom_trials`, `unsided_loom_trials`: raw animal-file loading, L/R-labelled loom-block parsing, and pulse-level parsing of unsided `CLsemi###` blocks.
- `functions.joh_loom_helpers.extract_loom_snippets`, `extract_legacy_loom_snippets`, `legacy_loom_response_metrics`: loom-window extraction, legacy coordinate alignment, and raw baseline/response escape metrics.
- `functions.joh_loom_helpers.select_loom_experiments`: metadata-driven selection of raw shoaling loom experiments using date mode, line, genotype, lineSet, and folder filters.
- `functions.joh_loom_helpers.normalize_genotype`, `loom_genotype_catalog`, `resolve_requested_genotypes`: normalized loom-genotype metadata and user-selection helpers.
- `functions.joh_loom_helpers.extract_trial_velocity_summary`, `trial_max_velocity_summary`: condition-trial-aligned velocity traces plus per-fish pre-loom mean and post-loom maximum-speed summaries.
- `functions.joh_loom_helpers.collect_or_load_loom_analysis_data`, `summarize_across_experiments`: per-experiment loom cache orchestration, including the editable pre-loom velocity window, and equal-experiment aggregation for the multi-experiment loom notebooks.
- `functions.video_functions.get_pixel_scaling`: video/ROI-derived scale owner.
- `functions.video_functions.extract_frames_ffmpeg`: ffmpeg frame extraction owner.
- `functions.video_functions.getAnimalLength`, `getAnimalSize`: animal size from frame/video owner.
- `functions.getAnimalSizeFromVideo.getAnimalSizeFromVideo`: video-backed animal-size extraction.
- `functions.getMedVideo.getMedVideo`: median/background video generation.
- `functions.getVideoProperties.getVideoProperties`: video metadata lookup.
- `functions.gui_circle.get_circle_rois`: interactive ROI definition.
- `functions.ImageProcessor`: reusable image segmentation, contour, skeleton, morphology, and crop helpers.
- `functions.plotFunctions_joh`: reusable plot helpers for maps and group comparisons.
- `functions.paperFigureProps.paper`: global paper-style plotting defaults.
- `functions.matrixUtilities_joh`: geometry/path/math utility helpers used across modules.

## Repository maintenance

- `scripts/sync_coding_doc.py`: regenerates `coding.md` from `setupAgents.md`.

## Archived helpers and models

- `legacy.models.AnimalBoutSeriesCollection.AnimalBoutSeriesCollection`: bout-series semantics.
- `legacy.models.AnimalShapeParameters.get_AnimalShapeParameters`: shape/tail/body extraction entrypoint.
- `legacy.models.AnimalShapeParameters_MP.get_AnimalShapeParameters`: multiprocessing variant of shape extraction.
- `legacy.functions.carryover_effects`: raw animal-stimulus distance extraction, 1-minute shifted-control shoaling index, and plotting helpers for `legacy/Analyses/others/ShoalingCarryoverRawASD_2026.ipynb`.
- `legacy.functions.vector_field_analysis.VectorFieldAnalysis`: vector-field analysis orchestration.
- `legacy.functions.vector_field_analysis.calc_*`, `plot_*`: vector-field statistics and plots.
- `legacy.functions.si_slowdown.estimate_si_slowdown`: SI slowdown inference for the archived exposure-duration notebook.

## Archived scripts and wrappers

- `legacy/root-scripts/main.py`: simple batch/single-experiment entrypoint.
- `legacy/root-scripts/reRunSizeAnalysis.py`: size rerun wrapper.
- `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`: ffmpeg split/background-division wrapper.
- `legacy/root-scripts/processingVideo.py`, `legacy/root-scripts/rotationWrapper.py`: video preprocessing orchestration.
- `legacy/root-scripts/vidGui.py`, `legacy/root-scripts/VidGui2ArenaShoal.py`: GUI entrypoints.
- `legacy/scripts/pairLists.py`, `legacy/scripts/calibration.py`, `legacy/scripts/splitMultiPairTxt.py`: operational helper scripts.
