# Symbol Index

Purpose: compact ownership index for public or workflow-called symbols.

Use this file when choosing the narrowest code owner before editing.

## Trusted implementation patterns

- `models/` owns experiment, animal, pair, time-series, and shape-analysis semantics.
- `functions/` owns reusable helpers for notebooks, plotting, image/video processing, statistics, and vector fields.
- Top-level scripts and GUIs own orchestration, command construction, and user interaction.
- Notebooks own parameter setup, stage order, and display/composition only.

## Models

- `models.experiment_set.experiment_set`: batch wrapper over experiment-definition CSV rows; owns `MissingOnly` and per-row experiment instantiation.
- `models.experiment.ExperimentMeta`: experiment metadata, file paths, defaults, ROI/scale discovery, and output directories.
- `models.experiment.experiment`: raw trajectory loading, animal/pair linking, summary/map/size output writing, overview plotting.
- `models.pair.Pair`: pair-level behavior such as inter-animal distance, shifted controls, and shoaling index.
- `models.animal.Animal`: animal-level wrapper around trajectories and derived time series.
- `models.AnimalTimeSeriesCollection.AnimalTimeSeriesCollection`: position, speed, curvature, thigmotaxis, neighborhood maps, force maps.
- `models.AnimalBoutSeriesCollection.AnimalBoutSeriesCollection`: bout-series semantics.
- `models.geometry`: `Trajectory`, `Circle`, `Region`, `Vector`, angle and distance geometry helpers.
- `models.AnimalShapeParameters.get_AnimalShapeParameters`: shape/tail/body extraction entrypoint.
- `models.AnimalShapeParameters_MP.get_AnimalShapeParameters`: multiprocessing variant of shape extraction.

## Functions

- `functions.arena_use.calculate_arena_use_evenness`: fixed expected-arena occupancy grids, probability distributions, normalized Shannon evenness, and coverage diagnostics from centered trajectories.
- `functions.notebookHelper.readExperiment`: notebook-facing experiment loading helper.
- `functions.notebookHelper.savedCsvToDf`: aggregation of saved summary CSVs.
- `functions.notebookHelper.computeExpTimeOfDay`, `cohend`, `groupCohen`, `groupPower`: notebook statistics/time helpers.
- `functions.carryover_effects`: raw animal-stimulus distance extraction, 1-minute shifted-control shoaling index, and plotting helpers for `Analyses/ShoalingCarryoverRawASD_2026.ipynb`.
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
- `functions.vector_field_analysis.VectorFieldAnalysis`: vector-field analysis orchestration.
- `functions.vector_field_analysis.calc_*`, `plot_*`: vector-field statistics and plots.
- `functions.matrixUtilities_joh`: geometry/path/math utility helpers used across modules.

## Scripts and wrappers

- `main.py`: simple batch/single-experiment entrypoint.
- `reRunSizeAnalysis.py`: size rerun wrapper.
- `ffmpegSplit4_module_bgDiv.py`: ffmpeg split/background-division wrapper.
- `processingVideo.py`, `rotationWrapper.py`: video preprocessing orchestration.
- `vidGui.py`, `VidGui2ArenaShoal.py`: GUI entrypoints.
- `scripts/pairLists.py`, `scripts/calibration.py`, `scripts/splitMultiPairTxt.py`: operational helper scripts.
