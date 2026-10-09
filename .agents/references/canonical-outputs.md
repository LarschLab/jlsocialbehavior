# Canonical Outputs

Purpose: identify authoritative output writers and avoid downstream schema drift.

Use this file before changing filenames, columns, array shapes, cache behavior, rerun logic, or notebook assumptions about generated artifacts.

## Social behavior outputs

- `*_siSummary_epi<episodeDur>.csv`
  - Writer: `models/experiment.py`.
  - Created from per-animal/per-episode statistics.
  - Current fields include identifiers, partner, `si`, episode/time fields, age, average speed, smoothed speed, animal size, thigmotaxis, bout duration, leadership, and sync metrics.

- `*MapData.npy`
  - Writer: `models/experiment.py`.
  - Depends on `SaveNeighborhoodMaps` and `filteredMaps`.
  - Contains neighborhood, speed, and turn maps for data and shifted controls.

- `*_anSize.csv`
  - Writers: `models/experiment.py`, `functions/getAnimalSizeFromVideo.py`, `functions/video_functions.py`.
  - Represents cached animal size estimates from video frames.

- `anSizeAll.csv`
  - Documented by `models/experiment_set.py` as aggregate size output.
  - Before changing behavior, find the current writer in helpers/notebooks.

## Notebook and figure outputs

- `LarschAndBaier2018/output` style locations are described in `readme.md`.
- Notebook-generated figures are owned by the notebook stage unless the repeated plot logic lives in `functions/`.
- `functions/vector_field_analysis.py` writes named PNG outputs for vector-field, thigmotaxis, and attraction analyses.
- `Analyses/Dish-useAnalysis.ipynb` owns `occupancy_heatmap_grid_<folder>_<window>` PNG/PDF figures, `_metrics.csv`, and `_dish_evenness.csv`, plus `individual_dish_evenness_distribution_<folder>_<window>` PNG/PDF figures and `_dish_evenness_distribution.csv`. Bin-count columns remain `reachable_bin_count` and `visited_reachable_bin_count`. Its notebook-specific `.npz` trajectory caches write `dish_radius_mm`; readers also accept legacy `arena_radius_mm`, preferring the current field. Existing cache names and files are preserved. QTL QC retains its own cache schema and `spaceEvenness`/`trajectory_valid_samples` table columns.

## Video and preprocessing outputs

- ROI files are owned by `functions/gui_circle.py` and discovery logic in `models/experiment.py`.
- Background/median/scale artifacts are owned by `functions/getMedVideo.py` and `functions/video_functions.py`.
- Split videos and rotated/preprocessed media are owned by top-level wrappers and preprocessing scripts.

## Rules

- Change the writer first when output semantics are wrong.
- Verify the first downstream consumer when a writer output changes.
- Do not silently rename canonical outputs from notebooks or wrappers.
- Record meaningful output changes in the relevant workflow recent-changes log.
