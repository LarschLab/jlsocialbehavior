# Social Behavior Stage Map

Purpose: navigate the ordered analysis flow from experiment definitions to social behavior outputs.

Use this file for questions about stage order, writer ownership, reruns, or downstream impact.

## End-to-end stages

1. Experiment list setup
   - Entry: notebook-generated CSV or user-selected CSV.
   - Owner: `models/experiment_set.py`.
   - Key behavior: reads rows, applies `MissingOnly`, instantiates `experiment(row)`.

2. Experiment metadata and raw load
   - Owner: `models/experiment.py`, `ExperimentMeta`, `experiment.loadData`.
   - Inputs: `txtPath`, `aviPath`, `pairList`, `epiDur`, `episodes`, `ProcessingDir`, `outputDir`, ROI/scale settings.
   - Handles old Bonsai, VR tracking, idTracker-like `.npy`/CSV cases.

3. Calibration and size caches
   - Owners: `models/experiment.py`, `functions/video_functions.py`, `functions/getAnimalSizeFromVideo.py`.
   - Outputs: `*_anSize.csv`, `anSizeAll.csv` where produced by helpers.

4. Animal and pair linking
   - Owners: `models/animal.py`, `models/pair.py`, `models/AnimalTimeSeriesCollection.py`.
   - Concepts: animal identity, pair episodes, shifted controls, speed, position, inter-animal distance.

5. Episode statistics
   - Owner: `models/experiment.py`.
   - Outputs: `*_siSummary_epi<episodeDur>.csv`.
   - Columns include animal/set identifiers, partner, `si`, episode fields, speed, size, thigmotaxis, bouts, leadership, sync fields.

6. Neighborhood maps
   - Owners: `models/experiment.py`, `models/AnimalTimeSeriesCollection.py`.
   - Output: `*MapData.npy`.
   - Shape convention in writer: episode-pair rows by measure by data/shuffle by map grid.

7. Notebook aggregation and figures
   - Owners: notebooks for orchestration, `functions/notebookHelper.py` and plotting helpers for reusable work.
   - Downstream consumers should not redefine writer-stage output semantics.

## Current selection-shoaling pipeline memory

- Primary current notebook: `Analyses/Selection/ShoalingSelectionAnalysis_2025_slim_clean.ipynb`.
- The active selection/QTL notebooks describe recordings as experiments containing 35 focal animal slots; each focal animal has trajectory columns used independently, then the notebook remaps local slot indices to metadata animal IDs.
- Visual stimulus blocks are handled as 5 minute episodes in the active notebooks (`epiDur = 5`, `episodes = 24`, `readLim = 24 * 5 * 60 * 30 + 11`).
- Metadata come from Excel sheets `AllExp` and `AllAn`. Notebook cells build `processingSettings.csv` with raw trajectory paths, pair-list paths, birth dates, arena settings, processing flags, and output folders.
- `models.experiment_set.experiment_set(csvFile=..., MissingOnly=True)` reuses existing matching `*_siSummary*.csv` files and only instantiates `models.experiment.experiment` for missing summaries.
- `models.experiment.experiment` loads trajectories, computes ROI-derived pixel scaling, optionally recomputes animal size, creates one `Pair` per animal-partner episode, and writes `*_siSummary_epi<episodeDur>.csv`.
- The notebook reloads those summary CSVs from `ProcessingDir`, offsets local `animalIndex` by `i * 35`, maps through `anIDsAll`, merges genotype/line/date/setup metadata, filters to `80 < inDishTime < 240`, and averages numeric columns into `dfEpiAn`.
- `dfEpiAn_selection.csv` is written by the slim selection notebook and is consumed by `Analyses/QTLmapping/ShoalingQTLMappingAnalysis_2026.ipynb`.

## Navigation notes

- Use `canonical-outputs.md` before changing filenames or columns.
- Use `symbol-index.md` to find the narrowest class/function owner.
- Open notebooks only after checking whether the writer module owns the behavior.
- For documentation-only or pipeline-understanding tasks, do not change analysis code or notebook code without explicit user instruction.
