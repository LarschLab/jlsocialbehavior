# Current State

Purpose: collect practical caveats and mixed-authority notes for future agents.

Use this file when repo authority is unclear or a task touches legacy/migration boundaries.

## Repo shape

- This is a Python 3.6-era scientific analysis repository with notebooks, reusable modules, wrappers, GUI tools, and historical paper reproduction assets.
- `readme.md` describes the historical Larsch and Baier 2018 analysis workflow and raw data source.
- Seven active notebooks remain under `Analyses/Selection/`, `Analyses/QTLmapping/`, and `Analyses/Dish-useAnalysis.ipynb`; see `repository-layout.md` for the exact set.
- `legacy/` consolidates inactive notebooks, historical collections, standalone scripts, and their exclusive helpers. Its move manifest records 149 relocations and original hashes.
- `scripts/sync_coding_doc.py` remains repository maintenance tooling; other operational scripts are archived.
- `.agents/` now provides routing and handoff docs; `coding.md` is the baseline coding behavior file.

## Authority notes

- `models/` and `functions/` are the authoritative reusable code layers.
- `legacy/DeprecatedAnalyses/LarschAndBaier2018/` notebooks are historical workflow surfaces and should be changed carefully.
- `Analyses/` notebooks are current analysis surfaces but should still avoid accumulating reusable logic.
- `legacy/`, including `legacy/playground/` and `legacy/obsolete/`, is outside active workflow authority unless directly targeted.
- Directories such as `Lib/`, `DLLs/`, `conda-meta/`, `Tools/`, and `Library/` look like local environment artifacts, not project source.

## Practical caveats

- Some code uses Python 2/early Python 3-era styles and old library conventions.
- Many validations may require local raw data, ffmpeg, OpenCV, GUI support, or notebook execution.
- If full validation is unavailable, run the smallest import/static/contract check possible and record what could not be verified.
- For smoke checks that import `PythonMeta`, configure console logging first: that dependency otherwise attempts to write `meta.log` inside its installed package directory. Active pipeline helpers also have existing import-order dependencies; check notebook entrypoint imports in their original order.

## Working tree note

- Preserve unrelated user changes. At setup time the worktree already had a deleted root `Copilot_FFMPEG_SETUP.md` and untracked `documentation_cr/`.
