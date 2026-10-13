# Repository Layout

Use this reference for active/legacy classification, file movement, or restoring an archived workflow. Read `refactor-rules.md` before editing.

## Active entrypoints

- `Analyses/Selection/ShoalingSelectionAnalysis_2025_slim_clean.ipynb`
- `Analyses/Selection/ShoalingSelectionAnalysis_2025History.ipynb`
- `Analyses/Selection/LoomAnalysisCR.ipynb`
- `Analyses/QTLmapping/ShoalingQTLMappingAnalysis_2026.ipynb`
- `Analyses/QTLmapping/QTLQualityControl.ipynb`
- `Analyses/QTLmapping/LoomAnalysis_QTL.ipynb`
- `Analyses/Dish-useAnalysis.ipynb`

These entrypoints reach 22 implementation modules in `functions/` and `models/`, excluding package initializers. Indirect video, ROI, background, camera-correction, image-processing, and animal-size imports are active dependencies. Keep shared import paths stable.

`scripts/sync_coding_doc.py` is repository maintenance tooling. Active tests live in `tests/`; use `python -B -m unittest discover -s tests -v`. `pytest.ini` restricts default pytest discovery to that directory.

## Archive

`legacy/README.md` explains the archive and execution requirements. `legacy/move-manifest.csv` records 149 original/destination paths and pre-/post-relocation SHA-256 hashes and content-update annotations. It covers 35 inactive notebooks, 112 Python files, and two supporting assets. Original hierarchies and filenames are preserved. Selected archived consumers received import-only changes.

- User-designated inactive collections: `legacy/Analyses/old/` and `legacy/Analyses/others/`.
- Historical reproduction: `legacy/DeprecatedAnalyses/`, including `LarschAndBaier2018/props.csv`.
- Archive-only modules: `legacy/functions/` and `legacy/models/`.
- Standalone/operational entrypoints: `legacy/root-scripts/` and `legacy/scripts/`.
- Exploratory/retired source: `legacy/playground/` and `legacy/obsolete/`.
- Other inactive notebook and tests: `legacy/temp/` and `legacy/tests/`.

Active code does not depend on `legacy/`. Archived code imports its exclusive helpers through `legacy.functions.*` and `legacy.models.*`, and shared runtime code through root `functions.*` and `models.*`. Keep the repository root on the Python import path; do not substitute `legacy/` as the code root. Historical pickles can retain old class module names; restoring their environments or regenerating them is separate from active canonical outputs.

## Future changes

Classify dependencies transitively from all active entrypoints, including nested/conditional imports and notebook execution commands. Maintenance tooling and tests are separate from notebook dependencies. Restore archived files deliberately with their consumers, update routing/symbol/output references, and validate the entrypoint. Preserve canonical output contracts and historical change-log entries.

Embedded environment directories (`Lib/`, `DLLs/`, `conda-meta/`, `Library/`, and similar) remain in place; they contain no tracked repository source in this inventory.
