# Legacy archive

This archive was created on 2026-10-12. Active entrypoints are the seven notebooks in `../Analyses/Selection/`, `../Analyses/QTLmapping/`, and `../Analyses/Dish-useAnalysis.ipynb`. Modules in the repository's root `functions/` and `models/` remain their runtime dependencies. Active analyses do not import this archive.

## Contents

| Location | Purpose |
| --- | --- |
| `Analyses/old/` | Five inactive notebook versions and exploratory analyses |
| `Analyses/others/` | Four inactive carryover, exposure-duration, episode-comparison, and size analyses |
| `DeprecatedAnalyses/` | Historical analyses, including the 2018 paper workflow and its `props.csv` |
| `functions/`, `models/` | Helpers and models outside the active notebook dependency tree |
| `root-scripts/` | Former root entrypoints for batch analysis, size reruns, video splitting, GUI inspection, and shape extraction |
| `scripts/` | Pair-list generation, calibration, trajectory-text splitting, TIFF processing, and an old multi-animal driver |
| `playground/`, `obsolete/` | Exploratory code, retired implementations, and associated assets |
| `temp/` | The former temporary analysis notebook |
| `tests/` | Tests for the archived SI slowdown helper |

`move-manifest.csv` records all 149 relocated files, their original and destination paths, the classification reason, SHA-256 hashes before and after relocation, and any content updates. Seven archived files received import-path updates; the other 142 files were moved byte for byte. Original filenames and collection hierarchies are preserved.

## Using an archived workflow

Keep the repository root on Python's import path. Archived consumers import relocated modules as `legacy.functions.*` or `legacy.models.*` and continue to import shared runtime code as `functions.*` or `models.*`. Do not add `legacy/` to the import path as a replacement repository root: its archived `functions/` and `models/` directories contain only archive-exclusive modules.

For a standalone script, start at the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python legacy/root-scripts/main.py
```

For notebooks, set their existing `codeDir`/`CODE_DIR` to the repository root or start Jupyter with that root on `PYTHONPATH`. The historical master notebook uses relative `%run` commands; run it with the working directory set to `legacy/DeprecatedAnalyses/LarschAndBaier2018/` and update `props.csv` for your data locations.

The archive preserves historical analysis settings, outputs, and metadata. Some workflows still require old Python/library versions, raw recordings, NAS paths, ffmpeg, or GUI support. Relocation does not establish that every historical workflow runs in the current environment. Stored pickles containing relocated class names may require their original environment or regeneration; the canonical active pipeline outputs retain their existing paths and schemas.

In the current `jlsocial` environment, `splineTest` and shape helpers require unavailable `scikit-learn`, `svg_stack` requires unavailable `lxml`, and `AnimalShapeParameters_MP.py` retains Python 2 syntax. These historical requirements are recorded in `.agents/references/remaining-work-video-preprocessing-gui.md`. The archived vector-field module also retains an existing invalid-escape syntax warning under Python 3.12.

## Tests

Run active tests from the repository root:

```powershell
python -B -m unittest discover -s tests -v
```

Run archived SI slowdown tests explicitly:

```powershell
python -B -m unittest discover -s legacy/tests -v
```

`pytest.ini` limits default pytest discovery to active `tests/`. Historical and experimental scripts named like tests are preserved in the archive.

## Restoring a workflow

Use the manifest to find its original location and check its shared and archived dependencies. Move the required files deliberately, update consumers and routing documentation, and validate the restored entrypoint. Moving a notebook alone does not make its dependencies active.
