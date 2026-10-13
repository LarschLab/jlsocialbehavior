# Video Preprocessing Stage Map

Purpose: navigate video, ROI, scale, size, and GUI preprocessing work.

Use this file for tasks involving videos, ffmpeg, ROI definitions, background/median videos, animal size extraction, or GUI wrappers.

Shared `functions/` helpers remain in the active dependency tree. Standalone entrypoints and shape modules below are historical workflows under `legacy/`.

## End-to-end stages

1. Video and trajectory discovery
   - Owners: archived standalone wrappers and GUI scripts, plus active `models/experiment.py`.
   - Inputs: `.avi`, trajectory `.txt`/CSV/`.npy`, ROI files, pair lists.

2. ROI, background, and scale
   - Owners: `functions/gui_circle.py`, `functions/getMedVideo.py`, `functions/video_functions.py`, `models/experiment.py`.
   - Outputs: ROI definition files, background/median files, scale metadata used for `pxPmm`.

3. Video splitting and preprocessing
   - Owners: `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`, `legacy/root-scripts/processingVideo.py`, `legacy/root-scripts/rotationWrapper.py`, `legacy/scripts/tiffProcessing.py`.
   - Output semantics depend on wrapper parameters and external ffmpeg availability.

4. Animal size and shape extraction
   - Owners: `functions/getAnimalSizeFromVideo.py`, `functions/video_functions.py`, `legacy/models/AnimalShapeParameters.py`, `legacy/models/AnimalShapeParameters_MP.py`.
   - Outputs: `*_anSize.csv`, shape-parameter `.npz`/video artifacts where requested.

5. GUI orchestration
   - Owners: `legacy/root-scripts/vidGui.py`, `legacy/root-scripts/VidGui2ArenaShoal.py`, `functions/gui_circle.py`.
   - GUIs own user interaction and parameter collection, not reusable processing semantics.

## Validation notes

- For ffmpeg/video changes, prefer a small dry run or command construction check.
- For image/ROI changes, inspect generated images, scale files, or ROI tables.
- For shape extraction changes, validate on a small frame range when possible.
