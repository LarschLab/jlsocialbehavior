# Video Preprocessing And GUI Router

Purpose: route tasks for video splitting, ffmpeg integration, ROI/background/scale detection, animal size extraction, shape parameters, and GUI wrappers.

Use this file when a task mentions `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`, `legacy/root-scripts/processingVideo.py`, `legacy/root-scripts/rotationWrapper.py`, `legacy/root-scripts/vidGui.py`, `legacy/root-scripts/VidGui2ArenaShoal.py`, `AnimalShapeParameters`, ROI files, background videos, scale files, or preprocessing scripts.

## Read order

1. `coding.md`
2. `.agents/workflows/jlsocialbehavior-router.md`
3. This router
4. `.agents/references/video-preprocessing-stage-map.md`
5. `.agents/references/canonical-outputs.md` for generated ROI, size, split-video, or preprocessing artifacts
6. `.agents/references/symbol-index.md` for helper/module owner lookup
7. Owning module or script
8. GUI or wrapper caller only for wrapper behavior

## Task routing table

| Query content | Read next | Likely owner |
| --- | --- | --- |
| ffmpeg split command, background division, split video outputs | `video-preprocessing-stage-map.md` | `legacy/root-scripts/ffmpegSplit4_module_bgDiv.py`, `legacy/root-scripts/processingVideo.py` |
| Animal size from video or cached `*_anSize.csv` | `canonical-outputs.md` | `functions/getAnimalSizeFromVideo.py`, `functions/video_functions.py`, `models/experiment.py` |
| Shape parameters, tail/body extraction, multiprocessing variant | `symbol-index.md` | `legacy/models/AnimalShapeParameters.py`, `legacy/models/AnimalShapeParameters_MP.py` |
| ROI circle selection or scale/background files | `video-preprocessing-stage-map.md` | `functions/gui_circle.py`, `functions/getMedVideo.py`, `functions/video_functions.py` |
| GUI behavior | `video-preprocessing-stage-map.md` | `legacy/root-scripts/vidGui.py`, `legacy/root-scripts/VidGui2ArenaShoal.py` |
| Calibration or pair-list helper scripts | `symbol-index.md` | `legacy/scripts/calibration.py`, `legacy/scripts/pairLists.py` |

Standalone GUI/video entrypoints and shape models are archived. Shared ROI, background, metadata, image, and animal-size helpers remain active runtime dependencies. See `legacy/README.md` for invocation and environment requirements.

## Ownership guidance

- Active video/image processing semantics belong in `functions/`; historical shape-parameter semantics live in `legacy/models/`.
- Archived standalone wrappers own CLI/GUI orchestration and user interaction.
- Do not move business logic into GUI scripts when a reusable helper already owns it.
- For generated media or visual output changes, inspect the artifact for blank frames, wrong scale, clipping, or incorrect ROI state.
- Log meaningful completed changes in `.agents/references/recent-changes-video-preprocessing-gui.md`.
- Track unresolved work in `.agents/references/remaining-work-video-preprocessing-gui.md`.
