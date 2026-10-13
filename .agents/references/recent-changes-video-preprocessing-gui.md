# Recent Changes - Video Preprocessing And GUI

Append meaningful completed changes here.

## Template

### YYYY-MM-DD - Short label

- Slice goal:
- Passes completed:
- What changed:
- Rerun implications:
- Validation performed:

### 2026-10-12 - Archive standalone preprocessing entrypoints

- What changed: moved standalone GUI/video wrappers to `legacy/root-scripts/`, operational scripts to `legacy/scripts/`, and exclusive shape/spline code to `legacy/models/` and `legacy/functions/`. Shared ROI, background, video metadata, camera correction, image processing, and size helpers remain active at existing paths. Archived shape consumers now use `legacy.*` imports and qualify shared geometry as `models.geometry`.
- Validation performed: all moved files are covered by the relocation/hash manifest; active video/size helper contents are unchanged and their notebook entrypoint imports pass. Five reusable archived modules import successfully. Shape/spline imports require unavailable scikit-learn, SVG stacking requires unavailable lxml, and the multiprocessing shape module retains its original Python 2 syntax; the vector-field module retains an existing invalid-escape syntax warning under Python 3.12. These limits are documented in the archive README and remaining-work tracker. No video, ffmpeg, GUI, or real-recording execution was performed.
- Rerun implications: run archived scripts with the repository root on Python's import path. Active animal-size and ROI/output schemas are unchanged.
