# Recent Changes - Video Preprocessing And GUI

Append meaningful completed changes here.

## Template

### YYYY-MM-DD - Short label

- Slice goal:
- Passes completed:
- What changed:
- Rerun implications:
- Validation performed:

### 2026-10-14 - Fix animal-size frame alignment and resource cleanup

- What changed: ROI animal-size sampling now selects 400 distinct, sorted frames from `[1000, 20000)`. The shared ffmpeg extractor sorts and deduplicates requested frame numbers before video-length filtering, so extracted images map back to the original frame and coordinate rows, including repeated requests. Incomplete extraction falls back to OpenCV after removing its temporary directory.
- Resource cleanup: `getAnimalLength` opens OpenCV only for fallback or explicitly disabled ffmpeg and releases its capture/removes extracted frames in `finally`, including image-loading and processing exceptions. Removed the three debug prints from metadata lookup without changing parsing or caching.
- Validation performed: 11 focused mocked-video regressions pass; `python -m unittest discover -s tests -v` passes all 30 active tests. Covered frame/coordinate alignment, distinct sampling, video-length filtering, fallback, cleanup, and matching silent cached/uncached metadata. CRLF-aware `git diff --check` passes. All validation fixtures and the temporary Matplotlib cache stayed inside the repository; no real-recording or real-ffmpeg extraction was performed.
- Existing caveats: directly importing `models.experiment` first exposes its existing circular import; the notebook entrypoint order (`functions.notebookHelper` first) imports the size helper and downstream consumer successfully. Metadata lookup retains its existing boolean-inversion deprecation warning.
- Rerun implications: function signatures, array/output schemas, filenames, percentile calculation, metadata cache, and non-collision sampling remain unchanged. Existing animal-size caches are not automatically regenerated; recompute sizes to apply corrected extraction and distinct ROI sampling. Existing short-recording limitations remain.

### 2026-10-12 - Archive standalone preprocessing entrypoints

- What changed: moved standalone GUI/video wrappers to `legacy/root-scripts/`, operational scripts to `legacy/scripts/`, and exclusive shape/spline code to `legacy/models/` and `legacy/functions/`. Shared ROI, background, video metadata, camera correction, image processing, and size helpers remain active at existing paths. Archived shape consumers now use `legacy.*` imports and qualify shared geometry as `models.geometry`.
- Validation performed: all moved files are covered by the relocation/hash manifest; active video/size helper contents are unchanged and their notebook entrypoint imports pass. Five reusable archived modules import successfully. Shape/spline imports require unavailable scikit-learn, SVG stacking requires unavailable lxml, and the multiprocessing shape module retains its original Python 2 syntax; the vector-field module retains an existing invalid-escape syntax warning under Python 3.12. These limits are documented in the archive README and remaining-work tracker. No video, ffmpeg, GUI, or real-recording execution was performed.
- Rerun implications: run archived scripts with the repository root on Python's import path. Active animal-size and ROI/output schemas are unchanged.
