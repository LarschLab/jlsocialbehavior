# Remaining Work - Video Preprocessing And GUI

Track unresolved actionable work here.

## Template

### YYYY-MM-DD - Short label

- Current status:
- What remains broken:
- Remaining in-slice work:
- Next likely breakpoint:
- Blocking context:
- Validation or rerun needed:

### 2026-10-12 - Historical archive environment validation

- Current status: repository relocation is complete; archived execution checks have historical environment limitations. All active import checks and tests pass.
- What remains broken: `legacy.functions.splineTest` and `legacy.models.AnimalShapeParameters` cannot import in the current jlsocial environment because `sklearn` is unavailable; `legacy.functions.svg_stack` requires unavailable `lxml`. `legacy/models/AnimalShapeParameters_MP.py` retains its original Python 2 print syntax and cannot import under Python 3.12. The archived vector-field module emits an existing invalid-escape syntax warning.
- Remaining in-slice work: none for restructuring. If these historical workflows are requested, restore or select an appropriate environment and check their entrypoints. Do not add archive dependencies to active analyses merely to silence these checks.
- Next likely breakpoint: historical library versions, raw-data paths, GUI/ffmpeg support, or pickles with original module-qualified class names.
- Blocking context: historical analysis/runtime requirements were deliberately preserved; external package installation, real recording/NAS processing, and historical workflow modernization were outside the restructuring task.
- Validation or rerun needed: archived shape/spline/SVG import checks, then a small requested historical workflow fixture before full reproduction.
