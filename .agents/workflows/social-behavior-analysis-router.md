# Social Behavior Analysis Router

Purpose: route tasks for experiment loading, pair/animal episode analysis, shoaling metrics, summaries, maps, bouts, and model-layer semantics.

Use this file when a task mentions `models/experiment_set.py`, `models/experiment.py`, pair/animal classes, `siSummary`, `MapData`, bouts, sync, leadership, thigmotaxis, inter-animal distance, or social index behavior.

## Read order

1. `coding.md`
2. `.agents/workflows/jlsocialbehavior-router.md`
3. This router
4. `.agents/references/canonical-outputs.md` for output/cache/schema tasks
5. `.agents/references/social-behavior-stage-map.md` for ordered workflow questions
6. `.agents/references/symbol-index.md` for public surface or owner lookup
7. Owning module in `models/` or `functions/`
8. Calling notebook or wrapper only if module context is insufficient

## Task routing table

| Query content | Read next | Likely owner |
| --- | --- | --- |
| CSV experiment list, batch processing, `MissingOnly` | `social-behavior-stage-map.md` | `models/experiment_set.py` |
| Metadata defaults, trajectories, episode splitting, `ProcessingDir`, `outputDir` | `social-behavior-stage-map.md` | `models/experiment.py` |
| `*_siSummary_epi*.csv` columns or schema | `canonical-outputs.md` | writer in `models/experiment.py` |
| `*MapData.npy` shape, shifted maps, neighborhood maps | `canonical-outputs.md` | `models/experiment.py`, `models/AnimalTimeSeriesCollection.py` |
| Shoaling index or inter-animal distance | `symbol-index.md` | `models/pair.py`, `models/AnimalTimeSeriesCollection.py` |
| Animal-level speed, thigmotaxis, bouts, sync, leadership | `symbol-index.md` | `models/animal.py`, `models/pair.py`, `models/AnimalTimeSeriesCollection.py` |
| Notebook reads saved summaries | `canonical-outputs.md` | writer first, notebook consumer second |
| Refactor of reusable analysis logic | `refactor-rules.md` | narrowest model/helper owner |

The active bout-duration path uses `models.pair.Pair.medBoutDur` and `models/AnimalTimeSeriesCollection.py`. The separate `legacy/models/AnimalBoutSeriesCollection.py` is archived.

## Ownership guidance

- `models/experiment_set.py` owns batch orchestration over rows of experiment definitions.
- `models/experiment.py` owns experiment metadata, raw trajectory loading, pair splitting, summary CSVs, map arrays, and animal size cache decisions.
- `models/pair.py`, `models/animal.py`, and time-series collection classes own pair/animal semantics.
- `functions/notebookHelper.py` owns notebook-friendly aggregation helpers, not writer-stage schema authority.
- Log meaningful completed changes in `.agents/references/recent-changes-social-behavior-analysis.md`.
- Track unresolved work in `.agents/references/remaining-work-social-behavior-analysis.md`.
