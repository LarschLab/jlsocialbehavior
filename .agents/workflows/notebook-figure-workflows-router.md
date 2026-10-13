# Notebook Figure Workflows Router

Purpose: route tasks for notebook orchestration, figure generation, paper-style plots, and saved analysis reports.

Use this file when a task targets notebooks in `Analyses/` or `legacy/DeprecatedAnalyses/LarschAndBaier2018/`, figure panels, composed reports, QTL/cFos/selection analyses, or notebook-specific plotting.

## Read order

1. `coding.md`
2. `.agents/workflows/jlsocialbehavior-router.md`
3. This router
4. `.agents/references/notebook-stage-map.md`
5. `.agents/references/canonical-outputs.md` when notebooks consume generated files
6. `.agents/references/symbol-index.md` when identifying module owners
7. Owning `models/` or `functions/` module before editing notebook code
8. Notebook region only when orchestration or display behavior is the true owner

## Task routing table

| Query content | Read next | Likely owner |
| --- | --- | --- |
| Master 2018 figure rerun or paper figure panel | `notebook-stage-map.md` | `legacy/DeprecatedAnalyses/LarschAndBaier2018/*.ipynb`, module callers |
| Notebook data loading from saved summaries | `canonical-outputs.md` | `functions/notebookHelper.py` or writer stage |
| Plot styling, figure properties, color maps | `symbol-index.md` | `functions/paperFigureProps.py`, `functions/plotFunctions_joh.py` |
| Vector field plots and derived social maps | `symbol-index.md` | `legacy/functions/vector_field_analysis.py` |
| QTL, cFos, or selection analysis notebook changes | `notebook-stage-map.md` | `Analyses/*.ipynb` plus helper modules |
| Notebook refactor | `refactor-rules.md` | extract reusable logic to modules; keep notebook orchestration-thin |

The active set has seven notebooks; consult `repository-layout.md` for exact paths. `legacy/Analyses/old/`, `legacy/Analyses/others/`, and `legacy/DeprecatedAnalyses/` are inactive collections.

## Ownership guidance

- Notebooks own ordering, parameter setup, explicit display choices, and final composition.
- Reusable loading, statistics, transformations, and plotting helpers belong in `functions/` or `models/`.
- Do not patch notebooks to compensate for writer-stage schema or semantic bugs.
- For plot/figure edits, validate by inspecting rendered output or generated artifacts.
- Log meaningful completed changes in `.agents/references/recent-changes-notebook-figure-workflows.md`.
- Track unresolved work in `.agents/references/remaining-work-notebook-figure-workflows.md`.
