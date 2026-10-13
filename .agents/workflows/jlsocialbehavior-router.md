# jlsocialbehavior Router

Purpose: dispatch future agents to the smallest useful workflow instructions for this Python and notebook analysis repository.

Use this file when any task targets repository code, notebooks, generated analysis outputs, video preprocessing tools, or agent workflow docs.

## Read this first

Always read `coding.md` before this router. Then choose one workflow profile below and follow that profile router before opening source files.

## Workflow profile dispatch

| Primary target or query content | Open first |
| --- | --- |
| Shoaling index, inter-animal distance, pair/animal episode semantics, summaries, maps, bouts, leadership, sync, experiment metadata | `.agents/workflows/social-behavior-analysis-router.md` |
| `Analyses/*.ipynb`, `legacy/DeprecatedAnalyses/LarschAndBaier2018/*.ipynb`, figure panels, composite reports, notebook plotting, QTL/cFos/selection notebook analysis | `.agents/workflows/notebook-figure-workflows-router.md` |
| Video splitting, ffmpeg wrappers, ROI/background/scale detection, animal size extraction, GUI scripts, shape parameters, preprocessing scripts | `.agents/workflows/video-preprocessing-gui-router.md` |
| Active/legacy classification, repository layout, restoring archived workflows | `.agents/references/repository-layout.md` |
| Generic refactor, ownership question, code movement, public symbol changes | `.agents/references/refactor-rules.md`, then the matching profile router |
| Canonical output filename/schema/cache/rerun confusion | `.agents/references/canonical-outputs.md`, then the writer workflow router |
| Active migration, legacy code, or unclear authority | `.agents/references/current-state.md`, then the matching profile router |

## Cross-workflow invariants

- Apply `coding.md` baseline behavior first, then layer repo-specific routing from `.agents/`.
- Prefer package/module edits over orchestration-layer edits for reusable behavior.
- Preserve canonical outputs and stage semantics unless the task is an intentional migration.
- Wrapper scripts and GUIs own wrapper behavior only; they are not business-logic authority.
- Data-table, cache, export, and figure semantics are owned by the stage that writes them.
- If a writer stage changes, verify the first downstream consumer.
- Validate every meaningful edit and update the relevant recent-changes or remaining-work reference.

## Compact scaling rule

Add a target to an existing profile when it shares entrypoints, owners, stage semantics, and validation. Create a new profile only when it needs its own stage map, semantic references, validation cadence, and handoff log.
