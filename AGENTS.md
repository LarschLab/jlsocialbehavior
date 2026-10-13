# Agent Entrypoint

This repository uses a routed instruction system under `.agents/`. Start with the short baseline coding rules, then use the router to find the smallest workflow-specific reference before opening large scripts or notebooks.

## Required startup order

1. Open `coding.md` for baseline coding behavior.
2. Open `.agents/workflows/jlsocialbehavior-router.md`.
3. Follow its dispatch table to the relevant workflow router.
4. Read the smallest relevant reference doc under `.agents/references/`.
5. Open stage maps or `symbol-index.md` only if needed.
6. Open owning modules before large notebooks or wrapper scripts.
7. Open large notebook/script regions only when owner-module context is insufficient.

## Non-negotiable repo rules

- Do not create, edit, or erase local files outside this repository.
- Treat `coding.md` as required baseline behavior for every coding task.
- For pipeline-understanding or documentation tasks, ground every claim in repository files, say what is unclear, and do not change existing analysis code or notebook code unless the user explicitly asks for code changes.
- Prefer edits in `models/` and `functions/` over notebook or wrapper edits when fixing reusable behavior.
- Keep notebooks and wrappers orchestration-thin; do not make them new sources of business logic.
- Preserve canonical output names, schemas, stage order, and legacy variable names unless a deliberate migration requires changing them.
- Fix data semantics at the writer stage, not by compensating in downstream consumers.
- Do not treat `legacy/` or embedded environment directories as authoritative unless the user directly targets them.
- Validate after edits with the smallest relevant smoke, contract, rerun, or artifact check.
- For new analysis notebooks or helpers, add a compact entry to the relevant `.agents/references/recent-changes-*.md` log and update routing docs only when a future agent would need the new entrypoint.

## Reference files

- `.agents/workflows/jlsocialbehavior-router.md`: top-level workflow dispatch.
- `.agents/references/symbol-index.md`: public module and helper ownership index.
- `.agents/references/canonical-outputs.md`: authoritative generated tables, arrays, figures, and local artifacts.
- `.agents/references/refactor-rules.md`: repo-specific ownership and edit-scope policy.
- `.agents/references/refactor-loop-policy.md`: how to continue or stop within an ownership slice.
- `.agents/references/current-state.md`: migration state, legacy areas, and practical caveats.
- `.agents/references/repository-layout.md`: active entrypoints, archive locations, and restoration policy.
- `.agents/references/recent-changes.md`: index for workflow-specific change logs.
- `.agents/references/remaining-work.md`: index for unresolved workflow work.

## Scope note

Details live under `.agents/`. Keep this entrypoint short; add routing or semantic detail to the relevant workflow router or reference doc instead.
