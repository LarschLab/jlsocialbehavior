# Recent Changes Index

Purpose: route agents to the correct append-only change log for meaningful completed work.

Use a workflow-specific log when a change affects behavior, public workflow semantics, canonical outputs, validation expectations, ownership boundaries, or future rerun/debugging decisions.

## Logs

- Social behavior analysis: `.agents/references/recent-changes-social-behavior-analysis.md`
- Notebook figure workflows: `.agents/references/recent-changes-notebook-figure-workflows.md`
- Video preprocessing and GUI: `.agents/references/recent-changes-video-preprocessing-gui.md`

Do not log purely mechanical formatting, typo-only edits, or regenerated artifacts unless they change future agent behavior.

## 2026-07-16 - Transparent working-log guidance

- Slice goal: make command failures and their resolution visible to users during agent work.
- What changed: added baseline guidance requiring agents to report every command or tool failure, its cause or best diagnosis, its impact, and whether and how it was resolved.
- Rerun implications: future tool-driven tasks should distinguish errors from warnings and keep failed attempts visible even when later validation succeeds.
- Validation performed: regenerated `coding.md` from `setupAgents.md` and verified the generated file matches the source block.

## 2026-06-23 - Baseline code comment guidance

- Slice goal: require future agents to write human-readable comments with their code.
- What changed: added baseline guidance in `setupAgents.md` to comment intent, assumptions, non-obvious logic, and workflow-sensitive behavior while avoiding obvious restatements; regenerated `coding.md`.
- Rerun implications: future coding tasks should include concise explanatory comments where they improve human readability.
- Validation performed: regenerated `coding.md` from `setupAgents.md`.

## 2026-06-17 - Routed agent workflow setup

- Slice goal: establish repo-specific agent routing, baseline coding guidance, references, and handoff logs from `setupAgents.md`.
- Passes completed: inspected top-level docs, scripts, modules, notebooks, outputs, and existing `.agents/` structure; created compact routers and references.
- What changed: added `AGENTS.md`, `coding.md`, workflow routers, semantic references, workflow logs, remaining-work trackers, and `scripts/sync_coding_doc.py`; expanded `.gitignore`.
- Rerun implications: future agents should start from `AGENTS.md`, then `coding.md`, then `.agents/workflows/jlsocialbehavior-router.md`.
- Validation performed: regenerated `coding.md` from `setupAgents.md`; checked sync equality; compiled the sync script; listed routed files and router references.
