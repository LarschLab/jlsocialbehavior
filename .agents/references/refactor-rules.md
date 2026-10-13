# Refactor Rules

Purpose: repo-specific ownership and edit-scope rules.

Use this file before moving code, extracting helpers, or changing public workflow behavior.

## Ownership rules

- Reusable analysis semantics belong in `models/` or `functions/`.
- Notebook code should stay orchestration-thin: parameters, calls, plotting display, and composition.
- Top-level scripts and GUIs own wrapper behavior, command setup, and interaction only.
- Writer stages own table schemas, cache names, array shapes, and export semantics.
- Downstream notebooks and helper consumers should not redefine writer-stage meanings.
- `legacy/`, including `legacy/obsolete/` and `legacy/playground/`, is not authority unless directly targeted.
- Embedded environment folders are dependency artifacts, not project source owners.

## Editing rules

- Fix at the narrowest owning layer.
- Preserve canonical filenames, variable names, stage order, and output schemas unless changing them is the explicit goal.
- Do not add notebook-local helpers when a reusable helper or model owner should hold the logic.
- Do not refactor adjacent legacy code just because it is nearby.
- If public function surface changes, update `symbol-index.md`.
- If stage/output ownership changes, update the relevant stage map and `canonical-outputs.md`.

## Validation rules

- Run the smallest relevant smoke, contract, or rerun check after edits.
- If a writer output changes, verify its first downstream consumer.
- If a plot, report, notebook, GUI, image, or video artifact changes, inspect the rendered artifact.
- Record completed meaningful changes in the workflow recent-changes log.
- Put unresolved actionable work in the workflow remaining-work tracker.
