# Refactor Loop Policy

Purpose: keep agents working through one coherent ownership slice instead of stopping after a superficial edit.

Use this file for refactors, cleanups, migrations, and multi-pass bug fixes.

## Default working unit

One working unit is the smallest owner plus validation surface that can be made coherent together. Examples: one writer output and its first consumer, one model class and its smoke check, one notebook stage and its helper function.

## Required sub-pass loop

1. Identify the semantic owner.
2. Read the smallest router/reference docs and the owning code.
3. Make the narrowest edit.
4. Remove only unused code made obsolete by that edit.
5. Validate the owner and first affected consumer.
6. Update reference docs or handoff logs when behavior changed.

## Keep-going rules

- Continue within the same owner when the next failure is caused by your change.
- Continue when the same output contract remains inconsistent in the same writer.
- Continue when validation reveals a nearby in-slice issue needed for correctness.

## Valid stop conditions

- The requested behavior is implemented and validated.
- Validation is blocked by missing data, missing external tools, or unavailable environment.
- Further work would cross into a different owner not required by the request.

## Invalid stop conditions

- One function was cleaned while the same requested behavior is still broken in that owner.
- Static reasoning says it should work but no relevant check was run.
- Unresolved breakage is mentioned only in the final response and not in the remaining-work tracker.

## Handoff requirements

When stopping with unresolved actionable work, update the matching `remaining-work-*.md` file with status, what remains broken, next likely breakpoint, blocking context, and validation needed.
