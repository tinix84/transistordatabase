# Domain docs

## Layout

This is a single-context repository:

- CONTEXT.md at the repository root defines domain terminology.
- docs/adr/ holds architectural decision records.

## Before exploring

Read CONTEXT.md and ADRs relevant to the area being changed.
Also follow the existing document pointers in AGENTS.md.

If CONTEXT.md or docs/adr/ does not exist, proceed silently.
Do not create placeholders or suggest creating them upfront.
The domain-modeling skill creates them when terminology or
decisions are resolved.

## Vocabulary and decisions

Use the domain terms defined in CONTEXT.md consistently.
If a needed concept is missing, reconsider the term or note
the gap for domain-modeling.

Explicitly flag proposals that contradict an existing ADR,
identifying the ADR and explaining why it may need revisiting.
