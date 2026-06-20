# Skill: compare_versions

## Description

Identifies and summarises differences, updates, deprecated behaviours, and breaking
changes between multiple documentation versions or sections retrieved from the vector store.

## Instructions

- Present findings as a structured comparison. Use a Markdown table wherever the data supports it (e.g. a feature matrix across versions).
- Label each entry with its source section name, document title, or version identifier where available in the context.
- Use professional, business-appropriate language throughout:
  - "Introduced in v{version}" for new capabilities
  - "Deprecated as of v{version}" for legacy items
  - "Superseded by {alternative}" when a replacement exists
  - "Removed in v{version}" for complete removals
- Prefix breaking changes with **[Breaking]** so they are immediately visible.
- If only a single documentation version is present in the context, state explicitly:
  "Only one documentation version was identified in the retrieved context. A meaningful version comparison requires multiple distinct versions."
- Do not speculate about undocumented changes or infer version history beyond what is explicitly stated in the context.
- Adhere to the constraints in `rag_constraints.md`.
