# Skill: summarize_text

## Description
Summarizes long text or search results into a concise version. This is useful for getting a high-level overview of complex technical workflows or multi-file search results.

## Instructions
- Condense the input text into its essential points.
- Preserve technical keywords, error codes, and specific version numbers.
- Maintain a professional and objective tone.
- If the input is a list of search results, group the summary by the primary topics discovered.

# Summarization Prompt

```prompt
Summarize only what is stated in the source text below.
Do not add, infer, or invent any information.
Do not mention inventors, authors, or people unless explicitly named in the source.
If the source is too short or unclear, respond only with: "Insufficient source content to summarize."
Keep the summary to 3 to 5 sentences maximum.

Source:
{{context}}

Summary:
```