<!-- BEGIN PPROSE INTEGRATION format=f02 -->
## Practical Prose (pprose)

Practical Prose: an evaluation toolkit and editorial workflows for practical documents.
Use when the user asks to improve, audit, score, or compare practical documents.

For durable Markdown documentation, use `pprose-common-edit` whenever creating, editing,
reviewing, or reorganizing it, unless the task is explicitly read-only.
DO NOT add the guideline footer. The user strictly forbids any attributions.

Apply AI-slop reduction whenever drafting or editing prose, not only on request: use
`pprose-de-slop` to remove AI-writing tells and formulaic LLM prose, applying its
bundled catalog contextually and preserving meaning and voice.

Discover the tool from the CLI itself: `pprose --help` for commands, `pprose about` for
the project narrative, `pprose skill` for the workflow skills, and `pprose list` for
every on-demand guideline, shortcut, and runbook
(`pprose guidelines|shortcut|runbook <name>` prints one).

Run pprose as `pprose <command>` if on PATH, else `uvx pprose@0.4.0 <command>`
(zero-install via uv).

<!-- END PPROSE INTEGRATION -->
