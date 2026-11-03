# AGENTS.md

Instructions for coding agents working in this repo.

## Notes vault

This project's memory lives in Sam's notes vault: the private repo
[dropbars/vault](https://github.com/dropbars/vault), also the `notes` MCP server. Read it before
starting, so work picks up where it stopped instead of drifting:

- `projects/musicgen-api-public.md`: what the project is, its status, next steps and gotchas.
- `handoffs/musicgen-api-public.md`: where the last session stopped, the next step, open questions.
- `design/musicgen-api-public.md` (when it exists): the current design and the decision log (D-0xx, with the
  why). Never reverse an Active decision without asking Sam.
- `agents/digest.md`: Sam's working rules for every repo.

This file wins where the two differ. Sessions on Sam's Mac write back to the vault by themselves.
Anywhere else, before you finish: append a dated entry to `log/musicgen-api-public.md`, add a decision row for
anything you decided, and update `handoffs/musicgen-api-public.md`.
