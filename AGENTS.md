# Liquid AI Hackathon Agent Guide

## Mission Anchor
- Deliver on the 2-day plan to evaluate and improve LFM2's JA↔EN translation.
- Treat `README.md` as the single source of truth for schedule, scope, and burndown status.
- Capture findings in the existing `RESEARCH.*.md` notebooks as work progresses.

## Burndown Snapshot (sync with README)
1. **llm-jp-eval MT validation**
   - Clarify task spec (`RESEARCH.llm-jp-eval-mt-analysis.md`, `MT-SUMMARY.md`).
   - Stand up baseline runs; replicate LiquidAI configurations (template, temperature, prompts).
   - Explore alternative judging (LLM-as-a-judge, feedback-mode).
2. **Other benchmark sweeps** — shisa-jp-tl-bench, chotto-eval, kiseki-eval, shaberi.
3. **Training iterations** — Shisa V2.1 SFT, Shisa V2.1 DPO, Chotto + Kiseki DPO.
4. **UI streamlit surface** — YAML-configured prompts per model, undo/clear/rewind affordances.
5. **Bonus: MT-Plus** — identify evaluation gaps, gather counter-examples, design unified scoring and visualisations, draft dashboard + deck content.

Update the bullets above whenever the README burndown shifts; keep the order stable so every agent knows the current focus.

## Shared Workflow Expectations
- **Before picking up work:** skim the latest README + research notes; confirm open checkboxes align with the plan.
- **During execution:** leave concise breadcrumbs (commands run, datasets touched, intermediate outputs) in the relevant `RESEARCH.*.md` file or a short scratch note linked from there.
- **After changes:** reflect status in `README.md` (check/annotate items), note deliverables or blockers here under a dated bullet if coordination changes are required.
- **Handoff rhythm:** morning (initial sync), mid-day (status bump), end-of-day (delta + next steps). Document these summaries in the README or the appropriate research log.

## Role-Focused Guidance
### Codex agents
- Use fast, traceable commands (`rg`, `sed`, targeted scripts) and prefer `apply_patch` for localized edits.
- Validate code or document changes locally when tooling permits; report skipped tests with rationale.
- When editing plans or burndown items, preserve existing context and call out assumptions inline.

### Claude agents
- Provide reasoning-first responses; surface uncertainties as explicit follow-up questions.
- When synthesizing research, cite file names and headings so Codex agents can jump directly to the source.
- Keep answers concise; redirect to README/AGENTS sections instead of duplicating large blocks unless they change.

## Coordination Hygiene
- Keep filenames ASCII and avoid destructive git actions.
- When a plan item completes or shifts, update both this guide and the README in the same session.
- Flag unexpected repository changes immediately so the team can triage before proceeding.
