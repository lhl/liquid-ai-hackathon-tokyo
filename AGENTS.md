# Liquid AI Hackathon Agent Guide

Please refer to `README.md` for all implementation details. This AGENTS.md/CLAUDE.md is specifically for ground rules, process and behavior notes.

## SUMMARY 
- Deliver on the 2-day plan to evaluate and improve LFM2's JA↔EN translation.
- Treat `README.md` as the single source of truth for schedule, scope, and burndown status.
- Capture findings in the existing `RESEARCH.*.md` notebooks as work progresses.
- Multiple Agents will be coordinating, so please be careful and sensitive to others' work. Prefer to append/patch the shared README.md file to not step over each other. Otherwise, each agent will be responsible for different aspects and the Lead dev will assist with any coordination issues.

## Shared Workflow Expectations
- **Before picking up work:** skim the latest README + research notes (git can help for tracking updates); confirm open checkboxes align with the plan.
- **During execution:** leave concise breadcrumbs (commands run, datasets touched, intermediate outputs) in the relevant `RESEARCH.*.md` file or a short scratch note linked from there.
- **After changes:** reflect status in `README.md` (check/annotate items), note deliverables or blockers here under a dated bullet if coordination changes are required.

## Coordination Hygiene
- Keep filenames ASCII and avoid destructive git actions, be *very* specific on all edits and commits.
- Flag unexpected repository changes immediately so the Lead can triage before proceeding.

## Committing Code
- ONLY commit working code, make sure that code is committed/clean before you start a new task
- Commit messages should be concise - one line summary, bullet points for changes. **CLAUDE ESPECIALLY**: do not include extra footers or BS in your commit logs
- If there is any confusion, flag the Lead for instruction
