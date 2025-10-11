# Liquid AI Hackathon Tokyo
- https://hackathons.liquid.ai/
- https://luma.com/we2jbbp3

October 11 + 12
9:30AM — 5:30PM

## TODO (Burndown List)
GOAL: Start with Evals, then training, then use the model!
TEAM: 1 Lead, 3-4 Agents including UI specialist
Compute Resources: As many full MI300X, H100 nodes as we need, not a problem, realistically

[x] README.md

llm-jp-eval MT Validation (see: `RESEARCH.LFM2.md`)
[x] what is MT?
   - `RESEARCH.llm-jp-eval-mt-analysis.md` (consolidated dataset + workflow notes)
[ ] Get it running
[ ] Replication runs
    [ ] Correct template, temperature, prompts
[ ] Alternative Judging (Feedback style LLM-as-a-Judge)

Other benchmarks
[ ] shisa-jp-tl-bench
[ ] chotto-eval
[ ] kiseki-eval
[ ] shaberi

Train
[x] Shisa V2.1 Chotto SFT
[ ] Shisa V2.1 Chotto DPO
[ ] Shisa V2.1 Chotto-only DPO

UI - Create Parallel Prompting Streamlit UI
[ ] YAML w/ optimal prompt (single turn support for LFM2) parameter endpoint per model
[ ] Undo/clear/rewind

Bonus: MT-Plus
   - Analyse what the llm-jp-eval MT tasks measure (sentence-level adequacy) and where they fall short (tone, politeness, persona, long-context).
   - Collate real failure examples from our runs to illustrate blind spots.
   - Findings feed into presentation and the MT-Plus spec.
   - Define complementary axes: keigo/politeness, ellipsis handling, gendered/role speech, domain tone, long-context coherence.
   - Combine llm-jp-eval metrics with our pairwise/rubric scores in a unified schema and viewer.
   - Deliverable: draft spec + prototype dashboard.
   - Evaluate a broad roster (GPT-4o, GPT-4.1, Gemini 2.5, Shisa production models, Gemma-3, Emma-3n, LFM2-305M, etc.) under MT-Plus.
   - Compare spread vs. existing Shisa ladders to validate discriminative power.
   - Summarise in comparative plots for the final deck.

## Presentation

Evaluating and improving LFM2's JA-EN Translation Quality


Hero Graphic
- GPT-4o: Wow awesome!
- Hmm...
-  0.75 Poop 

Run on our evals.
- Just open source, run data here
- shisa-jp-tl
  - dataset
  - Fixed BTL Pairwise comparison -> TL

LLM as a judge

Private Evals
Pairwise ; Fixed Quality 1-5 rubric; % Perfect Japanese

Revisit, harsh analyzing llm-jp-eval, critiquing it

What is llm-jp-eval MT?

How is it judged?

Was not published, so let's replicate.
- Range of models by strength, including the strongest Japanes models.

Not so useful for discrimination
Can we make it better?
Judge it differently? LLM as a judge
Pairwise comparison?

How do we make MT better? MT Plus

Categories

How do we make LFM2 better?
- Context

Let's process a few turns

base
shisa-v2.1-lmf2-350m
shisa-v2.1-lfm2-enjpmt-350m

-tl

enjp
shisa-v2.1-lmf2-350m
shisa-v2.1-lfm2-enjpmt-350m

-tl

versions

## PRESENTATION: Evaluating LFM2's EN+JA Translation Capabilities
Exploring Japanese translation LFM2-350M-ENJP-MT

Lately at Shisa AI, I've been focused on Japanese translation.

Japanese is pretty fascinating for translation:
- Heavily context and social relative
- Multiple formalities and forms
- Implicit communication, it is valid and common to exclude subject and object

First Replicate llm-jp-eval-mt results
- Analyze llm-jpe-eval
  - What is it 
  - Look at the data  (35K, my Japanese sucks, no one's go time for that...)
  - LLM as a judge for samples (TODO: Write) - Ultrafeedback like scoring

## Presentation Outline (Draft)
1. LiquidAI claim review + replication status.
2. Our translation benchmark results (Bradley-Terry, Chotto, Kiseki).
3. llm-jp-eval MT reproduction and comparison vs. LiquidAI chart.
4. MT benchmark audit: what it measures vs. what it misses.
5. MT-Plus: motivation, design, early findings.
6. Model sweep insights.
7. Shisa-LFM2 fine-tune outcomes and next steps.


> Keep detailed findings in the `RESEARCH.*.md` files; update this README as plans evolve or milestones are achieved.
