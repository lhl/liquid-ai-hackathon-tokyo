# LFM2-350M EN↔JP Translation Claims & Validation Plan

## 1. What LiquidAI Claims
- The model card’s hero graphic positions **LFM2-350M-ENJP-MT** at roughly COMET 0.85 on llm-jp-eval MT, neck-and-neck with GPT-4o and ahead of Gemma 3 (4-bit) and Exaone Emma-3n (`README.md`:22-49). LiquidAI presentations highlight this as evidence of near frontier performance despite a 350M parameter budget.
- LiquidAI also insists the model must be prompted with strict system directives (“Translate to Japanese.” / “Translate to English.”) and low-temperature decoding (`temperature=0`) to reproduce their numbers (`README.md`:182-199). Missing these guardrails will invalidate comparisons.
- They acknowledge outstanding weaknesses in context-sensitive domains (technical jargon, emerging proper nouns, company-specific nuance) despite the high score (`README.md`:165-178). This is important when judging whether the llm-jp-eval MT average really captures production-readiness.

## 2. llm-jp-eval MT Benchmark: How Saturated?
- llm-jp-eval’s MT category averages COMET across four translation datasets—ALT and WikiCorpus in both directions—when computing leaderboard scores (`eval_configs/all_datasets.yaml`:153-161).
- ALT samples are sentence-level, 4-shot prompts with output-length caps of 250–500 characters, prioritising literal accuracy (`src/llm_jp_eval/jaster/alt.py`:67-143).
- WikiCorpus samples are longer paragraphs but still short enough (<500 combined characters) and evaluated zero-shot (`src/llm_jp_eval/jaster/wikicorpus.py`:32-156).
- Metrics are BLEU, BERTScore, and COMET-WMT22 per sample (`src/llm_jp_eval/jaster/alt.py`:117,137; `src/llm_jp_eval/jaster/wikicorpus.py`:105-139). Leaderboards default to COMET-WMT22, averaged over the four datasets (`src/llm_jp_eval/metrics/metrics.py`:230-317; `src/llm_jp_eval/evaluator.py`:140-162).
- Given ALT/WikiCorpus coverage and the emphasis on literal adequacy, near-parity with GPT-4o at ~0.85 COMET largely reflects sentence-level lexical fidelity. Contextual fluency, tone, and discourse-level consistency remain out-of-scope, so multiple strong models cluster together—hence the apparent saturation the LiquidAI chart reveals.

## 3. Replicating LiquidAI’s Numbers
### 3.1 Environment & Data Preparation
1. **Materialise evaluation datasets** with the built-in pipeline so we share identical prompts/metrics:
   ```bash
   python -m llm_jp_eval.preprocess --targets alt-j-to-e alt-e-to-j wikicorpus-j-to-e wikicorpus-e-to-j
   ```
   The dataset processors handle downloads, filtering, and JSON export (`src/llm_jp_eval/jaster/base.py`:119-152).
2. **Cache COMET-WMT22** in a local resource directory before large sweeps to avoid repeated downloads (`src/llm_jp_eval/metrics/metrics.py`:457-465).
3. **Respect LiquidAI decoding guidance**: enforce `temperature=0`, `max_new_tokens` within dataset caps, and the required system prompts (`README.md`:182-199; `src/llm_jp_eval/jaster/alt.py`:112-137).

### 3.2 Running Evaluations
1. Configure `configs/config_default.yaml` to point at the new model endpoint; the MT tasks inherit COMET, BLEU, and BERTScore metrics by default (`configs/config_default.yaml`:1-66; `eval_configs/all_datasets.yaml`:153-161).
2. Launch inference (local or HTTP) to generate responses on the evaluation prompts. Make sure request templates include the LiquidAI system string and place the source text in `<user>` turns identical to the dataset prompts (`src/llm_jp_eval/prompts.py`:1-53).
3. Execute the evaluator to score outputs and produce `*.eval-generated.json` artefacts plus post-processed COMET averages (`src/llm_jp_eval/evaluator.py`:151-240).
4. Extract COMET MT scores and compare to LiquidAI’s claimed ~0.85. Because llm-jp-eval stores per-dataset metrics, we can also inspect BLEU/BERTScore to diagnose deviations (`src/llm_jp_eval/evaluator.py`:140-162).

### 3.3 Result Viewer & Cross-Checks
- Use the existing aggregated JSON to build a small Rich/Plotly dashboard showing COMET, BLEU, and BERTScore per dataset. The evaluator already emits per-dataset metrics, so this layer is purely presentational (`src/llm_jp_eval/evaluator.py`:140-162).
- Diff the new outputs against LiquidAI’s screenshot by plotting our scores alongside GPT-4o, Gemma-3, Emma-3n, and LFM2-305M (if numbers are available). If their chart lacks raw data, request CSV/JSON from LiquidAI or digitise the figure.

## 4. Cross-Evaluation Using Existing Shisa Tooling
### 4.1 Bradley-Terry Ladder (shisa-jp-tl-bench)
- Run `run_translation_bench.sh` to generate translations for LFM2 and feed them into the built-in Bradley-Terry ladder. This script already pairs new outputs against a bank of baselines and fits a Bradley-Terry model to judge picks (`shisa-jp-tl-bench/README.md`:36-95).
- Add LFM2’s translations to `base_translations/` for reproducible comparisons, then re-fit scores to see where it lands relative to Shisa’s production models.

### 4.2 Chotto Eval Pairwise Study
- `run_evaluation.sh` in `chotto-eval` automates generation + pairwise judging between a new model and a baseline using Gemini or GPT-4 (`chotto-eval/README.md`:10-180).
- Configure Model A as LFM2 (with the LiquidAI system prompt) and Model B as a trusted reference (e.g., shisa-v2). The script records judge rationales and win/loss/tie counts for deeper qualitative inspection.

### 4.3 Kiseki Rubric-Based Scoring
- The Kiseki evaluation harness scores translations on a 1–5 scale with explicit correctness flags and category rubrics (`client-kiseki/eval/README.md`:11-194).
- Add LFM2 as a model config, generate translations on the dating dataset, and judge with `gemini-2.5-pro`. The Rich viewer summarises tone, gender alignment, and “Native JA” correctness for quick QA (`client-kiseki/eval/README.md`:155-188).

### 4.4 Harmonising Outputs
- Normalise model IDs and metadata across the three systems so we can merge Bradley-Terry scores, pairwise tallies, and rubric averages into a single comparison spreadsheet.
- Capture prompts, decoding params, and evaluation latency for auditability.

## 5. Towards an "MT-Plus" Benchmark
- **Core idea**: extend llm-jp-eval’s literal COMET metric with layered qualitative checks:
  1. Run llm-jp-eval for baseline adequacy (COMET/BLEU/BERTScore).
  2. Pipe the same translations into the Bradley-Terry ladder for relative fluency competitiveness.
  3. Add Chotto Eval’s judge rationales to surface edge cases (idioms, omitted pronouns, tone shifts).
  4. Score with Kiseki’s rubric to quantify politeness, tone, and correctness for domain-specific prompts.
- **New scoring dimensions** to add on top of COMET:
  - Keigo/politeness adherence.
  - Proper handling of ellipsis (implicit subjects/objects).
  - Gendered speech and honorific consistency.
  - Role/relationship alignment (senpai/kouhai, customer/service).
  - Register match (casual vs. formal vs. professional).
- **Implementation sketch**:
  - Create a unified result schema with fields for COMET, BLEU, BERTScore, Bradley-Terry rating, pairwise win rate, rubric averages, and qualitative tags.
  - Build a viewer that can filter by dataset, source direction, or rubric category, with sparkline comparisons over time.
  - Use open-source judge prompts to ensure reproducibility (Gemini/GPT as default, but plan for local LLM judges).

## 6. Assessing LFM2’s Readiness Beyond COMET
Even if LFM2 matches GPT-4o on llm-jp-eval COMET, several Japanese-specific quality axes remain unchecked:
- **Keigo & Politeness**: llm-jp-eval’s references rarely probe honorific gradients; the benchmark does not penalise mixing plain and polite speech inappropriately.
- **Ellipsis Recovery**: Japanese often omits subjects/objects. COMET may forgive incorrect insertions or dropped context if overall semantics look close, but downstream users will notice.
- **Gendered & Persona Speech**: The benchmark lacks scenarios where feminine/masculine or senpai/kouhai speech must be respected—crucial for production chat.
- **Domain-Specific Tone**: LiquidAI itself notes technical and company-specific nuance as open issues (`README.md`:165-178). None of the benchmark datasets stress these axes.
- **Long-Context Coherence**: ALT/WikiCorpus limit combined length to <500 characters (`src/llm_jp_eval/jaster/alt.py`:93-103; `src/llm_jp_eval/jaster/wikicorpus.py`:80-83). Real dialogues require multi-paragraph memory.

## 7. Next Actions
1. Re-run llm-jp-eval MT using LiquidAI’s prescribed prompts/params and capture COMET/BLEU/BERTScore.
2. Feed the same translations through Shisa’s Bradley-Terry, Chotto, and Kiseki workflows to gauge relative and rubric-based quality.
3. Prototype the MT-Plus aggregator + viewer to combine quantitative and qualitative metrics.
4. Expand the dataset suite with politeness-sensitive and context-heavy prompts to stress-test LFM2 before the hackathon.
