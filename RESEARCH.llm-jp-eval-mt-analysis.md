# Machine Translation Benchmark Analysis

## Purpose & Scope
- Consolidate our understanding of the llm-jp-eval MT suite and the hackathon `run-mt.py` workflow.
- Serve as the single reference for dataset properties, prompt construction, scoring, and observed behavior.
- Anchor burndown tasks tied to MT validation, replication runs, and future metric extensions.

## Benchmark Stack
- **Inference driver** — `run-mt.py` prepares prompts, dispatches inference (supports `transformers`, `vllm`, synchronous `openai`, and the new asynchronous `openai-batch` backend), and collects scores with deterministic fallbacks (`run-mt.py:200-741`).
- **Prompt preparation** — Translation datasets expose task instructions, optional exemplars, and generation budgets. ALT injects four few-shot examples; WikiCorpus is strictly zero-shot (`run-mt.py:200-228`, `llm-jp-eval/src/llm_jp_eval/jaster/*.py`).
- **Normalization** — Outputs pass through Unicode NFKC normalization before metric computation (`llm-jp-eval/src/llm_jp_eval/utils.py:54-60`).
- **Scoring** — BLEU, BERTScore F1, and COMET WMT22 are logged per dataset; COMET is the headline metric used for category and overall averages (`run-mt.py:698-741`, `llm-jp-eval/src/llm_jp_eval/metrics/metrics.py:230-317`).

## Dataset Profiles

| Dataset | Direction | Source & Domain | Samples | Few-shot | Output cap | Avg. input chars | Avg. ref chars | Notes |
|---------|-----------|-----------------|---------|----------|------------|------------------|----------------|-------|
| `alt-e-to-j` | EN → JA | ALT Parallel Corpus (news) | 1,010 | 4 | 250 | 134 (max 378) | 59 (max 161) | Sentence-level news translation (`llm-jp-eval/src/llm_jp_eval/jaster/alt.py:46-143`). |
| `alt-j-to-e` | JA → EN | ALT Parallel Corpus | 1,010 | 4 | 500 | 59 | 134 | Mirror of `alt-e-to-j`. |
| `wikicorpus-e-to-j` | EN → JA | NICT WikiCorpus paragraphs | 35,436 | 0 | 300 | 193 (max 467) | 57 (max 212) | Zero-shot encyclopedic paragraphs (`llm-jp-eval/src/llm_jp_eval/jaster/wikicorpus.py:14-139`). |
| `wikicorpus-j-to-e` | JA → EN | NICT WikiCorpus | 35,436 | 0 | 550 | 57 | 193 | Mirror direction of WikiCorpus. |

### Dataset Details
- **ALT** — Downloads the official 2019 release, filters sentence pairs under 500 characters combined, and sets shared instructions for J↔E translation. Output caps (500 tokens J→E, 250 tokens E→J) mirror longest references. Few-shot exemplars default to four via the global eval config (`src/llm_jp_eval/jaster/alt.py`:67-143, `src/llm_jp_eval/schemas.py`:152-207).
- **WikiCorpus** — Fetches the Kyoto Wikipedia bi-text, patches XML entities, shuffles with seed 42, and enforces the same <500 character constraint. Prompts remain zero-shot with higher token caps (550 / 300) for longer paragraphs (`src/llm_jp_eval/jaster/wikicorpus.py`:32-156).
- **Licensing** — ALT is CC BY 4.0; WikiCorpus is CC BY-SA 3.0. Redistribution requires attribution, and WikiCorpus outputs inherit share-alike terms (`DATASET_en.md`:242-252).

## Prompt & Evaluation Pipeline
- Prompt templates are language-aware Jinja snippets that compose instruction, exemplars, and input before the output slot (`src/llm_jp_eval/prompts.py`:1-53, `run-mt.py:208-227`).
- By default the rendered prompt is passed as a plain text user turn with no chat wrapper (`run-mt.py:368-387`, `run-mt.py:620-707`).
- Added `--prompt-template lfm2` for models that expect Meta-style chat prompts; it injects `<|im_start|>` tags and swaps the system message to `Translate to Japanese.` or `Translate to English.` based on dataset direction (`run-mt.py:904-939`, `lfm2.j2`).
- The evaluator merges dataset metadata with global config, materialises JSON prompt packs, and writes them to disk for reuse during inference (`src/llm_jp_eval/evaluator.py`:60-162).
- Answer extraction leverages `AnswerPatternId.CUSTOM`, trimming to the first paragraph to form the candidate translation (`src/llm_jp_eval/jaster/base.py`:88-152).
- Length ceilings must be respected in generation configs to prevent truncation; align `max_new_tokens` with dataset caps.

## Metrics & Aggregation

| Metric | Implementation | Strengths | Limitations |
|--------|----------------|-----------|-------------|
| `bleu_{lang}` | SacreBLEU with language-specific tokenization (`metrics.py`:230-258) | Widely understood corpus metric; highlights n-gram omissions. | Penalises paraphrases; sensitive to tokenisation quirks (esp. Japanese). |
| `bert_score_{lang}_f1` | `bert-score` multilingual models (`metrics.py`:260-276) | Tolerant of paraphrasing; correlates better than BLEU for fluent systems. | Lexical bias remains; pretrained coverage of Japanese phenomena uneven. |
| `comet_wmt22` | COMET WMT22 DA model, CPU inference by default (`metrics.py`:290-317) | Strong alignment with WMT human judgments; reference-aware. | Trained mostly on non-Japanese pairs; ∼6 GB RAM footprint, correlation dips on JA. |

- Dataset-level metrics feed into category aggregates. The MT category averages COMET across the four datasets, which then rolls into the global `AVG` score (`src/llm_jp_eval/evaluator.py`:140-190, `eval_configs/all_datasets.yaml`:153-161).
- BLEU and BERTScore are retained for diagnostics, but leaderboard ordering follows COMET unless the config overrides the default metric.

## Observations & Considerations
- **Metric sensitivity** — COMET provides more spread than BLEU/BERTScore but still clusters in 0.6–0.85 for strong systems; bootstrap or significance testing is needed for tight races.
- **Sample size bias** — Small slices (e.g., `--max-samples 4`) produce volatile COMET means; rely on full-set runs (~73 k sentences) for trustworthy rankings.
- **Few-shot asymmetry** — ALT’s four exemplars vs. WikiCorpus zero-shot prompts can skew results toward models with stronger in-context learning.
- **Domain coverage gaps** — News and encyclopedic text dominate; conversational, technical, and long-context narratives remain under-tested.
- **Judge fidelity** — COMET may reward fluent hallucinations or penalise literal yet valid translations, particularly for Japanese tokenisation idiosyncrasies.
- **COMET GPU interplay** — Running COMET with `gpus>1` triggers PyTorch Lightning’s distributed launch, which re-executes `eval/run-mt.py` in worker ranks (`metrics/metrics.py:333-348`). Those child processes try to instantiate fresh vLLM engines, immediately exhausting VRAM (see `eval/results/shisa-ai--chotto-14b-20251007-dpo-openrlhf.log`). We now clamp COMET to a single GPU (`metrics/metrics.py:35-53`) to avoid extra processes; pinning `LLM_JP_EVAL_COMET_GPUS` to 0 or 1 still works when juggling inference/metric sharing.
- **OpenAI throughput** — The OpenAI backend fans requests across a thread pool sized by `--openai-max-concurrency` (default 20) while chunking prompts by `--batch-size` (`run-mt.py:523-543`). Increase the flag if the endpoint/quotas allow more parallelism.
- **Sampling defaults** — Generation now samples by default (`temperature=0.5`, `top_p=0.95`); pass `--no-sample` or set `--temperature 0` to fall back to greedy decoding.

## Strengths
- Broad bilingual coverage with deterministic preprocessing and prompt construction.
- Multi-metric reporting with cached assets reduces rerun friction.
- Compatible with CPU-only environments by forcing COMET to `gpus=0`.
- Clear traceability to source code for every stage of preprocessing, prompting, and scoring.

## Limitations & Improvement Ideas
- Tight COMET bands warrant confidence intervals or paired significance tests.
- Add domain-diverse corpora (e.g., JParaCrawl, TED, dialogue datasets) to address coverage gaps.
- Introduce reference-free metrics (COMETKiwi, xCOMET) or LLM-as-a-judge spot checks to catch hallucinations.
- Balance few-shot settings by offering optional exemplars for WikiCorpus or disabling them for ALT.
- Provide lighter judges for resource-constrained environments (BLEURT-style or distilled COMET).

## Practical Tips
1. Run full evaluations to avoid noisy metrics; document any subset runs with sample counts.
2. Pin COMET to CPU when GPU memory is scarce (`CUDA_VISIBLE_DEVICES=`) or confirm ~4 GB VRAM availability for GPU acceleration.
3. Use `--engine openai-batch` for cheap/large OpenAI runs; the helper uploads dataset-sized JSONL files, polls the Batch API, and reconciles `custom_id` keys when merging responses. Stick with `--engine openai` when you need immediate completions.
4. Cache datasets via `python -m llm_jp_eval.preprocess --targets alt-j-to-e ...` ahead of time to eliminate download latency.
5. Capture command history and key findings in the research logs referenced in `README.md` for smooth agent handoffs.
