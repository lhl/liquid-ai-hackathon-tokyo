# MT Evaluation Playbook

## llm-jp-eval Snapshot
- Repo path: `eval/llm-jp-eval`
- Commit: `ceb58dba00d27853671695ac967bba2fecd06326` (vendored on 2024-10-11)
- Results dir: `eval/results/` (auto-created for dataset metrics, prediction dumps, and logs).
- Default command writes dataset summaries to `results/<model>.scores.jsonl`, raw translations to `results/<model>.predictions.jsonl`, and console logs to `results/<model>.log` (override with `--scores` / `--predictions` / `--log-file` as needed).
- Filenames now include the decoding strategy (e.g. `.greedy`, `.sampled-t0_2-p0_9`); add extra context with `--run-tag timestamp` when you need fully unique artifacts per run.

## Generation Settings
- **Greedy / Liquid AI recommendation** — leave sampling disabled (`--do-sample` omitted, `--temperature 0`, `--top-p 1.0`).
- **Chotto baseline** — `--do-sample --temperature 0.2 --top-p 0.9`.
- **LFM2 chat formatting** — pass `--format lfm2` to wrap prompts with `<|im_start|>` chat tokens and enforce the required system prompts (`Translate to Japanese.` / `Translate to English.` selected per dataset direction).
- The script respects CLI overrides per backend (`transformers`, `vllm`, `openai`).
- COMET scoring now auto-detects GPU availability (respects `CUDA_VISIBLE_DEVICES`); override with `LLM_JP_EVAL_COMET_GPUS=0` for CPU or set an explicit count.
- WikiCorpus splits are skipped by default; pass `--include-wikicorpus` for full runs or `--exclude-datasets` to add/remove specific sets.
- vLLM runs default to `--vllm-gpu-memory-utilization 0.8`; tweak higher/lower if COMET (≈4 GB VRAM) needs more headroom or you want maximum throughput.

## Command Examples
```bash
# LFM2 models (greedy decoding)
CUDA_VISIBLE_DEVICES=1 python run-mt.py LiquidAI/LFM2-350M-ENJP-MT --format lfm2
CUDA_VISIBLE_DEVICES=2 python run-mt.py LiquidAI/LFM2-350M --format lfm2

# Chotto models with sampling
CUDA_VISIBLE_DEVICES=0 ./run-mt.py shisa-ai/chotto-14b-20251007-dpo-openrlhf \
  --do-sample --temperature 0.2 --top-p 0.9 \
  --predictions eval/llm-jp-eval/results/chotto-20251007-preds.jsonl

[2025-10-11 16:21:08] INFO - run-mt: alt-e-to-j scores: ool=0.0020, bleu_ja=13.4076, bert_score_ja_f1=0.8631, comet_wmt22=0.9127



# Gemma
CUDA_VISIBLE_DEVICES=3 ./run-mt.py google/gemma-3-4b-it --do-sample --temperature 1.0
CUDA_VISIBLE_DEVICES=3 ./run-mt.py google/gemma-3-4b-it --do-sample --temperature 0.2
```

## Result Utilities
- Markdown table of COMET scores: `python report_mt_scores.py` (use `--metric` / `--human-names` for customisation).
- Interactive TUI scoreboard (grouped by model & run): `python mt_scores_tui.py` (same flags as `report_mt_scores.py`).
- Random sample viewer: `python sample_predictions.py LiquidAI/LFM2-350M-ENJP-MT --count 10`.
- Override locations with `--results-dir`, `--scores`, `--predictions`, or `--log-file` when runs are stored elsewhere.

Record new runs by appending the command, model revision, and output artifacts here so the team can reproduce results when llm-jp-eval updates.

## Prompting
By default, llm-jp-eval does NOT use chat templates. Instead, it uses a custom Jinja2-based prompt template defined in `llm-jp-eval/src/llm_jp_eval/prompts.py`. The prompts look like this:
```
### 指示
これから提示する英語の文章を日本語に翻訳してください。必ず日本語の訳文を出力してください。

<examples>
  <example_1>
  ### 入力:
  Steve Wright, yesterday convicted of...
  ### 応答:
  昨日、スティーヴ・ライトは、5人の女性を...
  </example_1>
...
</examples>

### 入力:
Major British charity Comic Relief has invested money...
### 応答:
```
