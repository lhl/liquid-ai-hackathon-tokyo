# MT Evaluation Playbook

## llm-jp-eval Snapshot
- Repo path: `eval/llm-jp-eval`
- Commit: `ceb58dba00d27853671695ac967bba2fecd06326` (vendored on 2024-10-11)
- Results dir: `eval/results/` (auto-created for dataset metrics, prediction dumps, and logs).
- Default command writes dataset summaries to `results/<model>.scores.jsonl`, raw translations to `results/<model>.predictions.jsonl`, and console logs to `results/<model>.log` (override with `--scores` / `--predictions` / `--log-file` as needed).
- Filenames now include the decoding strategy (e.g. `.greedy`, `.sampled-t0_2-p0_9`); add extra context with `--run-tag timestamp` when you need fully unique artifacts per run.
- Install helper dependencies with `pip install -r requirements.txt` to run the local viewers, reports, and judging scripts.

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
 - Judge score distribution stats: `python report_judge_distribution.py` (add `--interactive` to choose models, `--ascii-only` if block glyphs are undesirable; output now highlights Useful ≥3 and Perfect =5 rates).
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

# Results
- We are using the new 2025-10-13 updated parameters from the README (thx guys!), however that only makes a marginal difference on the scores
- **HOWEVER**, if we modify llm-jp-eval setup to strictly obey the ENJP-MT prompt format, we are able to replicate the COMET scores! We can test this against different chat templates and see exactly how brittle this is with tiny models.
- In addition to the standard COMET scoring, I've added gemini-2.5-pro 1-5 LLM judging. This is extremely interesting, since the COMET basically saturate, however the Judge scoring and "Perfect" scoring doest not, and is much more useful for discriminating model performance

| Model                                                                                |   alt-e-to-j |   alt-j-to-e |   MT avg |   Judge |   Perfect |
|--------------------------------------------------------------------------------------|--------------|--------------|----------|---------|-----------|
| LiquidAI/LFM2-350M-ENJP-MT.greedy                                                    |       0.4510 |       0.4715 |   0.4612 |    1.27 |     3/200 |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_2-p0_9                                         |       0.4470 |       0.4801 |   0.4635 |    1.22 |     1/200 |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_chotto-lfm2                 |       0.6044 |       0.6444 |   0.6244 |    2.60 |    25/200 |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_chotto                      |       0.6960 |       0.6559 |   0.6759 |    2.83 |    35/200 |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2                        |       0.9046 |       0.8731 |   0.8889 |    3.96 |    76/200 |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05                                |       0.4346 |       0.4734 |   0.4540 |    1.35 |     5/200 |
| LiquidAI/LFM2-350M.sampled-t0_2-p0_9                                                 |       0.5684 |       0.6460 |   0.6072 |    2.10 |     1/200 |
| google/gemma-3-4b-it.sampled-t0_2-p0_9                                               |       0.8926 |       0.8694 |   0.8810 |    3.69 |    38/200 |
| gpt-4o.sampled-t0_2-p0_9                                                             |       0.9212 |       0.8973 |   0.9093 |    4.55 |   127/200 |
| shisa-ai/chotto-14b-20250922.sampled-t0_2-p0_9                                       |       0.8792 |       0.8695 |   0.8743 |    4.22 |    91/200 |
| shisa-ai/chotto-14b-20250922.sampled-t0_5-p1-m0_1-r1_05-ct_chotto                    |       0.8995 |       0.8810 |   0.8903 |    4.19 |    88/200 |
| shisa-ai/chotto-14b-20251007-dpo-openrlhf.sampled-t0_2-p0_9-ct_chotto                |       0.9073 |       0.8865 |   0.8969 |    4.37 |   106/200 |
| shisa-ai/chotto-14b-20251007-dpo-openrlhf.sampled-t0_2-p0_9                          |       0.9122 |       0.8855 |   0.8988 |    4.32 |   104/200 |
| shisa-ai/chotto-14b-20251007-dpo-openrlhf.sampled-t0_5-p1-m0_1-r1_05-ct_chotto       |       0.9067 |       0.8858 |   0.8963 |    4.34 |   108/200 |
| shisa-ai/chotto-14b-20251013-dpo.sampled-t0_2-p0_9-ct_chotto                         |       0.9091 |       0.8878 |   0.8984 |    4.39 |   113/200 |
| shisa-ai/chotto-14b-20251013-dpo.sampled-t0_5-p1-m0_1-r1_05-ct_chotto                |       0.9093 |       0.8878 |   0.8985 |    4.39 |   106/200 |
| shisa-ai/shisa-v2-llama3.1-405b.sampled-t0_2-p0_9                                    |       0.9165 |       0.8936 |   0.9050 |    4.57 |   128/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_chotto-lfm2          |       0.8993 |       0.8682 |   0.8837 |    3.63 |    41/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_chotto               |       0.9014 |       0.8716 |   0.8865 |    3.73 |    55/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_lfm2                 |       0.8848 |       0.7895 |   0.8372 |    2.95 |    25/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9                         |       0.4935 |       0.4672 |   0.4804 |    2.65 |    20/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_5-p1-m0_1-r1_05-ct_chotto-lfm2 |       0.8996 |       0.8668 |   0.8832 |    3.62 |    46/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_5-p1-m0_1-r1_05-ct_chotto      |       0.9001 |       0.8703 |   0.8852 |    3.67 |    45/200 |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2        |       0.8865 |       0.7791 |   0.8328 |    2.88 |    31/200 |


I built a statistical viewer for our ratings as well.

| Model                                                                       |   Samples |   Mean |   Median |   1% |   2% |   3% |   4% |   5% |   Useful% |   Perfect% | Hist   |
|-----------------------------------------------------------------------------|-----------|--------|----------|------|------|------|------|------|-----------|------------|--------|
| shisa-ai/shisa-v2-llama3.1-405b.sampled-t0_2-p0_9                           |       200 |   4.57 |      5.0 |  0.0 |  1.5 |  6.5 | 26.0 | 66.0 |      98.5 |       66.0 | ▁▁▂▄█  |
| gpt-4o.sampled-t0_2-p0_9                                                    |       200 |   4.55 |      5.0 |  0.0 |  0.5 |  7.5 | 28.0 | 64.0 |      99.5 |       64.0 | ▁▁▂▄█  |
| shisa-ai/chotto-14b-20251013-dpo.sampled-t0_2-p0_9-ct_chotto                |       200 |   4.39 |      5.0 |  0.0 |  5.5 |  7.5 | 29.5 | 57.5 |      94.5 |       57.5 | ▁▂▂▅█  |
| shisa-ai/chotto-14b-20251013-dpo.sampled-t0_5-p1-m0_1-r1_05-ct_chotto       |       200 |   4.39 |      5.0 |  0.0 |  1.5 | 12.0 | 32.5 | 54.0 |      98.5 |       54.0 | ▁▁▃▅█  |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2               |       200 |   3.96 |      4.0 |  0.0 | 12.5 | 18.5 | 29.5 | 39.5 |      87.5 |       39.5 | ▁▃▄▆█  |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_chotto      |       200 |   3.73 |      4.0 |  0.5 | 14.0 | 26.5 | 30.5 | 28.5 |      85.5 |       28.5 | ▁▄▇██  |
| google/gemma-3-4b-it.sampled-t0_2-p0_9                                      |       200 |   3.69 |      4.0 |  0.0 | 13.5 | 25.0 | 40.5 | 21.0 |      86.5 |       21.0 | ▁▃▅█▅  |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_chotto-lfm2 |       200 |   3.63 |      4.0 |  0.5 | 18.0 | 21.0 | 39.0 | 21.5 |      81.5 |       21.5 | ▁▄▅█▅  |
| shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly.sampled-t0_2-p0_9-ct_lfm2        |       200 |   2.95 |      3.0 |  6.5 | 36.5 | 25.0 | 19.5 | 12.5 |      57.0 |       12.5 | ▂█▆▅▃  |
| LiquidAI/LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05                       |       200 |   1.35 |      1.0 | 77.0 | 16.5 |  3.5 |  0.5 |  2.5 |       6.5 |        2.5 | █▃▁▁▁  |
| LiquidAI/LFM2-350M-ENJP-MT.greedy                                           |       200 |   1.27 |      1.0 | 82.5 | 12.5 |  2.0 |  1.5 |  1.5 |       5.0 |        1.5 | █▂▁▁▁  |

(Note: percentage columns show share of samples for each 1-5 rating; Useful% aggregates scores ≥3, and Perfect% is the share of 5s.)

Of course, you should use the `tui-viewer.py` to view the raw results, as it's even more illuminating. Even with the judge scoring, there are some discrpenacies on the results that make you go hmmm and I'd highly recommend anyone really interested in seeing the model behavior to look at the raw data (the one trick to better models!)
