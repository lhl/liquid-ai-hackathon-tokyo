# MT Evaluation Playbook

## llm-jp-eval Snapshot
- Repo path: `eval/llm-jp-eval`
- Commit: `ceb58dba00d27853671695ac967bba2fecd06326` (vendored on 2024-10-11)
- Results dir: `eval/results/` (auto-created for dataset metrics, prediction dumps, and logs).
- Default command writes dataset summaries to `results/<model>.scores.jsonl`, raw translations to `results/<model>.predictions.jsonl`, and console logs to `results/<model>.log` (override with `--scores` / `--predictions` / `--log-file` as needed).

## Generation Settings
- **Greedy / Liquid AI recommendation** — leave sampling disabled (`--do-sample` omitted, `--temperature 0`, `--top-p 1.0`).
- **Chotto baseline** — `--do-sample --temperature 0.2 --top-p 0.9`.
- The script respects CLI overrides per backend (`transformers`, `vllm`, `openai`).
- COMET scoring now auto-detects GPU availability (respects `CUDA_VISIBLE_DEVICES`); override with `LLM_JP_EVAL_COMET_GPUS=0` for CPU or set an explicit count.
- WikiCorpus splits are skipped by default; pass `--include-wikicorpus` for full runs or `--exclude-datasets` to add/remove specific sets.
- vLLM runs default to `--vllm-gpu-memory-utilization 0.8`; tweak higher/lower if COMET (≈4 GB VRAM) needs more headroom or you want maximum throughput.

## Command Examples
```bash
# LFM2 models (greedy decoding)
CUDA_VISIBLE_DEVICES=1 python run-mt.py LiquidAI/LFM2-350M-ENJP-MT
CUDA_VISIBLE_DEVICES=2 python run-mt.py LiquidAI/LFM2-350M

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
- Random sample viewer: `python sample_predictions.py LiquidAI/LFM2-350M-ENJP-MT --count 10`.
- Override locations with `--results-dir`, `--scores`, `--predictions`, or `--log-file` when runs are stored elsewhere.

Record new runs by appending the command, model revision, and output artifacts here so the team can reproduce results when llm-jp-eval updates.
