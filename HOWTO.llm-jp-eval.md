# Machine Translation Evaluation HOWTO

This short guide explains how to run the machine-translation (MT) benchmarks that ship with `llm-jp-eval`, using the new `run-mt.py` helper script.

## Prerequisites
- Python 3.10–3.12 with `pip` or [uv](https://docs.astral.sh/uv/) installed.
- The project dependencies installed:
  - `uv sync` (recommended), or
  - `pip install -e .` from the `llm-jp-eval` root.
- Optionally export your Hugging Face token before you download gated models:
  ```bash
  export HF_TOKEN=hf_...
  ```
- GPU access is recommended for the MT models and COMET scorer. To follow the provided examples, pin the evaluation to the requested devices:
  ```bash
  export CUDA_VISIBLE_DEVICES=2,3
  ```

All metric assets (BLEU, BERTScore, COMET) are cached under `llm-jp-eval/local_files/cache` the first time you run the script.

## Quick Start
Run the script from the repository root (the directory that contains `run-mt.py` and the `llm-jp-eval/` subfolder). It will evaluate the four MT datasets (`alt-e-to-j`, `alt-j-to-e`, `wikicorpus-e-to-j`, `wikicorpus-j-to-e`) and output dataset-level scores plus an MT average.

```bash
python run-mt.py shisa-ai/chotto-14b-20251007-sft \
  --output results/chotto-14b-mt.jsonl \
  --predictions results/chotto-14b-mt-preds.jsonl
```

Use the local model list as needed:
```bash
python run-mt.py LiquidAI/LFM2-350M-ENJP-MT
python run-mt.py google/gemma-3-4b-it
python run-mt.py shisa-ai/chotto-14b-20250922
```

On completion the script prints the COMET average for the MT category and writes a JSONL file under `results/` (or at the path you pass via `--output`). Each line is either a dataset record or the MT summary.

## Key Options
- `--max-samples N`: limit evaluation to the first `N` samples per dataset (default is all).
- `--batch-size B`: adjust generation batch size (default: 4).
- `--num-few-shots K`: override few-shot count when a dataset does not specify one (default: 4).
- `--split {dev,test}`: choose which split to evaluate (default: `test`).
- `--do-sample`, `--temperature`, `--top-p`: enable sampling instead of greedy decoding.
- `--extra-output-tokens M`: add extra generation allowance beyond the dataset’s requested maximum.
- `--predictions path`: store per-sample predictions and references for later analysis.
- `--device-map`, `--dtype`, `--trust-remote-code`: forwarded to `transformers` when loading models.
- `--engine {auto,transformers,vllm,openai}`: auto (default) prefers vLLM when it is installed, otherwise falls back to `transformers`. Select `vllm` to force the high-throughput backend, `transformers` for the reference implementation, or `openai` to call an OpenAI-compatible endpoint.
- `--vllm-tensor-parallel-size N`, `--vllm-gpu-memory-utilization F`, `--vllm-max-model-len L`: optional knobs when `--engine vllm` is active.
- `--openai-base-url`, `--openai-api-key`, `--openai-max-concurrency`, `--openai-request-timeout`, `--openai-max-retries`, `--openai-retry-wait-seconds`: configure OpenAI or OpenAI-compatible endpoints. Defaults read from `OPENAI_BASE_URL` / `OPENAI_API_KEY` when available.

Run `python run-mt.py --help` to see the full list.

## Understanding the Output
Each dataset line in the JSONL file looks like:
```json
{"type": "dataset", "dataset": "alt-e-to-j", "num_samples": 2000,
 "metrics": {"ool": 0.0, "bleu_ja": 25.43, "bert_score_ja_f1": 0.86, "comet_wmt22": 74.12},
 "default_metric": "comet_wmt22", "default_metric_score": 74.12}
```
The final summary line reports the COMET average across all evaluated MT datasets:
```json
{"type": "summary", "category": "MT", "metric": "comet_wmt22",
 "score": 71.88, "datasets": ["alt-e-to-j", "alt-j-to-e", "wikicorpus-e-to-j", "wikicorpus-j-to-e"]}
```

If you supplied `--predictions`, each JSONL row there includes the prompt, model output, gold answer, and quick match metrics—handy for error analysis.

## Troubleshooting
- **Missing dataset files**: ensure you run from the repository root. Processed datasets live under `datasets/datasets/2.0.0`.
- **Model load failures**: check that the model is available locally or via Hugging Face (authenticate if necessary). Add `--trust-remote-code` when the repository provides custom modeling code.
- **Out-of-memory**: the script now forces left padding for decoder-only models and dynamically shrinks the batch on CUDA OOM, then resumes once capacity is regained. If memory is still tight, set a smaller `--batch-size` or switch to `--engine vllm`.
- **Slow COMET scoring**: the first execution downloads the checkpoint; subsequent runs reuse the cached copy in `local_files/cache`.
- **COMET resource usage**: the `comet_wmt22` scorer defaults to CPU inference (≈6 GB RAM during startup). If `CUDA_VISIBLE_DEVICES` exposes a GPU, Lightning will try to offload the model; budget ~4 GB VRAM and keep at least 2–3 GB free for peak spikes. You can pin COMET to CPU by unsetting `CUDA_VISIBLE_DEVICES` or exporting it to an empty string when GPU memory is scarce.

Recent versions of `run-mt.py` addressed a failure where right-padded decoder prompts plus a fixed batch size exhausted GPU memory (`torch.OutOfMemoryError`). Padding now defaults to the left, and the generator gracefully halves the batch size until the request fits (this also applies when using the vLLM backend).

That’s it—`run-mt.py` now encapsulates preprocessing, generation, scoring, and result export for the translation benchmarks bundled with `llm-jp-eval`.

## Using the vLLM Backend
The script can drive vLLM directly—no separate inference server required.

1. Install vLLM (and ensure the pinned dependencies remain compatible with the rest of the environment):
   ```bash
   pip install "numpy<2"  # vLLM 0.11 currently requires NumPy 1.x
   pip install -U pyarrow
   pip install vllm
   ```
2. Run the evaluator with the vLLM engine (this is the default when vLLM is available):
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python run-mt.py shisa-ai/chotto-14b-20251007-dpo-openrlhf \
     --engine vllm \
     --vllm-tensor-parallel-size 2 \
     --max-samples 8
   ```
   Omit `--max-samples` for the full benchmark. vLLM streams requests to the GPU(s) and typically finishes the large corpora (e.g. `wikicorpus-e-to-j`) significantly faster.

3. Optional flags mirror vLLM’s constructor:
   - `--vllm-gpu-memory-utilization 0.9` to cap memory usage.
   - `--vllm-max-model-len 16384` to override the default context window.

When the vLLM backend is active, the script still honours `--batch-size` as the number of prompts sent per `generate` call; vLLM then applies its own continuous batching internally. Leaving the default (4) is fine for most runs, but you can raise it to issue larger chunks or lower it if GPU memory becomes tight.

If vLLM is unavailable, the script falls back to `transformers`. A missing vLLM installation while `--engine vllm` is set now produces a clear error message.

## Using OpenAI-Compatible APIs
You can evaluate hosted OpenAI models or any self-hosted endpoint that exposes the OpenAI Chat Completions API (including vLLM servers running in OpenAI mode).

1. Install the SDK if it is not already present:
   ```bash
   pip install openai
   ```
2. Configure credentials and the base URL (for self-hosted servers):
   ```bash
   export OPENAI_API_KEY=sk-your-key
   export OPENAI_BASE_URL=http://localhost:8000/v1
   ```
3. Run the evaluator:
   ```bash
   python run-mt.py gpt-4o-mini --engine openai --max-samples 16 --openai-max-concurrency 8
   ```

The script batches prompts according to `--batch-size`, then issues concurrent Chat Completion requests (up to `--openai-max-concurrency` workers). Retries with exponential backoff smooth over transient rate limits or network hiccups. Responses flow through the usual scoring pipeline, so metrics and output files remain identical to on-device runs.

To target a self-hosted vLLM server exposing the `/v1` API, set `--openai-base-url` (or `OPENAI_BASE_URL`) to that endpoint and pass the model identifier advertised by the server.
