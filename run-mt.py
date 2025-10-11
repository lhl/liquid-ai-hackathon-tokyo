#!/usr/bin/env python
"""Run machine translation evaluation for llm-jp-eval datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT / "llm-jp-eval"
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Expected project directory at {PROJECT_ROOT}, but it does not exist.")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_jp_eval import __version__ as DATASET_VERSION  # noqa: E402
from llm_jp_eval.answer_parser import AnswerPatternId  # noqa: E402
from llm_jp_eval.evaluator import get_evaluation_result  # noqa: E402
from llm_jp_eval.jaster.base import Sample  # noqa: E402
from llm_jp_eval.metrics.metrics import init_metrics  # noqa: E402
from llm_jp_eval.utils import GeneratedSample, get_evaluation_prompt, get_few_shot_samples, normalize, set_seed  # noqa: E402


LOGGER = logging.getLogger("run-mt")
DEFAULT_ANSWER_PATTERN = r"(?s)^(.*?)(?=\n\n|\Z)"
MT_DATASETS = ["alt-e-to-j", "alt-j-to-e", "wikicorpus-e-to-j", "wikicorpus-j-to-e"]


@dataclass
class DatasetConfig:
    name: str
    instruction: str
    output_length: int
    metrics: list[str]
    samples: list[dict]
    label_list: list[str]
    answer_pattern_id: AnswerPatternId
    answer_extract_pattern: str | None
    language: str
    custom_prompt_template: str | dict | None
    num_few_shots: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MT evaluation on llm-jp-eval datasets.")
    parser.add_argument("model", help="Hugging Face model ID or local path.")
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=MT_DATASETS,
        help="Dataset names to evaluate (default: all MT datasets).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit samples per dataset (-1 means all).")
    parser.add_argument(
        "--num-few-shots",
        type=int,
        default=4,
        help="Fallback number of few-shot examples when dataset does not define it.",
    )
    parser.add_argument("--output", type=Path, help="Path to save dataset-level results in JSONL format.")
    parser.add_argument("--predictions", type=Path, help="Optional path to save per-sample predictions JSONL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling cutoff.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (defaults to greedy).")
    parser.add_argument("--extra-output-tokens", type=int, default=0, help="Extra max tokens beyond dataset length.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype hint.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "transformers", "vllm", "openai"],
        default="auto",
        help="Generation backend: auto prefers vllm when available, otherwise falls back to transformers.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM backend.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        help="Optional vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        help="Optional vLLM max model length override.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL (defaults to the official API or OPENAI_BASE_URL env).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI-compatible API key (defaults to OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--openai-max-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent OpenAI requests per batch.",
    )
    parser.add_argument(
        "--openai-request-timeout",
        type=float,
        help="Optional OpenAI request timeout in seconds.",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=5,
        help="Retries per OpenAI request on recoverable errors.",
    )
    parser.add_argument(
        "--openai-retry-wait-seconds",
        type=float,
        default=1.0,
        help="Initial wait time between OpenAI retries; doubles after each failure.",
    )
    parser.add_argument("--device-map", default="auto", help="Device map passed to from_pretrained.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code from repo.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "local_files" / "cache",
        help="Metric cache dir.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def sanitize_run_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def resolve_model_device(model) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first_device = next(iter(device_map.values()))
        if isinstance(first_device, (list, tuple)):
            first_device = first_device[0]
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_root() -> Path:
    return PROJECT_ROOT / "datasets" / "datasets" / DATASET_VERSION / "evaluation"


def load_dataset_config(
    dataset_name: str, split: str, max_samples: int, fallback_few_shots: int
) -> tuple[DatasetConfig, Path]:
    path = dataset_root() / split / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    if max_samples >= 0:
        samples = samples[:max_samples]

    answer_pattern_id = AnswerPatternId(data.get("answer_pattern_id", "custom"))
    num_few_shots = fallback_few_shots
    if (dataset_defined := data.get("num_few_shots")) is not None:
        num_few_shots = dataset_defined

    config = DatasetConfig(
        name=dataset_name,
        instruction=data["instruction"],
        output_length=int(data["output_length"]),
        metrics=list(data["metrics"]),
        samples=samples,
        label_list=list(data.get("label_list", [])),
        answer_pattern_id=answer_pattern_id,
        answer_extract_pattern=data.get("answer_extract_pattern") or None,
        language=data.get("language", "ja"),
        custom_prompt_template=data.get("custom_prompt_template"),
        num_few_shots=num_few_shots if num_few_shots is not None else fallback_few_shots,
    )

    return config, path


def prepare_prompts(
    dataset: DatasetConfig,
    dataset_path: Path,
) -> list[GeneratedSample]:
    few_shot_samples: list[Sample] = []
    if dataset.num_few_shots:
        few_shot_samples = get_few_shot_samples(dataset_path, dataset.num_few_shots)

    prompt_template = get_evaluation_prompt(
        dataset.instruction,
        few_shot_samples,
        dataset.custom_prompt_template,
        dataset.answer_pattern_id,
        dataset.language,
    )

    generated_samples: list[GeneratedSample] = []
    for sample in dataset.samples:
        prompt = prompt_template.replace("<%input%>", sample["input"])
        generated_samples.append(
            GeneratedSample(
                input=sample["input"],
                prompt=prompt,
                generated="",  # placeholder, filled after inference
                gold=sample["output"],
                metadata=sample.get("metadata", {}),
            )
        )
    return generated_samples


def _batch_generate_transformers(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    outputs: list[str] = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = pad_token_id

    if not getattr(model.config, "is_encoder_decoder", False) and tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    device = resolve_model_device(model)
    max_positions = getattr(model.config, "max_position_embeddings", None)

    total = len(prompts)
    current_batch_size = max(1, batch_size)
    index = 0
    while index < total:
        attempt_size = min(current_batch_size, total - index)
        while True:
            batch_prompts = prompts[index : index + attempt_size]
            tokenized = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            attention_mask = tokenized["attention_mask"]
            input_lengths = attention_mask.sum(dim=1)

            effective_max_new_tokens = max_new_tokens
            if isinstance(max_positions, int):
                max_prompt = int(input_lengths.max().item())
                remaining = max_positions - max_prompt
                if remaining <= 0:
                    LOGGER.warning(
                        "Prompt length %s exceeds model max position %s; forcing minimal generation.",
                        max_prompt,
                        max_positions,
                    )
                    remaining = 1
                effective_max_new_tokens = min(effective_max_new_tokens, remaining)

            try:
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": effective_max_new_tokens,
                        "do_sample": do_sample,
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": pad_token_id,
                    }
                    if do_sample:
                        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
                    generated = model.generate(**tokenized, **gen_kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt_size == 1:
                    raise
                new_attempt = max(1, attempt_size // 2)
                LOGGER.warning(
                    "CUDA out of memory with batch size %d: %s; retrying with batch size %d",
                    attempt_size,
                    exc,
                    new_attempt,
                )
                attempt_size = new_attempt
                current_batch_size = new_attempt
                continue
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if attempt_size == 1:
                        raise
                    new_attempt = max(1, attempt_size // 2)
                    LOGGER.warning(
                        "CUDA OOM (RuntimeError) with batch size %d; retrying with batch size %d",
                        attempt_size,
                        new_attempt,
                    )
                    attempt_size = new_attempt
                    current_batch_size = new_attempt
                    continue
                raise
            else:
                break

        for i in range(generated.size(0)):
            gen_tokens = generated[i, input_lengths[i] :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            outputs.append(text.strip())

        index += attempt_size

    return outputs


def _batch_generate_vllm(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    try:
        from vllm import SamplingParams  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive path
        raise RuntimeError("vLLM is not installed. Re-run without --engine vllm or install vllm.") from exc

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None and eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id

    engine_model = getattr(model, "llm_engine", None)
    if engine_model is not None:
        engine_model = getattr(engine_model, "model", None)
    engine_config = getattr(engine_model, "config", None)
    max_positions = getattr(engine_config, "max_position_embeddings", None)

    outputs: list[str] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = prompts[start : start + max(1, batch_size)]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        attention_mask = tokenized["attention_mask"]
        input_lengths = attention_mask.sum(dim=1)

        effective_max_new_tokens = max_new_tokens
        if isinstance(max_positions, int):
            max_prompt = int(input_lengths.max().item())
            remaining = max_positions - max_prompt
            if remaining <= 0:
                LOGGER.warning(
                    "Prompt length %s exceeds model max position %s; forcing minimal generation.",
                    max_prompt,
                    max_positions,
                )
                remaining = 1
            effective_max_new_tokens = min(effective_max_new_tokens, remaining)

        sampling_kwargs = {
            "max_tokens": effective_max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "skip_special_tokens": True,
        }
        if eos_token_id is not None:
            sampling_kwargs["stop_token_ids"] = [eos_token_id]
        sampling_params = SamplingParams(**sampling_kwargs)
        results = model.generate(batch_prompts, sampling_params=sampling_params)
        for i, result in enumerate(results):
            text = result.outputs[0].text if result.outputs else ""
            outputs.append(text.strip())
    return outputs


def _batch_generate_openai(
    engine_config: dict[str, Any],
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    try:
        client = engine_config["client"]
        model_name = engine_config["model_name"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Missing OpenAI engine configuration entry: {exc}") from exc

    max_concurrency = max(1, int(engine_config.get("max_concurrency", 4)))
    max_retries = max(0, int(engine_config.get("max_retries", 5)))
    base_wait = max(0.0, float(engine_config.get("retry_wait", 1.0)))
    request_timeout = engine_config.get("request_timeout")

    outputs: list[str] = []
    effective_temperature = temperature if do_sample else 0.0
    effective_top_p = top_p if do_sample else 1.0

    def generate_prompt(prompt: str) -> str:
        wait = base_wait
        attempt = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    timeout=request_timeout,
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                return text.strip()
            except Exception as exc:  # pragma: no cover - depends on network failures
                if attempt >= max_retries:
                    raise
                delay = wait or 1.0
                LOGGER.warning("OpenAI request failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
                if wait:
                    wait *= 2
                attempt += 1

    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = prompts[start : start + max(1, batch_size)]
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [executor.submit(generate_prompt, p) for p in batch_prompts]
            for future in futures:
                outputs.append(future.result())
    return outputs


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    engine: str,
    engine_config: dict[str, Any] | None = None,
) -> list[str]:
    if engine == "vllm":
        return _batch_generate_vllm(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    if engine == "openai":
        if engine_config is None:
            raise RuntimeError("OpenAI engine requires engine_config with client information.")
        return _batch_generate_openai(
            engine_config=engine_config,
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    return _batch_generate_transformers(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )


def ensure_metrics_initialized(metrics: Iterable[str], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    init_metrics(list(metrics), cache_dir=cache_dir)


def save_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_seed()

    LOGGER.info(
        "Max samples per dataset: %s",
        "all" if args.max_samples < 0 else args.max_samples,
    )

    engine = args.engine
    if engine == "auto":
        if importlib.util.find_spec("vllm") is not None:
            engine = "vllm"
            LOGGER.info("Engine auto-detected: using vllm backend.")
        else:
            engine = "transformers"
            LOGGER.info("Engine auto-detected: vllm not found, falling back to transformers.")

    LOGGER.info("Loading model %s with engine %s", args.model, engine)
    dtype = resolve_dtype(args.dtype)

    engine_config: dict[str, Any] | None = None

    if engine == "vllm":
        try:
            from vllm import LLM  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive path
            raise RuntimeError("vLLM backend requested but vllm is not installed.") from exc

        vllm_kwargs: dict[str, object] = {
            "model": args.model,
            "tokenizer": args.model,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.dtype != "auto":
            vllm_kwargs["dtype"] = args.dtype
        if args.vllm_gpu_memory_utilization is not None:
            vllm_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        if args.vllm_max_model_len is not None:
            vllm_kwargs["max_model_len"] = args.vllm_max_model_len
        model = LLM(**vllm_kwargs)
        tokenizer = model.get_tokenizer()
    elif engine == "openai":
        try:
            from openai import OpenAI  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive path
            raise RuntimeError("OpenAI backend requested but openai package is not installed.") from exc

        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI engine requires an API key. Pass --openai-api-key or set OPENAI_API_KEY."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if args.openai_base_url:
            client_kwargs["base_url"] = args.openai_base_url
        if args.openai_request_timeout is not None:
            client_kwargs["timeout"] = args.openai_request_timeout

        client = OpenAI(**client_kwargs)
        model = client
        tokenizer = None
        engine_config = {
            "client": client,
            "model_name": args.model,
            "max_concurrency": args.openai_max_concurrency,
            "request_timeout": args.openai_request_timeout,
            "max_retries": args.openai_max_retries,
            "retry_wait": args.openai_retry_wait_seconds,
        }
    else:
        model_loading_kwargs: dict[str, object] = {
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
        }
        if dtype is not None:
            model_loading_kwargs["dtype"] = dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                **model_loading_kwargs,
            )
        except TypeError as exc:
            if "dtype" in str(exc) and dtype is not None:
                model_loading_kwargs.pop("dtype", None)
                model_loading_kwargs["torch_dtype"] = dtype
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    **model_loading_kwargs,
                )
            else:
                raise
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )

    all_dataset_configs: list[tuple[DatasetConfig, list[GeneratedSample]]] = []
    all_metrics: set[str] = set()
    total_samples = 0

    for dataset_name in args.datasets:
        dataset_cfg, dataset_path = load_dataset_config(
            dataset_name,
            args.split,
            args.max_samples,
            args.num_few_shots,
        )
        samples = prepare_prompts(dataset_cfg, dataset_path)
        sample_count = len(samples)
        total_samples += sample_count
        LOGGER.info(
            "Prepared dataset %s with %d samples (max output tokens %d)",
            dataset_cfg.name,
            sample_count,
            dataset_cfg.output_length,
        )
        all_dataset_configs.append((dataset_cfg, samples))
        all_metrics.update(dataset_cfg.metrics)

    ensure_metrics_initialized(all_metrics, args.cache_dir)
    LOGGER.info("Total samples queued across datasets: %d", total_samples)

    run_name = sanitize_run_name(args.model)
    score_results: dict[str, dict[str, float]] = {}
    prediction_records: list[dict] = []
    dataset_level_records: list[dict] = []
    default_metric = "comet_wmt22"

    for dataset_cfg, samples in all_dataset_configs:
        LOGGER.info("Evaluating %s with %d samples", dataset_cfg.name, len(samples))
        prompts = [sample["prompt"] for sample in samples]
        max_new_tokens = dataset_cfg.output_length + args.extra_output_tokens

        generations = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            engine=engine,
            engine_config=engine_config,
        )

        for sample, output in zip(samples, generations, strict=False):
            sample["generated"] = normalize(output)

        score_dict, _, output_records = get_evaluation_result(
            run_name=run_name,
            samples=samples,
            num_few_shots=dataset_cfg.num_few_shots,
            target_dataset_name=dataset_cfg.name,
            target_split=args.split,
            target_data_answer_extract_pattern=dataset_cfg.answer_extract_pattern or DEFAULT_ANSWER_PATTERN,
            answer_pattern_id=dataset_cfg.answer_pattern_id,
            metrics=dataset_cfg.metrics,
            label_list=dataset_cfg.label_list,
            metainfo={
                "basemodel_name": args.model,
                "model_type": "",
                "instruction_tuning_method_by_llm_jp": None,
                "instruction_tuning_data_by_llm_jp": None,
            },
            dataset_processor_name=dataset_cfg.name,
        )

        score_results[dataset_cfg.name] = score_dict

        dataset_level_records.append(
            {
                "type": "dataset",
                "dataset": dataset_cfg.name,
                "num_samples": len(samples),
                "metrics": score_dict,
                "default_metric": default_metric,
                "default_metric_score": score_dict.get(default_metric),
            }
        )

        if args.predictions:
            for record in output_records:
                prediction_records.append({"type": "prediction", **record.__dict__})

        LOGGER.info(
            "%s scores: %s",
            dataset_cfg.name,
            ", ".join(f"{k}={v:.4f}" for k, v in score_dict.items() if isinstance(v, (int, float))),
        )

    mt_scores = [
        score_results[name][default_metric]
        for name in score_results
        if default_metric in score_results[name]
    ]
    mt_average = mean(mt_scores) if mt_scores else 0.0

    dataset_level_records.append(
        {
            "type": "summary",
            "category": "MT",
            "metric": default_metric,
            "score": mt_average,
            "datasets": list(score_results.keys()),
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        results_dir = REPO_ROOT / "results"
        args.output = results_dir / f"mt_{sanitize_run_name(args.model)}_{timestamp}.jsonl"

    save_jsonl(args.output, dataset_level_records)
    LOGGER.info("Saved dataset-level results to %s", args.output)

    if args.predictions and prediction_records:
        save_jsonl(args.predictions, prediction_records)
        LOGGER.info("Saved prediction records to %s", args.predictions)

    LOGGER.info("MT average (%s): %.4f", default_metric, mt_average)


if __name__ == "__main__":
    main()
