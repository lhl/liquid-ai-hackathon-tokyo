#!/usr/bin/env python3
"""Dataset statistics helper for LiquidAI/LFM2-350M-ENJP-MT JSONL corpora."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from transformers import AutoTokenizer

MODEL_ID = "LiquidAI/LFM2-350M-ENJP-MT"

ROLE_ALIASES: Mapping[str, str] = {
    "assistant": "assistant",
    "assistant_en": "assistant",
    "assistant_jp": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "system": "system",
    "context": "system",
    "narrator": "system",
    "user": "user",
    "human": "user",
    "client": "user",
    "customer": "user",
    "question": "user",
    "prompt": "user",
}


def normalize_role(value: Optional[str]) -> str:
    if value is None:
        return "user"
    role = str(value).strip().lower()
    return ROLE_ALIASES.get(role, role)


def contains_japanese(text: str) -> bool:
    for char in text:
        code_point = ord(char)
        if (
            0x3040 <= code_point <= 0x309F
            or 0x30A0 <= code_point <= 0x30FF
            or 0x3400 <= code_point <= 0x4DBF
            or 0x4E00 <= code_point <= 0x9FFF
        ):
            return True
    return False


def contains_ascii_letters(text: str) -> bool:
    return any("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in text)


def infer_language(role_value: Optional[str], content: str) -> str:
    if isinstance(role_value, str):
        lowered = role_value.lower()
        if lowered.endswith(("_ja", "_jp", "_japanese")):
            return "ja"
        if lowered.endswith(("_en", "_english")):
            return "en"

    if contains_japanese(content):
        return "ja"
    if contains_ascii_letters(content):
        return "en"
    return "other"


class StatsAccumulator:
    def __init__(self) -> None:
        self.values: List[float] = []
        self.total: float = 0.0

    def add(self, value: float) -> None:
        if value is None:
            return
        self.values.append(float(value))
        self.total += float(value)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)

    def count(self) -> int:
        return len(self.values)

    def _percentile(self, sorted_vals: List[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        k = (len(sorted_vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[f]
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return d0 + d1

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {}
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        total = self.total
        mean = total / n
        summary = {
            "count": float(n),
            "sum": total,
            "mean": mean,
            "min": sorted_vals[0],
            "median": self._percentile(sorted_vals, 0.5),
            "p90": self._percentile(sorted_vals, 0.9),
            "p95": self._percentile(sorted_vals, 0.95),
            "p99": self._percentile(sorted_vals, 0.99),
            "max": sorted_vals[-1],
            "std": 0.0,
        }
        if n > 1:
            summary["std"] = statistics.pstdev(sorted_vals)
        return summary


class VariantStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sample_count = 0
        self.empty_samples = 0
        self.total_messages = 0
        self.chat_token_stats = StatsAccumulator()
        self.content_token_stats = StatsAccumulator()
        self.char_stats = StatsAccumulator()
        self.message_count_stats = StatsAccumulator()
        self.sample_role_token_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
        self.message_role_token_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
        self.message_role_char_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
        self.sample_language_token_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
        self.message_language_token_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
        self.message_language_char_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)

    def _normalize_messages(self, messages: Iterable) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                role_value = msg.get("role")
                content_value = msg.get("content")
            else:
                role_value = None
                content_value = msg
            role = normalize_role(role_value)
            if content_value is None:
                content = ""
            else:
                content = str(content_value)
            language = infer_language(role_value, content)
            normalized.append({"role": role, "content": content, "language": language})
        return normalized

    def process(
        self,
        messages: Iterable,
        tokenizer,
        record_index: int,
        dataset_path: Path,
        line_index: int,
    ) -> None:
        normalized = self._normalize_messages(messages)
        self.sample_count += 1
        if not normalized:
            self.empty_samples += 1
            self.message_count_stats.add(0)
            return

        self.total_messages += len(normalized)
        self.message_count_stats.add(len(normalized))

        char_total = 0
        role_token_totals: Dict[str, int] = defaultdict(int)
        language_token_totals: Dict[str, int] = defaultdict(int)

        for msg in normalized:
            role = msg["role"]
            text = msg["content"]
            language = msg["language"]
            char_len = len(text)
            char_total += char_len
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(token_ids)
            role_token_totals[role] += token_count
            self.message_role_token_stats[role].add(token_count)
            self.message_role_char_stats[role].add(char_len)
            language_token_totals[language] += token_count
            self.message_language_token_stats[language].add(token_count)
            self.message_language_char_stats[language].add(char_len)

        for role, total_tokens in role_token_totals.items():
            self.sample_role_token_stats[role].add(total_tokens)

        for language, total_tokens in language_token_totals.items():
            self.sample_language_token_stats[language].add(total_tokens)

        self.char_stats.add(char_total)
        self.content_token_stats.add(sum(role_token_totals.values()))

        try:
            chat_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in normalized
            ]
            tokenized = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception as exc:  # pragma: no cover - debugging aid
            raise RuntimeError(
                "Failed to apply chat template for "
                f"{dataset_path} record #{record_index} (line {line_index}) variant '{self.name}'."
            ) from exc

        chat_token_count = len(tokenized) if hasattr(tokenized, "__len__") else int(tokenized.shape[-1])
        self.chat_token_stats.add(chat_token_count)


class DatasetStats:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.record_count = 0
        self.unhandled_records = 0
        self.variants: Dict[str, VariantStats] = {}

    def get_variant(self, name: str) -> VariantStats:
        if name not in self.variants:
            self.variants[name] = VariantStats(name)
        return self.variants[name]

    def process(self, tokenizer, limit: Optional[int] = None) -> None:
        processed = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_index, raw_line in enumerate(handle, start=1):
                if limit is not None and processed >= limit:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to parse JSON in {self.path} on line {line_index}: {exc}"
                    ) from exc

                processed += 1
                self.record_count += 1

                variants = extract_conversations(record)
                if not variants:
                    self.unhandled_records += 1
                    continue

                for name, messages in variants.items():
                    variant_stats = self.get_variant(name)
                    variant_stats.process(
                        messages,
                        tokenizer=tokenizer,
                        record_index=self.record_count,
                        dataset_path=self.path,
                        line_index=line_index,
                    )


def extract_conversations(record: Mapping) -> Dict[str, List]:
    variants: Dict[str, List] = {}

    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        variants["conversations"] = conversations

    chosen = record.get("chosen")
    if isinstance(chosen, list) and chosen:
        variants["chosen"] = chosen

    rejected = record.get("rejected")
    if isinstance(rejected, list) and rejected:
        variants["rejected"] = rejected

    if "prompt" in record and "response" in record:
        prompt = record.get("prompt")
        response = record.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            variants.setdefault(
                "prompt_response",
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            )

    return variants


def format_number(value: float) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return f"{int(round(value)):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_metric_block(
    label: str,
    stats: StatsAccumulator,
    *,
    indent: int = 6,
    unit: str = "",
    count_label: Optional[str] = None,
    include_sum: bool = True,
) -> None:
    summary = stats.summary()
    prefix = " " * indent
    unit_suffix = f" {unit}" if unit else ""
    if not summary:
        print(f"{prefix}{label}: n/a")
        return

    print(f"{prefix}{label}:")
    count_value = int(summary["count"]) if "count" in summary else 0
    count_text = format_number(count_value)
    if count_label:
        print(f"{prefix}  count: {count_text} {count_label}")
    else:
        print(f"{prefix}  count: {count_text}")

    if include_sum and "sum" in summary:
        print(f"{prefix}  total: {format_number(summary['sum'])}{unit_suffix}")

    for key in ("min", "mean", "median", "p90", "p95", "p99", "max"):
        if key in summary:
            print(f"{prefix}  {key}: {format_number(summary[key])}{unit_suffix}")

    std_value = summary.get("std")
    if std_value:
        print(f"{prefix}  std: {format_number(std_value)}{unit_suffix}")


def print_dataset_summary(dataset: DatasetStats) -> None:
    print(f"Dataset: {dataset.path}")
    print(f"  Records processed: {format_number(dataset.record_count)}")
    if dataset.unhandled_records:
        print(
            f"  Records without recognised conversations: {format_number(dataset.unhandled_records)}"
        )

    for variant_name, variant in sorted(dataset.variants.items()):
        print(f"  Variant: {variant_name}")
        print(f"    Samples: {format_number(variant.sample_count)}")
        if variant.empty_samples:
            print(f"    Empty samples: {format_number(variant.empty_samples)}")
        print(f"    Total messages: {format_number(variant.total_messages)}")

        print_metric_block(
            "Messages per sample",
            variant.message_count_stats,
            indent=6,
            unit="messages",
            count_label="samples",
            include_sum=False,
        )
        print_metric_block(
            "Characters per sample",
            variant.char_stats,
            indent=6,
            unit="chars",
            count_label="samples",
        )
        print_metric_block(
            "Content tokens per sample",
            variant.content_token_stats,
            indent=6,
            unit="tokens",
            count_label="samples",
        )
        print_metric_block(
            "Chat-template tokens per sample",
            variant.chat_token_stats,
            indent=6,
            unit="tokens",
            count_label="samples",
        )

        if variant.sample_role_token_stats:
            print(f"    Role token totals per sample:")
            for role, stats in sorted(variant.sample_role_token_stats.items()):
                print_metric_block(
                    f"{role}",
                    stats,
                    indent=8,
                    unit="tokens",
                    count_label="samples",
                )

        if variant.message_role_token_stats:
            print(f"    Tokens per message by role:")
            for role, stats in sorted(variant.message_role_token_stats.items()):
                print_metric_block(
                    f"{role}",
                    stats,
                    indent=8,
                    unit="tokens",
                    count_label="messages",
                )

        if variant.message_role_char_stats:
            print(f"    Characters per message by role:")
            for role, stats in sorted(variant.message_role_char_stats.items()):
                print_metric_block(
                    f"{role}",
                    stats,
                    indent=8,
                    unit="chars",
                    count_label="messages",
                )

        if variant.sample_language_token_stats:
            print(f"    Language token totals per sample:")
            for language, stats in sorted(variant.sample_language_token_stats.items()):
                print_metric_block(
                    f"{language}",
                    stats,
                    indent=8,
                    unit="tokens",
                    count_label="samples",
                )

        if variant.message_language_token_stats:
            print(f"    Tokens per message by language:")
            for language, stats in sorted(variant.message_language_token_stats.items()):
                print_metric_block(
                    f"{language}",
                    stats,
                    indent=8,
                    unit="tokens",
                    count_label="messages",
                )

        if variant.message_language_char_stats:
            print(f"    Characters per message by language:")
            for language, stats in sorted(variant.message_language_char_stats.items()):
                print_metric_block(
                    f"{language}",
                    stats,
                    indent=8,
                    unit="chars",
                    count_label="messages",
                )

        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics using the LiquidAI/LFM2 tokenizer.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSONL dataset files to analyse.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of records to process per file.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Tokenizer to load (defaults to LiquidAI/LFM2-350M-ENJP-MT).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force loading the tokenizer from the local cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"File not found: {path}")
            continue
        dataset_stats = DatasetStats(path)
        dataset_stats.process(tokenizer, limit=args.limit)
        print_dataset_summary(dataset_stats)


if __name__ == "__main__":  # pragma: no cover
    main()
