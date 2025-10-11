#!/usr/bin/env python3
"""Sample translated examples from run-mt.py predictions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample translation records from <model>.predictions.jsonl."
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model identifier (e.g., LiquidAI/LFM2-350M-ENJP-MT). Used to locate default predictions file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Explicit path to a predictions JSONL file. Overrides the model argument.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of samples to display.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing results/*.predictions.jsonl files.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Include prompt text in the output (can be verbose).",
    )
    return parser.parse_args()


def sanitize_run_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "prediction":
                records.append(record)
    if not records:
        raise SystemExit(f"No prediction records found in {path}")
    return records


def pick_samples(records: Iterable[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    records_list = list(records)
    if count >= len(records_list):
        return records_list
    rng = random.Random(seed)
    return rng.sample(records_list, count)


def render_sample(record: dict[str, Any], index: int, show_prompt: bool) -> str:
    dataset = record.get("target_dataset", "unknown")
    sample_id = record.get("id", "n/a")
    header = f"### Sample {index}: {dataset} (id={sample_id})"
    blocks = [
        header,
        f"**Input**:\n{record.get('input', '').strip()}",
        f"**Prediction**:\n{record.get('pred', '').strip()}",
        f"**Reference**:\n{record.get('true', '').strip()}",
    ]
    if show_prompt:
        blocks.append(f"**Prompt**:\n{record.get('prompt', '').strip()}")
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions
    if predictions_path is None:
        if not args.model:
            raise SystemExit("Provide a model name or --predictions path.")
        sanitized = sanitize_run_name(args.model)
        predictions_path = args.results_dir / f"{sanitized}.predictions.jsonl"
    if not predictions_path.exists():
        raise SystemExit(f"Predictions file not found: {predictions_path}")

    records = load_predictions(predictions_path)
    samples = pick_samples(records, args.count, args.seed)
    for idx, record in enumerate(samples, start=1):
        print(render_sample(record, idx, args.show_prompt))
        print()


if __name__ == "__main__":
    main()
