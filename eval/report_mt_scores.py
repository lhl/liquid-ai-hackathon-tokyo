#!/usr/bin/env python3
"""Produce a Markdown table of MT scores from run-mt.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, OrderedDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise COMET (or other) scores from *.scores.jsonl files into a Markdown table."
    )
    parser.add_argument(
        "scores",
        nargs="*",
        type=Path,
        help="Paths to <model>.scores.jsonl files. Defaults to all files under eval/results/ ending with .scores.jsonl.",
    )
    parser.add_argument(
        "--metric",
        default="comet_wmt22",
        help="Metric key to extract from each dataset record (default: comet_wmt22).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to search when score files are not explicitly provided.",
    )
    parser.add_argument(
        "--human-names",
        nargs="*",
        default=[],
        help=(
            "Optional mapping in the form file_stem=DisplayName. Example: "
            "LiquidAI--LFM2-350M-ENJP-MT=LiquidAI LFM2 MT."
        ),
    )
    return parser.parse_args()


def load_scores(path: Path, metric: str) -> tuple[str, dict[str, float | None], float | None]:
    datasets: dict[str, float | None] = {}
    summary: float | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "dataset":
                dataset = record["dataset"]
                datasets[dataset] = record["metrics"].get(metric)
            elif record.get("type") == "summary" and record.get("metric") == metric:
                summary = record.get("score")
    return path.stem, datasets, summary


def resolve_display_names(mappings: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for mapping in mappings:
        if "=" not in mapping:
            raise ValueError(f"Invalid --human-names entry (expected key=value): {mapping}")
        key, value = mapping.split("=", 1)
        result[key] = value
    return result


def format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def markdown_table(
    rows: list[tuple[str, dict[str, float | None], float | None]],
    datasets: list[str],
) -> str:
    headers = ["Model"] + datasets + ["MT avg"]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if h == "Model" else "---:" for h in headers) + " |",
    ]
    for model, dataset_scores, summary in rows:
        cells = [model]
        for dataset in datasets:
            cells.append(format_score(dataset_scores.get(dataset)))
        cells.append(format_score(summary))
        md_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(md_lines)


def main() -> None:
    args = parse_args()
    score_files: list[Path] = []
    if args.scores:
        score_files = list(args.scores)
    else:
        score_files = sorted(args.results_dir.glob("*.scores.jsonl"))
    if not score_files:
        raise SystemExit("No score files found. Run eval/run-mt.py first or specify paths explicitly.")

    display_names = resolve_display_names(args.human_names)
    collected_rows: list[tuple[str, dict[str, float | None], float | None]] = []
    all_datasets: "OrderedDict[str, None]" = OrderedDict()

    for path in score_files:
        stem, datasets, summary = load_scores(path, args.metric)
        all_datasets.update({name: None for name in sorted(datasets)})
        model_name = display_names.get(stem, stem.replace("--", "/").removesuffix(".scores"))
        collected_rows.append((model_name, datasets, summary))

    table = markdown_table(collected_rows, list(all_datasets.keys()))
    print(table)


if __name__ == "__main__":
    main()
