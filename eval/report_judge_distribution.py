#!/usr/bin/env python3
"""Summarise LLM judge score distributions across MT runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from tabulate import tabulate


RATING_RANGE = range(1, 6)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SELECTION_FILE = Path(__file__).resolve().parent / ".judge_report_selection.json"
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class JudgeSource:
    identifier: str
    path: Path
    display_name: str


@dataclass
class DistributionStats:
    source: JudgeSource
    samples: int
    ratings: list[int]
    counts: Counter[int]
    correct: int

    @property
    def mean(self) -> float:
        return mean(self.ratings)

    @property
    def median(self) -> float:
        return median(self.ratings)

    @property
    def correct_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.correct / self.samples * 100

    @property
    def useful_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        useful_total = sum(self.counts.get(score, 0) for score in range(3, 6))
        return useful_total / self.samples * 100

    @property
    def perfect_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.counts.get(5, 0) / self.samples * 100

    def percentage(self, rating: int) -> float:
        if self.samples == 0:
            return 0.0
        return self.counts.get(rating, 0) / self.samples * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report judge score distributions (1-5) for MT runs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=str,
        help=(
            "Optional judge score files or glob patterns. Defaults to "
            "eval/results/*.llmjudge-scores.jsonl"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory that holds *.llmjudge-scores.jsonl files (default: eval/results/).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive selector to choose which models to include.",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=DEFAULT_SELECTION_FILE,
        help="Path to remember interactive selections between runs (JSON list).",
    )
    parser.add_argument(
        "--reset-selection",
        action="store_true",
        help="Ignore any previously saved selections and include all models.",
    )
    parser.add_argument(
        "--sort",
        choices=["mean", "median", "correct", "samples", "name"],
        default="mean",
        help="Sort output by this column (default: mean).",
    )
    parser.set_defaults(ascending=None)
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument(
        "--ascending",
        dest="ascending",
        action="store_true",
        help="Force ascending order when sorting.",
    )
    order_group.add_argument(
        "--descending",
        dest="ascending",
        action="store_false",
        help="Force descending order when sorting.",
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Render histograms with plain ASCII characters instead of density glyphs.",
    )
    parser.add_argument(
        "--human-names",
        nargs="*",
        default=[],
        help=(
            "Optional mapping in the form safe_name=Display Name. "
            "Example: LiquidAI--LFM2-350M-ENJP-MT.greedy=LiquidAI LFM2 MT (greedy)."
        ),
    )
    return parser.parse_args()


def resolve_display_names(mappings: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in mappings:
        if "=" not in entry:
            raise ValueError(f"Invalid --human-names entry (expected key=value): {entry}")
        key, value = entry.split("=", 1)
        result[key] = value
    return result


def friendly_name_from_path(path: Path) -> str:
    name = path.name.removesuffix(".jsonl").removesuffix(".llmjudge-scores")
    return name.replace("--", "/")


def discover_sources(inputs: Sequence[str], results_dir: Path, human_names: dict[str, str]) -> list[JudgeSource]:
    candidates: list[Path] = []

    if inputs:
        for raw in inputs:
            path = Path(raw)
            if path.is_dir():
                candidates.extend(sorted(path.glob("*.llmjudge-scores.jsonl")))
                continue
            if path.exists():
                candidates.append(path)
                continue

            relative = results_dir / raw
            if relative.exists():
                if relative.is_dir():
                    candidates.extend(sorted(relative.glob("*.llmjudge-scores.jsonl")))
                else:
                    candidates.append(relative)
                continue

            candidates.extend(sorted(results_dir.glob(raw)))
    else:
        candidates = sorted(results_dir.glob("*.llmjudge-scores.jsonl"))

    sources: list[JudgeSource] = []
    for path in candidates:
        if not path.name.endswith(".llmjudge-scores.jsonl"):
            continue
        identifier = path.name
        display = human_names.get(identifier.removesuffix(".jsonl"), None)
        if display is None:
            display = human_names.get(identifier, None)
        if display is None:
            display = friendly_name_from_path(path)
        sources.append(JudgeSource(identifier=identifier, path=path, display_name=display))

    return sources


def load_selection(selection_path: Path) -> set[str]:
    if not selection_path.exists():
        return set()
    try:
        data = json.loads(selection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if isinstance(data, dict) and "selected" in data:
        data = data["selected"]
    if not isinstance(data, list):
        return set()
    return {str(item) for item in data}


def save_selection(selection_path: Path, identifiers: Iterable[str]) -> None:
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(sorted(set(identifiers)), handle, indent=2)
        handle.write("\n")


def ascii_sparkline(values: Sequence[int], ascii_only: bool) -> str:
    if not values:
        return ""
    max_value = max(values)
    if max_value == 0:
        return ""

    if ascii_only:
        charset = " .:-=+*#%@"  # ASCII-friendly density ramp
    else:
        charset = SPARKLINE_CHARS

    scale = len(charset) - 1
    return "".join(charset[int(round((value / max_value) * scale))] for value in values)


def load_distribution(source: JudgeSource) -> DistributionStats | None:
    ratings: list[int] = []
    correct_total = 0
    counts: Counter[int] = Counter()

    try:
        with source.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "judgment":
                    continue
                score = record.get("score")
                if not isinstance(score, int) or score not in RATING_RANGE:
                    continue
                ratings.append(score)
                counts[score] += 1
                correct_value = record.get("correct")
                if isinstance(correct_value, bool):
                    correct_total += 1 if correct_value else 0
                elif isinstance(correct_value, int):
                    correct_total += 1 if correct_value else 0
    except OSError:
        return None

    if not ratings:
        return None

    return DistributionStats(
        source=source,
        samples=len(ratings),
        ratings=ratings,
        counts=counts,
        correct=correct_total,
    )


def format_percentage(value: float) -> str:
    return f"{value:4.1f}"


def interactive_selection(sources: Sequence[JudgeSource], selected: set[str]) -> set[str]:
    if not sources:
        print("No judge score files found.")
        return set()

    current = set(selected) if selected else {source.identifier for source in sources}

    help_message = (
        "Toggle selections by typing numbers (e.g. 1 3 5), "
        "'all' to select all, 'none' to clear, or press Enter when done."
    )

    while True:
        print("\nAvailable judge runs:")
        for index, source in enumerate(sources, start=1):
            mark = "x" if source.identifier in current else " "
            print(f"  [{mark}] {index:>2} {source.display_name} ({source.identifier})")
        print(help_message)
        response = input("> ").strip().lower()

        if response in {"", "done", "ok", "exit", "quit"}:
            break

        if response in {"all", "a"}:
            current = {source.identifier for source in sources}
            continue

        if response in {"none", "clear", "n"}:
            current.clear()
            continue

        toggled = False
        for token in response.replace(",", " ").split():
            if not token.isdigit():
                continue
            index = int(token)
            if 1 <= index <= len(sources):
                identifier = sources[index - 1].identifier
                if identifier in current:
                    current.remove(identifier)
                else:
                    current.add(identifier)
                toggled = True
        if not toggled and response:
            print("  Unrecognised input, keeping current selection.")

    return current


def build_table(distributions: Sequence[DistributionStats], ascii_only: bool) -> str:
    headers = [
        "Model",
        "Samples",
        "Mean",
        "Median",
        "1%",
        "2%",
        "3%",
        "4%",
        "5%",
        "Useful%",
        "Perfect%",
        "Hist",
    ]
    rows = []
    for stats in distributions:
        row = [
            stats.source.display_name,
            stats.samples,
            stats.mean,
            stats.median,
            stats.percentage(1),
            stats.percentage(2),
            stats.percentage(3),
            stats.percentage(4),
            stats.percentage(5),
            stats.useful_rate,
            stats.perfect_rate,
            ascii_sparkline([stats.counts.get(r, 0) for r in RATING_RANGE], ascii_only),
        ]
        rows.append(row)

    colalign = [
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
    ]
    floatfmt = [
        "",  # Model
        "",  # Samples
        ".2f",
        ".1f",
        ".1f",
        ".1f",
        ".1f",
        ".1f",
        ".1f",
        ".1f",
        ".1f",
        "",   # Hist
    ]
    table = tabulate(rows, headers=headers, tablefmt="github", colalign=colalign, floatfmt=floatfmt)
    return (
        table
        + "\n\nNote: percentage columns show share of samples for each 1-5 rating; "
        "Useful% aggregates scores ≥3, and Perfect% is the share of 5s."
    )


def sort_distributions(
    distributions: list[DistributionStats], sort_key: str, explicit_order: bool | None
) -> list[DistributionStats]:
    if explicit_order is None:
        ascending = True if sort_key == "name" else False
    else:
        ascending = explicit_order

    if sort_key == "name":
        key_func = lambda stats: stats.source.display_name.lower()
    elif sort_key == "samples":
        key_func = lambda stats: stats.samples
    elif sort_key == "median":
        key_func = lambda stats: stats.median
    elif sort_key == "correct":
        key_func = lambda stats: stats.correct_rate
    else:  # mean
        key_func = lambda stats: stats.mean

    reverse = not ascending
    return sorted(distributions, key=key_func, reverse=reverse)


def main() -> None:
    args = parse_args()

    try:
        human_names = resolve_display_names(args.human_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sources = discover_sources(args.inputs, args.results_dir, human_names)
    if not sources:
        raise SystemExit("No judge score files found. Run judge-mt.py first or provide paths explicitly.")

    saved_selection = set() if args.reset_selection else load_selection(args.selection_file)
    if args.interactive:
        selected_ids = interactive_selection(sources, saved_selection)
        save_selection(args.selection_file, selected_ids)
    elif saved_selection:
        selected_ids = saved_selection
    else:
        selected_ids = {source.identifier for source in sources}

    filtered_sources = [source for source in sources if source.identifier in selected_ids]
    if not filtered_sources:
        filtered_sources = sources

    distributions: list[DistributionStats] = []
    skipped: list[str] = []
    for source in filtered_sources:
        stats = load_distribution(source)
        if stats is None:
            skipped.append(str(source.path))
            continue
        distributions.append(stats)

    if not distributions:
        message = "No valid judgments found in the selected files."
        if skipped:
            message += " Skipped: " + ", ".join(skipped)
        raise SystemExit(message)

    distributions = sort_distributions(distributions, args.sort, args.ascending)

    table = build_table(distributions, args.ascii_only)
    print(table)

    if skipped:
        print("\nSkipped files with no valid judgments:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
