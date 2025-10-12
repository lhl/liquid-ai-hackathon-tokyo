#!/usr/bin/env python3
"""Interactive TUI for browsing MT run scores by model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

try:  # Defer Textual import to provide a helpful error when missing.
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, ListItem, ListView, Static
except ImportError as exc:  # pragma: no cover - executed only when Textual is absent
    App = None  # type: ignore[assignment]
    ComposeResult = None  # type: ignore[assignment]
    Horizontal = Vertical = VerticalScroll = None  # type: ignore[assignment]
    Footer = Header = ListItem = ListView = Static = None  # type: ignore[assignment]
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None

from report_mt_scores import load_scores, resolve_display_names


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


def load_predictions(path: Path) -> List[Dict]:
    """Load predictions from a .predictions.jsonl file."""
    predictions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "prediction":
                predictions.append(record)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display MT evaluation scores for all runs grouped by model in an interactive TUI."
    )
    parser.add_argument(
        "scores",
        nargs="*",
        type=Path,
        help="Optional paths to <model>.scores.jsonl files. Defaults to all results/*.scores.jsonl files.",
    )
    parser.add_argument(
        "--metric",
        default="comet_wmt22",
        help="Metric key to display (default: comet_wmt22).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to scan when score files are not passed explicitly.",
    )
    parser.add_argument(
        "--human-names",
        nargs="*",
        default=[],
        help="Optional mapping in the form file_stem=DisplayName. Example: LiquidAI--LFM2-350M=LiquidAI LFM2.",
    )
    return parser.parse_args()


def extend_display_name_map(mappings: Dict[str, str]) -> Dict[str, str]:
    """Allow lookups by both <stem> and <stem>.scores to match score filenames."""
    extended = dict(mappings)
    for key, value in list(mappings.items()):
        if not key.endswith(".scores"):
            extended[f"{key}.scores"] = value
    return extended


def split_base_and_variant(raw_safe_name: str) -> tuple[str, str]:
    """Split a raw stem (without .scores suffix) into base model and run variant."""
    if "." in raw_safe_name:
        base, variant = raw_safe_name.split(".", 1)
        return base, variant
    return raw_safe_name, ""


def normalise_model_name(raw_safe_name: str) -> str:
    """Convert the raw safe filename stem into a human-friendly display string."""
    return raw_safe_name.replace("--", "/")


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


@dataclass
class ScoreRun:
    key: str
    label: str
    display_name: str
    dataset_scores: Dict[str, Optional[float]]
    summary: Optional[float]
    path: Path
    predictions: List[Dict[str, Any]]
    predictions_path: Optional[Path]


@dataclass
class ModelGroup:
    base_key: str
    display_name: str
    runs: List[ScoreRun]

    def dataset_names(self) -> List[str]:
        datasets: set[str] = set()
        for run in self.runs:
            datasets.update(run.dataset_scores.keys())
        return sorted(datasets)


def lookup_display(display_map: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        if key in display_map:
            return display_map[key]
    return None


def collect_model_groups(
    score_files: Sequence[Path],
    metric: str,
    display_map: Dict[str, str],
) -> List[ModelGroup]:
    grouped: Dict[str, ModelGroup] = {}
    for path in sorted(score_files):
        stem, datasets, summary = load_scores(path, metric)
        raw_safe_name = stem.removesuffix(".scores")
        base_key, variant = split_base_and_variant(raw_safe_name)

        base_display = lookup_display(display_map, base_key, f"{base_key}.scores")
        run_display_override = lookup_display(display_map, raw_safe_name, stem)

        human_base = base_display or normalise_model_name(base_key)
        run_label = run_display_override or (variant if variant else "default")
        run_display_name = (
            run_display_override
            or (f"{human_base} · {variant}" if variant else human_base)
        )

        # Load predictions if available
        predictions_path = path.parent / path.name.replace(".scores.jsonl", ".predictions.jsonl")
        predictions = []
        if predictions_path.exists():
            predictions = load_predictions(predictions_path)

        score_run = ScoreRun(
            key=raw_safe_name,
            label=run_label,
            display_name=run_display_name,
            dataset_scores=datasets,
            summary=summary,
            path=path,
            predictions=predictions,
            predictions_path=predictions_path if predictions_path.exists() else None,
        )

        group = grouped.get(base_key)
        if group is None:
            group = ModelGroup(base_key=base_key, display_name=human_base, runs=[])
            grouped[base_key] = group
        group.runs.append(score_run)

    # Sort runs (best summary first) and groups alphabetically.
    for group in grouped.values():
        group.runs.sort(key=lambda run: (run.summary is None, -(run.summary or float("-inf"))))
    return sorted(grouped.values(), key=lambda group: group.display_name.lower())


def build_table(group: ModelGroup, metric: str) -> Table:
    datasets = group.dataset_names()
    table = Table(
        title=f"{group.display_name} — {metric}",
        expand=True,
        show_edge=False,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Run", style="bold")
    for dataset in datasets:
        table.add_column(dataset, justify="right")
    table.add_column("MT avg", justify="right")

    best_dataset_scores: Dict[str, float] = {}
    for dataset in datasets:
        scores = [run.dataset_scores.get(dataset) for run in group.runs]
        cleaned = [score for score in scores if score is not None]
        if cleaned:
            best_dataset_scores[dataset] = max(cleaned)
    best_summary = max(
        (run.summary for run in group.runs if run.summary is not None),
        default=None,
    )

    for run in group.runs:
        cells: List[Text] = [Text(run.label)]
        for dataset in datasets:
            value = run.dataset_scores.get(dataset)
            is_best = best_dataset_scores.get(dataset) is not None and value == best_dataset_scores.get(dataset)
            text = Text(format_score(value))
            if is_best:
                text.stylize("bold green")
            cells.append(text)
        summary_text = Text(format_score(run.summary))
        if best_summary is not None and run.summary == best_summary:
            summary_text.stylize("bold green")
        cells.append(summary_text)
        table.add_row(*cells)
    return table


class ScoreboardApp(App[None]):  # pragma: no cover - interactive application
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #model-panel {
        width: 32;
        border: tall;
    }
    #run-panel {
        width: 40;
        border: tall;
    }
    #detail-panel {
        border: tall;
    }
    .panel-title {
        padding: 0 1;
        text-style: bold;
    }
    ListView {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, groups: Sequence[ModelGroup], metric: str) -> None:
        super().__init__()
        self._groups = list(groups)
        self._metric = metric
        self._detail: Optional[VerticalScroll] = None
        self._model_list: Optional[ListView] = None
        self._run_list: Optional[ListView] = None
        self._current_group: Optional[ModelGroup] = None
        self._current_run: Optional[ScoreRun] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="model-panel"):
                yield Static("Models", classes="panel-title")
                self._model_list = ListView()
                yield self._model_list
            with Vertical(id="run-panel"):
                yield Static("Runs", classes="panel-title")
                self._run_list = ListView()
                yield self._run_list
            self._detail = VerticalScroll(id="detail-panel")
            yield self._detail
        yield Footer()

    def on_mount(self) -> None:
        assert self._model_list is not None
        for group in self._groups:
            label = Text(f"{group.display_name} ({len(group.runs)})")
            item = ListItem(Static(label))
            item.data = group  # type: ignore[attr-defined]
            self._model_list.append(item)

        if self._groups and self._model_list.children:
            self._model_list.index = 0
            first_item = self._model_list.children[0]
            if isinstance(first_item, ListItem):
                self._show_group(first_item.data)  # type: ignore[arg-type]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Check which list was selected
        if event.list_view == self._model_list:
            group = getattr(event.item, "data", None)
            if group:
                self._show_group(group)
        elif event.list_view == self._run_list:
            run = getattr(event.item, "data", None)
            if run:
                self._show_run_predictions(run)

    def _show_group(self, group: ModelGroup) -> None:
        assert self._run_list is not None
        assert self._detail is not None
        self._current_group = group
        self._current_run = None

        # Update run list
        self._run_list.clear()
        for run in group.runs:
            pred_count = len(run.predictions)
            label = Text(f"{run.label} ({pred_count} predictions)")
            item = ListItem(Static(label))
            item.data = run  # type: ignore[attr-defined]
            self._run_list.append(item)

        # Show summary table in detail panel
        self._detail.remove_children()
        table = build_table(group, self._metric)
        run_paths = "\n".join(f"- {run.label}: {run.path.name}" for run in group.runs)
        info = Text(f"{len(group.runs)} run(s) • metric: {self._metric}")
        paths = Text.from_markup(f"[dim]Files:[/dim]\n{run_paths}") if run_paths else Text("")
        content = Static(Group(info, table, paths))
        self._detail.mount(content)

        # Auto-select first run
        if group.runs and self._run_list.children:
            self._run_list.index = 0
            first_run = self._run_list.children[0]
            if isinstance(first_run, ListItem):
                self._show_run_predictions(first_run.data)  # type: ignore[arg-type]

    def _show_run_predictions(self, run: ScoreRun) -> None:
        assert self._detail is not None
        self._current_run = run

        # Clear and show predictions
        self._detail.remove_children()

        if not run.predictions:
            no_data = Static(Text("No predictions available", style="italic dim"))
            self._detail.mount(no_data)
            return

        # Header
        header = Static(Text(f"{run.display_name} — {len(run.predictions)} predictions", style="bold"))
        self._detail.mount(header)

        # Show each prediction
        for i, pred in enumerate(run.predictions):
            separator = Static(Text("─" * 80, style="dim"))
            self._detail.mount(separator)

            # Prediction info
            dataset = pred.get("target_dataset", "unknown")
            pred_id = pred.get("id", i)
            exact = pred.get("exact", 0)
            char_f1 = pred.get("char_f1", 0.0)

            info_text = Text()
            info_text.append(f"[{i+1}/{len(run.predictions)}] ", style="bold cyan")
            info_text.append(f"Dataset: {dataset} | ID: {pred_id} | ", style="dim")
            info_text.append(f"Exact: {exact} | Char F1: {char_f1:.4f}", style="yellow")

            info = Static(info_text)
            self._detail.mount(info)

            # Original prompt (truncated for display)
            prompt_text = pred.get("prompt", "")
            if len(prompt_text) > 500:
                prompt_display = prompt_text[:500] + "..."
            else:
                prompt_display = prompt_text

            prompt_section = Static(
                Text.from_markup(f"[bold]Prompt:[/bold]\n[dim]{prompt_display}[/dim]")
            )
            self._detail.mount(prompt_section)

            # Input
            input_text = pred.get("input", "")
            input_section = Static(
                Text.from_markup(f"[bold]Input:[/bold]\n{input_text}")
            )
            self._detail.mount(input_section)

            # Prediction
            pred_text = pred.get("pred", "")
            pred_section = Static(
                Text.from_markup(f"[bold green]Prediction:[/bold green]\n{pred_text}")
            )
            self._detail.mount(pred_section)

            # True/Reference
            true_text = pred.get("true", "")
            true_section = Static(
                Text.from_markup(f"[bold blue]Reference:[/bold blue]\n{true_text}")
            )
            self._detail.mount(true_section)

    def action_back(self) -> None:
        """Handle back/escape action."""
        if self._current_run is not None and self._current_group is not None:
            # Currently viewing predictions, go back to group view
            self._show_group(self._current_group)


def main() -> None:
    if TEXTUAL_IMPORT_ERROR is not None:
        raise SystemExit(
            "Textual is required for mt_scores_tui.py. "
            "Install with `pip install textual[dev]` or add it to your environment."
        )

    args = parse_args()
    if args.scores:
        score_files = list(args.scores)
    else:
        score_files = sorted(args.results_dir.glob("*.scores.jsonl"))
    if not score_files:
        raise SystemExit("No score files found. Run eval/run-mt.py first or specify score files explicitly.")

    display_names = extend_display_name_map(resolve_display_names(args.human_names))
    groups = collect_model_groups(score_files, args.metric, display_names)
    if not groups:
        raise SystemExit("No model groups could be constructed from the provided score files.")

    app = ScoreboardApp(groups, args.metric)
    app.run()


if __name__ == "__main__":
    main()
