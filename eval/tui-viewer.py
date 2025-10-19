#!/usr/bin/env python3
"""Interactive Textual TUI for browsing translation judgments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SCORES_DIR = BASE_DIR / "scores"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_DATASET = "all"


try:  # Defer Textual import so users get a clear message if it's missing.
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, ListItem, ListView, Select, Static
except ImportError as exc:  # pragma: no cover - executed only when Textual is absent
    App = None  # type: ignore[assignment]
    ComposeResult = None  # type: ignore[assignment]
    Horizontal = Vertical = VerticalScroll = None  # type: ignore[assignment]
    reactive = None  # type: ignore[assignment]
    Footer = Header = ListItem = ListView = Select = Static = None  # type: ignore[assignment]
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None


def display_model_name(safe_name: str) -> str:
    prefix, dot, suffix = safe_name.partition(".")
    prefix = prefix.replace("--", "/")
    return prefix + (dot + suffix if dot else "")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as src:
            return json.load(src)
    except json.JSONDecodeError:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def list_datasets(results_dir: Path) -> List[str]:
    """List all unique datasets from scores files in results directory"""
    datasets_set = set()

    if not results_dir.exists():
        return ["all"]

    for scores_file in results_dir.glob("*.scores.jsonl"):
        dataset_scores, _ = load_model_scores(scores_file)
        datasets_set.update(dataset_scores.keys())

    datasets = sorted(datasets_set)
    if datasets:
        datasets.insert(0, "all")  # Add "all" option at the beginning
    else:
        datasets = ["all"]

    return datasets


@dataclass
class PredictionRecord:
    index: int
    dataset: str
    pred_id: str
    input_text: str
    prediction: str
    reference: str
    exact_match: Optional[int]
    char_f1: Optional[float]
    metadata: Dict[str, Any]
    raw: Dict[str, Any]
    judge_score: Optional[int] = None
    judge_correct: Optional[int] = None
    judge_justification: Optional[str] = None

    @property
    def is_failure(self) -> bool:
        if self.exact_match is not None:
            return self.exact_match == 0
        if self.char_f1 is not None:
            return self.char_f1 < 0.5
        return False


@dataclass
class DatasetRun:
    dataset: str
    safe_model: str
    predictions: List[PredictionRecord]
    metrics: Dict[str, float]
    num_samples: int
    default_metric: str
    default_metric_score: float
    judge_average_score: Optional[float] = None
    judge_correct_rate: Optional[float] = None
    judge_samples: int = 0
    judge_average_score: Optional[float]
    judge_correct_rate: Optional[float]
    judge_samples: int

    @property
    def total(self) -> int:
        return len(self.predictions)


@dataclass
class ModelData:
    safe_name: str
    display_name: str
    dataset_runs: List[DatasetRun]
    summary_score: Optional[float]
    summary_metric: Optional[str]
    summary_datasets: List[str]
    judge_summary_average: Optional[float] = None
    judge_summary_correct_rate: Optional[float] = None
    judge_summary_samples: int = 0


def load_model_scores(scores_path: Path) -> tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load scores.jsonl and return (dataset_scores, summary)"""
    dataset_scores: Dict[str, Dict[str, Any]] = {}
    summary = None

    if not scores_path.exists():
        return dataset_scores, summary

    with scores_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("type") == "dataset":
                dataset_scores[data["dataset"]] = data
            elif data.get("type") == "summary":
                summary = data

    return dataset_scores, summary


def load_judge_data(
    safe_model: str,
    results_dir: Path,
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load LLM judge results if available for a given run."""
    judge_path = results_dir / f"{safe_model}.llmjudge-scores.jsonl"
    if not judge_path.exists():
        return {}, {}, None

    per_prediction: Dict[tuple[str, str], Dict[str, Any]] = {}
    per_dataset_accumulator: Dict[str, Dict[str, Any]] = {}
    summary_record: Optional[Dict[str, Any]] = None

    with judge_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_type = data.get("type")
            if record_type == "judgment":
                dataset = data.get("target_dataset", "unknown")
                pred_id = str(data.get("id"))
                per_prediction[(dataset, pred_id)] = data

                score = safe_int(data.get("score"))
                correct = safe_int(data.get("correct"))
                accumulator = per_dataset_accumulator.setdefault(
                    dataset,
                    {"scores": [], "correct": 0, "count": 0},
                )
                if score is not None:
                    accumulator["scores"].append(score)
                if correct is not None:
                    accumulator["correct"] += correct
                accumulator["count"] += 1
            elif record_type == "summary":
                summary_record = data

    per_dataset_summary: Dict[str, Dict[str, Any]] = {}
    for dataset, stats in per_dataset_accumulator.items():
        count = stats["count"]
        if count <= 0:
            continue
        avg_score = mean(stats["scores"]) if stats["scores"] else None
        correct_rate = (
            stats["correct"] / count if count > 0 else None
        )
        per_dataset_summary[dataset] = {
            "judge_average_score": avg_score,
            "judge_correct_rate": correct_rate,
            "judge_samples": count,
        }

    return per_prediction, per_dataset_summary, summary_record


def load_model_predictions(
    predictions_path: Path,
    dataset_filter: Optional[str] = None,
    judge_lookup: Optional[Dict[tuple[str, str], Dict[str, Any]]] = None,
) -> List[PredictionRecord]:
    """Load predictions.jsonl, optionally filtering by dataset"""
    records: List[PredictionRecord] = []

    if not predictions_path.exists():
        return records

    with predictions_path.open("r", encoding="utf-8") as src:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("type") != "prediction":
                continue

            dataset = data.get("target_dataset", "unknown")
            if dataset_filter and dataset != dataset_filter:
                continue

            pred_id = str(data.get("id", str(idx)))

            judge_data = judge_lookup.get((dataset, pred_id)) if judge_lookup else None

            record = PredictionRecord(
                index=idx,
                dataset=dataset,
                pred_id=pred_id,
                input_text=data.get("input", ""),
                prediction=data.get("pred", ""),
                reference=data.get("true", ""),
                exact_match=safe_int(data.get("exact")),
                char_f1=safe_float(data.get("char_f1")),
                metadata=data.get("metadata") or {},
                judge_score=safe_int(judge_data.get("score")) if judge_data else None,
                judge_correct=safe_int(judge_data.get("correct")) if judge_data else None,
                judge_justification=judge_data.get("justification") if judge_data else None,
                raw=data,
            )
            records.append(record)

    return records


def load_models(dataset: str, results_dir: Path) -> List[ModelData]:
    """Load models from results directory.

    Args:
        dataset: Dataset filter ("all" for all datasets, or specific dataset name)
        results_dir: Path to results directory with *.scores.jsonl and *.predictions.jsonl files
    """
    models: Dict[str, ModelData] = {}

    if not results_dir.exists():
        return []

    # Find all *.scores.jsonl files
    for scores_file in sorted(results_dir.glob("*.scores.jsonl")):
        # Extract model name from filename (remove .scores.jsonl)
        safe_model = scores_file.name.replace(".scores.jsonl", "")

        # Load scores and summary
        dataset_scores, summary = load_model_scores(scores_file)

        if not dataset_scores:
            continue

        # Load predictions
        predictions_file = results_dir / f"{safe_model}.predictions.jsonl"
        judge_lookup, judge_dataset_stats, judge_summary = load_judge_data(safe_model, results_dir)

        # Create model data
        model = ModelData(
            safe_name=safe_model,
            display_name=display_model_name(safe_model),
            dataset_runs=[],
            summary_score=summary.get("score") if summary else None,
            summary_metric=summary.get("metric") if summary else None,
            summary_datasets=summary.get("datasets", []) if summary else [],
            judge_summary_average=safe_float(judge_summary.get("average_score")) if judge_summary else None,
            judge_summary_correct_rate=safe_float(judge_summary.get("fully_correct_rate")) if judge_summary else None,
            judge_summary_samples=safe_int(judge_summary.get("num_samples")) if judge_summary else 0,
        )

        # Create dataset runs
        for dataset_name, dataset_info in dataset_scores.items():
            # Skip if filtering by dataset and this doesn't match
            if dataset != "all" and dataset != dataset_name:
                continue

            predictions = load_model_predictions(
                predictions_file,
                dataset_name,
                judge_lookup,
            )

            judge_metrics = judge_dataset_stats.get(dataset_name, {})

            dataset_run = DatasetRun(
                dataset=dataset_name,
                safe_model=safe_model,
                predictions=predictions,
                metrics=dataset_info.get("metrics", {}),
                num_samples=dataset_info.get("num_samples", 0),
                default_metric=dataset_info.get("default_metric", ""),
                default_metric_score=dataset_info.get("default_metric_score", 0.0),
                judge_average_score=safe_float(judge_metrics.get("judge_average_score")),
                judge_correct_rate=safe_float(judge_metrics.get("judge_correct_rate")),
                judge_samples=safe_int(judge_metrics.get("judge_samples")) or 0,
            )
            model.dataset_runs.append(dataset_run)

        # Only add model if it has dataset runs
        if model.dataset_runs:
            model.dataset_runs.sort(key=lambda run: run.dataset)
            models[safe_model] = model

    ordered = sorted(models.values(), key=lambda m: m.display_name.lower())
    return ordered


def format_run_title(run: DatasetRun) -> Text:
    text = Text()
    text.append(run.dataset, style="bold cyan")
    text.append(f" • {run.total} samples")
    if run.default_metric and run.default_metric_score is not None:
        text.append(f" • {run.default_metric}: {run.default_metric_score:.4f}")
    if run.judge_average_score is not None:
        text.append(f" • judge_avg: {run.judge_average_score:.2f}")
    if run.judge_correct_rate is not None:
        text.append(f" • judge_acc: {run.judge_correct_rate:.1%}")
    return text


def build_record_renderable(record: PredictionRecord) -> Group:
    icon = "✅" if record.exact_match == 1 else "⚠️" if record.exact_match == 0 else "❔"
    icon_style = "green" if record.exact_match == 1 else "bold red" if record.exact_match == 0 else "yellow"
    header = Text()
    header.append(icon + " ", style=icon_style)
    header.append(f"ID: {record.pred_id}", style="bold")
    if record.char_f1 is not None:
        header.append(f" · char_f1: {record.char_f1:.3f}")
    if record.exact_match is not None:
        header.append(f" · exact: {record.exact_match}")
    if record.judge_score is not None:
        judge_status = "✓" if record.judge_correct == 1 else "✗" if record.judge_correct == 0 else "?"
        header.append(f" · judge: {record.judge_score}{judge_status}")

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=12, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1)
    table.add_row("Input", "|", Text(record.input_text))
    table.add_row("Prediction", "|", Text(record.prediction, style="cyan"))
    table.add_row("Reference", "|", Text(record.reference, style="green"))
    if record.judge_score is not None:
        judge_summary = f"Score {record.judge_score}"
        if record.judge_correct is not None:
            judge_summary += " (correct)" if record.judge_correct == 1 else " (incorrect)"
        table.add_row("LLM Judge", "|", Text(judge_summary, style="magenta"))
        if record.judge_justification:
            table.add_row("Justification", "|", Text(record.judge_justification, style="dim"))
    return Group(header, table)


def build_run_renderable(records: Sequence[PredictionRecord]) -> Group:
    if not records:
        return Group(Text("No predictions match the current filter."))

    renderables: List[Any] = []
    for idx, record in enumerate(records):
        renderables.append(build_record_renderable(record))
        if idx != len(records) - 1:
            renderables.append(Text(""))
    return Group(*renderables)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive viewer for translation predictions.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name (default: all)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Path to results directory")
    parser.add_argument("--model", help="Model name to preselect (safe or display name)")
    args = parser.parse_args(argv)

    dataset = args.dataset
    results_dir = args.results_dir

    datasets = list_datasets(results_dir)
    if dataset not in datasets:
        datasets.append(dataset)
        datasets.sort()

    models = load_models(dataset, results_dir)
    if App is None or TEXTUAL_IMPORT_ERROR is not None:
        print("This viewer requires the 'textual' package. Install it with `pip install textual`.")
        if TEXTUAL_IMPORT_ERROR is not None:
            print(f"Import error: {TEXTUAL_IMPORT_ERROR}")
        return 1

    app = EvaluationViewerApp(
        dataset=dataset,
        datasets=datasets,
        models=models,
        results_dir=results_dir,
        preselect_model=args.model,
    )
    app.run()
    return 0


if App is not None:

    class ModelListItem(ListItem):
        def __init__(self, model: ModelData):
            # Format model name with summary score if available
            label = model.display_name
            if model.summary_score is not None and model.summary_metric:
                label += f" ({model.summary_metric}: {model.summary_score:.4f})"
            if model.judge_summary_average is not None:
                judge_label = f"judge_avg {model.judge_summary_average:.2f}"
                if model.judge_summary_correct_rate is not None:
                    judge_label += f", acc {model.judge_summary_correct_rate:.1%}"
                label += f" [{judge_label}]"
            super().__init__(Static(label))
            self.model = model

    class DatasetRunListItem(ListItem):
        def __init__(self, run: DatasetRun):
            super().__init__(Static(format_run_title(run)))
            self.run = run

    class EvaluationViewerApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            layout: horizontal;
            height: 1fr;
        }
        #sidebar {
            width: 32;
            min-width: 28;
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 1 0;
        }
        #dataset-select {
            margin: 0 1 1 1;
        }
        #model-list {
            height: 1fr;
            margin: 0 1 0 1;
            overflow: auto;
        }
        #main {
            layout: vertical;
            width: 1fr;
            height: 1fr;
            padding: 1;
        }
        #run-list {
            height: 3;
            overflow: auto;
            border: solid $surface-darken-1;
            margin-bottom: 1;
        }
        #content-summary {
            min-height: 1;
            margin-bottom: 1;
        }
        #details-panel {
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("f", "toggle_failures", "Toggle failures"),
            ("j", "toggle_judged", "Toggle judged"),
        ]

        show_failures_only = reactive(False)
        show_judged_only = reactive(False)

        def __init__(
            self,
            dataset: str,
            datasets: List[str],
            models: List[ModelData],
            results_dir: Path,
            preselect_model: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.dataset = dataset
            self.datasets = datasets
            self.models = models
            self.results_dir = results_dir
            self.preselect_model = preselect_model
            self.selected_model: Optional[ModelData] = None
            self.selected_run: Optional[DatasetRun] = None

        def compose(self) -> ComposeResult:
            dataset_options = [(name, name) for name in self.datasets]
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Select(options=dataset_options, value=self.dataset, id="dataset-select")
                    yield ListView(id="model-list")
                with Vertical(id="main"):
                    yield ListView(id="run-list")
                    yield Static("", id="content-summary")
                    with VerticalScroll(id="details-panel"):
                        yield Static("Select a model to view predictions.", id="details-content")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"Predictions · {self.dataset}"
            if not self.models:
                self.models = load_models(self.dataset, self.results_dir)
            self._populate_models()

        def action_toggle_failures(self) -> None:
            self.show_failures_only = not self.show_failures_only

        def action_toggle_judged(self) -> None:
            self.show_judged_only = not self.show_judged_only

        def watch_show_failures_only(self, _: bool) -> None:
            self._refresh_content()

        def watch_show_judged_only(self, _: bool) -> None:
            self._refresh_content()

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            if isinstance(event.item, ModelListItem):
                self._select_model(event.item.model)
            elif isinstance(event.item, DatasetRunListItem):
                self._select_run(event.item.run)

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "dataset-select" and event.value:
                self._change_dataset(str(event.value))

        def _change_dataset(self, dataset: str) -> None:
            if dataset == self.dataset:
                return
            self.dataset = dataset
            self.title = f"Predictions · {self.dataset}"
            self.models = load_models(self.dataset, self.results_dir)
            self.preselect_model = None
            self.selected_model = None
            self.selected_run = None
            self.show_failures_only = False
            self.show_judged_only = False
            self._populate_models()

        def _populate_models(self) -> None:
            model_list = self.query_one("#model-list", ListView)
            run_list = self.query_one("#run-list", ListView)
            details = self.query_one("#details-content", Static)
            summary = self.query_one("#content-summary", Static)
            model_list.clear()
            run_list.clear()
            details.update(Text("Select a model to view predictions."))
            summary.update("")
            self.selected_model = None
            self.selected_run = None

            if not self.models:
                model_list.append(ListItem(Static("No models found for this dataset.")))
                return

            for model in self.models:
                model_list.append(ModelListItem(model))

            index = self._find_model_index(self.preselect_model)
            index = max(0, min(index, len(self.models) - 1))
            model_list.index = index
            self._select_model(self.models[index])

        def _find_model_index(self, target: Optional[str]) -> int:
            if not target:
                return 0
            target_lower = target.lower()
            for idx, model in enumerate(self.models):
                if model.safe_name.lower() == target_lower or model.display_name.lower() == target_lower:
                    return idx
            return 0

        def _select_model(self, model: Optional[ModelData]) -> None:
            if model is None:
                self.selected_model = None
                self.selected_run = None
                self._refresh_content()
                return
            self.selected_model = model
            self._populate_runs()

        def _populate_runs(self) -> None:
            run_list = self.query_one("#run-list", ListView)
            run_list.clear()
            self.selected_run = None

            if not self.selected_model:
                run_list.append(ListItem(Static("Select a model.")))
                return

            runs = self.selected_model.dataset_runs
            if not runs:
                run_list.append(ListItem(Static("No dataset runs for this model.")))
                self._refresh_content()
                return

            for run in runs:
                run_list.append(DatasetRunListItem(run))

            # Auto-select first run
            run_list.index = 0
            self._select_run(runs[0])

        def _select_run(self, run: Optional[DatasetRun]) -> None:
            self.selected_run = run
            self._refresh_content()

        def _refresh_content(self) -> None:
            summary = self.query_one("#content-summary", Static)
            details = self.query_one("#details-content", Static)

            if not self.selected_run:
                summary.update("Select a dataset run to view predictions.")
                details.update(Text("Select a model and dataset to display predictions."))
                return

            records = self.selected_run.predictions
            if self.show_failures_only:
                records = [record for record in records if record.is_failure]
            if self.show_judged_only:
                records = [record for record in records if record.judge_score is not None]
            summary_text = f"Showing {len(records)} of {self.selected_run.total} predictions"
            if self.show_failures_only:
                summary_text += " (failures only)"
            if self.show_judged_only:
                summary_text += " (judged only)"
            if self.selected_run.judge_samples:
                summary_text += f" · judged: {self.selected_run.judge_samples}"
            summary.update(summary_text)

            if not records:
                details.update(Text("No predictions match the current filter."))
                return

            details.update(build_run_renderable(records))


if __name__ == "__main__":
    sys.exit(main())
