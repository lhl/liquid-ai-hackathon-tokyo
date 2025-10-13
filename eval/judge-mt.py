#!/usr/bin/env python3
"""LLM-as-a-Judge for machine translation evaluation."""

from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Configuration
REPO_ROOT = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"
JUDGE_TEMPLATE = REPO_ROOT / "judge.j2"
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash-exp"
DEFAULT_SAMPLES = 100
FIXED_SEED = 42
DEFAULT_CONCURRENCY = 20
DEFAULT_THINKING_BUDGET = 512

# Regex patterns for parsing judge responses
SCORE_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>", re.IGNORECASE)
CORRECT_PATTERN = re.compile(r"<correct>\s*(\d)\s*</correct>", re.IGNORECASE)
JUSTIFICATION_PATTERN = re.compile(r"<justification>(.*?)</justification>", re.IGNORECASE | re.DOTALL)


class JudgingError(RuntimeError):
    """Raised when judging cannot proceed."""


def parse_response(text: str) -> tuple[int, int, str]:
    """Parse score, correct, and justification from judge response."""
    score_match = SCORE_PATTERN.search(text)
    correct_match = CORRECT_PATTERN.search(text)
    justification_match = JUSTIFICATION_PATTERN.search(text)

    if not score_match or not correct_match:
        raise JudgingError(f"Could not parse score/correct tags from response: {text}")

    score = int(score_match.group(1))
    correct = int(correct_match.group(1))

    if score not in {1, 2, 3, 4, 5}:
        raise JudgingError(f"Score {score} outside 1-5 range")
    if correct not in {0, 1}:
        raise JudgingError(f"Correct value {correct} invalid")

    justification = justification_match.group(1).strip() if justification_match else ""
    return score, correct, justification


def extract_response_text(response: Any) -> str:
    """Extract text from Gemini API response."""
    if response is None:
        return ""

    # Try direct text attribute
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    # Try candidates
    candidates = getattr(response, "candidates", None)
    if candidates:
        collected = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        text_value = getattr(part, "text", None)
                        if isinstance(text_value, str) and text_value.strip():
                            collected.append(text_value)
        if collected:
            return "\n".join(collected)

    return str(response)


def render_prompt(template_path: Path, context: dict[str, Any]) -> str:
    """Render Jinja2 template with context."""
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    return template.render(**context).strip()


def request_judgment(
    client: genai.Client,
    judge_model: str,
    prompt_text: str,
    gen_config: types.GenerateContentConfig,
    retries: int = 3,
) -> tuple[str, int, int, str]:
    """Request judgment from LLM judge with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=judge_model,
                contents=prompt_text,
                config=gen_config,
            )
            response_text = extract_response_text(response)
            if not response_text.strip():
                raise JudgingError("Judge response was empty")

            score, correct, justification = parse_response(response_text)
            return response_text, score, correct, justification

        except Exception as exc:
            if attempt < retries:
                wait_time = min(2 ** (attempt - 1), 4.0)
                time.sleep(wait_time)
            else:
                raise JudgingError(f"Failed after {retries} attempts: {exc}")

    raise JudgingError(f"Failed after {retries} attempts")


def sanitize_model_name(name: str) -> str:
    """Convert model name to safe filesystem format."""
    return name.replace("/", "--")


def resolve_model_to_predictions(model_identifier: str, results_dir: Path) -> list[Path]:
    """Resolve model name/pattern to prediction files.

    Tries in order:
    1. Direct path if it exists
    2. Safe model name conversion (e.g., LiquidAI/LFM2 -> LiquidAI--LFM2*.predictions.jsonl)
    3. Glob pattern expansion
    """
    # If it's already a valid path, return it
    identifier_path = Path(model_identifier)
    if identifier_path.exists():
        return [identifier_path]

    # Try as path relative to results_dir
    relative_path = results_dir / model_identifier
    if relative_path.exists():
        return [relative_path]

    # Try safe model name conversion
    safe_name = sanitize_model_name(model_identifier)
    if "/" in model_identifier:  # Only try pattern if it looked like a model name
        pattern = f"{safe_name}*.predictions.jsonl"
        matches = list(results_dir.glob(pattern))
        if matches:
            return sorted(matches)

    # Try as glob pattern in results_dir
    matches = list(results_dir.glob(model_identifier))
    if matches:
        return sorted(matches)

    # Try glob pattern with .predictions.jsonl extension if not present
    if not model_identifier.endswith(".predictions.jsonl") and not model_identifier.endswith(".jsonl"):
        pattern = f"{model_identifier}*.predictions.jsonl"
        matches = list(results_dir.glob(pattern))
        if matches:
            return sorted(matches)

    return []


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "prediction":
                predictions.append(record)
    return predictions


def sample_predictions(predictions: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    """Sample predictions using fixed seed."""
    if samples <= 0 or samples >= len(predictions):
        return predictions

    rng = random.Random(seed)
    return rng.sample(predictions, samples)


def group_by_dataset(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group predictions by target dataset."""
    groups = {}
    for pred in predictions:
        dataset = pred.get("target_dataset", "unknown")
        groups.setdefault(dataset, []).append(pred)
    return groups


def load_existing_judgments(output_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing judgments from output file."""
    if not output_path.exists():
        return {}

    judgments = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "judgment":
                # Use id + target_dataset as key for uniqueness
                key = f"{record.get('target_dataset')}::{record.get('id')}"
                judgments[key] = record
    return judgments


def save_judgments(output_path: Path, judgments: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Save judgments and summary to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        # Write all judgments sorted by dataset and id
        for judgment in sorted(judgments, key=lambda x: (x.get("target_dataset", ""), x.get("id", ""))):
            f.write(json.dumps(judgment, ensure_ascii=False) + "\n")

        # Write summary at the end
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def judge_predictions(
    predictions: list[dict[str, Any]],
    judge_model: str,
    template_path: Path,
    client: genai.Client,
    gen_config: types.GenerateContentConfig,
    concurrency: int,
    existing_judgments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Judge predictions in parallel."""
    judgments = []
    errors = []

    def worker(pred: dict[str, Any]) -> dict[str, Any] | None:
        pred_id = pred.get("id", "")
        dataset = pred.get("target_dataset", "unknown")
        key = f"{dataset}::{pred_id}"

        # Skip if already judged
        if key in existing_judgments:
            return existing_judgments[key]

        # Extract fields
        source_text = pred.get("input", "")
        translated_text = pred.get("pred", "")
        reference_text = pred.get("true", "")

        # Render prompt
        context = {
            "source_text": source_text,
            "translated_text": translated_text,
            "reference_text": reference_text,
        }
        prompt_text = render_prompt(template_path, context)

        try:
            response_text, score, correct, justification = request_judgment(
                client, judge_model, prompt_text, gen_config
            )

            return {
                "type": "judgment",
                "id": pred_id,
                "target_dataset": dataset,
                "run_name": pred.get("run_name"),
                "score": score,
                "correct": correct,
                "justification": justification,
                "source_text": source_text,
                "translation": translated_text,
                "reference_text": reference_text,
                "judge_model": judge_model,
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }
        except JudgingError as exc:
            errors.append(f"Error judging {dataset}::{pred_id}: {exc}")
            return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Judging {len(predictions)} predictions...", total=len(predictions))

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, pred) for pred in predictions]
            for future in futures:
                result = future.result()
                if result:
                    judgments.append(result)
                progress.advance(task)

    if errors:
        console.print(f"[yellow]Encountered {len(errors)} errors during judging[/yellow]")
        for error in errors[:5]:  # Show first 5
            console.print(f"  [dim]{error}[/dim]")

    return judgments


def compute_summary(judgments: list[dict[str, Any]], judge_model: str, samples: int, seed: int) -> dict[str, Any]:
    """Compute summary statistics from judgments."""
    if not judgments:
        return {
            "type": "summary",
            "judge_model": judge_model,
            "num_samples": 0,
            "average_score": 0.0,
            "fully_correct": 0,
            "fully_correct_rate": 0.0,
            "sample_seed": seed,
            "max_samples_per_dataset": samples,
            "datasets": [],
            "judged_at": datetime.now(timezone.utc).isoformat(),
        }

    total_score = sum(j["score"] for j in judgments)
    total_correct = sum(j["correct"] for j in judgments)
    datasets = sorted(set(j["target_dataset"] for j in judgments))

    return {
        "type": "summary",
        "judge_model": judge_model,
        "num_samples": len(judgments),
        "average_score": total_score / len(judgments),
        "fully_correct": total_correct,
        "fully_correct_rate": total_correct / len(judgments),
        "sample_seed": seed,
        "max_samples_per_dataset": samples,
        "datasets": datasets,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


@click.command(help="Judge machine translation predictions with LLM-as-a-Judge.")
@click.argument("predictions", type=click.Path(exists=True, path_type=Path), nargs=-1, required=True)
@click.option("--judge", default=DEFAULT_JUDGE_MODEL, show_default=True, help="Judge model identifier")
@click.option("--samples", type=int, default=DEFAULT_SAMPLES, show_default=True,
              help="Number of samples to judge per dataset (0 = all)")
@click.option("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, show_default=True,
              help="Maximum parallel judge requests")
@click.option("--rerun", is_flag=True, help="Force re-judge and overwrite existing judgments")
@click.option("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET, show_default=True,
              help="Gemini thinking budget")
@click.option("--dry-run", is_flag=True, help="Show what would be judged without making API calls")
def main(
    predictions: tuple[Path, ...],
    judge: str,
    samples: int,
    concurrency: int,
    rerun: bool,
    thinking_budget: int,
    dry_run: bool,
) -> None:
    """Judge MT predictions with LLM-as-a-Judge."""

    # Verify template exists
    if not JUDGE_TEMPLATE.exists():
        console.print(f"[red]Judge template not found: {JUDGE_TEMPLATE}[/red]")
        raise SystemExit(1)

    # Setup API client
    if not dry_run:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]GEMINI_API_KEY environment variable not set[/red]")
            raise SystemExit(1)
        client = genai.Client(api_key=api_key)
        gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        )
    else:
        client = None
        gen_config = None

    console.print(f"[bold green]Judge model:[/bold green] {judge}")
    console.print(f"[bold green]Max samples per dataset:[/bold green] {samples if samples > 0 else 'all'}")
    console.print(f"[bold green]Fixed seed:[/bold green] {FIXED_SEED}")
    console.print()

    # Process each predictions file
    for pred_path in predictions:
        console.rule(f"[bold cyan]{pred_path.name}[/bold cyan]")

        # Determine output path
        if pred_path.name.endswith(".predictions.jsonl"):
            base_name = pred_path.name[:-len(".predictions.jsonl")]
        else:
            base_name = pred_path.stem
        output_path = RESULTS_DIR / f"{base_name}.llmjudge-scores.jsonl"

        # Load predictions
        all_predictions = load_predictions(pred_path)
        if not all_predictions:
            console.print("[yellow]No predictions found, skipping[/yellow]")
            continue

        # Group by dataset and sample
        grouped = group_by_dataset(all_predictions)
        sampled_predictions = []
        for dataset, preds in sorted(grouped.items()):
            sampled = sample_predictions(preds, samples, FIXED_SEED)
            sampled_predictions.extend(sampled)
            console.print(f"  [cyan]{dataset}[/cyan]: {len(sampled)} / {len(preds)} samples")

        console.print(f"[green]Total to judge:[/green] {len(sampled_predictions)}")

        if dry_run:
            console.print(f"[yellow]Would output to:[/yellow] {output_path}")
            continue

        # Load existing judgments
        existing_judgments = {} if rerun else load_existing_judgments(output_path)
        if existing_judgments:
            console.print(f"[blue]Found {len(existing_judgments)} existing judgments, resuming...[/blue]")

        # Judge predictions
        judgments = judge_predictions(
            sampled_predictions,
            judge,
            JUDGE_TEMPLATE,
            client,
            gen_config,
            concurrency,
            existing_judgments,
        )

        # Compute summary
        summary = compute_summary(judgments, judge, samples, FIXED_SEED)

        # Save results
        save_judgments(output_path, judgments, summary)

        console.print(f"[bold green]Saved {len(judgments)} judgments → {output_path}[/bold green]")
        console.print(f"  Average score: [cyan]{summary['average_score']:.3f}[/cyan]")
        console.print(f"  Fully correct: [cyan]{summary['fully_correct']} / {summary['num_samples']} ({summary['fully_correct_rate']:.1%})[/cyan]")
        console.print()


if __name__ == "__main__":
    main()
