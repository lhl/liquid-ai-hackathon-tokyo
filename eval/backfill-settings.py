#!/usr/bin/env python3
"""Backfill settings.json files for existing result files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def parse_run_name(run_name: str) -> dict[str, Any]:
    """Parse run name to extract settings."""
    settings: dict[str, Any] = {}

    # Extract model name (everything before first dot)
    parts = run_name.split(".")
    if not parts:
        return settings

    model = parts[0].replace("--", "/")
    settings["model"] = model

    # Default values
    settings["split"] = "test"
    settings["datasets"] = ["alt-e-to-j", "alt-j-to-e"]
    settings["temperature"] = 0.2
    settings["top_p"] = 0.9
    settings["min_p"] = 0.0
    settings["repetition_penalty"] = 1.0
    settings["do_sample"] = True
    settings["max_samples"] = -1
    settings["format"] = None

    # Parse variant components
    for part in parts[1:]:
        # Greedy decoding
        if part == "greedy":
            settings["do_sample"] = False
            continue

        # Sampled with temperature and top_p
        if part.startswith("sampled-"):
            settings["do_sample"] = True
            # Extract t and p values
            match = re.search(r"t([0-9_]+)-p([0-9_]+)", part)
            if match:
                t_str = match.group(1).replace("_", ".")
                p_str = match.group(2).replace("_", ".")
                settings["temperature"] = float(t_str)
                settings["top_p"] = float(p_str)
            continue

        # Min-p
        if part.startswith("m") and part[1:].replace("_", "").replace(".", "").isdigit():
            m_str = part[1:].replace("_", ".")
            settings["min_p"] = float(m_str)
            continue

        # Repetition penalty
        if part.startswith("r") and part[1:].replace("_", "").replace(".", "").isdigit():
            r_str = part[1:].replace("_", ".")
            settings["repetition_penalty"] = float(r_str)
            continue

        # Chat template / format
        if part.startswith("ct_"):
            template_name = part[3:]
            if template_name and template_name != "none":
                settings["format"] = template_name
            continue

        # Split
        if part.startswith("split-"):
            settings["split"] = part[6:]
            continue

        # Max samples
        if part.startswith("max") and part[3:].isdigit():
            settings["max_samples"] = int(part[3:])
            continue

    return settings


def create_settings_json(run_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """Create a settings.json structure from parsed settings."""
    # Defaults
    defaults = {
        "split": "test",
        "datasets": ["alt-e-to-j", "alt-j-to-e"],
        "temperature": 0.2,
        "top_p": 0.9,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "max_samples": -1,
        "format": None,
        "batch_size": 256,
        "num_few_shots": 4,
        "extra_output_tokens": 0,
        "dtype": "auto",
        "engine": "auto",
    }

    settings_dict: dict[str, Any] = {
        "model": parsed.get("model", "unknown"),
        "run_name": run_name,
        "timestamp": "unknown (backfilled)",
        "settings": {},
        "flags": [],
        "note": "This settings file was backfilled from the run name. Some settings may be approximate.",
    }

    # Build settings with defaults
    for key, default_value in defaults.items():
        actual_value = parsed.get(key, default_value)
        is_default = actual_value == default_value

        settings_dict["settings"][key] = {
            "value": actual_value,
            "default": default_value,
            "is_default": is_default,
        }

    # Build command-line flags
    flags = [parsed.get("model", "unknown")]

    if parsed.get("split") != defaults["split"]:
        flags.extend(["--split", parsed["split"]])

    if parsed.get("datasets") != defaults["datasets"]:
        flags.append("--datasets")
        flags.extend(parsed["datasets"])

    if parsed.get("max_samples", -1) != defaults["max_samples"]:
        flags.extend(["--max-samples", str(parsed["max_samples"])])

    if parsed.get("format"):
        flags.extend(["--format", parsed["format"]])

    if parsed.get("temperature") != defaults["temperature"]:
        flags.extend(["--temperature", str(parsed["temperature"])])

    if parsed.get("top_p") != defaults["top_p"]:
        flags.extend(["--top-p", str(parsed["top_p"])])

    if parsed.get("min_p", 0.0) != defaults["min_p"]:
        flags.extend(["--min-p", str(parsed["min_p"])])

    if parsed.get("repetition_penalty", 1.0) != defaults["repetition_penalty"]:
        flags.extend(["--repetition-penalty", str(parsed["repetition_penalty"])])

    if not parsed.get("do_sample"):
        flags.append("--no-sample")

    settings_dict["flags"] = flags

    return settings_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill settings.json files for existing runs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing settings.json files",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all .scores.jsonl files
    score_files = list(results_dir.glob("*.scores.jsonl"))
    if not score_files:
        print(f"No score files found in {results_dir}")
        return

    print(f"Found {len(score_files)} score file(s)")

    created = 0
    skipped = 0

    for score_file in sorted(score_files):
        run_name = score_file.stem.removesuffix(".scores")
        settings_file = results_dir / f"{run_name}.settings.json"

        if settings_file.exists() and not args.force:
            skipped += 1
            continue

        # Parse run name to extract settings
        parsed = parse_run_name(run_name)
        settings = create_settings_json(run_name, parsed)

        if args.dry_run:
            print(f"\nWould create: {settings_file.name}")
            print(f"  Model: {parsed.get('model')}")
            print(f"  Sampling: {parsed.get('do_sample')}")
            if parsed.get("do_sample"):
                print(f"  Temperature: {parsed.get('temperature')}, top_p: {parsed.get('top_p')}")
                if parsed.get("min_p", 0.0) > 0:
                    print(f"  Min-p: {parsed.get('min_p')}")
                if parsed.get("repetition_penalty", 1.0) != 1.0:
                    print(f"  Repetition penalty: {parsed.get('repetition_penalty')}")
            if parsed.get("format"):
                print(f"  Format: {parsed.get('format')}")
        else:
            with settings_file.open("w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"Created: {settings_file.name}")

        created += 1

    print(f"\n{'Would create' if args.dry_run else 'Created'}: {created}")
    if skipped > 0:
        print(f"Skipped (already exist): {skipped}")
    if args.dry_run:
        print("\nRun without --dry-run to create files")


if __name__ == "__main__":
    main()
