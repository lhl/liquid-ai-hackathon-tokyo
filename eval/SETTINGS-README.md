# Run Settings and File Naming

## Overview

All runs now include a `.settings.json` file alongside the results that contains complete configuration for replicability. File names now include all non-default parameters.

## Enhanced File Naming

### Pattern
```
{model}.{sampling}-{params}-{template}-{other}
```

### Components

- **Model**: `LiquidAI--LFM2-350M-ENJP-MT` (slashes→double-dash)
- **Sampling**:
  - Sampled: `sampled-t{temp}-p{top_p}` (e.g., `sampled-t0_5-p1`)
  - Greedy: `greedy`
- **Non-default params** (only if set):
  - Min-p: `m{value}` (e.g., `m0_1` for 0.1)
  - Repetition penalty: `r{value}` (e.g., `r1_05` for 1.05)
- **Chat format**:
  - With format: `ct_{format}` (e.g., `ct_lfm2` for `--format lfm2`)
  - Without format: `ct_none`
- **Other**:
  - Split: `split-{name}` (e.g., `split-dev`)
  - Max samples: `max{n}` (e.g., `max100`)
  - Run tag: `{tag}`

### Examples

```bash
# LFM2 with optimal settings + chat template
LiquidAI--LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2.scores.jsonl
LiquidAI--LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2.predictions.jsonl
LiquidAI--LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2.settings.json
LiquidAI--LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2.log

# Gemma with defaults (no extra params)
google--gemma-3-4b-it.sampled-t0_2-p0_9.scores.jsonl
google--gemma-3-4b-it.sampled-t0_2-p0_9.predictions.jsonl
google--gemma-3-4b-it.sampled-t0_2-p0_9.settings.json
google--gemma-3-4b-it.sampled-t0_2-p0_9.log

# Greedy decoding
LiquidAI--LFM2-350M-ENJP-MT.greedy.scores.jsonl
LiquidAI--LFM2-350M-ENJP-MT.greedy.predictions.jsonl
LiquidAI--LFM2-350M-ENJP-MT.greedy.settings.json
```

## Settings JSON Format

Each `.settings.json` file contains:

```json
{
  "model": "google/gemma-3-4b-it",
  "run_name": "google--gemma-3-4b-it.sampled-t0_2-p0_9",
  "timestamp": "2025-10-13T15:30:00+0000",
  "settings": {
    "temperature": {
      "value": 0.2,
      "default": 0.2,
      "is_default": true
    },
    "min_p": {
      "value": 0.1,
      "default": 0.0,
      "is_default": false
    }
    // ... all other settings with is_default markers
  },
  "flags": [
    "google/gemma-3-4b-it",
    "--min-p", "0.1"
    // ... only non-default flags
  ]
}
```

### Key Features

1. **Complete settings**: All parameters with their values and defaults
2. **Default markers**: `is_default` flag shows which settings differ from defaults
3. **Reproducible command**: `flags` array contains exact command to reproduce
4. **Timestamp**: When the run was executed

## Backfilling Existing Runs

For runs that don't have `.settings.json` files yet:

```bash
# Dry run to see what would be created
python backfill-settings.py --dry-run

# Create settings files for all existing runs
python backfill-settings.py

# Force overwrite existing settings files
python backfill-settings.py --force
```

**Note**: Backfilled settings are marked with `"timestamp": "unknown (backfilled)"` and include a note that settings were inferred from filenames.

## Replicating a Run

From any `.settings.json` file:

```bash
# View the exact command to replicate
jq -r '.flags | join(" ")' results/model.settings.json | xargs python run-mt.py

# Or directly:
python run-mt.py $(jq -r '.flags | join(" ")' results/model.settings.json)
```

## Example: Running LFM2 with Chat Format

```bash
python run-mt.py LiquidAI/LFM2-350M-ENJP-MT \
  --format lfm2 \
  --temperature 0.5 \
  --top-p 1.0 \
  --min-p 0.1 \
  --repetition-penalty 1.05
```

This creates:
- `LiquidAI--LFM2-350M-ENJP-MT.sampled-t0_5-p1-m0_1-r1_05-ct_lfm2.{scores,predictions,settings,log}.{jsonl,json}`

## Chat Format Files

For `--format NAME`, the runner looks for:
- `NAME.system.j2` (optional): renders the system message. Empty file ⇒ no system message. Missing file ⇒ dataset instruction fallback with “Return ONLY the translated text.” appended when absent.
- `NAME.user.j2` (optional): renders the user message. Empty or missing file ⇒ raw sample text.

Template context:
- `{{ language }}` / `{{ target_language }}`: inferred from dataset direction (`alt-e-to-j` → Japanese, `alt-j-to-e` → English).
- `{{ instruction }}`: dataset instruction text.
- `{{ text }}`: sample input.
- `{{ conversation_history }}`: optional list from sample metadata.
