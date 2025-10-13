# LLM-as-a-Judge for Machine Translation Evaluation

## Overview

`judge-mt.py` provides LLM-as-a-Judge scoring for machine translation predictions, complementing automated metrics (BLEU, COMET) with nuanced quality assessments.

## Quick Start

```bash
# Judge a single model using model name (auto-converts to safe filename)
python judge-mt.py "LiquidAI/LFM2-350M"  # Finds LiquidAI--LFM2-350M*.predictions.jsonl

# Judge specific model variant by path
python judge-mt.py results/google--gemma-3-4b-it.sampled-t0_2-p0_9.predictions.jsonl

# Judge multiple models at once using model names
python judge-mt.py "LiquidAI/LFM2-350M" "google/gemma-3-4b-it"

# Judge using glob patterns
python judge-mt.py "LiquidAI*" "google*"

# Judge all models
python judge-mt.py results/*.predictions.jsonl

# Quick test with fewer samples
python judge-mt.py "LiquidAI/LFM2-350M" --samples 10

# Force re-judge (overwrite existing)
python judge-mt.py "LiquidAI/LFM2-350M" --rerun

# Use different judge model
python judge-mt.py "LiquidAI/LFM2-350M" --judge gemini-2.5-pro
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Required for Gemini API access

### Default Settings
- **Judge model**: `gemini-2.0-flash-exp` (fast and cost-effective)
- **Samples per dataset**: 100 (alt-e-to-j: 100, alt-j-to-e: 100)
- **Seed**: 42 (fixed for reproducibility)
- **Concurrency**: 20 parallel requests
- **Thinking budget**: 512 tokens

## Output Format

Each model gets a corresponding `*.llmjudge-scores.jsonl` file in `results/`:

```
results/
├── google--gemma-3-4b-it.sampled-t0_2-p0_9.predictions.jsonl
├── google--gemma-3-4b-it.sampled-t0_2-p0_9.scores.jsonl         # BLEU/COMET
└── google--gemma-3-4b-it.sampled-t0_2-p0_9.llmjudge-scores.jsonl  # LLM judge
```

### Judgment Record Format

```json
{
  "type": "judgment",
  "id": "123",
  "target_dataset": "alt-e-to-j",
  "run_name": "google--gemma-3-4b-it.sampled-t0_2-p0_9",
  "score": 4,
  "correct": 0,
  "justification": "Translation is accurate and natural...",
  "source_text": "Original English text",
  "translation": "Japanese translation",
  "reference_text": "Reference translation",
  "judge_model": "gemini-2.0-flash-exp",
  "judged_at": "2025-10-13T12:34:56.789Z"
}
```

### Summary Record Format

```json
{
  "type": "summary",
  "judge_model": "gemini-2.0-flash-exp",
  "num_samples": 200,
  "average_score": 3.85,
  "fully_correct": 23,
  "fully_correct_rate": 0.115,
  "sample_seed": 42,
  "max_samples_per_dataset": 100,
  "datasets": ["alt-e-to-j", "alt-j-to-e"],
  "judged_at": "2025-10-13T12:45:00.123Z"
}
```

## Scoring Rubric

The judge uses a strict 1-5 scale defined in `judge.j2`:

- **1** – Completely wrong, untranslated, or incomprehensible
- **2** – Major errors in meaning, facts, or grammar
- **3** – Adequate with main idea conveyed, but noticeable issues
- **4** – Good translation, accurate and natural with minor imperfections
- **5** – Excellent, professional-quality translation

Additionally, `correct` (0/1) indicates whether the translation is perfect/native-quality.

## Idempotent Behavior

By default, the script **resumes** from existing judgments:
- Skips already-judged samples
- Only judges new samples
- Use `--rerun` to force complete re-judging

This makes it safe to run multiple times and cheap to add more models incrementally.

## Cost Considerations

Judging is API-based and incurs costs:
- **100 samples/dataset × 2 datasets = 200 API calls per model**
- Default judge (`gemini-2.0-flash-exp`) is cost-effective
- Use `--samples 10` for quick testing
- Use `--samples 0` to judge all predictions (expensive!)

## Integration with Existing Metrics

The flat file structure makes it easy to combine metrics:

```python
import json

# Load COMET scores
with open('results/model.scores.jsonl') as f:
    comet_data = [json.loads(line) for line in f]

# Load LLM judge scores
with open('results/model.llmjudge-scores.jsonl') as f:
    judge_data = [json.loads(line) for line in f]

# Zip together by id/dataset
# Both use same sample_seed=42 for alignment
```

## Customizing the Judge Prompt

Edit `judge.j2` to adjust:
- Evaluation criteria
- Scoring rubric
- Output format

The template receives:
- `source_text`: Original text
- `translated_text`: Model's translation
- `reference_text`: Gold standard

## Command-Line Options

```
--judge TEXT              Judge model identifier [default: gemini-2.0-flash-exp]
--samples INTEGER         Samples per dataset (0=all) [default: 100]
--concurrency INTEGER     Parallel requests [default: 20]
--rerun                   Force re-judge, overwrite existing
--thinking-budget INTEGER Gemini thinking budget [default: 512]
--dry-run                Show what would be judged without API calls
```

## Model Name Resolution

The script intelligently resolves model identifiers to prediction files:

1. **Model names with slashes**: `LiquidAI/LFM2-350M` → `LiquidAI--LFM2-350M*.predictions.jsonl`
2. **Direct paths**: `results/model.predictions.jsonl` → as-is
3. **Glob patterns**: `LiquidAI*` → all matching files
4. **Partial names**: `LFM2` → `LFM2*.predictions.jsonl`

All patterns search in the `results/` directory by default.

## Examples

### Testing New Judge Prompt
```bash
# Test with 1 sample to verify prompt works
python judge-mt.py "LiquidAI/LFM2-350M" --samples 1 --dry-run

# Verify output looks good
python judge-mt.py "LiquidAI/LFM2-350M" --samples 1
```

### Production Run
```bash
# Judge all models with default settings using model names
export GEMINI_API_KEY="your-key-here"
python judge-mt.py "LiquidAI/LFM2-350M" "google/gemma-3-4b-it" "gpt-4o"

# Or judge everything with glob
python judge-mt.py results/*.predictions.jsonl

# Monitor progress in real-time
# Output shows per-dataset sampling and progress bars
```

### Incremental Updates
```bash
# Run initial judging
python judge-mt.py "LiquidAI/LFM2-350M" "google/gemma-3-4b-it"

# Add new model later - only new file gets judged
python judge-mt.py "openai/gpt-4o"

# Re-run safely - existing judgments are preserved
python judge-mt.py "LiquidAI/LFM2-350M" "google/gemma-3-4b-it" "openai/gpt-4o"
```

## Troubleshooting

### Import Error: google.genai
```bash
pip install google-genai
```

### GEMINI_API_KEY not set
```bash
export GEMINI_API_KEY="your-api-key"
```

### Rate Limiting
Reduce concurrency:
```bash
python judge-mt.py results/*.predictions.jsonl --concurrency 5
```

### API Errors
The script automatically retries with exponential backoff (up to 3 attempts per request).

## Design Decisions

1. **Flat structure**: Outputs alongside predictions for easy discovery
2. **Fixed seed (42)**: Ensures reproducible sampling across runs
3. **Per-dataset sampling**: Balances coverage across alt-e-to-j and alt-j-to-e
4. **Idempotent by default**: Safe to re-run, cost-effective
5. **No category guidance**: General MT evaluation (removed dating-specific categories)
6. **Local template**: `judge.j2` in repo root for easy customization
7. **Rich metadata**: Each judgment includes full context for analysis

## Future Enhancements

- [ ] Per-category breakdown in summary
- [ ] Agreement metrics when multiple judges used
- [ ] Bootstrap confidence intervals for scores
- [ ] Integration script to merge with COMET scores
- [ ] HTML report generation
