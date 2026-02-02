# HOWTO: Running TranslateGemma (and other non-standard chat template models)

## The Problem

TranslateGemma uses a **structured chat template** that differs from standard OpenAI-style models. Instead of plain text `content`, it requires:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": "ja",
                "text": "Hello world",
            }
        ],
    }
]
```

The tokenizer's `apply_chat_template()` converts this into the actual prompt:

```
<bos><start_of_turn>user
You are a professional English (en) to Japanese (ja) translator. Your goal is to
accurately convey the meaning and nuances of the original English text while
adhering to Japanese grammar, vocabulary, and cultural sensitivities.
Produce only the Japanese translation, without any additional explanations or
commentary. Please translate the following English text into Japanese:


Hello world<end_of_turn>
<start_of_turn>model
```

This creates **two incompatibilities** with the standard eval pipeline:

1. **vLLM's `/v1/chat/completions` endpoint** validates messages server-side using the model's chat template. It rejects plain text `content` strings — even raw HTTP requests fail with `400 Bad Request`.

2. **The eval framework's `prepare_prompts()`** constructs standard `{"role": "user", "content": "plain text"}` messages, which the tokenizer's `apply_chat_template()` also rejects.

## The Solution

We solved this with three changes:

### 1. Load tokenizer for openai engine when `--format` is used

In `run-mt.py`, when `--engine openai` and `--format` are both set, we load the tokenizer client-side (without loading model weights) so `prepare_prompts` can call `apply_chat_template()`:

```python
if chat_template_mode:
    tokenizer = AutoTokenizer.from_pretrained(args.model, ...)
    use_completions = True
```

### 2. Build structured messages for TranslateGemma

In `prepare_prompts()`, when `format_name == "translategemma"`, we override the standard message construction with TranslateGemma's required structured format:

```python
if format_name == "translategemma" and tokenizer is not None:
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": src_code,  # inferred from dataset direction
            "target_lang_code": tgt_code,
            "text": sample["input"],
        }],
    }]
```

The tokenizer's `apply_chat_template()` then renders this into the full prompt with special tokens.

### 3. Use `/v1/completions` instead of `/v1/chat/completions`

Since the pre-rendered prompt already includes chat template tokens (`<bos>`, `<start_of_turn>`, etc.), we send it to vLLM's `/v1/completions` endpoint which accepts raw text:

```python
if use_completions:
    response = client.completions.create(model=model_name, prompt=prompt, ...)
```

## Files

| File | Purpose |
|---|---|
| `translategemma.system.j2` | Empty file — prevents dataset instruction fallback (TranslateGemma doesn't support system messages) |
| `translategemma.user.j2` | Jinja2 template matching TranslateGemma's prompt style (used as fallback if tokenizer unavailable) |
| `run-mt.py` | Structured message construction + completions endpoint routing |

## Running the Eval

```bash
# Start vLLM server
vllm serve google/translategemma-12b-it --tensor-parallel-size 2

# Run eval
python run-mt.py google/translategemma-12b-it \
  --format translategemma \
  --engine openai \
  --openai-api-key unused \
  --openai-base-url http://localhost:8000/v1
```

## Adding Other Non-Standard Models

If you encounter another model with a custom chat template that vLLM's chat endpoint rejects:

### Option A: Model needs structured `content` (like TranslateGemma)

1. Add a special-case block in `prepare_prompts()` that constructs the model's required message format
2. The existing `use_completions` path will handle the rest — tokenizer renders the prompt client-side, completions endpoint sends it as raw text

### Option B: Model works with plain text but has unusual template logic

1. Create `{format}.user.j2` (and optionally `{format}.system.j2`) templates
2. The `--format` flag + openai engine will:
   - Load the tokenizer to call `apply_chat_template()`
   - Send the fully-rendered prompt via `/v1/completions`

### Option C: Fix vLLM's chat endpoint validation

vLLM's chat completions endpoint runs the model's chat template Jinja2 to validate and format messages. For models with non-standard templates, vLLM would need to:

- Accept arbitrary keys in `content` items (currently it validates against a fixed schema)
- Pass them through to the Jinja2 template unchanged

This would require a vLLM PR. The `use_completions` workaround avoids this entirely by doing template rendering client-side.

### General Pattern

The key insight: when a model's chat template is non-standard, **render the prompt client-side** using the model's tokenizer and send the raw result via the completions endpoint. This bypasses vLLM's chat endpoint validation entirely while still using the model's official chat template.

```
Standard model:   J2 template → plain messages → vLLM chat endpoint → chat template → generation
Non-standard:     tokenizer.apply_chat_template(structured msgs) → vLLM completions endpoint → generation
```
