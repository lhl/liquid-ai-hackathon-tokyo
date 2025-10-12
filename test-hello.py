#!/usr/bin/env python3
"""
Simple test script to load Shisa models from /data/outputs/shisa*
and test chat_template with a basic "hello" message.

Usage:
    python test-hello.py                    # Test all models in /data/outputs/shisa*
    python test-hello.py <model_path>       # Test specific model (full path)
    python test-hello.py shisa-v2.1c-lfm2-350m-dpo  # Test model by folder name
"""

import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import argparse

def test_model_chat_template(model_path: str):
    """Load a model and test its chat template with a hello message."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path}")
    print(f"{'='*80}")

    try:
        # Load tokenizer and model
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare chat messages
        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        # Apply chat template
        print("\nApplying chat template...")
        print(f"Input messages: {messages}")

        # Apply template and tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"\nFormatted prompt:\n{repr(text)}")

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate response
        print("\nGenerating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\nFull response:\n{response}")

        # Extract just the assistant's response
        response_only = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\nAssistant response only:\n{response_only}")

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"\n✓ Test completed successfully for {model_path}")
        return True

    except Exception as e:
        print(f"\n✗ Error testing {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def resolve_model_path(model_arg: str) -> str:
    """
    Resolve model argument to a full path.
    First check if it's a full path, then check /data/outputs/
    """
    # If it's an absolute path that exists, use it directly
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return model_arg

    # If it's a relative path that exists, resolve it
    if os.path.exists(model_arg):
        return os.path.abspath(model_arg)

    # Otherwise, check if it's a folder name in /data/outputs/
    candidate = os.path.join("/data/outputs", model_arg)
    if os.path.exists(candidate):
        return candidate

    # Not found
    raise ValueError(f"Model not found: {model_arg}\nTried:\n  - {model_arg}\n  - {candidate}")

def main():
    parser = argparse.ArgumentParser(description="Test Shisa models with chat template")
    parser.add_argument(
        "model",
        nargs="?",
        help="Specific model to test (full path or folder name in /data/outputs/)"
    )
    args = parser.parse_args()

    # Determine which models to test
    if args.model:
        # Test specific model
        try:
            model_path = resolve_model_path(args.model)
            model_paths = [model_path]
            print(f"Testing specific model: {model_path}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Test all shisa models
        model_paths = sorted(glob.glob("/data/outputs/shisa*"))
        if not model_paths:
            print("No shisa models found in /data/outputs/shisa*")
            sys.exit(1)
        print(f"Found {len(model_paths)} model(s):")
        for path in model_paths:
            print(f"  - {path}")

    # Test each model
    results = {}
    for model_path in model_paths:
        results[model_path] = test_model_chat_template(model_path)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model_path, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_path}")

if __name__ == "__main__":
    main()
