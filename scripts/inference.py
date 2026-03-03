"""Quick inference script for testing trained models."""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input_file", default=None, help="JSONL file for batch inference")
    parser.add_argument("--output_file", default=None, help="Output JSONL file (batch mode)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    return parser.parse_args()


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    """Generate a response from chat messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        use_cache=True,
    )

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def interactive_mode(model, tokenizer, args):
    """Interactive chat loop."""
    print("Interactive mode. Type 'quit' to exit.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages = [{"role": "user", "content": user_input}]
        response = generate(model, tokenizer, messages, args.max_new_tokens, args.temperature)
        print(f"\nAssistant: {response}")


def batch_mode(model, tokenizer, args):
    """Batch inference from JSONL file."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output_file) if args.output_file else input_path.with_suffix(".results.jsonl")

    data = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Processing {len(data)} examples...")

    results = []
    for i, example in enumerate(data):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        user_content = f"{instruction}\n{input_text}" if input_text else instruction

        messages = [{"role": "user", "content": user_content}]
        if example.get("system"):
            messages.insert(0, {"role": "system", "content": example["system"]})

        generated = generate(model, tokenizer, messages, args.max_new_tokens, args.temperature)

        result = {
            "instruction": instruction,
            "input": input_text,
            "expected": example.get("output", ""),
            "generated": generated,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(data)}]")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint}")
    FastLanguageModel.for_inference(None)  # Enable native 2x faster inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if args.input_file:
        batch_mode(model, tokenizer, args)
    else:
        interactive_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
