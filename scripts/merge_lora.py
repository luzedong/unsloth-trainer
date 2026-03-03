"""Merge LoRA adapter weights into base model and export."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights and export model")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
    parser.add_argument(
        "--export_format",
        default="huggingface",
        choices=["huggingface", "gguf", "vllm"],
        help="Export format (default: huggingface)",
    )
    parser.add_argument("--gguf_quant", default="q4_k_m", help="GGUF quantization method (default: q4_k_m)")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading model from {checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint),
        load_in_4bit=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_format == "huggingface":
        print(f"Saving merged HuggingFace model to {output_dir}")
        model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
        print("Done. Model can be loaded with transformers or used for further training.")

    elif args.export_format == "gguf":
        print(f"Exporting GGUF ({args.gguf_quant}) to {output_dir}")
        model.save_pretrained_gguf(str(output_dir), tokenizer, quantization_method=args.gguf_quant)
        print("Done. Model can be used with llama.cpp or ollama.")

    elif args.export_format == "vllm":
        # vLLM loads standard HF format, but needs merged weights (not adapter)
        print(f"Saving merged model for vLLM to {output_dir}")
        model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
        print("Done. Model can be loaded directly by vLLM.")


if __name__ == "__main__":
    main()
