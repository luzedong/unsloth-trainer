"""Unsloth model loading and LoRA configuration."""

from unsloth import FastLanguageModel


def load_model(config: dict):
    """Load model and tokenizer via Unsloth, then apply LoRA.

    Returns:
        tuple: (model, tokenizer)
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg.get("dtype"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg["use_rslora"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
    )

    return model, tokenizer
