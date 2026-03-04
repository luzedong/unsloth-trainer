"""Unsloth model loading and LoRA configuration."""

from unsloth import FastLanguageModel


def load_model(config: dict):
    """Load model and tokenizer via Unsloth, then apply LoRA.

    Returns:
        tuple: (model, tokenizer)
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    load_kwargs = dict(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )

    # Qwen3.5+ supports load_in_16bit and full_finetuning
    if model_cfg.get("load_in_16bit"):
        load_kwargs["load_in_16bit"] = True
    if model_cfg.get("full_finetuning"):
        load_kwargs["full_finetuning"] = True

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        use_rslora=lora_cfg["use_rslora"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg.get("random_state", 3407),
        max_seq_length=model_cfg["max_seq_length"],
    )

    return model, tokenizer
