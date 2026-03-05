"""Unsloth model loading and LoRA configuration."""

from unsloth import FastLanguageModel


def _extract_tokenizer(tokenizer_or_processor):
    """Separate the plain tokenizer from the processor.

    VLM models (e.g. Qwen3.5) return a processor that wraps a tokenizer;
    pure text LLMs (e.g. Qwen3) return a plain tokenizer directly.

    Returns:
        tuple: (tokenizer, processor)
    """
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer, tokenizer_or_processor
    return tokenizer_or_processor, tokenizer_or_processor


def load_model(config: dict):
    """Load model and tokenizer via Unsloth, then apply LoRA.

    Returns:
        tuple: (model, tokenizer, processor)
            - tokenizer: always the plain tokenizer, safe for SFTTrainer
            - processor: the full processor (== tokenizer for text LLMs)
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    load_kwargs = dict(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )

    if model_cfg.get("load_in_16bit"):
        load_kwargs["load_in_16bit"] = True
    if model_cfg.get("full_finetuning"):
        load_kwargs["full_finetuning"] = True

    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(**load_kwargs)
    tokenizer, processor = _extract_tokenizer(tokenizer_or_processor)

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

    return model, tokenizer, processor


def get_tokenizer(config: dict):
    """Load only the tokenizer (without model weights) for data preprocessing."""
    model_cfg = config["model"]
    _, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )
    tokenizer, _ = _extract_tokenizer(tokenizer_or_processor)
    return tokenizer
