"""Dataset loading for SFT (Alpaca format) and DPO training."""

import json
from pathlib import Path

from datasets import Dataset


def _load_data(file_path: str) -> list[dict]:
    """Load a JSON or JSONL file into a list of dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.suffix == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list of objects: {path}")
            return data

        # Default: JSONL (one JSON object per line)
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data


def _format_alpaca_to_messages(example: dict) -> list[dict]:
    """Convert Alpaca format to chat messages list.

    Concatenates system, instruction, and input into a single user message.
    """
    parts = [p for p in (
        example.get("system", ""),
        example.get("instruction", ""),
        example.get("input", ""),
    ) if p]
    user_content = "\n".join(parts)
    output = example.get("output", "")

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _format_sft_example(example: dict, tokenizer) -> dict:
    """Format a single SFT example using the tokenizer's chat template."""
    messages = _format_alpaca_to_messages(example)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def load_sft_dataset(config: dict, tokenizer) -> dict:
    """Load and format SFT dataset from Alpaca-format JSONL.

    Returns:
        dict with 'train' and optionally 'val' Dataset objects.
    """
    data_cfg = config["data"]

    train_data = _load_data(data_cfg["train_file"])
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        lambda x: _format_sft_example(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    result = {"train": train_dataset}

    if data_cfg.get("val_file"):
        val_data = _load_data(data_cfg["val_file"])
        val_dataset = Dataset.from_list(val_data)
        val_dataset = val_dataset.map(
            lambda x: _format_sft_example(x, tokenizer),
            remove_columns=val_dataset.column_names,
        )
        result["val"] = val_dataset
    elif data_cfg.get("val_size", 0) > 0:
        split = train_dataset.train_test_split(test_size=data_cfg["val_size"], seed=config["training"]["seed"])
        result["train"] = split["train"]
        result["val"] = split["test"]

    return result


def _format_dpo_example(example: dict, tokenizer) -> dict:
    """Format a single DPO example into prompt/chosen/rejected.

    Input: {"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}
    """
    parts = [p for p in (
        example.get("system", ""),
        example.get("instruction", ""),
        example.get("input", ""),
    ) if p]
    user_content = "\n".join(parts)

    prompt_messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    chosen_messages = prompt_messages + [{"role": "assistant", "content": example["chosen"]}]
    rejected_messages = prompt_messages + [{"role": "assistant", "content": example["rejected"]}]

    chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
    rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_dpo_dataset(config: dict, tokenizer) -> dict:
    """Load and format DPO dataset.

    Returns:
        dict with 'train' and optionally 'val' Dataset objects.
    """
    data_cfg = config["data"]

    train_data = _load_data(data_cfg["train_file"])
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        lambda x: _format_dpo_example(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    result = {"train": train_dataset}

    if data_cfg.get("val_file"):
        val_data = _load_data(data_cfg["val_file"])
        val_dataset = Dataset.from_list(val_data)
        val_dataset = val_dataset.map(
            lambda x: _format_dpo_example(x, tokenizer),
            remove_columns=val_dataset.column_names,
        )
        result["val"] = val_dataset
    elif data_cfg.get("val_size", 0) > 0:
        split = train_dataset.train_test_split(test_size=data_cfg["val_size"], seed=config["training"]["seed"])
        result["train"] = split["train"]
        result["val"] = split["test"]

    return result
