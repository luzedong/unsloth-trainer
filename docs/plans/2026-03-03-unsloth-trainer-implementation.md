# Unsloth Training Scaffold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable, YAML-driven Unsloth training scaffold supporting SFT and DPO for Qwen models on single-GPU.

**Architecture:** Pure Unsloth + trl stack. YAML configs with inheritance for experiment management. Four core modules (config, data, model, callbacks) consumed by entry scripts. No heavy frameworks.

**Tech Stack:** Python 3.10+, unsloth, trl, transformers, peft, datasets, pyyaml

**Design doc:** `docs/plans/2026-03-03-unsloth-trainer-scaffold-design.md`

---

### Task 1: Project skeleton + requirements

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `configs/` directory
- Create: `scripts/` directory
- Create: `data/raw/` and `data/processed/` directories
- Create: `outputs/.gitkeep`

**Step 1: Create directory structure**

```bash
cd /Users/zedong/Documents/huimei/unsloth-trainer
mkdir -p configs scripts src data/raw data/processed outputs
```

**Step 2: Write requirements.txt**

Create `requirements.txt`:

```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
trl>=0.12.0
transformers>=4.46.0
datasets>=3.0.0
peft>=0.13.0
accelerate>=1.1.0
bitsandbytes>=0.44.0
pyyaml>=6.0
torch>=2.1.0
```

**Step 3: Write src/__init__.py**

Create `src/__init__.py`:

```python
from src.config import load_config
from src.model import load_model
from src.data import load_sft_dataset, load_dpo_dataset
from src.callbacks import ExperimentCallback
```

**Step 4: Create placeholder .gitkeep files**

```bash
touch data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep
```

**Step 5: Initialize git repo and commit**

```bash
cd /Users/zedong/Documents/huimei/unsloth-trainer
git init
git add requirements.txt src/__init__.py data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep
git commit -m "chore: initialize project skeleton with directory structure and requirements"
```

---

### Task 2: YAML config system (`src/config.py`)

**Files:**
- Create: `src/config.py`
- Create: `configs/base.yaml`

**Step 1: Write src/config.py**

Create `src/config.py`:

```python
"""YAML config loading with inheritance and CLI override support."""

import copy
import sys
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_cli_overrides(args: list[str]) -> dict:
    """Parse CLI args like --training.learning_rate 1e-4 into nested dict."""
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and arg != "--config":
            key_path = arg[2:]  # Remove --
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = _parse_value(args[i + 1])
                i += 2
            else:
                value = True
                i += 1
            _set_nested(overrides, key_path, value)
        else:
            i += 1
    return overrides


def _parse_value(value_str: str):
    """Parse a CLI value string into the appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _set_nested(d: dict, key_path: str, value):
    """Set a value in a nested dict using dot-notation key path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _resolve_inherit(config: dict, config_dir: Path) -> dict:
    """Resolve the 'inherit' field by loading and merging parent config."""
    if "inherit" not in config:
        return config

    parent_path = config_dir / config["inherit"]
    if not parent_path.exists():
        raise FileNotFoundError(f"Inherited config not found: {parent_path}")

    with open(parent_path) as f:
        parent_config = yaml.safe_load(f) or {}

    # Recursively resolve parent's inheritance
    parent_config = _resolve_inherit(parent_config, parent_path.parent)

    # Remove inherit key before merging
    child_config = {k: v for k, v in config.items() if k != "inherit"}

    return _deep_merge(parent_config, child_config)


def load_config(config_path: str, cli_args: list[str] | None = None) -> dict:
    """Load config from YAML file with inheritance and CLI overrides.

    Priority: CLI args > experiment YAML > inherited base YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Resolve inheritance
    config = _resolve_inherit(config, config_path.parent)

    # Apply CLI overrides
    if cli_args:
        overrides = _parse_cli_overrides(cli_args)
        config = _deep_merge(config, overrides)

    return config
```

**Step 2: Write configs/base.yaml**

Create `configs/base.yaml`:

```yaml
# Default configuration - all experiment configs inherit from this file.

model:
  max_seq_length: 2048
  load_in_4bit: true
  dtype: null

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  use_rslora: false
  use_gradient_checkpointing: "unsloth"

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 10
  num_train_epochs: 3
  learning_rate: 2.0e-4
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
  seed: 42
  bf16: true
  optim: "adamw_8bit"

data:
  train_file: null
  val_file: null
  val_size: 0.05
  format: "alpaca"

output:
  dir: "outputs"
  experiment_name: null
```

**Step 3: Commit**

```bash
git add src/config.py configs/base.yaml
git commit -m "feat: add YAML config system with inheritance and CLI overrides"
```

---

### Task 3: Model loading (`src/model.py`)

**Files:**
- Create: `src/model.py`

**Step 1: Write src/model.py**

Create `src/model.py`:

```python
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
```

**Step 2: Commit**

```bash
git add src/model.py
git commit -m "feat: add Unsloth model loading with LoRA configuration"
```

---

### Task 4: Data loading (`src/data.py`)

**Files:**
- Create: `src/data.py`

**Step 1: Write src/data.py**

Create `src/data.py`:

```python
"""Dataset loading for SFT (Alpaca format) and DPO training."""

import json
from pathlib import Path

from datasets import Dataset


def _load_jsonl(file_path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _format_alpaca_to_messages(example: dict) -> list[dict]:
    """Convert Alpaca format to chat messages list.

    Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    Messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    system = example.get("system", None)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})

    return messages


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

    train_data = _load_jsonl(data_cfg["train_file"])
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        lambda x: _format_sft_example(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    result = {"train": train_dataset}

    if data_cfg.get("val_file"):
        val_data = _load_jsonl(data_cfg["val_file"])
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
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")

    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    system = example.get("system", None)
    prompt_messages = []
    if system:
        prompt_messages.append({"role": "system", "content": system})
    prompt_messages.append({"role": "user", "content": user_content})

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

    train_data = _load_jsonl(data_cfg["train_file"])
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        lambda x: _format_dpo_example(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    result = {"train": train_dataset}

    if data_cfg.get("val_file"):
        val_data = _load_jsonl(data_cfg["val_file"])
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
```

**Step 2: Commit**

```bash
git add src/data.py
git commit -m "feat: add SFT and DPO dataset loading with Alpaca format support"
```

---

### Task 5: Training callbacks (`src/callbacks.py`)

**Files:**
- Create: `src/callbacks.py`

**Step 1: Write src/callbacks.py**

Create `src/callbacks.py`:

```python
"""Custom training callbacks for experiment tracking."""

import json
import shutil
from pathlib import Path

import torch
from transformers import TrainerCallback


class ExperimentCallback(TrainerCallback):
    """Callback that logs experiment info and saves config on completion."""

    def __init__(self, config: dict, config_path: str | None = None):
        self.config = config
        self.config_path = config_path

    def on_train_begin(self, args, state, control, **kwargs):
        print("=" * 60)
        print("Experiment Configuration")
        print("=" * 60)
        print(json.dumps(self.config, indent=2, default=str))
        print("=" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            logs["gpu_memory_gb"] = round(gpu_mem, 2)

    def on_train_end(self, args, state, control, **kwargs):
        # Copy config YAML to output dir for reproducibility
        if self.config_path:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / "train_config.yaml"
            shutil.copy2(self.config_path, dest)
            print(f"Config saved to {dest}")

        # Also save as JSON for programmatic access
        output_dir = Path(args.output_dir)
        config_json = output_dir / "train_config.json"
        with open(config_json, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        print(f"Training complete. Output dir: {args.output_dir}")
```

**Step 2: Commit**

```bash
git add src/callbacks.py
git commit -m "feat: add experiment callback for logging and config backup"
```

---

### Task 6: SFT training script (`scripts/train_sft.py`)

**Files:**
- Create: `scripts/train_sft.py`
- Create: `configs/sft_qwen2.5_7b.yaml`

**Step 1: Write scripts/train_sft.py**

Create `scripts/train_sft.py`:

```python
"""SFT training entry script."""

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trl import SFTTrainer
from transformers import TrainingArguments

from src import load_config, load_model, load_sft_dataset, ExperimentCallback


def get_config_path() -> str:
    """Extract --config value from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    print("Usage: python scripts/train_sft.py --config <config.yaml> [overrides]")
    sys.exit(1)


def main():
    config_path = get_config_path()
    config = load_config(config_path, sys.argv[1:])

    # Build output directory
    output_cfg = config["output"]
    experiment_name = output_cfg.get("experiment_name") or config["model"]["model_name"].split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_cfg["dir"]) / f"{experiment_name}_{timestamp}")

    # Load model and data
    model, tokenizer = load_model(config)
    datasets = load_sft_dataset(config, tokenizer)

    # Training arguments
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("val"),
        args=training_args,
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field="text",
        callbacks=[ExperimentCallback(config, config_path)],
    )

    # Train
    trainer.train()

    # Save final model
    final_dir = str(Path(output_dir) / "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Write configs/sft_qwen2.5_7b.yaml**

Create `configs/sft_qwen2.5_7b.yaml`:

```yaml
inherit: base.yaml

model:
  model_name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  max_seq_length: 4096

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4

data:
  train_file: "data/processed/sft_train.jsonl"
  val_file: "data/processed/sft_val.jsonl"
  format: "alpaca"

output:
  experiment_name: "sft_qwen2.5_7b_medical"
```

**Step 3: Commit**

```bash
git add scripts/train_sft.py configs/sft_qwen2.5_7b.yaml
git commit -m "feat: add SFT training script with example Qwen2.5 config"
```

---

### Task 7: DPO training script (`scripts/train_dpo.py`)

**Files:**
- Create: `scripts/train_dpo.py`
- Create: `configs/dpo_qwen2.5_7b.yaml`

**Step 1: Write scripts/train_dpo.py**

Create `scripts/train_dpo.py`:

```python
"""DPO training entry script."""

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trl import DPOConfig, DPOTrainer

from src import load_config, load_model, load_dpo_dataset, ExperimentCallback


def get_config_path() -> str:
    """Extract --config value from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    print("Usage: python scripts/train_dpo.py --config <config.yaml> [overrides]")
    sys.exit(1)


def main():
    config_path = get_config_path()
    config = load_config(config_path, sys.argv[1:])

    # Build output directory
    output_cfg = config["output"]
    experiment_name = output_cfg.get("experiment_name") or config["model"]["model_name"].split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_cfg["dir"]) / f"{experiment_name}_{timestamp}")

    # Load model and data
    model, tokenizer = load_model(config)
    datasets = load_dpo_dataset(config, tokenizer)

    # DPO config
    train_cfg = config["training"]
    dpo_cfg = config.get("dpo", {})

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        max_length=config["model"]["max_seq_length"],
        report_to="none",
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("val"),
        args=training_args,
        callbacks=[ExperimentCallback(config, config_path)],
    )

    # Train
    trainer.train()

    # Save final model
    final_dir = str(Path(output_dir) / "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Write configs/dpo_qwen2.5_7b.yaml**

Create `configs/dpo_qwen2.5_7b.yaml`:

```yaml
inherit: base.yaml

model:
  model_name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  max_seq_length: 4096

training:
  num_train_epochs: 1
  learning_rate: 5.0e-5

dpo:
  beta: 0.1
  loss_type: "sigmoid"

data:
  train_file: "data/processed/dpo_train.jsonl"
  format: "dpo"

output:
  experiment_name: "dpo_qwen2.5_7b_medical"
```

**Step 3: Commit**

```bash
git add scripts/train_dpo.py configs/dpo_qwen2.5_7b.yaml
git commit -m "feat: add DPO training script with example config"
```

---

### Task 8: LoRA merge script (`scripts/merge_lora.py`)

**Files:**
- Create: `scripts/merge_lora.py`

**Step 1: Write scripts/merge_lora.py**

Create `scripts/merge_lora.py`:

```python
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
```

**Step 2: Commit**

```bash
git add scripts/merge_lora.py
git commit -m "feat: add LoRA merge script with HuggingFace/GGUF/vLLM export"
```

---

### Task 9: Inference script (`scripts/inference.py`)

**Files:**
- Create: `scripts/inference.py`

**Step 1: Write scripts/inference.py**

Create `scripts/inference.py`:

```python
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
```

**Step 2: Commit**

```bash
git add scripts/inference.py
git commit -m "feat: add inference script with interactive and batch modes"
```

---

### Task 10: Sample data files + final integration

**Files:**
- Create: `data/processed/sft_sample.jsonl` (sample data for testing)
- Create: `data/processed/dpo_sample.jsonl` (sample data for testing)

**Step 1: Write sample SFT data**

Create `data/processed/sft_sample.jsonl`:

```jsonl
{"instruction": "请根据以下临床信息，生成首次主治医师查房记录。", "input": "患者，男，65岁，因\"反复咳嗽、咳痰1月余\"入院。查体：T 36.8℃，P 80次/分，R 20次/分，BP 130/80mmHg。双肺呼吸音粗，右下肺可闻及少量湿啰音。", "output": "首次主治医师查房记录\n\n查房时间：入院后48小时内\n\n患者，男性，65岁，以\"反复咳嗽、咳痰1月余\"为主诉入院。\n\n病情摘要：患者1月余前无明显诱因出现咳嗽、咳痰，痰为白色粘痰，量中等，无痰中带血，无发热、胸痛、呼吸困难等不适。入院查体：T 36.8℃，P 80次/分，R 20次/分，BP 130/80mmHg，双肺呼吸音粗，右下肺可闻及少量湿啰音。\n\n初步诊断：肺部感染\n\n诊疗计划：\n1. 完善相关检查：血常规、CRP、PCT、胸部CT、痰培养+药敏等。\n2. 予以经验性抗感染治疗。\n3. 止咳化痰对症处理。\n4. 待检查结果回报后调整治疗方案。"}
{"instruction": "请根据以下临床信息，书写主诉。", "input": "患者女性，45岁，3天前出现上腹部疼痛，呈持续性胀痛，伴恶心、呕吐胃内容物2次，无发热、腹泻。", "output": "上腹部疼痛伴恶心呕吐3天。"}
```

**Step 2: Write sample DPO data**

Create `data/processed/dpo_sample.jsonl`:

```jsonl
{"instruction": "请根据以下临床信息，书写主诉。", "input": "患者男性，58岁，反复胸闷气促2年，加重伴双下肢水肿1周。", "chosen": "反复胸闷气促2年，加重伴双下肢水肿1周。", "rejected": "患者2年前开始出现胸闷和气促的症状，最近1周症状加重了，还出现了双下肢水肿的情况。"}
{"instruction": "请根据以下临床信息，生成鉴别诊断。", "input": "患者女性，30岁，反复发热伴关节肿痛2月。查体：面部蝶形红斑，双手近端指间关节肿胀压痛。实验室：ANA(+), 抗dsDNA抗体(+), 补体C3降低。", "chosen": "1. 系统性红斑狼疮（SLE）：患者青年女性，面部蝶形红斑、关节炎、ANA及抗dsDNA抗体阳性、补体降低，高度符合SLE诊断标准。\n2. 类风湿关节炎：可有关节肿痛，但通常以小关节对称性受累为主，RF及抗CCP抗体阳性，一般无蝶形红斑及抗dsDNA抗体阳性。\n3. 混合性结缔组织病：可有多系统受累，但以抗U1-RNP抗体阳性为特征，与本例抗体谱不符。", "rejected": "鉴别诊断：可能是红斑狼疮或者类风湿关节炎，需要进一步检查确认。"}
```

**Step 3: Update src/__init__.py (verify imports are correct)**

Re-read and verify `src/__init__.py` imports match the actual module exports. No changes needed if Task 1 was followed.

**Step 4: Commit**

```bash
git add data/processed/sft_sample.jsonl data/processed/dpo_sample.jsonl
git commit -m "feat: add sample SFT and DPO data files for testing"
```

---

### Task Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Project skeleton | dirs, requirements.txt, `__init__.py` |
| 2 | Config system | `src/config.py`, `configs/base.yaml` |
| 3 | Model loading | `src/model.py` |
| 4 | Data loading | `src/data.py` |
| 5 | Callbacks | `src/callbacks.py` |
| 6 | SFT training | `scripts/train_sft.py`, `configs/sft_qwen2.5_7b.yaml` |
| 7 | DPO training | `scripts/train_dpo.py`, `configs/dpo_qwen2.5_7b.yaml` |
| 8 | LoRA merge | `scripts/merge_lora.py` |
| 9 | Inference | `scripts/inference.py` |
| 10 | Sample data | `data/processed/sft_sample.jsonl`, `data/processed/dpo_sample.jsonl` |

**Dependencies:** Tasks 2-5 are independent core modules. Tasks 6-7 depend on all of 2-5. Tasks 8-9 are independent utilities. Task 10 is independent sample data.
