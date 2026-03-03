# Unsloth Training Scaffold Design

## Overview

A lightweight, YAML-driven training scaffold based on Unsloth + trl, supporting SFT and DPO training for Qwen series models on single A100/H100 GPUs.

## Requirements

- **Models**: Qwen series (Qwen2.5, Qwen3, etc.)
- **Tasks**: SFT supervised fine-tuning + DPO preference alignment
- **Data format**: Alpaca format (instruction/input/output)
- **Hardware**: Single A100/H100 GPU
- **Config**: YAML-driven, one file per experiment

## Project Structure

```
unsloth-trainer/
├── configs/
│   ├── base.yaml                   # Shared defaults
│   ├── sft_qwen2.5_7b.yaml        # SFT example config
│   └── dpo_qwen2.5_7b.yaml        # DPO example config
├── src/
│   ├── __init__.py
│   ├── config.py                   # YAML loading + merge + CLI override
│   ├── data.py                     # Alpaca/DPO data loading
│   ├── model.py                    # Unsloth model + LoRA setup
│   └── callbacks.py                # Logging + config backup callback
├── scripts/
│   ├── train_sft.py                # SFT training entry
│   ├── train_dpo.py                # DPO training entry
│   ├── merge_lora.py               # LoRA merge + export
│   └── inference.py                # Interactive/batch inference
├── data/
│   ├── raw/
│   └── processed/
├── outputs/                        # Auto-organized by experiment_name + timestamp
├── requirements.txt
└── README.md
```

## YAML Config System

### Inheritance

Configs support `inherit: base.yaml` to inherit defaults. Override priority:

```
CLI args > experiment YAML > base.yaml
```

### base.yaml Schema

```yaml
model:
  max_seq_length: 2048
  load_in_4bit: true
  dtype: null                       # Auto-detect (bfloat16 on A100)

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
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

### SFT Config Example

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

output:
  experiment_name: "sft_qwen2.5_7b_medical"
```

### DPO Config Example

```yaml
inherit: base.yaml

model:
  model_name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

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

## Core Modules

### src/config.py

- `load_config(config_path, cli_overrides) -> dict`
- Loads YAML, resolves `inherit` chain, deep-merges dicts
- CLI overrides via dot-notation: `--training.learning_rate 1e-4`

### src/data.py

- `load_sft_dataset(config, tokenizer) -> Dataset`
  - Reads Alpaca JSONL, applies `tokenizer.apply_chat_template` for format conversion
  - Auto train/val split when val_file is not specified
- `load_dpo_dataset(config, tokenizer) -> Dataset`
  - Reads DPO JSONL with chosen/rejected fields
  - Converts to prompt/chosen/rejected columns for DPOTrainer
- Empty `input` field is automatically ignored (no extra newlines)

### src/model.py

- `load_model(config) -> (model, tokenizer)`
- Calls `FastLanguageModel.from_pretrained` + `FastLanguageModel.get_peft_model`
- All parameters read from config dict

### src/callbacks.py

- `ExperimentCallback(TrainerCallback)`
  - On train begin: print config summary
  - Every N steps: log loss/lr/GPU memory
  - On train end: save final LoRA weights + copy config YAML to output dir

## Scripts

### scripts/train_sft.py

```
python scripts/train_sft.py --config configs/sft_qwen2.5_7b.yaml [--training.learning_rate 1e-4]
```

Flow: load_config -> load_model -> load_sft_dataset -> SFTTrainer -> train

### scripts/train_dpo.py

```
python scripts/train_dpo.py --config configs/dpo_qwen2.5_7b.yaml
```

Flow: load_config -> load_model -> load_dpo_dataset -> DPOTrainer -> train

### scripts/merge_lora.py

```
python scripts/merge_lora.py --checkpoint outputs/xxx/checkpoint-500 --output_dir models/merged --export_format huggingface|gguf|vllm
```

### scripts/inference.py

```
# Interactive mode
python scripts/inference.py --checkpoint outputs/xxx/checkpoint-500

# Batch mode
python scripts/inference.py --checkpoint outputs/xxx/checkpoint-500 --input_file data/test.jsonl --output_file results.jsonl
```

Batch output format:
```jsonl
{"instruction": "...", "input": "...", "expected": "...", "generated": "..."}
```

## Typical Workflow

```
1. Prepare data    -> data/processed/sft_train.jsonl
2. Write config    -> configs/sft_qwen2.5_7b.yaml
3. SFT training    -> python scripts/train_sft.py --config configs/sft_qwen2.5_7b.yaml
4. Verify          -> python scripts/inference.py --checkpoint outputs/xxx/checkpoint-500
5. DPO training    -> python scripts/train_dpo.py --config configs/dpo_qwen2.5_7b.yaml
6. Export          -> python scripts/merge_lora.py --checkpoint outputs/xxx --export_format vllm
```

## Module Dependencies

```
train_sft.py / train_dpo.py
    ├── config.load_config()     -> dict
    ├── model.load_model()       -> (model, tokenizer)
    ├── data.load_*_dataset()    -> Dataset
    └── trl.SFTTrainer / DPOTrainer
            + callbacks.ExperimentCallback
            -> trainer.train()
```
