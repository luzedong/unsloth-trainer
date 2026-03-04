# Unsloth Trainer

A lightweight LLM fine-tuning scaffold built on [Unsloth](https://github.com/unslothai/unsloth) + [trl](https://github.com/huggingface/trl), supporting SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) training.

## Project Structure

```
unsloth-trainer/
├── configs/                    # Training configs (one self-contained YAML per experiment)
│   ├── sft_qwen3.5_27b.yaml
│   ├── sft_qwen3.5_4b.yaml
│   ├── dpo_qwen3.5_27b.yaml
│   └── dpo_qwen3.5_4b.yaml
├── src/                        # Core modules
│   ├── config.py               # YAML config loading + CLI overrides
│   ├── model.py                # Unsloth model loading + LoRA
│   ├── data.py                 # SFT/DPO data loading (Alpaca format)
│   └── callbacks.py            # Training callbacks (logging, config backup)
├── scripts/                    # Entry scripts
│   ├── train_sft.py            # SFT training
│   ├── train_dpo.py            # DPO training
│   ├── merge_lora.py           # LoRA merge & export
│   └── inference.py            # Inference (interactive / batch)
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed training data
└── outputs/                    # Training outputs (checkpoints, logs)
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### SFT Training

```bash
python scripts/train_sft.py --config configs/sft_qwen3.5_4b.yaml
```

### DPO Training

```bash
python scripts/train_dpo.py --config configs/dpo_qwen3.5_4b.yaml
```

### CLI Overrides

Any parameter in the YAML config can be overridden via command line:

```bash
python scripts/train_sft.py --config configs/sft_qwen3.5_4b.yaml \
    --training.learning_rate 1e-4 \
    --training.num_train_epochs 5 \
    --lora.r 32
```

### Inference

```bash
# Interactive mode
python scripts/inference.py --checkpoint outputs/sft_qwen3.5_4b_medical_20260304/final

# Batch mode
python scripts/inference.py --checkpoint outputs/sft_qwen3.5_4b_medical_20260304/final \
    --input_file data/processed/sft_sample.jsonl \
    --output_file results.jsonl
```

### LoRA Merge & Export

```bash
# Export as HuggingFace format
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format huggingface

# Export as GGUF (for llama.cpp / ollama)
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format gguf

# Export for vLLM
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format vllm
```

## Data Format

### SFT (Alpaca Format)

One JSON object per line in a `.jsonl` file:

```json
{"instruction": "Write a chief complaint.", "input": "Female, 45 years old, abdominal pain for 3 days.", "output": "Abdominal pain for 3 days."}
```

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | Yes | Task instruction |
| `input` | No | Input context (ignored when empty) |
| `output` | Yes | Expected output |
| `system` | No | System prompt |

### DPO (Preference Format)

```json
{"instruction": "Write a chief complaint.", "input": "Male, 58 years old, chest tightness for 2 years.", "chosen": "Chest tightness for 2 years.", "rejected": "The patient began experiencing chest tightness two years ago."}
```

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | Yes | Task instruction |
| `input` | No | Input context |
| `chosen` | Yes | Preferred response |
| `rejected` | Yes | Dispreferred response |

## YAML Config Sections

Each YAML config is self-contained with all parameters:

| Section | Description |
|---------|-------------|
| `model` | Model name, sequence length, precision settings |
| `lora` | LoRA rank, alpha, target modules, etc. |
| `training` | Batch size, learning rate, epochs, optimizer, etc. |
| `dpo` | DPO-specific parameters (DPO configs only) |
| `data` | Train/val data paths, data format |
| `output` | Output directory, experiment name |

## Typical Workflow

```
1. Prepare data     → data/processed/sft_train.jsonl
2. Copy config      → cp configs/sft_qwen3.5_4b.yaml configs/my_experiment.yaml
3. Edit config      → Adjust model, data paths, hyperparameters
4. SFT training     → python scripts/train_sft.py --config configs/my_experiment.yaml
5. Verify           → python scripts/inference.py --checkpoint outputs/xxx/final
6. DPO training     → python scripts/train_dpo.py --config configs/dpo_xxx.yaml
7. Export            → python scripts/merge_lora.py --checkpoint outputs/xxx/final --export_format vllm
```
