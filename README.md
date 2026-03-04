# Unsloth Trainer

基于 [Unsloth](https://github.com/unslothai/unsloth) + [trl](https://github.com/huggingface/trl) 的轻量级 LLM 微调脚手架，支持 SFT 监督微调和 DPO 偏好对齐训练。

## 项目结构

```
unsloth-trainer/
├── configs/                    # 训练配置（每个实验一个独立 YAML）
│   ├── sft_qwen3.5_27b.yaml
│   ├── sft_qwen3.5_4b.yaml
│   ├── dpo_qwen3.5_27b.yaml
│   └── dpo_qwen3.5_4b.yaml
├── src/                        # 核心模块
│   ├── config.py               # YAML 配置加载 + CLI 参数覆盖
│   ├── model.py                # Unsloth 模型加载 + LoRA
│   ├── data.py                 # SFT/DPO 数据加载（Alpaca 格式）
│   └── callbacks.py            # 训练回调（日志、配置备份）
├── scripts/                    # 入口脚本
│   ├── train_sft.py            # SFT 训练
│   ├── train_dpo.py            # DPO 训练
│   ├── merge_lora.py           # LoRA 合并导出
│   └── inference.py            # 推理测试（交互/批量）
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后的训练数据
└── outputs/                    # 训练产出（checkpoints、logs）
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### SFT 训练

```bash
python scripts/train_sft.py --config configs/sft_qwen3.5_4b.yaml
```

### DPO 训练

```bash
python scripts/train_dpo.py --config configs/dpo_qwen3.5_4b.yaml
```

### CLI 参数覆盖

任何 YAML 中的参数都可以通过命令行覆盖：

```bash
python scripts/train_sft.py --config configs/sft_qwen3.5_4b.yaml \
    --training.learning_rate 1e-4 \
    --training.num_train_epochs 5 \
    --lora.r 32
```

### 推理验证

```bash
# 交互模式
python scripts/inference.py --checkpoint outputs/sft_qwen3.5_4b_medical_20260304/final

# 批量模式
python scripts/inference.py --checkpoint outputs/sft_qwen3.5_4b_medical_20260304/final \
    --input_file data/processed/sft_sample.jsonl \
    --output_file results.jsonl
```

### LoRA 合并导出

```bash
# 导出 HuggingFace 格式
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format huggingface

# 导出 GGUF（用于 llama.cpp / ollama）
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format gguf

# 导出 vLLM 格式
python scripts/merge_lora.py --checkpoint outputs/xxx/final \
    --output_dir models/merged --export_format vllm
```

## 数据格式

### SFT（Alpaca 格式）

每行一个 JSON 对象，文件后缀 `.jsonl`：

```json
{"instruction": "请根据以下临床信息，书写主诉。", "input": "患者女性，45岁，上腹部疼痛3天。", "output": "上腹部疼痛3天。"}
```

- `instruction`：任务指令（必填）
- `input`：输入上下文（可选，为空时忽略）
- `output`：期望输出（必填）
- `system`：系统提示词（可选）

### DPO（偏好对齐格式）

```json
{"instruction": "请书写主诉。", "input": "患者男性，58岁，胸闷气促2年。", "chosen": "胸闷气促2年。", "rejected": "患者2年前开始出现胸闷气促的症状。"}
```

- `chosen`：优质回答（必填）
- `rejected`：劣质回答（必填）

## YAML 配置说明

每个 YAML 配置文件包含以下部分：

| 配置段 | 说明 |
|--------|------|
| `model` | 模型名称、序列长度、精度设置 |
| `lora` | LoRA 秩、alpha、目标模块等 |
| `training` | 批大小、学习率、epoch 数等训练超参 |
| `dpo` | DPO 专用参数（仅 DPO 配置需要） |
| `data` | 训练/验证数据路径、数据格式 |
| `output` | 输出目录、实验名称 |

## 典型工作流

```
1. 准备数据    → data/processed/sft_train.jsonl
2. 复制配置    → cp configs/sft_qwen3.5_4b.yaml configs/my_experiment.yaml
3. 修改配置    → 调整模型、数据路径、超参数
4. SFT 训练    → python scripts/train_sft.py --config configs/my_experiment.yaml
5. 推理验证    → python scripts/inference.py --checkpoint outputs/xxx/final
6. DPO 训练    → python scripts/train_dpo.py --config configs/dpo_xxx.yaml
7. 合并导出    → python scripts/merge_lora.py --checkpoint outputs/xxx/final --export_format vllm
```
