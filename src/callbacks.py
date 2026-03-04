"""Custom training callbacks for experiment tracking."""

import json
import shutil
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class ExperimentCallback(TrainerCallback):
    """Callback that logs experiment info and saves config on completion."""

    def __init__(self, config: dict, config_path: str | None = None):
        self.config = config
        self.config_path = config_path
        self.start_time = None
        self.best_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

        config = self.config
        model_cfg = config.get("model", {})
        lora_cfg = config.get("lora", {})
        data_cfg = config.get("data", {})
        training_cfg = config.get("training", {})
        output_cfg = config.get("output", {})

        # Determine task type
        task_type = "DPO" if "dpo" in config else "SFT"
        experiment_name = output_cfg.get("experiment_name") or model_cfg.get("model_name", "unknown").split("/")[-1]

        # Compute effective batch size
        per_device_bs = training_cfg.get("per_device_train_batch_size", 1)
        grad_accum = training_cfg.get("gradient_accumulation_steps", 1)
        effective_bs = per_device_bs * grad_accum

        # Determine precision
        if model_cfg.get("load_in_4bit"):
            precision = "4-bit"
        elif model_cfg.get("load_in_16bit"):
            precision = "16-bit"
        elif training_cfg.get("bf16"):
            precision = "bf16"
        else:
            precision = str(model_cfg.get("dtype", "auto"))

        print()
        print("=" * 60)
        print(f"  {experiment_name} | {task_type} Training")
        print("=" * 60)

        # Model & LoRA
        print(f"  Model:          {model_cfg.get('model_name', 'N/A')}")
        print(f"  Max seq length: {model_cfg.get('max_seq_length', 'N/A')}")
        print(f"  Precision:      {precision}")
        if not model_cfg.get("full_finetuning"):
            print(f"  LoRA r/alpha:   {lora_cfg.get('r', 'N/A')}/{lora_cfg.get('lora_alpha', 'N/A')}")
            print(f"  LoRA dropout:   {lora_cfg.get('lora_dropout', 0)}")
            print(f"  LoRA targets:   {', '.join(lora_cfg.get('target_modules', []))}")
        else:
            print("  Mode:           Full fine-tuning")
        print("-" * 60)

        # Dataset
        print(f"  Train file:     {data_cfg.get('train_file', 'N/A')}")
        if data_cfg.get("val_file"):
            print(f"  Val file:       {data_cfg['val_file']}")
        elif data_cfg.get("val_size"):
            print(f"  Val split:      {data_cfg['val_size']}")
        print(f"  Format:         {data_cfg.get('format', 'N/A')}")
        print("-" * 60)

        # Training hyperparams
        print(f"  Batch size:     {per_device_bs} x {grad_accum} = {effective_bs} effective")
        print(f"  Learning rate:  {training_cfg.get('learning_rate', 'N/A')}")
        print(f"  Epochs:         {training_cfg.get('num_train_epochs', 'N/A')}")
        print(f"  Optimizer:      {training_cfg.get('optim', 'N/A')}")
        print(f"  LR scheduler:   {training_cfg.get('lr_scheduler_type', 'N/A')}")
        print(f"  Warmup steps:   {training_cfg.get('warmup_steps', 0)}")

        # DPO-specific
        if task_type == "DPO":
            dpo_cfg = config.get("dpo", {})
            print(f"  DPO beta:       {dpo_cfg.get('beta', 0.1)}")
            print(f"  DPO loss type:  {dpo_cfg.get('loss_type', 'sigmoid')}")

        print("-" * 60)
        print(f"  Output dir:     {args.output_dir}")
        print("=" * 60)
        print()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                logs["gpu_memory_gb"] = round(gpu_mem, 2)

            loss = logs.get("loss")
            if loss is not None:
                if self.best_loss is None or loss < self.best_loss:
                    self.best_loss = loss

    def on_train_end(self, args, state, control, **kwargs):
        output_dir = Path(args.output_dir)

        # Copy config YAML to output dir for reproducibility
        config_backup_path = None
        if self.config_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / "train_config.yaml"
            shutil.copy2(self.config_path, dest)
            config_backup_path = str(dest)

        # Also save as JSON for programmatic access
        config_json = output_dir / "train_config.json"
        with open(config_json, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        # Gather metrics
        final_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    final_loss = entry["loss"]
                    break

        # Training duration
        duration_str = "N/A"
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"

        # Peak GPU memory
        gpu_mem_str = "N/A"
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            gpu_mem_str = f"{gpu_mem:.2f} GB"

        # List checkpoint directories
        checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []

        print()
        print("=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"  Total steps:    {state.global_step}")
        print(f"  Epochs:         {state.epoch:.2f}" if state.epoch else "  Epochs:         N/A")
        if final_loss is not None:
            print(f"  Final loss:     {final_loss:.4f}")
        if self.best_loss is not None:
            print(f"  Best loss:      {self.best_loss:.4f}")
        print(f"  Peak GPU mem:   {gpu_mem_str}")
        print(f"  Duration:       {duration_str}")
        print("-" * 60)
        if checkpoints:
            print("  Checkpoints:")
            for ckpt in checkpoints:
                print(f"    - {ckpt}")
        print(f"  Final model:    {output_dir}")
        if config_backup_path:
            print(f"  Config backup:  {config_backup_path}")
        print("=" * 60)
        print()
