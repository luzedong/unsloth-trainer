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
