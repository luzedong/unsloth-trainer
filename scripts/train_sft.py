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
