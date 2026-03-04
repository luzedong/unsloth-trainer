"""YAML config loading with CLI override support."""

import copy
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


def load_config(config_path: str, cli_args: list[str] | None = None) -> dict:
    """Load config from YAML file with optional CLI overrides.

    Priority: CLI args > YAML config
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_args:
        overrides = _parse_cli_overrides(cli_args)
        config = _deep_merge(config, overrides)

    return config
