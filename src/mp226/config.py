"""Read params.yaml once and hand the settings back."""

from pathlib import Path

import yaml

# src/mp226/config.py -> parents[2] is the repository root
ROOT = Path(__file__).resolve().parents[2]


def load_params(path=None):
    """Return the settings in params.yaml as a dictionary."""
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
