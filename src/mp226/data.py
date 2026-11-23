"""Loading the raw files and fingerprinting them."""

import hashlib

import pandas as pd

from .config import ROOT, load_params


def sha256_of_file(path):
    """SHA-256 fingerprint of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(which):
    """Path of a raw file; `which` is 'train', 'later' or 'holdout'."""
    return ROOT / load_params()["data"][which]


def load_raw(which):
    """Read a raw CSV exactly as given (no cleaning)."""
    return pd.read_csv(raw_path(which))
