"""Helper to build dataloaders for supported public datasets in RLDAL."""
from typing import Tuple

from data.data_utils import get_data

from .config import RLDALConfig


SUPPORTED_DATASETS = {"ACDC", "TUI", "KVASIR", "TN3K"}


def build_loaders(cfg: RLDALConfig):
    """Return dataloaders/datasets configured for supported datasets.

    The return signature matches ``get_data`` in ``data.data_utils`` to keep
    compatibility with the existing training utilities.
    """
    if cfg.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{cfg.dataset}'. Supported: {sorted(SUPPORTED_DATASETS)}"
        )

    return get_data(**cfg.to_data_kwargs())
