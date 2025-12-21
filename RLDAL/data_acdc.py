"""Helper to build ACDC dataloaders for RLDAL."""
from typing import Tuple

from data.data_utils import get_data

from .config import RLDALConfig


def build_acdc_loaders(cfg: RLDALConfig):
    """Return dataloaders/datasets configured for ACDC.

    The return signature matches ``get_data`` in ``data.data_utils`` to keep
    compatibility with the existing training utilities.
    """
    if cfg.dataset != "ACDC":
        raise ValueError(f"build_acdc_loaders expects dataset='ACDC', got {cfg.dataset}")

    return get_data(**cfg.to_data_kwargs())
