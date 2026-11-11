"""Helpers for simulation-based model identification."""

from .inspect_samples import build_index_plan, print_feature_samples, print_label_samples

# Submodules are imported on demand (data_prep pulls in TensorFlow).
from . import io
from . import params

__all__ = [
    "io",
    "params",
    "data_prep",
    "label_setup",
    "rnn",
    "plotting",
    "build_index_plan",
    "print_feature_samples",
    "print_label_samples",
]


def __getattr__(name: str):
    if name in {"data_prep", "label_setup", "rnn", "plotting"}:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
