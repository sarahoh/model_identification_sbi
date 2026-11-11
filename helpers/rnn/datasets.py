"""Datasets for model-identification (seqidRNN) training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SessionNpyDataset(Dataset):
    """
    Memmap-backed features + labels for seqid-style **model-ID** training.

    - ``X``: ``(N, S, T, F)`` float array or memmap (one row per agent).
    - ``y``: ``(N, C)`` one-hot class labels, or ``(N,)`` integer class indices.

    Each sample returns ``(sessions, label, n_real_sessions)`` — the tuple layout
    ``seqidRNN`` / ``train_epoch`` expect from ``DataLoader``.

    ``n_real_sessions`` counts session slabs whose first trial is **not** all padding.
    """

    def __init__(
        self,
        features_path: Union[str, Path],
        labels_path: Union[str, Path],
        *,
        trial_pad_value: float = -1.0,
    ) -> None:
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.X = np.load(self.features_path, mmap_mode="r")
        self.y = np.load(self.labels_path, mmap_mode="r")
        if int(self.X.shape[0]) != int(self.y.shape[0]):
            raise ValueError(
                f"features rows {self.X.shape[0]} != label rows {self.y.shape[0]}"
            )
        self.n = int(self.X.shape[0])
        self.trial_pad_value = float(trial_pad_value)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_np = np.asarray(self.X[idx], dtype=np.float32, order="C")
        if not (x_np.flags.c_contiguous and x_np.flags.writeable):
            x_np = x_np.copy()
        sessions = torch.from_numpy(x_np)

        pad = self.trial_pad_value
        is_real_session = (sessions[:, 0, :] != pad).any(dim=-1)
        n_real = torch.tensor(int(is_real_session.sum().item()), dtype=torch.long)

        y_row = np.asarray(self.y[idx])
        if y_row.ndim == 0 or y_row.shape[-1] == 1:
            y_i = int(y_row.reshape(-1)[0])
        else:
            y_i = int(np.argmax(y_row))

        label = torch.tensor(y_i, dtype=torch.long)
        return sessions, label, n_real
