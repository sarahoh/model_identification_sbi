"""
seqid_RNN: Sequence Identification RNN
 
GRU/LSTM-based classifier that takes sequences of (cue_vis, cue_aud, action, reward)
and predicts which model generated the sequence.
 
Training: multi-session with per-session loss, hidden state detached between sessions.
Testing: use final session prediction.

note: try additional hidden layer with dropout if this simple architecture can't learn well
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Literal, Any, Dict, List, Sequence, Union, cast
 
 
class seqidRNN(nn.Module):
    def __init__(
        self,
        input_size: int,           # dimension of per-trial observation vector
        hidden_size: int = 64,
        n_models: int = 2,         # number of model identities to classify
        cell_type: Literal["gru", "lstm"] = "gru",
        bidirectional: bool = False,
        dropout = 0.2,             # float (same for all) or list of n_classifier_layers+1 values
        n_classifier_layers: int = 0,  # 0: linear only, 1: one hidden, 2: two hidden
        fc_width_mult: Union[float, Sequence[float]] = 0.5,  # like dropout: one value or one per FC layer
        trial_pad_value: Optional[float] = -1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_models = n_models
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.trial_pad_value = trial_pad_value # If set, trials where all features equal this value are trailing padding and are skipped (pack_padded_sequence).

        # Normalize dropout to a list of length n_classifier_layers + 1
        n_dropouts = n_classifier_layers + 1
        if isinstance(dropout, (int, float)):
            dropouts = [dropout] * n_dropouts
        else:
            assert len(dropout) == n_dropouts, (
                f"dropout list must have {n_dropouts} elements "
                f"(n_classifier_layers + 1), got {len(dropout)}"
            )
            dropouts = list(dropout)
 
        rnn_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        self.rnn_fwd = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        if bidirectional:
            self.rnn_bwd = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )
 
        cls_input_size = hidden_size * 2 if bidirectional else hidden_size
        layers = []
        in_size = cls_input_size
        # Normalize fc_width_mult to a list of length n_classifier_layers.
        if n_classifier_layers <= 0:
            mults: List[float] = []
        elif isinstance(fc_width_mult, (int, float)):
            mults = [float(fc_width_mult)] * int(n_classifier_layers)
        else:
            if len(fc_width_mult) != int(n_classifier_layers):
                raise ValueError(
                    f"fc_width_mult must have {n_classifier_layers} elements "
                    f"(one per FC layer), got {len(fc_width_mult)}"
                )
            mults = [float(x) for x in fc_width_mult]

        base = int(cls_input_size)
        for i in range(n_classifier_layers):
            out_size = max(1, int(round(float(base) * mults[i])))
            layers += [nn.Dropout(dropouts[i]), nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        layers += [nn.Dropout(dropouts[-1]), nn.Linear(in_size, n_models)]
        self.classifier = nn.Sequential(*layers)
 
    def _init_h(self, batch_size: int, device: torch.device):
        """Zero-initialized hidden state for one direction."""
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        if self.cell_type == "lstm":
            c = torch.zeros(1, batch_size, self.hidden_size, device=device)
            return (h, c)
        return h
 
    def _detach_h(self, hidden):
        """Detach hidden state from computation graph."""
        if self.cell_type == "lstm":
            return (hidden[0].detach(), hidden[1].detach())
        return hidden.detach()
 
    def _extract_h(self, hidden):
        """Extract h vector from hidden state (handles LSTM tuple)."""
        if self.cell_type == "lstm":
            return hidden[0]
        return hidden
 
    def _run_rnn(self, rnn, x, hidden):
        """Run one RNN (forward or backward) on one session. Returns new hidden."""
        _, hidden = rnn(x, hidden)
        return hidden


def _trailing_valid_lengths(x: torch.Tensor, pad_value: float) -> torch.Tensor:
    """
    Per-batch sequence length for trailing-padded sessions (B, T, F).

    A trial is padding if every feature equals ``pad_value``. Assumes real trials are
    a prefix [0 : L) with no gaps (matches get_modelID_features_from_dataframe layout).
    """
    is_pad = (x == pad_value).all(dim=-1)
    lengths = (~is_pad).sum(dim=1).to(torch.long)
    return lengths.clamp(min=1)


def _compact_time_reversed(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """For bidirectional backward pass: (B, T, F) -> (B, maxL, F) with time reversed per row."""
    B, T, Fdim = x.shape
    max_l = int(lengths.max().item())
    out = x.new_zeros(B, max_l, Fdim)
    for b in range(B):
        L = int(lengths[b].item())
        if L > 0:
            out[b, :L, :] = x[b, :L, :].flip(0)
    return out


def _run_rnn_session(
    model: seqidRNN,
    rnn: nn.Module,
    x: torch.Tensor,
    hidden: Any,
    *,
    reverse_time: bool,
) -> Any:
    """
    One session slab (B, T, F). Uses packed RNN when ``model.trial_pad_value`` is not None
    so trailing pad trials are not stepped. When None, legacy dense path (all T steps).
    """
    pad_v = model.trial_pad_value
    if pad_v is None:
        if reverse_time:
            x = x.flip(dims=[1])
        _, hidden = rnn(x, hidden)
        return hidden

    lengths = _trailing_valid_lengths(x, float(pad_v))
    t_max = x.size(1)
    if bool((lengths == t_max).all()):
        if reverse_time:
            x = _compact_time_reversed(x, lengths)
        _, hidden = rnn(x, hidden)
        return hidden

    if not reverse_time:
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = rnn(packed, hidden)
        return hidden

    x_rev = _compact_time_reversed(x, lengths)
    packed = pack_padded_sequence(
        x_rev, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
    )
    _, hidden = rnn(packed, hidden)
    return hidden



def _session_loop(
    model: seqidRNN,
    sessions: torch.Tensor,     # (B, n_sessions, session_len, input_size)
    labels: torch.Tensor,       # (B,)
    n_real_per_sub: Optional[torch.Tensor] = None, # (B,) actual session counts per subject
    session_loss_weights: Optional[torch.Tensor] = None,
    compute_loss: bool = True,
):
    """
    Run the multi-session loop (forward + optional backward pass).
 
    Returns:
        final_logits: (B, n_models) — last session's prediction
        total_loss: scalar tensor (or 0.0 if compute_loss=False)
    """
    device = sessions.device
    batch_size, max_sessions = sessions.shape[0], sessions.shape[1]
    max_sess = int(max_sessions) if n_real_per_sub is None else int(n_real_per_sub.max().item())
 
    # --- Forward pass: sessions 0 → max_sess-1 ---
    h_fwd = model._init_h(batch_size, device)
    fwd_hiddens = []
 
    for s in range(max_sess):
        h_fwd = model._detach_h(h_fwd)
        h_fwd = _run_rnn_session(
            model, model.rnn_fwd, sessions[:, s], h_fwd, reverse_time=False
        )
        fwd_hiddens.append(h_fwd)
 
    bwd_hiddens: Optional[List[Any]] = None
    if model.bidirectional:
        # --- Backward pass: sessions max_sess-1 → 0 ---
        h_bwd = model._init_h(batch_size, device)
        bwd_hiddens = [None] * max_sess
 
        for s in reversed(range(max_sess)):
            h_bwd = model._detach_h(h_bwd)
            h_bwd = _run_rnn_session(
                model, model.rnn_bwd, sessions[:, s], h_bwd, reverse_time=True
            )
            bwd_hiddens[s] = h_bwd
 
    # --- Classify at each session and compute loss ---
    total_loss = torch.tensor(0.0, device=device)
    final_logits = torch.zeros(batch_size, model.n_models, device=device)
 
    for s in range(max_sess):
        h_fwd_s = model._extract_h(fwd_hiddens[s]).squeeze(0)
        if model.bidirectional:
            assert bwd_hiddens is not None
            h_bwd_s = model._extract_h(bwd_hiddens[s]).squeeze(0)
            h_cat = torch.cat([h_fwd_s, h_bwd_s], dim=-1)
        else:
            h_cat = h_fwd_s
        logits = model.classifier(h_cat)
 
        if n_real_per_sub is not None:
            mask = (s < n_real_per_sub).float()
        else:
            mask = torch.ones(batch_size, device=device)
 
        if compute_loss:
            loss_s = F.cross_entropy(logits, labels, reduction="none")
            weight = 1.0 if session_loss_weights is None else session_loss_weights[s]
            total_loss = total_loss + weight * (loss_s * mask).sum() / mask.sum().clamp(min=1)
 
        # Track each subject's last-valid-session logits (overwrites until last valid)
        active = mask.bool()
        final_logits[active] = logits[active] # store logits for this session (replacing previous), if valid 
 
    return final_logits, total_loss



def train_epoch(
    model: seqidRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    session_loss_weights: Optional[torch.Tensor] = None,
) -> dict:
    """Train one epoch with per-session predictions and detached hidden states."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
 
    for batch in loader:
        sessions = batch[0].to(device)
        labels = batch[1].to(device)
        n_real_per_sub = batch[2].to(device) if len(batch) > 2 else None
 
        final_logits, batch_loss = _session_loop(
            model, sessions, labels, n_real_per_sub, session_loss_weights,
            compute_loss=True,
        )
 
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 
        total_loss += batch_loss.item()
        preds = final_logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += sessions.size(0)
 
    return {
        "loss": total_loss / len(loader),
        "accuracy": total_correct / total_samples,
    }
 
 
 
@torch.no_grad()
def evaluate(
    model: seqidRNN,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate using each subject's last valid session prediction."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
 
    for batch in loader:
        sessions = batch[0].to(device)
        labels = batch[1].to(device)
        n_real_per_sub = batch[2].to(device) if len(batch) > 2 else None
 
        final_logits, _ = _session_loop(
            model, sessions, labels, n_real_per_sub, compute_loss=False,
        )
 
        loss = F.cross_entropy(final_logits, labels)
        preds = final_logits.argmax(dim=-1)
        batch_size = sessions.size(0)
 
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size
 
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def gather_predictions(
    model: seqidRNN,
    loader: DataLoader,
    device: torch.device,
    *,
    return_logits: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run the model on a DataLoader and collect per-batch **last-session** predictions.

    Matches :func:`evaluate` forward logic (same ``_session_loop``), returning NumPy arrays
    suitable for sklearn metrics / confusion matrices.

    Parameters
    ----------
    return_logits
        If True, include ``logits`` with shape ``(n_samples, n_models)`` (float32 on CPU).

    Returns
    -------
    dict
        ``y_true`` and ``y_pred`` int64 arrays of shape ``(n_samples,)``; optionally ``logits``.
    """
    model.eval()
    y_parts: List[torch.Tensor] = []
    p_parts: List[torch.Tensor] = []
    z_parts: List[torch.Tensor] = []

    for batch in loader:
        sessions = batch[0].to(device)
        labels = batch[1].to(device)
        n_real_per_sub = batch[2].to(device) if len(batch) > 2 else None

        final_logits, _ = _session_loop(
            model, sessions, labels, n_real_per_sub, compute_loss=False,
        )
        y_parts.append(labels.detach().cpu())
        p_parts.append(final_logits.argmax(dim=-1).detach().cpu())
        if return_logits:
            z_parts.append(final_logits.detach().cpu())

    y_true = torch.cat(y_parts, dim=0).numpy().astype(np.int64, copy=False)
    y_pred = torch.cat(p_parts, dim=0).numpy().astype(np.int64, copy=False)
    out: Dict[str, np.ndarray] = {"y_true": y_true, "y_pred": y_pred}
    if return_logits:
        out["logits"] = torch.cat(z_parts, dim=0).numpy().astype(np.float32, copy=False)
    return out


def train(
    model: seqidRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 3e-4,
    patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    session_loss_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Full training loop with early stopping and LR reduction on plateau.
 
    Returns:
        dict with training history and best model state.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
    )
    model.to(device)
 
    # Everything (LR schedule, early stopping, checkpoint selection) is driven by
    # validation loss for stability and consistency with sweep ranking.
    best_val_loss = float("inf")
    best_epoch_loss = 0
    best_state_loss = None
    epochs_no_improve = 0
    stop_reason = "max_epochs"
    history: Dict[str, Any] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }
 
    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    epoch = -1
    for epoch in pbar:
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, session_loss_weights
        )
        val_metrics = evaluate(model, val_loader, device)
 
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
 
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)
 
        pbar.set_postfix(
            train_acc=f"{train_metrics['accuracy']:.3f}",
            val_acc=f"{val_metrics['accuracy']:.3f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            lr=f"{current_lr:.1e}",
        )
 
        # Track best-by-loss (for early stopping and for saving).
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch_loss = epoch + 1
            best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stop_reason = "early_stopping"
                pbar.write(f"Early stopping at epoch {epoch+1}")
                break
 
    history["stop_reason"] = stop_reason
    history["n_epochs_trained"] = epoch + 1
    # Backwards-compatible: keep best_epoch meaning "best by val loss"
    history["best_epoch"] = best_epoch_loss
    history["best_epoch_loss"] = best_epoch_loss
    history["best_val_loss"] = best_val_loss

    # Expose best-loss checkpoint to callers without changing the return type.
    # These are CPU tensors and safe to torch.save; they are not JSON-serializable.
    # Attach to model without confusing type checkers / nn.Module setattr expectations.
    object.__setattr__(model, "_best_state_by_loss", best_state_loss)
    object.__setattr__(model, "_best_epoch_loss", best_epoch_loss)
 
    # Default: leave model at best-by-loss (matches ranking and scheduler signal).
    if best_state_loss is not None:
        model.load_state_dict(best_state_loss)
    return history