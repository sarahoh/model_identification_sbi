"""
Hyperparameter sweep utilities for seqid_RNN.

Provides config generation, single-config training, and result tracking.
Designed to be used from a notebook — call functions directly, not as a script.
"""

import itertools
import random
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from helpers.rnn.seqidRNN import seqidRNN, train


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def build_configs(config_dict: Dict) -> List[Dict]:
    """
    Split a config dict into grid params (lists) and fixed params (everything else).
    Returns list of full config dicts with all combinations of grid params
    plus all fixed params.
 
    Convention:
        list    → grid param, expanded combinatorially
        tuple   → single value, kept as-is (use for multi-value params like dropout)
        scalar  → fixed value
        _paired → list of dicts whose keys must vary together (not crossed independently)
 
    Example:
        config = {
            'rnn_type': ['gru', 'lstm'],           # grid — 2 options
            '_paired': [                            # 3 paired options
                {'n_classifier_layers': 0, 'dropout': 0.2},
                {'n_classifier_layers': 1, 'dropout': (0.2, 0.1)},
                {'n_classifier_layers': 2, 'dropout': (0.2, 0.1, 0.05)},
            ],
            'hidden_size': 128,                     # fixed
        }
        # Returns 6 configs (2 × 3)
    """
    # Extract _paired before processing
    config_dict = dict(config_dict)  # don't mutate the original
    paired_options = config_dict.pop("_paired", None)
 
    grid_params = {k: v for k, v in config_dict.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in config_dict.items() if not isinstance(v, list)}
 
    keys = list(grid_params.keys())
    combos = list(itertools.product(*[grid_params[k] for k in keys]))
 
    configs = [{**fixed_params, **dict(zip(keys, combo))} for combo in combos]
 
    if paired_options:
        configs = [{**cfg, **paired} for cfg in configs for paired in paired_options]
 
    return configs


def sample_configs(config_dict: Dict, n: int, seed: int = 42) -> List[Dict]:
    """Random sample of n configs from the full grid (or all if fewer than n)."""
    all_cfgs = build_configs(config_dict)
    rng = random.Random(seed)
    return rng.sample(all_cfgs, min(n, len(all_cfgs)))


def config_tag(cfg: Dict) -> str:
    """Short human-readable tag from a config dict."""
    cell = cfg.get("cell_type") or cfg.get("rnn_type", "?")
    bidir = "bi" if cfg.get("bidirectional", False) else "uni"
    h = cfg.get("hidden_size", "?")
    n_cls = cfg.get("n_classifier_layers", 0)
    d = cfg.get("dropout", 0.2)
    fc_mult = cfg.get("fc_width_mult", None)
    if isinstance(d, (list, tuple)):
        d_str = "-".join(f"{x:g}" for x in d)
    else:
        d_str = f"{d:g}"
    if n_cls and fc_mult is not None:
        if isinstance(fc_mult, (list, tuple)):
            m_str = "-".join(f"{float(x):g}" for x in fc_mult)
        else:
            m_str = f"{float(fc_mult):g}"
        return f"{cell}_{bidir}_{h}u_fc{n_cls}_m{m_str}_d{d_str}"
    return f"{cell}_{bidir}_{h}u_fc{n_cls}_d{d_str}"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_config(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    n_models: int,
    device: torch.device,
    n_epochs: int = 100,
    session_loss_weights: Optional[torch.Tensor] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Train one config. Pulls all hyperparameters from cfg.
    If save_dir is provided, skips training if results already exist and
    saves results incrementally after training.
 
    Expected cfg keys (with defaults):
        cell_type / rnn_type        (required)
        hidden_size                 (default 64)
        bidirectional               (default False)
        dropout                     (default 0.2) — float or tuple for per-layer values
        n_classifier_layers         (default 0)
        lr                          (default 3e-4)
        patience                    (default 10)
        reduce_lr_patience          (default 5)
        reduce_lr_factor            (default 0.5)
        min_lr                      (default 1e-6)
 
    Returns dict with config, history, metrics, and model_state.
    """
    tag = config_tag(cfg)
 
    # Check if already completed
    if save_dir is not None:
        save_dir = Path(save_dir)
        pt_path = save_dir / f"{tag}.pt"
        if not pt_path.exists():
            # Cross-stage cache: if this config was trained in another stage directory,
            # copy it into the current save_dir so normal skip/load logic applies.
            parent = save_dir.parent
            try:
                # Prefer stage* siblings (stage1/stage2/stage3). Fallback to any sibling dir.
                candidates = list(parent.glob("stage*/" + pt_path.name))
                if not candidates:
                    candidates = list(parent.glob("*/" + pt_path.name))
                # Exclude current path if it appears in glob results.
                candidates = [p for p in candidates if p.resolve() != pt_path.resolve()]
            except Exception:
                candidates = []
            if candidates:
                src = candidates[0]
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src, pt_path)
                    print(f"    [CACHE COPY] Copied {src} -> {pt_path}")
                except Exception as e:
                    print(f"    [CACHE COPY] Failed to copy cached result from {src}: {e}")
        if pt_path.exists():
            print(f"    [SKIP] Found {pt_path.name}, loading cached result")
            saved = torch.load(pt_path, weights_only=False)
            result = {
                "tag": tag,
                "config": saved["config"],
                "best_val_acc": float(saved["history"].get("best_val_acc", max(saved["history"]["val_acc"]))),
                "best_val_loss": float(saved["history"].get("best_val_loss", min(saved["history"]["val_loss"]))),
                "best_epoch": int(saved["history"].get("best_epoch", 0)),
                "best_epoch_loss": int(saved["history"].get("best_epoch_loss", saved["history"].get("best_epoch", 0))),
                "n_epochs_trained": saved["history"]["n_epochs_trained"],
                "stop_reason": saved["history"]["stop_reason"],
                "history": saved["history"],
                "model_state": saved.get("model_state_by_loss", saved.get("model_state")),
            }
            _append_to_summary_json(save_dir, result)
            return result
 
    model = seqidRNN(
        input_size=input_size,
        hidden_size=cfg.get("hidden_size", 64),
        n_models=n_models,
        cell_type=cfg.get("cell_type") or cfg.get("rnn_type", "gru"),
        bidirectional=cfg.get("bidirectional", False),
        dropout=cfg.get("dropout", 0.2),
        n_classifier_layers=cfg.get("n_classifier_layers", 0),
        fc_width_mult=cfg.get("fc_width_mult", 0.5),
    )
 
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=n_epochs,
        lr=cfg.get("lr", 3e-4),
        patience=cfg.get("patience", 10),
        reduce_lr_patience=cfg.get("reduce_lr_patience", 5),
        reduce_lr_factor=cfg.get("reduce_lr_factor", 0.5),
        min_lr=cfg.get("min_lr", 1e-6),
        session_loss_weights=session_loss_weights,
    )
 
    # train() leaves model loaded at best-by-loss and exposes best-loss checkpoint on the model.
    model_state_by_loss = getattr(model, "_best_state_by_loss", None) or model.state_dict()

    result = {
        "tag": tag,
        "config": cfg,
        "best_val_acc": float(history.get("best_val_acc", max(history["val_acc"]))),
        "best_val_loss": float(history.get("best_val_loss", min(history["val_loss"]))),
        "best_epoch": int(history.get("best_epoch", 0)),
        "best_epoch_loss": int(history.get("best_epoch_loss", history.get("best_epoch", 0))),
        "n_epochs_trained": history["n_epochs_trained"],
        "stop_reason": history["stop_reason"],
        "history": history,
        # Backwards-compatible: keep only best-by-loss weights.
        "model_state": model_state_by_loss,
    }
 
    # Save incrementally
    if save_dir is not None:
        _save_result(result, save_dir)
 
    return result


# ---------------------------------------------------------------------------
# Result utilities
# ---------------------------------------------------------------------------

def rank_results(results: List[Dict], by: str = "best_val_loss") -> List[Dict]:
    """Sort results by metric (descending for acc, ascending for loss)."""
    reverse = "loss" not in by
    return sorted(results, key=lambda r: r[by], reverse=reverse)


def print_results(results: List[Dict], title: str = "Results"):
    """Print a ranked summary table."""
    ranked = rank_results(results)
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    for i, r in enumerate(ranked):
        print(
            f"  {i+1}. {r['tag']:16s}  val_acc={r['best_val_acc']:.4f}  "
            f"epoch={r['best_epoch']}/{r['n_epochs_trained']} "
            f"({r['stop_reason']})"
        )


def top_configs(results: List[Dict], k: int) -> List[Dict]:
    """Return top-k configs by val loss; break ties by smaller model."""
    def _param_count(result: Dict) -> int:
        model_state = result.get("model_state", None)
        if model_state is None:
            # Fallback for results loaded without weights: keep deterministic ordering.
            return 10**18
        return int(sum(v.numel() for v in model_state.values()))

    ranked = sorted(
        results,
        key=lambda r: (float(r["best_val_loss"]), _param_count(r)),
    )
    return [r["config"] for r in ranked[:k]]


def _append_to_summary_json(save_dir: Path, result: Dict) -> None:
    """Merge one run into summary.json (no .pt I/O). Skipped/cached runs use this."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    entry = {k: v for k, v in result.items() if k != "model_state"}
    existing = [e for e in existing if e.get("tag") != result["tag"]]
    existing.append(entry)

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)


def _save_result(result: Dict, path: Path):
    """Save one result: .pt file for weights, append to summary.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": result["config"],
            "model_state_by_loss": result["model_state"],
            "model_state": result["model_state"],
            "history": result["history"],
        },
        path / f"{result['tag']}.pt",
    )
    _append_to_summary_json(path, result)


def rebuild_summary_json(save_dir: Path) -> List[Dict]:
    """
    Rebuild summary.json from every ``*.pt`` in ``save_dir``. Use if summary.json
    was missing or out of sync (e.g. many configs were [SKIP] before summary merge
    was fixed). Returns ranked-style result dicts with ``model_state`` loaded.
    """
    save_dir = Path(save_dir)
    results: List[Dict] = []
    for pt_path in sorted(save_dir.glob("*.pt")):
        saved = torch.load(pt_path, weights_only=False)
        hist = saved["history"]
        cfg = saved["config"]
        tag = pt_path.stem
        results.append(
            {
                "tag": tag,
                "config": cfg,
                "best_val_acc": float(hist.get("best_val_acc", max(hist["val_acc"]))),
                "best_val_loss": float(hist.get("best_val_loss", min(hist["val_loss"]))),
                "best_epoch": int(hist.get("best_epoch", 0)),
                "best_epoch_loss": int(
                    hist.get("best_epoch_loss", hist.get("best_epoch", 0))
                ),
                "n_epochs_trained": hist["n_epochs_trained"],
                "stop_reason": hist["stop_reason"],
                "history": hist,
                "model_state": saved.get("model_state_by_loss", saved.get("model_state")),
            }
        )
    _write_summary_json(results, save_dir)
    return results
 
 
def _write_summary_json(results: List[Dict], path: Path):
    """Overwrite summary.json with ranked results (called at end of sweep)."""
    path = Path(path)
    summary = [
        {k: v for k, v in r.items() if k != "model_state"}
        for r in rank_results(results)
    ]
    with open(path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def save_results(results: List[Dict], path: Path):
    """Save all results to JSON and .pt files. Overwrites summary with ranked results."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
 
    for r in results:
        _save_result(r, path)
 
    _write_summary_json(results, path)

def load_results(path: Path) -> List[Dict]:
    """Load results from a save directory. Returns list of result dicts."""
    path = Path(path)
    summary_path = path / "summary.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found in {path}")
    
    with open(summary_path, "r") as f:
        results = json.load(f)
    
    # Load model_state from .pt files if available
    for r in results:
        pt_path = path / f"{r['tag']}.pt"
        if pt_path.exists():
            saved = torch.load(pt_path, weights_only=False)
            r["model_state"] = saved["model_state"]
    
    return results