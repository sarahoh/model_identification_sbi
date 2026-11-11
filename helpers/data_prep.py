"""
Data preparation utilities for RNN training.
Functions for extracting features and labels from simulation data.

KEY CONCEPTS:
-------------
- Observation tensors ``X`` for RNNs must use the numeric ``mask_value`` (default -1) only;
  :func:`helpers.observation_grid.finalize_observation_tensor` runs when saving features from
  :func:`get_features_from_dataframe` so NaNs from upstream data never reach training.
- write_train_val_test_from_chunks(): End-to-end streaming of feature chunks plus
  label files and ``split_info.pkl`` (uses :func:`save_features_from_chunks` internally).
- extract_and_curate_labels(): Memory-safe function that processes agents in chunks
  to avoid OOM errors. Supports single-head and multi-head architectures.
- output_head_config: Unified configuration dict specifying what labels to extract
  and how to transform them
- Label sources: metadata keys, trial data columns, or parameter values (all/free/subset)
"""
import gc
import shutil
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import one_hot
from tensorflow.python.keras.layers import Concatenate
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from pathlib import Path

from .observation_grid import DEFAULT_OBSERVATION_MASK_VALUE, finalize_observation_tensor

_MISSING = object()


def _lookup_metadata_value(metadata: Dict[str, Any], key: str) -> Any:
    """
    Resolve a label_key_metadata string against raw agent metadata (as in sim_metadata.pkl).

    Supports:
    - Top-level keys (e.g. base_model)
    - params_dict short names (e.g. Ap) when key has no dots
    - Dotted paths into nested dicts (e.g. variant_settings.defaults.L for LR presence heads)
    """
    if key in metadata:
        return metadata[key]
    pd = metadata.get("params_dict")
    if isinstance(pd, dict) and key in pd:
        return pd[key]
    if "." not in key:
        return _MISSING
    cur: Any = metadata
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _stack_per_agent_labels(head_name: str, label_values: List[Any]) -> np.ndarray:
    """
    Build a single array from one value per agent.

    Scalars become shape (n_agents,). Same-shaped 1D/2D arrays become (n_agents, T) or higher.
    Fails with a clear error if per-agent shapes differ (e.g. variable trial counts).
    """
    if not label_values:
        return np.array([])
    arrs = [np.asarray(x) for x in label_values]
    try:
        return np.stack(arrs, axis=0)
    except ValueError as e:
        shapes = sorted({a.shape for a in arrs})
        raise ValueError(
            f"Head '{head_name}': cannot stack per-agent labels — inconsistent shapes "
            f"(unique shapes, up to 15): {shapes[:15]}"
        ) from e


def _encode_object_grid_to_int_categories(
    values: np.ndarray,
    category_names: Sequence[str],
    *,
    head_name: str,
) -> np.ndarray:
    """
    Map string / object trial grids to ``int64`` class indices in ``[0, n_cat)``;
    padding / missing cells become ``-1``.

    Uses pandas for vectorized mapping over the flattened grid (large arrays).
    """
    mp: Dict[str, int] = {str(name): i for i, name in enumerate(category_names)}
    flat = np.asarray(values).reshape(-1)
    s = pd.Series(flat, dtype="object")
    is_pad = s.isna().to_numpy()
    out = np.full(flat.shape[0], -1, dtype=np.int64)
    if (~is_pad).any():
        keys_ser = cast(pd.Series, s[~is_pad].astype(str))
        codes = cast(pd.Series, keys_ser.map(mp))
        if codes.isna().any():
            bad = cast(pd.Series, keys_ser[codes.isna()]).unique()
            bad_list = [str(x) for x in bad[:15]]
            raise ValueError(
                f"Head '{head_name}': trial values not in category_names "
                f"(examples): {bad_list}"
            )
        out[~is_pad] = codes.astype(np.int64, copy=False)
    return out.reshape(values.shape)


def _sort_agent_rows_chronological(agent_data: pd.DataFrame) -> pd.DataFrame:
    """
    Order an agent's trials in true time order before session stacking.

    Simulators emit ``trial_idx`` as 0..T-1 **within each session**, so it resets every
    session. Sorting by ``trial_idx`` alone groups all first-in-session trials
    together, which breaks ``new_session`` boundaries (each segment collapses to one
    row) and fills only column 0 in each session row of the (session × trial) grid.

    Prefer ``(session_idx, trial_idx)`` when ``session_idx`` is present; otherwise
    use ``(session_id_from_new_session, trial_idx)`` where session_id increments at
    each ``new_session == 1`` row.
    """
    if len(agent_data) == 0:
        return agent_data

    trial_col = "trial_idx" if "trial_idx" in agent_data.columns else (
        "trial" if "trial" in agent_data.columns else None
    )

    if "session_idx" in agent_data.columns:
        if trial_col is None:
            return agent_data.sort_values("session_idx").reset_index(drop=True)
        return agent_data.sort_values(["session_idx", trial_col]).reset_index(drop=True)

    if trial_col is not None and "new_session" in agent_data.columns:
        ns = agent_data["new_session"].astype(np.float32)
        sess_ord = (ns == 1.0).cumsum()
        return (
            agent_data.assign(_sess_ord=sess_ord)
            .sort_values(["_sess_ord", trial_col])
            .drop(columns=["_sess_ord"])
            .reset_index(drop=True)
        )

    if trial_col is not None:
        return agent_data.sort_values(trial_col).reset_index(drop=True)

    return agent_data.reset_index(drop=True)


def _pad_agent_trial_columns_to_session_grid(
    agent_data: pd.DataFrame,
    columns: List[str],
    max_sessions: int,
    max_trials_per_session: int,
) -> Dict[str, np.ndarray]:
    """
    Map variable-length trial columns to a fixed (max_sessions, max_trials_per_session) grid
    using new_session boundaries — same **shape** as ``get_features_from_dataframe``, but for
    **label / trial-metadata** columns used inside ``extract_and_curate_labels``, not for model
    input ``X``.

    Numeric columns are initialized with **NaN** so padded cells are omitted by
    ``torch.isfinite`` in masked regression losses. Model **observation** tensors must come
    from :func:`get_features_from_dataframe` / :func:`finalize_observation_tensor` instead.

    Non-numeric columns use an object array with None padding.
    """
    out: Dict[str, np.ndarray] = {}
    if len(agent_data) == 0:
        for col in columns:
            out[col] = np.full(
                (max_sessions, max_trials_per_session), np.nan, dtype=np.float32
            )
        return out

    agent_data = _sort_agent_rows_chronological(agent_data)

    new_session_flags = agent_data["new_session"].values.astype(np.float32)
    start_positions = np.where(new_session_flags == 1.0)[0].astype(int)
    if len(start_positions) == 0:
        start_positions = np.array([0], dtype=int)
    elif start_positions[0] != 0:
        start_positions = np.concatenate([np.array([0], dtype=int), start_positions])

    n_real_sessions = min(len(start_positions), max_sessions)

    for col in columns:
        if col not in agent_data.columns:
            raise KeyError(
                f"Column {col!r} not in parquet for this agent (have: {list(agent_data.columns)})"
            )
        series = agent_data[col]
        if pd.api.types.is_numeric_dtype(series):
            grid = np.full(
                (max_sessions, max_trials_per_session), np.nan, dtype=np.float32
            )
            vals = series.values.astype(np.float32, copy=False)
        else:
            grid = np.full((max_sessions, max_trials_per_session), None, dtype=object)
            vals = series.values

        for sess_i in range(n_real_sessions):
            start = int(start_positions[sess_i])
            end = (
                int(start_positions[sess_i + 1])
                if sess_i + 1 < len(start_positions)
                else len(agent_data)
            )
            sess_trials = agent_data.iloc[start:end]
            if len(sess_trials) == 0:
                continue
            n_real_trials = min(len(sess_trials), max_trials_per_session)
            raw = vals[start : start + n_real_trials]
            if pd.api.types.is_numeric_dtype(series):
                grid[sess_i, :n_real_trials] = np.asarray(raw, dtype=np.float32)
            else:
                grid[sess_i, :n_real_trials] = np.asarray(raw, dtype=object)
        out[col] = grid

    return out


def _concatenate_curated_label_chunks(
    chunk_curated_labels: List[Union[Dict[str, np.ndarray], np.ndarray]],
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Merge per-chunk curated labels without ``np.concatenate(list_of_100_chunks)``, which
    can briefly need ~2× the final size. Preallocate each output and copy chunk-wise.

    For multi-head dict chunks, delete each head key from every chunk dict after that
    head is copied so large heads (e.g. Qlearn trial grids) are freed before the next head.
    """
    if not chunk_curated_labels:
        raise ValueError("chunk_curated_labels is empty")
    print("\nConcatenating curated labels from all chunks...")
    first = chunk_curated_labels[0]
    if isinstance(first, dict):
        result: Dict[str, np.ndarray] = {}
        head_names = list(first.keys())
        for head_name in head_names:
            total_n = sum(int(chunk[head_name].shape[0]) for chunk in chunk_curated_labels)
            first_arr = chunk_curated_labels[0][head_name]
            out = np.empty((total_n,) + first_arr.shape[1:], dtype=first_arr.dtype)
            off = 0
            for chunk in chunk_curated_labels:
                arr = chunk[head_name]
                n = int(arr.shape[0])
                out[off : off + n] = arr
                off += n
                del chunk[head_name]
            result[head_name] = out
        return result

    total_n = sum(int(a.shape[0]) for a in chunk_curated_labels)
    first_a = chunk_curated_labels[0]
    out = np.empty((total_n,) + first_a.shape[1:], dtype=first_a.dtype)
    off = 0
    for arr in chunk_curated_labels:
        n = int(arr.shape[0])
        out[off : off + n] = arr
        off += n
    return out


# ---------------------------------------------------------------------------
# Label scaling utilities (params + continuous latents)
# ---------------------------------------------------------------------------


def _finite_mean_std(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over finite entries only, aggregated over all elements.
    Returns scalar mean/std (as 0-d arrays) for arbitrary-shaped input.
    """
    mask = np.isfinite(x)
    if not mask.any():
        return np.array(0.0, dtype=np.float64), np.array(1.0, dtype=np.float64)
    count = float(mask.sum())
    mean = np.where(mask, x, 0.0).sum() / count
    var = np.where(mask, (x - mean) ** 2, 0.0).sum() / count
    std = np.sqrt(var)
    if std <= 0.0:
        std = 1.0
    return np.array(mean, dtype=np.float64), np.array(std, dtype=np.float64)


def _finite_mean_std_over_arrays(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a single scalar mean/std over finite entries across multiple arrays,
    without concatenating them (to reduce peak memory).
    """
    total_count = 0.0
    total_sum = 0.0
    total_sumsq = 0.0
    for x in arrays:
        if x.size == 0:
            continue
        mask = np.isfinite(x)
        if not mask.any():
            continue
        count = float(mask.sum())
        total_count += count
        # Use full-tensor ops (avoid boolean indexing copies).
        xf = np.asarray(x, dtype=np.float64)
        total_sum += np.where(mask, xf, 0.0).sum()
        total_sumsq += np.where(mask, xf * xf, 0.0).sum()

    if total_count <= 0.0:
        return np.array(0.0, dtype=np.float64), np.array(1.0, dtype=np.float64)
    mean = total_sum / total_count
    var = (total_sumsq / total_count) - (mean * mean)
    if var < 0.0:
        var = 0.0
    std = float(np.sqrt(var))
    if std <= 0.0:
        std = 1.0
    return np.array(mean, dtype=np.float64), np.array(std, dtype=np.float64)


def fit_label_scalers_from_npz(
    npz_path: Path,
    *,
    param_key: str = "params_all",
    latent_prefix: str = "Q_",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Fit simple mean/std scalers for param head and continuous latent heads in a labels_*.npz file.

    - ``param_key`` is expected to be a 2D array (n_agents, n_params).
    - Latent heads are any keys starting with ``latent_prefix``.
      By default we fit **one shared scalar mean/std across all latent keys** (common scale).
    """
    data = np.load(npz_path, allow_pickle=True)
    scalers: Dict[str, Dict[str, np.ndarray]] = {}

    # Params: per-parameter mean/std (columns), fit only on training labels.
    if param_key in data.files:
        arr = np.asarray(data[param_key], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{param_key!r} must be 2D (n_agents, n_params), got shape {arr.shape}")
        mask = np.isfinite(arr)
        if not mask.any():
            mean = np.zeros(arr.shape[1], dtype=np.float64)
            std = np.ones(arr.shape[1], dtype=np.float64)
        else:
            count = mask.sum(axis=0).clip(min=1)
            mean = np.where(mask, arr, 0.0).sum(axis=0) / count
            var = np.where(mask, (arr - mean) ** 2, 0.0).sum(axis=0) / count
            std = np.sqrt(var)
            std[std <= 0.0] = 1.0
        scalers[param_key] = {"mean": mean, "std": std}

    # Latents: use a single shared scalar mean/std across **all** Q_* keys,
    # enforcing a common numerical scale for all rule-values.
    latent_keys: List[str] = [k for k in data.files if k.startswith(latent_prefix)]
    if latent_keys:
        latent_arrays: List[np.ndarray] = []
        for k in latent_keys:
            arr = np.asarray(data[k])
            if arr.size == 0:
                continue
            latent_arrays.append(arr)
        if latent_arrays:
            mean, std = _finite_mean_std_over_arrays(latent_arrays)
            for k in latent_keys:
                scalers[k] = {"mean": mean, "std": std}

    return scalers


def fit_label_scalers_from_label_dir(
    label_dir: Path,
    *,
    param_key: str = "params_all",
    latent_prefix: str = "Q_",
    agent_block: int = 256,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Fit mean/std scalers from a directory of per-key ``.npy`` files (memmap-friendly).

    This is the recommended path for very large trial-level heads (e.g. many ``Q_*`` arrays),
    because it can stream over the memmaps without loading whole arrays into RAM.

    Scaling semantics:
    - ``param_key``: per-parameter mean/std over axis 0 (columns).
    - ``latent_prefix`` keys: **one shared scalar** mean/std across all matching keys.
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise ValueError(f"label_dir must be a directory, got {label_dir}")

    scalers: Dict[str, Dict[str, np.ndarray]] = {}

    # ---- params_all: per-column mean/std (stream axis 0 in blocks) ----
    param_path = label_dir / f"{param_key}.npy"
    if param_path.exists():
        arr = np.load(param_path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{param_key!r} must be 2D (n_agents, n_params), got shape {arr.shape}")

        n_agents, n_params = arr.shape
        count = np.zeros(n_params, dtype=np.float64)
        s1 = np.zeros(n_params, dtype=np.float64)
        s2 = np.zeros(n_params, dtype=np.float64)
        for start in range(0, n_agents, agent_block):
            sl = np.asarray(arr[start : start + agent_block], dtype=np.float64)
            mask = np.isfinite(sl)
            count += mask.sum(axis=0)
            s1 += np.where(mask, sl, 0.0).sum(axis=0)
            s2 += np.where(mask, sl * sl, 0.0).sum(axis=0)

        count = np.clip(count, 1.0, None)
        mean = s1 / count
        var = (s2 / count) - (mean * mean)
        var[var < 0.0] = 0.0
        std = np.sqrt(var)
        std[std <= 0.0] = 1.0
        scalers[param_key] = {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}

    # ---- Q_*: one shared scalar mean/std across all Q keys ----
    latent_paths = sorted(label_dir.glob(f"{latent_prefix}*.npy"))
    if latent_paths:
        total_count = 0.0
        total_sum = 0.0
        total_sumsq = 0.0

        for p in latent_paths:
            a = np.load(p, mmap_mode="r")
            if a.size == 0:
                continue
            n_agents = int(a.shape[0]) if a.ndim >= 1 else 1
            for start in range(0, n_agents, agent_block):
                block = np.asarray(a[start : start + agent_block], dtype=np.float64)
                mask = np.isfinite(block)
                if not mask.any():
                    continue
                c = float(mask.sum())
                total_count += c
                total_sum += np.where(mask, block, 0.0).sum()
                total_sumsq += np.where(mask, block * block, 0.0).sum()

        if total_count > 0.0:
            mean = total_sum / total_count
            var = (total_sumsq / total_count) - (mean * mean)
            if var < 0.0:
                var = 0.0
            std = float(np.sqrt(var))
            if std <= 0.0:
                std = 1.0
            mean_a = np.array(mean, dtype=np.float64)
            std_a = np.array(std, dtype=np.float64)
        else:
            mean_a = np.array(0.0, dtype=np.float64)
            std_a = np.array(1.0, dtype=np.float64)

        for p in latent_paths:
            k = p.stem
            scalers[k] = {"mean": mean_a, "std": std_a}

    return scalers


def apply_label_scalers_to_npz(
    in_path: Path,
    out_path: Path,
    scalers: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """
    Load labels npz, apply per-key mean/std scaling where scalers are provided, and save to out_path.

    Keys without a scaler entry are copied unchanged.
    """
    data = np.load(in_path, allow_pickle=True)
    out_arrays: Dict[str, np.ndarray] = {}
    for k in data.files:
        arr = np.asarray(data[k])
        if k in scalers:
            s = scalers[k]
            mean = np.asarray(s["mean"], dtype=np.float64)
            std = np.asarray(s["std"], dtype=np.float64)
            if k == "params_all":
                if arr.ndim != 2:
                    raise ValueError(f"{k!r} must be 2D to scale, got shape {arr.shape}")
                if mean.shape[0] != arr.shape[1]:
                    raise ValueError(
                        f"{k!r} scaler dim mismatch: mean has {mean.shape[0]} cols, labels have {arr.shape[1]}"
                    )
                arr_scaled = (arr - mean[None, :]) / std[None, :]
            else:
                # Broadcast scalar mean/std over full latent tensor.
                arr_scaled = (arr - mean) / std
            out_arrays[k] = arr_scaled.astype(arr.dtype, copy=False)
        else:
            out_arrays[k] = arr
    np.savez(out_path, **out_arrays)


def save_label_scalers_npz(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    path: Path,
) -> None:
    """
    Persist scalers to disk as a single .npz file: for each key, store mean/std arrays.
    """
    to_save: Dict[str, np.ndarray] = {}
    for k, s in scalers.items():
        to_save[f"{k}_mean"] = np.asarray(s["mean"], dtype=np.float64)
        to_save[f"{k}_std"] = np.asarray(s["std"], dtype=np.float64)
    np.savez(path, **to_save)


def load_label_scalers_npz(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Reload scalers saved by :func:`save_label_scalers_npz`.
    """
    data = np.load(path, allow_pickle=True)
    scalers: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.endswith("_mean"):
            base = key[: -len("_mean")]
            mean = np.asarray(data[key], dtype=np.float64)
            std_key = f"{base}_std"
            if std_key not in data.files:
                raise KeyError(f"Missing std for scaler key {base!r} in {path}")
            std = np.asarray(data[std_key], dtype=np.float64)
            scalers[base] = {"mean": mean, "std": std}
    return scalers


def _head_name_for_memmap_file(head_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in head_name)
    return safe[:200] if len(safe) > 200 else safe


def _dtype_allows_memmap(dt: np.dtype) -> bool:
    """Object / void dtypes cannot be streamed into a binary memmap reliably."""
    if dt == np.dtype("O") or dt.hasobject:
        return False
    return True


def _concatenate_curated_label_chunks_from_disk(
    spool_paths: List[Path],
    merged_memmap_dir: Path,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Merge curated chunks written as ``.npz`` (multi-head dict) or ``.npy`` (single array).

    Loads **one chunk file at a time**. For numeric dtypes, each merged head is built in a
    **disk-backed memmap** under ``merged_memmap_dir`` so peak RAM stays near one chunk
    (not the full merged tensor). Returns read-only ``np.memmap`` views of those files.

    Object-dtype heads fall back to a full in-RAM ``np.empty`` merge (rare); may OOM if huge.
    """
    if not spool_paths:
        raise ValueError("spool_paths is empty")
    spool_paths = sorted(spool_paths)
    merged_memmap_dir = Path(merged_memmap_dir)
    if merged_memmap_dir.exists():
        shutil.rmtree(merged_memmap_dir, ignore_errors=True)
    merged_memmap_dir.mkdir(parents=True, exist_ok=True)

    print("\nConcatenating curated labels from all chunks (disk-backed memmap merge)...")

    # Curated heads may include object-dtype arrays; loading them requires allow_pickle=True.
    first_path = spool_paths[0]
    if first_path.suffix == ".npy":
        total_n = sum(int(np.load(p, mmap_mode="r", allow_pickle=True).shape[0]) for p in spool_paths)
        first = np.load(spool_paths[0], mmap_mode="r", allow_pickle=True)
        dt, tail = first.dtype, first.shape[1:]
        del first
        out_path = merged_memmap_dir / "merged_labels.npy"
        if _dtype_allows_memmap(dt):
            out = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dt, shape=(total_n,) + tail
            )
            off = 0
            for p in spool_paths:
                arr = np.load(p, mmap_mode="r", allow_pickle=True)
                n = int(arr.shape[0])
                out[off : off + n] = arr
                off += n
                del arr
            del out
            gc.collect()
            return np.load(out_path, mmap_mode="r", allow_pickle=True)

        out = np.empty((total_n,) + tail, dtype=dt)
        off = 0
        for p in spool_paths:
            arr = np.load(p, allow_pickle=True)
            n = int(arr.shape[0])
            out[off : off + n] = arr
            off += n
            del arr
        return out

    # Multi-head: one .npz per chunk, keys = head names
    with np.load(spool_paths[0], allow_pickle=True) as z0:
        head_names = list(z0.files)
        templates = {h: (z0[h].dtype, z0[h].shape[1:]) for h in head_names}

    totals: Dict[str, int] = {h: 0 for h in head_names}
    for p in spool_paths:
        with np.load(p, allow_pickle=True) as z:
            for h in head_names:
                totals[h] += int(z[h].shape[0])

    result: Dict[str, np.ndarray] = {}
    for h in head_names:
        dt, tail = templates[h][0], templates[h][1]
        off = 0
        if _dtype_allows_memmap(dt):
            out_path = merged_memmap_dir / f"head_{_head_name_for_memmap_file(h)}.npy"
            out = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dt, shape=(totals[h],) + tail
            )
            for p in spool_paths:
                with np.load(p, allow_pickle=True) as z:
                    src = z[h]
                    n = int(src.shape[0])
                    out[off : off + n] = np.asarray(src)
                    off += n
            del out
            gc.collect()
            result[h] = np.load(out_path, mmap_mode="r", allow_pickle=True)
        else:
            print(
                f"  Warning: head {h!r} has object dtype; merging in RAM (may use a lot of memory)."
            )
            out_ram = np.empty((totals[h],) + tail, dtype=dt)
            for p in spool_paths:
                with np.load(p, allow_pickle=True) as z:
                    arr = np.array(z[h], copy=True)
                    n = int(arr.shape[0])
                    out_ram[off : off + n] = arr
                    off += n
                    del arr
            result[h] = out_ram
            gc.collect()

    return result


def extract_and_curate_labels(
    parquet_path: Path,
    all_metadata: Dict[int, Dict[str, Any]],
    output_head_config: Dict[str, Dict[str, Any]],
    chunk_size: int = 1000,
    pad_trial_labels_to_grid: bool = True,
    max_sessions: Optional[int] = None,
    max_trials_per_session: Optional[int] = None,
    spool_curated_chunks_to: Optional[Path] = None,
    remove_spool_dir_after: bool = True,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract and curate labels from parquet + metadata in memory-safe streaming fashion.
    
    This function processes agents in chunks to avoid loading all trial data into memory at once.
    
    Parameters:
    -----------
    parquet_path : Path
        Path to parquet file containing trial data
    all_metadata : Dict[int, Dict[str, Any]]
        Dictionary mapping agent_id to metadata dict
    output_head_config : Dict[str, Dict[str, Any]]
        Configuration for each output head. Keys are output names, values are config dicts.
        Each config dict supports:
        - label_key_metadata : str, optional
            Key to extract from metadata (e.g., 'model_id', 'base_model', 'alpha')
        - label_column_trialdata : str, optional
            Column name from trial data (e.g., 'attention_state')
        - extract_params : str or List[str], optional
            'all', 'free', or list of specific param names
        - param_transforms : Dict[str, callable], optional
            Per-parameter transformations
        - category_names : List[str], optional
            Ordered class vocabulary for categorical labels (e.g. model or rule names).
        - return_one_hot : bool, default True
            Whether to one-hot encode categorical labels
        - label_transform_fn : callable, optional
            Global transformation function
    chunk_size : int, optional
        Number of agents to process in each chunk (default: 1000).
        Larger chunks are faster but use more memory.
    pad_trial_labels_to_grid : bool, default True
        If True, trial-level heads are placed on a fixed
        ``(max_sessions, max_trials_per_session)`` grid (session boundaries from ``new_session``),
        matching ``get_features_from_dataframe``. Set False only for debugging / variable-length
        exports.
    max_sessions : int, optional
        Grid height when padding (default: ``MAX_SESSIONS - 1`` from simulation.constants).
    max_trials_per_session : int, optional
        Grid width when padding (default: ``MAX_TRIALS_PER_SESSION``).
    spool_curated_chunks_to : Path, optional
        If set, each curated chunk is written to this directory as ``.npz`` (multi-head) or
        ``.npy`` (single-head) and merged **from disk**. Use this for large multi-head
        trial-heavy exports (e.g. Qlearn param labels) to avoid holding all chunks in RAM.
        The merged result for numeric dtypes is stored under a sibling directory
        (same parent as the spool path, name suffixed with ``_memmap``) as ``.npy`` files
        and returned as read-only memory maps (low RAM peak during merge).
    remove_spool_dir_after : bool, default True
        If True, delete ``spool_curated_chunks_to`` after a successful merge (ignored if
        spooling was not used). The ``_memmap`` directory is **not** removed (it holds the
        returned arrays).

    Returns:
    --------
    labels : np.ndarray or Dict[str, np.ndarray]
        - If 1 output: Returns np.ndarray
        - If 2+ outputs: Returns Dict[str, np.ndarray]
    
    Examples:
    ---------
    # Model classification (safe for 50k+ agents)
    labels = extract_and_curate_labels(
        parquet_path=path,
        all_metadata=metadata,
        output_head_config={
            'model': {'label_key_metadata': 'base_model', 'category_names': ['Qlearn', 'ruleQhyp']}
        },
        chunk_size=1000
    )
    
    # Multi-head with latent variables (processes trial data in chunks)
    labels = extract_and_curate_labels(
        parquet_path=path,
        all_metadata=metadata,
        output_head_config={
            'model': {'label_key_metadata': 'base_model', 'category_names': ['Qlearn', 'ruleQhyp']},
            'attention': {'label_column_trialdata': 'attention_state', 'return_one_hot': False}
        },
        chunk_size=500  # Smaller chunks if trial data is large
    )
    """
    import pyarrow.dataset as ds
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc=None: x
    
    # Inspect output_head_config to determine what data is needed
    needs_trial_data = False
    trial_columns_needed = set()
    metadata_keys_needed = set()
    needs_model_spec = False
    needs_params_dict = False
    
    for head_name, head_config in output_head_config.items():
        if 'label_column_trialdata' in head_config and head_config['label_column_trialdata'] is not None:
            needs_trial_data = True
            col = head_config['label_column_trialdata']
            if col != 'Q_*':
                trial_columns_needed.add(col)
        
        if 'label_key_metadata' in head_config and head_config['label_key_metadata'] is not None:
            metadata_keys_needed.add(head_config['label_key_metadata'])
        
        if 'extract_params' in head_config and head_config['extract_params'] is not None:
            needs_params_dict = True
            if head_config['extract_params'] == 'free':
                needs_model_spec = True
    
    # Check for Q-values
    include_q_values = any('Q_*' in head_config.get('label_column_trialdata', '') 
                          for head_config in output_head_config.values())
    
    # Log analysis
    print(f"Analyzing output_head_config...")
    print(f"  Needs trial data: {needs_trial_data}")
    if trial_columns_needed:
        print(f"  Trial columns: {sorted(trial_columns_needed)}")
    if metadata_keys_needed:
        print(f"  Metadata keys: {sorted(metadata_keys_needed)}")
    print(f"  Needs params_dict: {needs_params_dict}")
    print(f"  Needs model_spec: {needs_model_spec}")

    from simulation.constants import MAX_SESSIONS, MAX_TRIALS_PER_SESSION

    label_ms = max_sessions if max_sessions is not None else MAX_SESSIONS - 1
    label_mt = max_trials_per_session if max_trials_per_session is not None else MAX_TRIALS_PER_SESSION
    if needs_trial_data and pad_trial_labels_to_grid:
        print(
            f"  Trial label grid (padding): ({label_ms}, {label_mt}) "
            f"[align with get_features_from_dataframe defaults]"
        )

    # Get agent IDs
    agent_ids = sorted(all_metadata.keys())
    n_agents = len(agent_ids)
    
    # Setup for streaming
    if needs_trial_data:
        dataset = ds.dataset(str(parquet_path), format='parquet')
        needed_columns = ['agent_id'] + list(trial_columns_needed)
        if include_q_values:
            needed_columns += [col for col in dataset.schema.names if col.startswith('Q_')]
        if pad_trial_labels_to_grid:
            needed_columns.append('new_session')
            if 'session_idx' in dataset.schema.names:
                needed_columns.append('session_idx')
            if 'trial_idx' in dataset.schema.names:
                needed_columns.append('trial_idx')
            elif 'trial' in dataset.schema.names:
                needed_columns.append('trial')
        needed_columns = list(dict.fromkeys(needed_columns))

    # Process agents in chunks
    n_chunks = (n_agents + chunk_size - 1) // chunk_size
    print(f"\nProcessing {n_agents:,} agents in {n_chunks} chunks of {chunk_size}...")
    
    use_spool = spool_curated_chunks_to is not None
    spool_dir: Optional[Path] = Path(spool_curated_chunks_to) if use_spool else None
    if use_spool and spool_dir is not None:
        spool_dir.mkdir(parents=True, exist_ok=True)
    spool_paths: List[Path] = []
    chunk_curated_labels: List[Union[Dict[str, np.ndarray], np.ndarray]] = []

    chunk_seq = 0
    for chunk_idx in tqdm(range(0, n_agents, chunk_size), desc="Processing chunks", total=n_chunks):
        chunk_end = min(chunk_idx + chunk_size, n_agents)
        chunk_agent_ids = agent_ids[chunk_idx:chunk_end]
        
        # Build labels dict for this chunk only
        chunk_labels_dict = {}
        
        # Load trial data for this chunk only
        if needs_trial_data:
            chunk_data = dataset.to_table(
                filter=ds.field('agent_id').isin(chunk_agent_ids),
                columns=needed_columns
            ).to_pandas()
        
        # Process each agent in chunk
        for agent_id in chunk_agent_ids:
            agent_id = int(agent_id)
            metadata = all_metadata.get(agent_id, {})
            
            label_dict = {}
            
            # Add requested metadata keys (including nested paths, e.g. variant_settings.defaults.L)
            for key in metadata_keys_needed:
                val = _lookup_metadata_value(metadata, key)
                if val is _MISSING:
                    raise KeyError(
                        f"metadata key {key!r} not found for agent {agent_id}. "
                        f"Top-level keys: {sorted(k for k in metadata.keys() if k != 'model_spec')}. "
                        "Dotted keys walk nested dicts (e.g. variant_settings.defaults.L)."
                    )
                label_dict[key] = val
            
            # Add params_dict if needed
            if needs_params_dict and 'params_dict' in metadata:
                label_dict['parameters'] = metadata['params_dict']
            
            # Add model_spec if needed
            if needs_model_spec and 'model_spec' in metadata:
                label_dict['model_spec'] = metadata['model_spec']
            
            # Add trial data if needed
            if needs_trial_data:
                trial_data_dict: Dict[str, Any] = {}
                agent_data = chunk_data[chunk_data['agent_id'] == agent_id]

                if pad_trial_labels_to_grid:
                    pad_cols = [
                        c
                        for c in needed_columns
                        if c
                        not in (
                            'agent_id',
                            'new_session',
                            'session_idx',
                            'trial_idx',
                            'trial',
                        )
                    ]
                    padded = _pad_agent_trial_columns_to_session_grid(
                        agent_data, pad_cols, label_ms, label_mt
                    )
                    trial_data_dict.update(padded)
                    if include_q_values:
                        trial_data_dict['q_values'] = {}
                        for q_col in agent_data.columns:
                            if q_col.startswith('Q_') and q_col in trial_data_dict:
                                trial_data_dict['q_values'][q_col[2:]] = trial_data_dict[
                                    q_col
                                ]
                else:
                    if len(agent_data) > 0:
                        for col in trial_columns_needed:
                            if col in agent_data.columns:
                                if pd.api.types.is_numeric_dtype(agent_data[col]):
                                    trial_data_dict[col] = agent_data[col].values.astype(
                                        np.float32
                                    )
                                else:
                                    trial_data_dict[col] = agent_data[col].values

                        if include_q_values:
                            trial_data_dict['q_values'] = {}
                            q_cols = [
                                col for col in agent_data.columns if col.startswith('Q_')
                            ]
                            for q_col in q_cols:
                                rule_name = q_col[2:]
                                trial_data_dict['q_values'][rule_name] = agent_data[
                                    q_col
                                ].values.astype(np.float32)

                label_dict['trial_data'] = trial_data_dict
            
            chunk_labels_dict[agent_id] = label_dict
        
        # Curate labels for this chunk (transforms raw data to final format)
        curated = curate_labels(chunk_labels_dict, output_head_config)
        if use_spool:
            assert spool_dir is not None
            if isinstance(curated, dict):
                spool_p = spool_dir / f"chunk_{chunk_seq:05d}.npz"
                np.savez_compressed(spool_p, **cast(Dict[str, Any], curated))
            else:
                spool_p = spool_dir / f"chunk_{chunk_seq:05d}.npy"
                np.save(spool_p, curated)
            spool_paths.append(spool_p)
            chunk_seq += 1
            del curated
            gc.collect()
        else:
            chunk_curated_labels.append(curated)
        
        # Explicitly delete raw trial data to free memory
        if needs_trial_data:
            del chunk_data
    
    if use_spool:
        assert spool_dir is not None
        merged_memmap_dir = spool_dir.parent / f"{spool_dir.name}_memmap"
        merged = _concatenate_curated_label_chunks_from_disk(spool_paths, merged_memmap_dir)
        if remove_spool_dir_after and spool_dir is not None:
            shutil.rmtree(spool_dir, ignore_errors=True)
            print(f"  Removed label spool directory {spool_dir} (chunks merged)")
        return merged
    return _concatenate_curated_label_chunks(chunk_curated_labels)


def curate_labels(
    all_labels_dict: Dict[int, Dict],
    output_head_config: Dict[str, Dict[str, Any]]
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract and transform labels from all_labels_dict for model training.
    
    Supports both single-head and multi-head architectures using a unified config interface.
    For single-head: specify 1 output in config, returns np.ndarray.
    For multi-head: specify 2+ outputs in config, returns Dict[str, np.ndarray].
    
    Parameters:
    -----------
    all_labels_dict : Dict[int, Dict]
        Dictionary mapping agent_id to labels dict
    output_head_config : Dict[str, Dict[str, Any]]
        Configuration for each output head. Keys are output names, values are config dicts.
        Each config dict supports:
        - label_key_metadata : str, optional
            Key to extract from metadata (top-level or 'parameters' dict).
            Examples: 'model_id', 'base_model', 'alpha', 'beta'
        - label_column_trialdata : str, optional
            Column name to extract from 'trial_data' dict.
            Examples: 'attention_state', 'exploit_mode'
        - extract_params : str or List[str], optional
            Extract parameters from params_dict:
            - 'all': Extract all params (including fixed)
            - 'free': Extract only free params (using model_spec)
            - List[str]: Extract only specified params (free or fixed)
            Returns concatenated array in sorted order.
        - param_transforms : Dict[str, callable], optional
            Per-parameter transformations applied before concatenation.
            Example: {'alpha': lambda x: (x - mean) / std}
        - category_names : List[str], optional
            Ordered class vocabulary for categorical labels. Required if return_one_hot=True.
        - return_one_hot : bool, default True
            If True, return one-hot encoded labels for categorical data. Requires category_names.
            If False, return continuous labels (for parameters, latent variables, etc.).
            Note: Cannot be True when using extract_params (parameter values are continuous).
        - categorical_as_index : bool, default False
            If True (with ``return_one_hot=False``), map string/object trial grids to ``int64``
            indices using ``category_names`` order; padding / missing cells are ``-1``.
            Incompatible with ``return_one_hot=True``.
        - label_transform_fn : callable, optional
            Function to transform labels: fn(labels: List[Any]) -> np.ndarray
            Applied after param_transforms for extract_params, or directly for other modes.
            Examples: lambda x: (np.array(x) - mean) / std
        
    Returns:
    --------
    labels : np.ndarray or Dict[str, np.ndarray]
        - If 1 output in config: Returns np.ndarray with shape (n_agents, ...) 
        - If 2+ outputs in config: Returns Dict[str, np.ndarray] for multi-head training
    
    Examples:
    ---------
    # Single-head: Model classification
    labels = curate_labels(all_labels, output_head_config={
        'model': {
            'label_key_metadata': 'base_model',
            'category_names': ['Qlearn', 'ruleQhyp'],
            'return_one_hot': True
        }
    })
    # Returns: np.ndarray (n_agents, 2)
    
    # Single-head: Parameter estimation
    labels = curate_labels(all_labels, output_head_config={
        'alpha': {
            'label_key_metadata': 'alpha',
            'return_one_hot': False,
            'label_transform_fn': lambda x: (np.array(x, dtype=np.float32) - 0.5) / 0.5
        }
    })
    # Returns: np.ndarray (n_agents,)
    
    # Multi-head: Model + multi-param estimation + latent vars
    labels = curate_labels(all_labels, output_head_config={
        'model_classification': {
            'label_key_metadata': 'base_model',
            'category_names': ['Qlearn', 'ruleQhyp']  # return_one_hot=True by default for categorical
        },
        'free_params': {
            'extract_params': 'free',  # 'all', 'free', or ['alpha', 'beta']
            'param_transforms': {
                'alpha': lambda x: (x - mean_alpha) / std_alpha,
                'beta': lambda x: (x - mean_beta) / std_beta
            }
        },
        'attention': {
            'label_column_trialdata': 'attention_state',
            'return_one_hot': False  # Trial data is usually continuous
        }
    })
    # Returns: Dict[str, np.ndarray] = {
    #     'model_classification': (n_agents, 2),
    #     'free_params': (n_agents, n_params),
    #     'attention': (n_agents, max_trials)
    # }
    """
    # Validate config
    if not output_head_config or len(output_head_config) == 0:
        raise ValueError(
            "output_head_config must contain at least one output configuration.\n"
            "Example:\n"
            "  output_head_config={'model': {'label_key_metadata': 'model_id', 'category_names': [...]}}"
        )
    
    # Process each output head
    results = {}
    for head_name, head_config in output_head_config.items():
        # Extract configuration for this head
        label_key_metadata = head_config.get('label_key_metadata')
        label_column_trialdata = head_config.get('label_column_trialdata')
        extract_params = head_config.get('extract_params')
        param_transforms = head_config.get('param_transforms', {})
        category_names = head_config.get("category_names")
        return_one_hot = head_config.get('return_one_hot', True)
        label_transform_fn = head_config.get('label_transform_fn')
        categorical_as_index = bool(head_config.get("categorical_as_index", False))
        
        # Validate mutually exclusive parameters for this head
        specified_sources = sum([
            label_key_metadata is not None,
            label_column_trialdata is not None,
            extract_params is not None
        ])
        
        if specified_sources > 1:
            raise ValueError(
                f"Head '{head_name}': label_key_metadata, label_column_trialdata, and extract_params "
                "are mutually exclusive. Specify only one."
            )
        
        if specified_sources == 0:
            raise ValueError(
                f"Head '{head_name}': Must specify one of label_key_metadata, label_column_trialdata, "
                "or extract_params."
            )
        
        # Validate return_one_hot is only used with categorical labels
        if extract_params is not None and return_one_hot:
            raise ValueError(
                f"Head '{head_name}': return_one_hot=True is not valid for extract_params. "
                "Parameter values are continuous. Use param_transforms or label_transform_fn instead."
            )
        
        # Extract labels for each agent (sorted by agent_id)
        label_values = []
        for agent_id in sorted(all_labels_dict.keys()):
            agent_labels = all_labels_dict[agent_id]
            
            # Extract based on which source was specified
            if extract_params is not None:
                # Extract parameters from params_dict
                if 'parameters' not in agent_labels:
                    raise KeyError(
                        f"Head '{head_name}': 'parameters' (params_dict) not found in agent {agent_id}'s metadata. "
                        "Required for extract_params."
                    )
                
                params_dict = agent_labels['parameters']
                
                # Determine which params to extract
                if extract_params == 'all':
                    # Extract all params (including fixed)
                    param_names_to_extract = sorted(params_dict.keys())
                    
                elif extract_params == 'free':
                    # Extract only free params using model_spec
                    if 'model_spec' not in agent_labels:
                        raise KeyError(
                            f"Head '{head_name}': 'model_spec' not found in agent {agent_id}'s metadata. "
                            "Required for extract_params='free'."
                        )
                    model_spec = agent_labels['model_spec']
                    param_names_to_extract = sorted(model_spec.get_params_to_fit())
                    
                elif isinstance(extract_params, list):
                    # Extract specified subset (free or fixed)
                    param_names_to_extract = sorted(extract_params)
                    # Validate that all specified params exist
                    invalid_params = [p for p in param_names_to_extract if p not in params_dict]
                    if invalid_params:
                        raise ValueError(
                            f"Head '{head_name}': Specified params {invalid_params} not found in params_dict. "
                            f"Available params for agent {agent_id}: {sorted(params_dict.keys())}"
                        )
                else:
                    raise ValueError(
                        f"Head '{head_name}': extract_params must be 'all', 'free', or a list of param names, "
                        f"got {type(extract_params)}: {extract_params}"
                    )
                
                # Extract values from params_dict in sorted order
                param_values = np.array([params_dict[name] for name in param_names_to_extract], dtype=np.float32)
                
                # Apply per-parameter transforms
                if param_transforms:
                    transformed_values = []
                    for param_name, param_val in zip(param_names_to_extract, param_values):
                        if param_name in param_transforms:
                            transformed_values.append(param_transforms[param_name](param_val))
                        else:
                            transformed_values.append(param_val)
                    label_value = np.array(transformed_values, dtype=np.float32)
                else:
                    label_value = param_values
                
            elif label_key_metadata is not None:
                # Try to get label from metadata sources (in order of priority):
                # 1. Top-level dict (e.g., 'model_id', 'base_model')
                # 2. 'parameters' dict (e.g., 'alpha', 'beta')
                if label_key_metadata in agent_labels:
                    label_value = agent_labels[label_key_metadata]
                elif 'parameters' in agent_labels and label_key_metadata in agent_labels['parameters']:
                    label_value = agent_labels['parameters'][label_key_metadata]
                else:
                    raise KeyError(
                        f"Head '{head_name}': Label key '{label_key_metadata}' not found in agent {agent_id}'s metadata. "
                        f"Available keys: {list(agent_labels.keys())}. "
                        f"Parameters: {list(agent_labels.get('parameters', {}).keys())}"
                    )
            else:
                # Extract from trial data
                if 'trial_data' in agent_labels and label_column_trialdata in agent_labels['trial_data']:
                    label_value = agent_labels['trial_data'][label_column_trialdata]
                else:
                    raise KeyError(
                        f"Head '{head_name}': Label column '{label_column_trialdata}' not found in agent {agent_id}'s trial data. "
                        f"Available trial data columns: {list(agent_labels.get('trial_data', {}).keys())}"
                    )
            
            label_values.append(label_value)
        
        # Apply custom transformation if provided
        if extract_params is not None:
            # label_values is a list of numpy arrays, stack them
            labels_array = np.stack(label_values, axis=0)  # (n_agents, n_params)
            # Apply global transform if provided (after per-param transforms)
            if label_transform_fn is not None:
                labels_array = label_transform_fn(labels_array)
        elif label_transform_fn is not None:
            # Per-agent transform (e.g. presence heads: float(v is None) per metadata value)
            transformed = [label_transform_fn(v) for v in label_values]
            labels_array = _stack_per_agent_labels(head_name, transformed)
        else:
            # Trial-level columns: one array per agent — use stack, not np.array(list of arrays)
            labels_array = _stack_per_agent_labels(head_name, label_values)

        if categorical_as_index:
            if return_one_hot:
                raise ValueError(
                    f"Head '{head_name}': categorical_as_index=True is incompatible with "
                    "return_one_hot=True (use one-hot or integer indices, not both)."
                )
            if category_names is None:
                raise ValueError(
                    f"Head '{head_name}': category_names is required when categorical_as_index=True"
                )
            if labels_array.dtype == object or np.issubdtype(labels_array.dtype, np.str_):
                labels_array = _encode_object_grid_to_int_categories(
                    labels_array, category_names, head_name=head_name
                )
            elif np.issubdtype(labels_array.dtype, np.integer):
                pass
            else:
                raise ValueError(
                    f"Head '{head_name}': categorical_as_index expects object or string "
                    f"labels, got dtype={labels_array.dtype}"
                )
        
        # One-hot encode for categorical classification
        if return_one_hot:
            if category_names is None:
                raise ValueError(
                    f"Head '{head_name}': category_names must be provided when return_one_hot=True"
                )

            # Create mapping from label value to index
            label_to_idx = {name: idx for idx, name in enumerate(category_names)}

            # Map label values to indices
            label_indices = []
            for label_val in label_values:
                if label_val not in label_to_idx:
                    raise ValueError(
                        f"Head '{head_name}': Label value '{label_val}' not in category_names: {category_names}"
                    )
                label_indices.append(label_to_idx[label_val])

            # Convert to one-hot
            results[head_name] = tf.keras.utils.to_categorical(
                label_indices, num_classes=len(category_names)
            )
        else:
            # Return continuous labels (already transformed if label_transform_fn was provided)
            results[head_name] = labels_array
    
    # Return single array if only one head, otherwise return dict
    if len(results) == 1:
        return list(results.values())[0]
    else:
        return results


# def get_features_from_dataframe(
#     parquet_path: Path,
#     all_metadata: Dict[int, Dict[str, Any]],
#     n_actions: int = 2,
#     include_prev_state: bool = False,  # Default to False - let RNN learn temporal dynamics
#     max_sequence_length: Optional[int] = None,
#     chunk_size: int = 1000,
#     save_dir: Optional[Path] = None
# ) -> Tuple[Optional[tf.Tensor], int, Optional[Path]]:
#     """
#     Extract features from parquet file in memory-safe chunked fashion.
    
#     This function processes agents in chunks to avoid loading all data into memory at once.
    
#     Features extracted (per trial):
#     - visual_cue (one-hot, 2 dims)
#     - auditory_cue (one-hot, 2 dims)
#     - action (one-hot, 2 dims)
#     - reward (scalar, 1 dim)
#     - new_session (binary, 1 dim) - indicates first trial of a new session
#     - prev_action (one-hot, 2 dims) [if include_prev_state=True]
#     - prev_reward (scalar, 1 dim) [if include_prev_state=True]
    
#     Total: 8 features (base) or 11 features (with prev_state)
    
#     Parameters:
#     -----------
#     parquet_path : Path
#         Path to parquet file containing trial_data
#     all_metadata : Dict[int, Dict[str, Any]]
#         Dictionary mapping agent_id to metadata (used to get agent IDs)
#     n_actions : int, optional
#         Number of possible actions (default: 2)
#     include_prev_state : bool, optional
#         If True, include prev_action and prev_reward as explicit features (default: True)
#     max_sequence_length : int, optional
#         Maximum sequence length to truncate to
#     chunk_size : int, optional
#         Number of agents to process in each chunk (default: 1000)
#     save_dir : Path, optional
#         If provided, save features incrementally to disk instead of returning in memory.
#         Features will be saved as chunk files (features_chunk_000.npy, etc.)
    
#     Returns:
#     --------
#     features : tf.Tensor or None
#         Tensor with shape (n_agents, max_trials, n_features) if save_dir is None,
#         where n_features = 8 (base) or 11 (with prev_state).
#         None if save_dir is provided (features saved to disk instead).
#     max_trials : int
#         Maximum number of trials across all agents
#     save_dir : Path or None
#         Directory where features were saved (if save_dir was provided), None otherwise
#     """
#     import pyarrow.parquet as pq
#     import pyarrow.dataset as ds
#     try:
#         from tqdm import tqdm
#     except ImportError:
#         tqdm = lambda x, desc=None: x
    
#     # Get all agent IDs from metadata
#     agent_ids = sorted(all_metadata.keys())
#     n_agents = len(agent_ids)
#     dataset = ds.dataset(str(parquet_path), format='parquet')
    
#     # Find max_trials efficiently
#     if max_sequence_length is not None:
#         # If max_sequence_length is provided, use it directly (skip scanning)
#         max_trials = max_sequence_length
#         print(f"Using provided max_sequence_length: {max_trials}")
#     else:
#         # Need to find max - do it efficiently by loading all agent_ids at once
#         print("Finding max sequence length (loading agent trial counts)...")
#         agent_trial_counts = dataset.to_table(
#             columns=['agent_id']
#         ).to_pandas()['agent_id'].value_counts()
#         max_trials = int(agent_trial_counts.max())
#         print(f"Max trials per agent: {max_trials}")
    
#     # Second pass: process agents in chunks and extract features
#     print(f"Processing {n_agents} agents in chunks of {chunk_size}...")
#     all_features = [] if save_dir is None else None  # Only accumulate if not saving to disk
    
#     n_chunks = (n_agents + chunk_size - 1) // chunk_size
#     pbar = tqdm(range(0, n_agents, chunk_size), desc="Extracting features", total=n_chunks)
#     for chunk_start in pbar:
#         chunk_end = min(chunk_start + chunk_size, n_agents)
#         chunk_agent_ids = agent_ids[chunk_start:chunk_end]
        
#         # Load chunk of agents
#         chunk_data = dataset.to_table(
#             filter=ds.field('agent_id').isin(chunk_agent_ids),
#             columns=['agent_id', 'trial_idx', 'visual_cue', 'auditory_cue', 
#                     'action', 'reward', 'prev_action', 'prev_reward', 'new_session']
#         ).to_pandas()
        
#         # Process each agent in chunk
#         chunk_features = []
#         for agent_id in chunk_agent_ids:
#             agent_data = chunk_data[chunk_data['agent_id'] == agent_id].copy()
            
#             if len(agent_data) == 0:
#                 # Empty agent - create zero-padded features
#                 # Features: visual(2) + auditory(2) + action(2) + reward(1) + new_session(1) = 8 base
#                 #           + prev_action(2) + prev_reward(1) = +3 if include_prev_state
#                 n_features = 8 if not include_prev_state else 11
#                 agent_features = tf.zeros((max_trials, n_features), dtype=tf.float32)
#                 chunk_features.append(agent_features)
#                 continue
            
#             # Sort by trial
#             trial_col = 'trial_idx' if 'trial_idx' in agent_data.columns else 'trial'
#             agent_data = agent_data.sort_values(trial_col).reset_index(drop=True)
            
#             # Truncate if needed
#             if max_sequence_length is not None and len(agent_data) > max_sequence_length:
#                 agent_data = agent_data.iloc[:max_sequence_length]
            
#             # Extract features (same logic as get_features_from_dataframe)
#             visual_cues = agent_data['visual_cue'].values.astype(np.int32)
#             auditory_cues = agent_data['auditory_cue'].values.astype(np.int32)
#             visual_onehot = one_hot(visual_cues, 2)
#             auditory_onehot = one_hot(auditory_cues, 2)
            
#             actions = agent_data['action'].values.astype(np.int32)
#             rewards = agent_data['reward'].values.astype(np.float32)
#             action_onehot = one_hot(actions, n_actions)
#             reward_expanded = tf.expand_dims(rewards, axis=1)
            
#             # Extract new_session indicator (1 if first trial of session, 0 otherwise)
#             new_session = agent_data['new_session'].values.astype(np.float32)
#             new_session_expanded = tf.expand_dims(new_session, axis=1)
            
#             feature_list = [visual_onehot, auditory_onehot, action_onehot, reward_expanded, new_session_expanded]
            
#             if include_prev_state:
#                 prev_actions = agent_data['prev_action'].values
#                 nan_mask = np.isnan(prev_actions)
#                 prev_actions_clean = np.where(nan_mask, 0, prev_actions).astype(np.int32)
#                 prev_action_onehot = one_hot(prev_actions_clean, n_actions)
#                 if np.any(nan_mask):
#                     nan_mask_tf = tf.constant(nan_mask, dtype=tf.bool)
#                     prev_action_onehot = tf.where(
#                         tf.expand_dims(nan_mask_tf, axis=1),
#                         tf.zeros_like(prev_action_onehot),
#                         prev_action_onehot
#                     )
                
#                 prev_rewards = agent_data['prev_reward'].values.astype(np.float32)
#                 prev_rewards = np.where(np.isnan(prev_rewards), 0.0, prev_rewards).astype(np.float32)
#                 prev_reward_expanded = tf.expand_dims(prev_rewards, axis=1)
                
#                 feature_list.extend([prev_action_onehot, prev_reward_expanded])
            
#             agent_features = Concatenate(axis=1)(feature_list)
            
#             # Pad to max_trials
#             n_trials = tf.shape(agent_features)[0]
#             padding_size = max_trials - n_trials
#             if padding_size > 0:
#                 padding = tf.fill([padding_size, tf.shape(agent_features)[1]], -1.0)
#                 agent_features = tf.concat([agent_features, padding], axis=0)
            
#             chunk_features.append(agent_features)
        
#         # Convert chunk features to numpy and save incrementally (avoid OOM)
#         chunk_features_np = []
#         for feat in chunk_features:
#             chunk_features_np.append(feat.numpy() if isinstance(feat, tf.Tensor) else feat)
        
#         # Stack chunk and save to disk immediately (don't accumulate in memory)
#         if save_dir is not None:
#             save_dir = Path(save_dir)
#             save_dir.mkdir(parents=True, exist_ok=True)
#             chunk_array = np.stack(chunk_features_np, axis=0)
#             chunk_file = save_dir / f"features_chunk_{chunk_start//chunk_size:03d}.npy"
#             np.save(chunk_file, chunk_array)
#             # Update progress bar with chunk info instead of printing
#             pbar.set_postfix({'saved': f"{chunk_file.name} ({chunk_array.shape[0]} agents)"})
#             del chunk_array, chunk_features_np  # Free memory immediately
#         else:
#             # Accumulate in memory (only if save_dir not provided - may cause OOM)
#             all_features.extend(chunk_features_np)
    
#     # Handle return value
#     if save_dir is not None:
#         # Features saved to disk - return None for features, path for loading
#         print(f"All features saved to {save_dir}")
#         return None, max_trials, save_dir
#     else:
#         # Return features in memory (may cause OOM for large datasets)
#         print("Stacking all features in memory (this may cause OOM for large datasets)...")
#         if len(all_features) > 0:
#             features_array = np.stack(all_features, axis=0)
#             return tf.constant(features_array, dtype=tf.float32), max_trials, None
#         else:
#             return tf.zeros((0, max_trials, 7), dtype=tf.float32), max_trials, None


def get_features_from_dataframe(
    parquet_path: Path,
    all_metadata: Dict[int, Dict[str, Any]],
    feature_names: Optional[List[str]] = None,
    max_sessions: int = 35,
    max_trials_per_session: int = 300,
    chunk_size: int = 1000,
    save_dir: Optional[Path] = None,
    mask_value: float = -1.0,
    skip_existing_chunks: bool = True,
) -> Tuple[Optional[tf.Tensor], int, Optional[Path]]:
    """
    Extract session-stacked features for the seqid / modelID RNN.

    Output feature tensor shape (per split):
        (n_agents, max_sessions, max_trials_per_session, n_features)

    where default n_features = 2 (visual) + 2 (auditory) + 2 (action) + 1 (reward) = 7.

    feature_names
    -------------
    Uses the `new_session` column to detect session boundaries.

    The `feature_names` list controls which trial features are *included in the tensor*.
    Default (if None) is the historical `(B, 35, 300, 7)` input layout:
        ["visual", "auditory", "action", "reward"]

    - If you include "prev_action" and/or "prev_reward", the last dimension grows accordingly.
    - If you include "new_session", the tensor includes the per-trial `new_session` indicator as a feature.

    Notes
    -----
    - Pads missing trials/sessions with `mask_value` (default: -1.0). Each agent tensor is
      passed through :func:`finalize_observation_tensor` so any stray NaNs from parquet are
      replaced with ``mask_value`` before chunks are saved.
    - If an agent has more than `n_sessions` sessions, extra sessions are truncated.
    - If `save_dir` is provided and `skip_existing_chunks=True`, existing
      `features_chunk_*.npy` files are skipped so interrupted runs can resume.
    """
    import pyarrow.dataset as ds

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc=None: x

    agent_ids = sorted(all_metadata.keys())
    n_agents = len(agent_ids)
    dataset = ds.dataset(str(parquet_path), format="parquet")

    available_features = [
        "visual",
        "auditory",
        "action",
        "reward",
        "new_session",
        "prev_visual",
        "prev_auditory",
        "prev_action",
        "prev_reward",
    ]
    if feature_names is None:
        feature_names = ["visual", "auditory", "action", "reward"]
    feature_names = [f.lower() for f in feature_names]
    unknown = [f for f in feature_names if f not in available_features]
    if unknown:
        raise ValueError(
            f"Requested feature(s) not in available_features: {unknown}. Available={available_features}"
        )

    def _feature_dim_modelid(feature: str) -> int:
        if feature in ("visual", "auditory"):
            return 2
        if feature in ("prev_visual", "prev_auditory"):
            return 2
        if feature in ("action", "prev_action"):
            return 2
        if feature in ("reward", "new_session", "prev_reward"):
            return 1
        raise ValueError(f"Unhandled feature: {feature}")

    n_features = int(sum(_feature_dim_modelid(f) for f in feature_names))

    def _one_hot_np(idxs: np.ndarray, depth: int) -> np.ndarray:
        idxs = idxs.astype(np.int64)
        out = np.zeros((idxs.shape[0], depth), dtype=np.float32)
        if out.shape[0] == 0:
            return out
        out[np.arange(idxs.shape[0]), idxs] = 1.0
        return out

    def _is_valid_chunk_file(path: Path, expected_n_agents: int) -> bool:
        """Check that an existing chunk file is readable and has expected shape."""
        try:
            arr = np.load(path, mmap_mode="r")
            ok = (
                arr.ndim == 4
                and arr.shape[0] == expected_n_agents
                and arr.shape[1] == max_sessions
                and arr.shape[2] == max_trials_per_session
                and arr.shape[3] == n_features
            )
            del arr
            return bool(ok)
        except Exception:
            return False

    print(f"Processing {n_agents} agents in chunks of {chunk_size}...")
    all_sessions_features = [] if save_dir is None else None

    pbar = tqdm(range(0, n_agents, chunk_size), desc="Extracting modelID session features")
    for chunk_start in pbar:
        chunk_end = min(chunk_start + chunk_size, n_agents)
        chunk_agent_ids = agent_ids[chunk_start:chunk_end]
        chunk_idx = chunk_start // chunk_size
        chunk_file = None

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            chunk_file = save_dir / f"features_chunk_{chunk_idx:03d}.npy"
            if skip_existing_chunks and chunk_file.exists():
                expected_n_agents = len(chunk_agent_ids)
                if _is_valid_chunk_file(chunk_file, expected_n_agents):
                    pbar.set_postfix({"skipped": chunk_file.name})
                    continue
                # Corrupt/partial file from interrupted run: remove and regenerate.
                try:
                    chunk_file.unlink(missing_ok=True)
                except OSError:
                    pass
                pbar.set_postfix({"regenerating": chunk_file.name})

        # Load only the columns we need.
        # We always load `new_session` for session boundary detection.
        base_cols = [
            "agent_id",
            "visual_cue",
            "auditory_cue",
            "action",
            "new_session",
        ]
        if "reward" in feature_names:
            base_cols.append("reward")
        if "prev_action" in feature_names:
            base_cols.append("prev_action")
        if "prev_reward" in feature_names:
            base_cols.append("prev_reward")

        # Some datasets use `trial_idx`, others use `trial`. Optional `session_idx`
        # is loaded for chronological ordering (see _sort_agent_rows_chronological).
        schema_names = set(dataset.schema.names)
        if "trial_idx" in schema_names:
            trial_col = "trial_idx"
        elif "trial" in schema_names:
            trial_col = "trial"
        else:
            raise ValueError("Parquet must contain column 'trial_idx' or 'trial'")
        needed_cols = ["agent_id", trial_col]
        if "session_idx" in schema_names:
            needed_cols.append("session_idx")
        needed_cols.extend([c for c in base_cols if c != "agent_id"])
        table = dataset.to_table(
            filter=ds.field("agent_id").isin(chunk_agent_ids),
            columns=needed_cols,
        )
        chunk_data = table.to_pandas()

        chunk_features_np = []
        for agent_id in chunk_agent_ids:
            agent_data = chunk_data[chunk_data["agent_id"] == agent_id]

            session_tensor = np.full(
                (max_sessions, max_trials_per_session, n_features),
                fill_value=mask_value,
                dtype=np.float32,
            )

            if len(agent_data) == 0:
                chunk_features_np.append(
                    finalize_observation_tensor(session_tensor, mask_value=float(mask_value))
                )
                continue

            agent_data = _sort_agent_rows_chronological(agent_data)

            new_session_flags = agent_data["new_session"].values.astype(np.float32)
            start_positions = np.where(new_session_flags == 1.0)[0].astype(int)
            if len(start_positions) == 0:
                start_positions = np.array([0], dtype=int)
            elif start_positions[0] != 0:
                start_positions = np.concatenate([np.array([0], dtype=int), start_positions])

            # Iterate sessions and fill up to n_sessions
            n_real_sessions = min(len(start_positions), max_sessions)
            for sess_i in range(n_real_sessions):
                start = int(start_positions[sess_i])
                end = int(start_positions[sess_i + 1]) if sess_i + 1 < len(start_positions) else len(agent_data)

                sess_trials = agent_data.iloc[start:end]
                if len(sess_trials) == 0:
                    continue

                n_real_trials = min(len(sess_trials), max_trials_per_session)
                sess_trials = sess_trials.iloc[:n_real_trials]

                # Trial features (NOT including new_session indicator)
                trial_feat_parts = []
                for feat in feature_names:
                    if feat == "visual":
                        visual = sess_trials["visual_cue"].values.astype(np.int64)
                        trial_feat_parts.append(_one_hot_np(visual, 2))
                    elif feat == "auditory":
                        auditory = sess_trials["auditory_cue"].values.astype(np.int64)
                        trial_feat_parts.append(_one_hot_np(auditory, 2))
                    elif feat == "prev_visual":
                        prev_visual = sess_trials["visual_cue"].shift(1).values
                        nan_mask = np.isnan(prev_visual)
                        prev_visual_clean = np.where(nan_mask, 0, prev_visual).astype(np.int64)
                        prev_visual_oh = _one_hot_np(prev_visual_clean, 2)
                        if nan_mask.any():
                            prev_visual_oh[nan_mask, :] = 0.0
                        trial_feat_parts.append(prev_visual_oh)
                    elif feat == "prev_auditory":
                        prev_auditory = sess_trials["auditory_cue"].shift(1).values
                        nan_mask = np.isnan(prev_auditory)
                        prev_auditory_clean = np.where(nan_mask, 0, prev_auditory).astype(np.int64)
                        prev_auditory_oh = _one_hot_np(prev_auditory_clean, 2)
                        if nan_mask.any():
                            prev_auditory_oh[nan_mask, :] = 0.0
                        trial_feat_parts.append(prev_auditory_oh)
                    elif feat == "action":
                        actions = sess_trials["action"].values.astype(np.int64)
                        trial_feat_parts.append(_one_hot_np(actions, 2))
                    elif feat == "reward":
                        rewards = sess_trials["reward"].values.astype(np.float32).reshape(-1, 1)
                        trial_feat_parts.append(rewards)
                    elif feat == "new_session":
                        ns = sess_trials["new_session"].values.astype(np.float32).reshape(-1, 1)
                        trial_feat_parts.append(ns)
                    elif feat == "prev_action":
                        prev_actions = sess_trials["prev_action"].values
                        nan_mask = np.isnan(prev_actions)
                        prev_actions_clean = np.where(nan_mask, 0, prev_actions).astype(np.int64)
                        prev_action_oh = _one_hot_np(prev_actions_clean, 2)
                        if nan_mask.any():
                            prev_action_oh[nan_mask, :] = 0.0
                        trial_feat_parts.append(prev_action_oh)
                    elif feat == "prev_reward":
                        prev_rewards = sess_trials["prev_reward"].values.astype(np.float32)
                        prev_rewards_clean = np.where(np.isnan(prev_rewards), 0.0, prev_rewards).astype(np.float32).reshape(-1, 1)
                        trial_feat_parts.append(prev_rewards_clean)
                    else:
                        raise ValueError(f"Unhandled feature name: {feat}")

                trial_feat = np.concatenate(trial_feat_parts, axis=1).astype(np.float32)
                session_tensor[sess_i, :n_real_trials, :] = trial_feat

            chunk_features_np.append(
                finalize_observation_tensor(session_tensor, mask_value=float(mask_value))
            )

        chunk_features_np = np.stack(chunk_features_np, axis=0)

        if save_dir is not None:
            # chunk_file is guaranteed to be set above when save_dir is not None
            np.save(chunk_file, chunk_features_np)
            pbar.set_postfix({"saved": chunk_file.name, "shape": tuple(chunk_features_np.shape)})
            del chunk_features_np
        else:
            all_sessions_features.extend(chunk_features_np)

    if save_dir is not None:
        print(f"All modelID session features saved to {save_dir}")
        return None, max_sessions, save_dir

    if len(all_sessions_features) == 0:
        return tf.zeros((0, max_sessions, max_trials_per_session, n_features), dtype=tf.float32), max_sessions, None

    features_array = np.stack(all_sessions_features, axis=0).astype(np.float32)
    return tf.constant(features_array, dtype=tf.float32), max_sessions, None


def load_features_and_labels(
    save_dir: Path,
    split: str = 'train'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from disk.
    
    Parameters:
    -----------
    save_dir : Path
        Directory containing saved files
    split : str, optional
        Which split to load: 'train', 'val', 'test', or 'all' (default: 'train')
    
    Returns:
    --------
    features : np.ndarray
        Features array with shape (n_agents, max_trials, n_features)
    labels : np.ndarray
        Labels array with shape (n_agents, n_models)
    """
    save_dir = Path(save_dir)
    
    features_path = save_dir / f"features_{split}.npy"
    labels_path = save_dir / f"labels_{split}.npy"
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Files not found: {features_path} or {labels_path}. "
            f"Available splits: {[f.stem.replace('features_', '') for f in save_dir.glob('features_*.npy')]}"
        )
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Loaded {split} set: {features.shape[0]} agents")
    
    return features, labels


def _encode_stratify_labels(strings: Sequence[str]) -> np.ndarray:
    """Map string stratum labels to contiguous int codes for sklearn stratify."""
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder()
    return np.asarray(enc.fit_transform(list(strings)), dtype=np.int64)


def _stratify_plan(
    metadata_spec: Union[
        Dict[int, Dict[str, Any]],
        Sequence[Tuple[Dict[int, Dict[str, Any]], str]],
    ],
    agent_ids: Optional[Sequence[int]],
) -> List[Tuple[Dict[int, Dict[str, Any]], str, List[int]]]:
    """
    Build rows ``(metadata, namespace, agent_ids_in_order)``.

    ``namespace`` is ``""`` for a single metadata dict (stratum string = variant ``key`` only).
    For stacked ``(meta, base_name)`` blocks it is ``base_name`` (stratum = ``"{namespace}||{variant}"``).
    """
    if isinstance(metadata_spec, dict):
        meta = metadata_spec
        ids = list(agent_ids) if agent_ids is not None else sorted(meta.keys())
        return [(meta, "", ids)]

    if agent_ids is not None:
        raise ValueError(
            "agent_ids is only supported when metadata_spec is a single dict; "
            "use None for stacked (concatenated) metadata blocks."
        )

    plan: List[Tuple[Dict[int, Dict[str, Any]], str, List[int]]] = []
    for block in metadata_spec:
        if not (isinstance(block, tuple) and len(block) == 2):
            raise TypeError(
                "Stacked metadata_spec must be a sequence of (all_metadata, base_model_name) tuples, "
                f"got entry: {type(block).__name__!r}"
            )
        meta, base_name = block
        plan.append((meta, str(base_name), sorted(meta.keys())))
    return plan


def stratify_indices_from_metadata(
    metadata_spec: Union[
        Dict[int, Dict[str, Any]],
        Sequence[Tuple[Dict[int, Dict[str, Any]], str]],
    ],
    *,
    agent_ids: Optional[Sequence[int]] = None,
    key: str = "model_id",
) -> np.ndarray:
    """
    Integer class labels per agent for stratified train/val/test (e.g. by simulator variant).

    **Single cohort:** pass ``metadata_spec`` as one ``all_metadata`` dict (agent_id → row).
    The stratum for each agent is ``str(metadata[key])`` (default ``key='model_id'``).
    Agent order is ``agent_ids`` if given, otherwise ``sorted(all_metadata.keys())``, matching
    :func:`extract_and_curate_labels` / chunk iteration order.

    **Stacked cohorts:** pass a sequence of ``(all_metadata, base_model_name)`` in the **same**
    order as ``labels_list`` / ``chunks_dirs_list`` in :func:`write_train_val_test_from_chunks`.
    Each stratum is ``"{base_model}||{metadata[agent][key]}"`` so the split balances both
    generative family and variant. Rare strata (fewer than two samples) fall back to random
    splits inside :func:`compute_stratified_train_val_test_split_indices`.

    Parameters
    ----------
    metadata_spec
        Either a single metadata dict or a list/tuple of ``(metadata_dict, base_model_name)``.
    agent_ids
        Supported only for the **single-dict** case; must be ``None`` when passing stacked blocks.
    key
        Metadata field for the variant id (default ``model_id``).
    """
    missing = f"__missing_{key}__"
    parts: List[str] = []
    for meta, namespace, ordered_ids in _stratify_plan(metadata_spec, agent_ids):
        for aid in ordered_ids:
            mid = str(meta[aid].get(key, missing))
            parts.append(f"{namespace}||{mid}" if namespace else mid)
    return _encode_stratify_labels(parts)


def load_split_indices_from_pickle(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load ``split_indices`` from ``split_info.pkl`` (see :func:`save_split_info_pickle`).

    Row ``i`` of ``features_train.npy`` / ``labels_train_dir`` matches global agent id
    ``split_indices['train'][i]`` when features were written with that split.
    """
    import pickle

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split info not found: {path}")
    with open(path, "rb") as f:
        info = pickle.load(f)
    si = info.get("split_indices")
    if not isinstance(si, dict):
        raise KeyError(
            f"{path} must contain dict key 'split_indices'; got {type(si).__name__!r}"
        )
    out: Dict[str, np.ndarray] = {}
    for k, v in si.items():
        out[str(k)] = np.asarray(v, dtype=np.int64)
    return out


def stratify_labels_for_agent_rows(
    agent_ids_per_row: np.ndarray,
    metadata_by_agent: Mapping[int, Any],
    *,
    key: str = "model_id",
    missing: str = "__missing_model_id__",
) -> np.ndarray:
    """
    Integer stratum per **row** (same length as ``agent_ids_per_row``) for sklearn-style stratify.

    ``agent_ids_per_row[i]`` is the global simulation agent id for row ``i`` of a feature/label
    matrix (e.g. ``split_indices['train']`` from :func:`load_split_indices_from_pickle`).
    """
    agent_ids_per_row = np.asarray(agent_ids_per_row, dtype=np.int64).reshape(-1)
    parts: List[str] = []
    for aid in agent_ids_per_row.tolist():
        row = metadata_by_agent.get(int(aid))
        if not isinstance(row, dict):
            parts.append(missing)
            continue
        parts.append(str(row.get(key, missing)))
    return _encode_stratify_labels(parts)


def sample_stratified_row_indices(
    stratify_labels: np.ndarray,
    subset_size: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """
    Choose ``subset_size`` row indices in ``[0, n_rows)`` approximately stratified by
    ``stratify_labels`` (same convention as :func:`compute_stratified_train_val_test_split_indices`).

    Uses :class:`sklearn.model_selection.StratifiedShuffleSplit` when each stratum has at
    least two samples; otherwise falls back to uniform random **without** stratification.
    """
    stratify_labels = np.asarray(stratify_labels).reshape(-1)
    n = int(stratify_labels.shape[0])
    k = int(subset_size)
    if k <= 0:
        raise ValueError("subset_size must be positive")
    if k > n:
        raise ValueError(f"subset_size={k} exceeds n_rows={n}")
    if k == n:
        return np.arange(n, dtype=np.int64)

    from sklearn.model_selection import StratifiedShuffleSplit

    X_dummy = np.zeros((n, 1), dtype=np.float32)
    test_size = n - k
    if _every_stratum_has_at_least_two(stratify_labels):
        try:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=k,
                test_size=test_size,
                random_state=int(random_state),
            )
            train_idx, _ = next(sss.split(X_dummy, stratify_labels))
            return np.asarray(train_idx, dtype=np.int64)
        except ValueError:
            pass

    rng = np.random.RandomState(int(random_state))
    return rng.choice(n, size=k, replace=False)


def _every_stratum_has_at_least_two(y: np.ndarray) -> bool:
    """Return True iff sklearn stratify= y is valid (each class appears at least twice)."""
    y = np.asarray(y)
    if y.size < 2:
        return False
    _, counts = np.unique(y, return_counts=True)
    return bool(np.min(counts) >= 2)


def _train_test_split_stratify_fallback(
    *arrays: Any,
    test_size: float,
    random_state: int,
    stratify_source: np.ndarray,
) -> Tuple[Any, ...]:
    """
    ``train_test_split`` with ``stratify=stratify_source`` when each class has ≥2 samples;
    otherwise unstratified. Retries without stratify if sklearn still raises (e.g. rare edge cases).
    """
    from sklearn.model_selection import train_test_split

    use_strat = stratify_source if _every_stratum_has_at_least_two(stratify_source) else None
    try:
        return cast(
            Tuple[Any, ...],
            train_test_split(
                *arrays,
                test_size=test_size,
                random_state=random_state,
                stratify=use_strat,
            ),
        )
    except ValueError as e:
        if use_strat is None:
            raise
        print(f"  Warning: stratify failed ({e!s}); retrying split without stratify.")
        return cast(
            Tuple[Any, ...],
            train_test_split(
                *arrays,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            ),
        )


def compute_stratified_train_val_test_split_indices(
    n_total: int,
    stratify_source: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_source_name: str = "stratify_source",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Global train/val/test index arrays (0 .. n_total-1) with the same stratified
    fallback behavior as :func:`write_train_val_test_from_chunks`.

    Intended workflow: compute split once → :func:`save_split_info_pickle` → stream
    features with :func:`save_features_from_chunks` → write labels with
    :func:`subset_labels_by_split_indices` and :func:`save_train_val_test_labels`.
    """
    stratify_source = np.asarray(stratify_source).reshape(-1)
    if stratify_source.shape[0] != n_total:
        raise ValueError(
            f"stratify_source must have length n_agents={n_total}, got {stratify_source.shape[0]}"
        )

    if verbose:
        if _every_stratum_has_at_least_two(stratify_source):
            print(
                f"  Stratifying first split by {stratify_source_name} "
                f"({len(np.unique(stratify_source)):,} classes)."
            )
        else:
            print(
                f"  Note: cannot stratify first split ({stratify_source_name}) "
                "— using random split for train+val vs test."
            )

    agent_indices = np.arange(n_total)
    idx_temp, idx_test, strat_temp, _ = _train_test_split_stratify_fallback(
        agent_indices,
        stratify_source,
        test_size=test_size,
        random_state=random_state,
        stratify_source=stratify_source,
    )

    val_size_adjusted = val_size / (1 - test_size)
    strat_temp_arr = np.asarray(strat_temp)
    if verbose:
        if _every_stratum_has_at_least_two(strat_temp_arr):
            print("  Stratifying train vs val (using first-split strata).")
        else:
            print(
                "  Note: cannot stratify second split — using random split for train vs val."
            )

    idx_train, idx_val, _, _ = _train_test_split_stratify_fallback(
        idx_temp,
        strat_temp_arr,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify_source=strat_temp_arr,
    )

    split_indices: Dict[str, np.ndarray] = {
        "train": np.asarray(idx_train, dtype=np.int64),
        "val": np.asarray(idx_val, dtype=np.int64),
        "test": np.asarray(idx_test, dtype=np.int64),
    }
    if verbose:
        n_tr = len(split_indices["train"])
        n_va = len(split_indices["val"])
        n_te = len(split_indices["test"])
        print(f"  Split computed:")
        print(f"    Train: {n_tr:,} agents ({100 * n_tr / n_total:.1f}%)")
        print(f"    Val:   {n_va:,} agents ({100 * n_va / n_total:.1f}%)")
        print(f"    Test:  {n_te:,} agents ({100 * n_te / n_total:.1f}%)")

    return split_indices


def save_features_from_chunks(
    chunks_dirs: Union[Path, List[Path]],
    n_agents_per_dataset: List[int],
    save_dir: Path,
    *,
    split_indices: Optional[Dict[str, np.ndarray]] = None,
    split_train_val_test: bool = True,
    cleanup_chunks: bool = True,
    quiet: bool = False,
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, np.ndarray]]:
    """
    Stream ``features_chunk_*.npy`` into train/val/test (or a single ``features_all``)
    **without** reading or writing label arrays.

    ``split_indices`` must contain ``train``, ``val``, and ``test`` when
    ``split_train_val_test`` is True (e.g. from
    :func:`compute_stratified_train_val_test_split_indices` or
    :func:`random_train_val_test_indices`). Agent order must match
    ``n_agents_per_dataset`` (concatenation order of chunk directories).

    quiet
        If True, omit the decorative banner lines so callers (e.g.
        :func:`write_train_val_test_from_chunks`) can print one combined log.
    """
    if not isinstance(chunks_dirs, list):
        chunks_dirs = [chunks_dirs]
    if len(chunks_dirs) != len(n_agents_per_dataset):
        raise ValueError(
            f"len(chunks_dirs)={len(chunks_dirs)} must match "
            f"len(n_agents_per_dataset)={len(n_agents_per_dataset)}"
        )

    n_total = int(sum(n_agents_per_dataset))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 70)
        print("STREAMING SAVE (features only): Processing chunks incrementally")
        print("=" * 70)

    saved_paths: Dict[str, Dict[str, Path]] = {}
    out_split_indices: Dict[str, np.ndarray]

    train_pos: Dict[int, int] = {}
    val_pos: Dict[int, int] = {}
    test_pos: Dict[int, int] = {}
    X_train: Optional[np.ndarray] = None
    X_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    features_all: Optional[np.ndarray] = None

    if split_train_val_test:
        if split_indices is None:
            raise ValueError("split_indices is required when split_train_val_test=True")
        for k in ("train", "val", "test"):
            if k not in split_indices:
                raise ValueError(f"split_indices must contain '{k}'")
        out_split_indices = {
            "train": np.asarray(split_indices["train"], dtype=np.int64),
            "val": np.asarray(split_indices["val"], dtype=np.int64),
            "test": np.asarray(split_indices["test"], dtype=np.int64),
        }

        print("\nStep 1: Preparing feature memmaps (split mode)...")
        idx_train = out_split_indices["train"]
        idx_val = out_split_indices["val"]
        idx_test = out_split_indices["test"]

        first_chunks_dir = Path(chunks_dirs[0])
        first_chunk_file = sorted(first_chunks_dir.glob("features_chunk_*.npy"))[0]
        # Use mmap_mode to avoid loading a full chunk into RAM (chunks can be hundreds of MB).
        first_chunk = np.load(first_chunk_file, mmap_mode="r")
        feature_shape = first_chunk.shape[1:]
        feature_dtype = first_chunk.dtype
        print(f"  Feature shape per agent: {feature_shape}")
        del first_chunk

        train_pos = {int(idx): pos for pos, idx in enumerate(idx_train)}
        val_pos = {int(idx): pos for pos, idx in enumerate(idx_val)}
        test_pos = {int(idx): pos for pos, idx in enumerate(idx_test)}

        X_train = np.lib.format.open_memmap(
            save_dir / "features_train.npy",
            mode="w+",
            dtype=feature_dtype,
            shape=(len(idx_train),) + feature_shape,
        )
        X_val = np.lib.format.open_memmap(
            save_dir / "features_val.npy",
            mode="w+",
            dtype=feature_dtype,
            shape=(len(idx_val),) + feature_shape,
        )
        X_test = np.lib.format.open_memmap(
            save_dir / "features_test.npy",
            mode="w+",
            dtype=feature_dtype,
            shape=(len(idx_test),) + feature_shape,
        )
    else:
        out_split_indices = {"all": np.arange(n_total, dtype=np.int64)}
        print("\nStep 1: Preparing feature memmap (all agents)...")
        first_chunks_dir = Path(chunks_dirs[0])
        first_chunk_file = sorted(first_chunks_dir.glob("features_chunk_*.npy"))[0]
        first_chunk = np.load(first_chunk_file, mmap_mode="r")
        feature_shape = first_chunk.shape[1:]
        feature_dtype = first_chunk.dtype
        print(f"  Feature shape per agent: {feature_shape}")
        del first_chunk
        features_all = np.lib.format.open_memmap(
            save_dir / "features_all.npy",
            mode="w+",
            dtype=feature_dtype,
            shape=(n_total,) + feature_shape,
        )

    if split_train_val_test:
        if X_train is None or X_val is None or X_test is None:
            raise RuntimeError("internal error: split memmaps not initialized")
        Xt_m: np.ndarray = X_train
        Xv_m: np.ndarray = X_val
        Xte_m: np.ndarray = X_test
        fall_m = np.empty(0)  # unused; satisfies static analysis in non-split inner branch
    else:
        if features_all is None:
            raise RuntimeError("internal error: features_all not initialized")
        fall_m = features_all
        Xt_m = Xv_m = Xte_m = np.empty(0)  # unused when split_train_val_test is False

    print("\nStep 2: Streaming through feature chunks...")
    global_idx = 0

    for dataset_idx, chunks_dir in enumerate(chunks_dirs):
        chunks_dir = Path(chunks_dir)
        chunk_files = sorted(chunks_dir.glob("features_chunk_*.npy"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

        n_agents_this_dataset = n_agents_per_dataset[dataset_idx]
        print(f"\n  Dataset {dataset_idx + 1}/{len(chunks_dirs)}: {chunks_dir.name}")
        print(f"    Agents: {n_agents_this_dataset:,}")
        print(f"    Chunks: {len(chunk_files)}")

        for chunk_file in chunk_files:
            # Memory-map chunk files: a single 500-agent chunk can exceed 200MB.
            # Indexing reads only the requested slice, keeping peak RAM low.
            chunk_features = np.load(chunk_file, mmap_mode="r")
            n_agents_chunk = len(chunk_features)

            for i in range(n_agents_chunk):
                agent_global_idx = global_idx + i
                if split_train_val_test:
                    if agent_global_idx in train_pos:
                        Xt_m[train_pos[agent_global_idx]] = chunk_features[i]
                    elif agent_global_idx in val_pos:
                        Xv_m[val_pos[agent_global_idx]] = chunk_features[i]
                    elif agent_global_idx in test_pos:
                        Xte_m[test_pos[agent_global_idx]] = chunk_features[i]
                else:
                    fall_m[agent_global_idx] = chunk_features[i]

            global_idx += n_agents_chunk
            del chunk_features

        print(f"    ✓ Processed {n_agents_this_dataset:,} agents from dataset {dataset_idx + 1}")

    if global_idx != n_total:
        raise ValueError(
            f"Agent count mismatch: streamed {global_idx} rows, expected n_total={n_total} "
            f"from n_agents_per_dataset={n_agents_per_dataset}"
        )

    print("\nStep 3: Finalizing feature files...")
    if split_train_val_test:
        saved_paths["train"] = {"features": save_dir / "features_train.npy"}
        saved_paths["val"] = {"features": save_dir / "features_val.npy"}
        saved_paths["test"] = {"features": save_dir / "features_test.npy"}
        print(f"    ✓ Train: {Xt_m.shape[0]:,} agents")
        print(f"    ✓ Val:   {Xv_m.shape[0]:,} agents")
        print(f"    ✓ Test:  {Xte_m.shape[0]:,} agents")
        del Xt_m, Xv_m, Xte_m
    else:
        saved_paths["all"] = {"features": save_dir / "features_all.npy"}
        print(f"    ✓ All: {fall_m.shape[0]:,} agents")
        del fall_m

    if not quiet:
        print("\n" + "=" * 70)
        print("✅ FEATURE STREAMING COMPLETE")
        print("=" * 70)

    if cleanup_chunks:
        print("\nStep 4: Cleaning up chunk files...")
        n_removed_files = 0
        n_removed_dirs = 0
        for cdir in chunks_dirs:
            cdir = Path(cdir)
            for chunk_file in sorted(cdir.glob("features_chunk_*.npy")):
                chunk_file.unlink(missing_ok=True)
                n_removed_files += 1
            try:
                cdir.rmdir()
                n_removed_dirs += 1
            except OSError:
                pass
        print(f"  ✓ Removed {n_removed_files} chunk files")
        print(f"  ✓ Removed {n_removed_dirs} empty chunk directories")

    return saved_paths, out_split_indices


def per_dataset_agent_counts(
    labels_list: List[Union[np.ndarray, Dict[str, np.ndarray]]],
) -> List[int]:
    """Number of agents per dataset block (same order as ``labels_list`` / chunk directories)."""
    counts: List[int] = []
    for lab in labels_list:
        if isinstance(lab, dict):
            counts.append(int(next(iter(lab.values())).shape[0]))
        else:
            counts.append(int(lab.shape[0]))
    return counts


def write_train_val_test_from_chunks(
    chunks_dirs: Union[Path, List[Path]],
    labels_list: Union[
        np.ndarray,
        Dict[str, np.ndarray],
        List[Union[np.ndarray, Dict[str, np.ndarray]]],
    ],
    save_dir: Path,
    *,
    split_train_val_test: bool = True,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    cleanup_chunks: bool = True,
    stratify_by: Optional[np.ndarray] = None,
    cleanup_spool_dirs: Optional[Sequence[Union[str, Path]]] = None,
    label_dict_format: str = "npz",
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, np.ndarray]]:
    """
    Stream feature chunks to train/val/test (or a single pool), write matching label
    files, and save ``split_info.pkl`` when using a three-way split.

    Composes :func:`concatenate_per_dataset_labels`,
    :func:`compute_stratified_train_val_test_split_indices` (or an ``{'all': ...}``
    index pool), :func:`save_features_from_chunks`,     :func:`subset_labels_by_split_indices`,
    :func:`save_train_val_test_labels`, and :func:`save_split_info_pickle`.

    Label **chunk spool** from :func:`extract_and_curate_labels` (``spool_curated_chunks_to``)
    is removed there after merging when ``remove_spool_dir_after=True`` (default), which happens
    **before** this function runs. To delete spool only **after** train/val/test label files
    exist, use ``remove_spool_dir_after=False`` in extraction and pass the same path(s) as
    ``cleanup_spool_dirs`` here.

    For multi-head label dicts, pass ``stratify_by`` (there is no single ``argmax(one_hot)``
    across heads). When ``split_train_val_test`` is False and labels are dicts, writes
    ``labels_all.npz`` instead of ``labels_all.npy``.
    """
    if isinstance(chunks_dirs, list):
        chunks_list: List[Path] = [Path(d) for d in chunks_dirs]
    else:
        chunks_list = [Path(chunks_dirs)]

    if isinstance(labels_list, np.ndarray):
        labels_norm: List[Union[np.ndarray, Dict[str, np.ndarray]]] = [labels_list]
    elif isinstance(labels_list, dict):
        labels_norm = [labels_list]
    elif isinstance(labels_list, list):
        labels_norm = labels_list
    else:
        raise TypeError(
            "labels_list must be an ndarray, a label dict, or a list of ndarrays/dicts"
        )

    if len(chunks_list) != len(labels_norm):
        raise ValueError(
            f"Number of chunk directories ({len(chunks_list)}) must match "
            f"number of label blocks ({len(labels_norm)})"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STREAMING SAVE: chunks → features + labels (memory-safe)")
    print("=" * 70)

    print("\nStep 1: Combining labels and computing split indices...")
    all_labels = concatenate_per_dataset_labels(labels_norm)
    n_per = per_dataset_agent_counts(labels_norm)
    if isinstance(all_labels, dict):
        n_total = int(next(iter(all_labels.values())).shape[0])
        print(f"  Total agents: {n_total:,}")
        print(f"  Combined labels: multi-head dict keys {list(all_labels.keys())}")
    else:
        n_total = int(all_labels.shape[0])
        print(f"  Total agents: {n_total:,}")
        print(f"  Combined labels shape: {all_labels.shape}")

    if sum(n_per) != n_total:
        raise ValueError(
            f"Sum of per-dataset label rows ({sum(n_per)}) != concatenated n_total ({n_total})"
        )

    stratify_source_name = "argmax(one_hot_labels)"

    if split_train_val_test:
        if stratify_by is not None:
            stratify_source = np.asarray(stratify_by).reshape(-1)
            if stratify_source.shape[0] != n_total:
                raise ValueError(
                    f"stratify_by must have length n_agents={n_total}, got {stratify_source.shape[0]}"
                )
            stratify_source_name = "stratify_by (e.g. model variant)"
        else:
            if isinstance(all_labels, dict):
                raise ValueError(
                    "stratify_by is required when labels are a multi-head dict "
                    "(no single argmax(one_hot) across heads)"
                )
            stratify_source = np.argmax(all_labels, axis=1)

        split_indices = compute_stratified_train_val_test_split_indices(
            n_total,
            stratify_source,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify_source_name=stratify_source_name,
            verbose=True,
        )

        feat_saved, _ = save_features_from_chunks(
            chunks_list,
            n_per,
            save_dir,
            split_indices=split_indices,
            split_train_val_test=True,
            cleanup_chunks=cleanup_chunks,
            quiet=True,
        )

        print("\nStep 3: Finalizing and saving label files...")
        if isinstance(all_labels, dict) and label_dict_format == "dir":
            label_paths = _save_labels_dir_streaming(all_labels, split_indices, save_dir)
        else:
            labels_by_split = subset_labels_by_split_indices(all_labels, split_indices)
            label_paths = save_train_val_test_labels(
                labels_by_split, save_dir, dict_format=label_dict_format
            )
        for split_name in ("train", "val", "test"):
            n_s = len(split_indices[split_name])
            print(f"    ✓ {split_name}: {n_s:,} agents (labels + features)")
        saved_paths: Dict[str, Dict[str, Path]] = {
            k: {"features": feat_saved[k]["features"], "labels": label_paths[k]}
            for k in ("train", "val", "test")
        }

        save_split_info_pickle(
            save_dir,
            split_indices,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            datasets=[str(d) for d in chunks_list],
            extra={
                "stratify": stratify_source_name,
                "stratify_by_passed": stratify_by is not None,
            },
        )
        print("  ✓ Saved split info")

    else:
        split_indices = {"all": np.arange(n_total, dtype=np.int64)}
        feat_saved, _ = save_features_from_chunks(
            chunks_list,
            n_per,
            save_dir,
            split_train_val_test=False,
            cleanup_chunks=cleanup_chunks,
            quiet=True,
        )
        print("\nStep 3: Finalizing and saving label files...")
        if isinstance(all_labels, dict):
            labels_all_path = save_dir / "labels_all.npz"
            np.savez_compressed(labels_all_path, **cast(Dict[str, Any], all_labels))
        else:
            labels_all_path = save_dir / "labels_all.npy"
            np.save(labels_all_path, all_labels)
        saved_paths = {
            "all": {
                "features": feat_saved["all"]["features"],
                "labels": labels_all_path,
            }
        }
        print(f"    ✓ All: {n_total:,} agents (labels + features)")

    if cleanup_spool_dirs:
        for raw in cleanup_spool_dirs:
            sp = Path(raw)
            if sp.exists():
                shutil.rmtree(sp, ignore_errors=True)
                print(f"  ✓ Removed spool directory {sp}")

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)

    return saved_paths, split_indices


def save_features_and_labels(
    chunks_dirs: Union[Path, List[Path]],
    labels_list: Union[
        np.ndarray,
        Dict[str, np.ndarray],
        List[Union[np.ndarray, Dict[str, np.ndarray]]],
    ],
    save_dir: Path,
    split_train_val_test: bool = True,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    cleanup_chunks: bool = True,
    stratify_by: Optional[np.ndarray] = None,
    cleanup_spool_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, np.ndarray]]:
    """Deprecated: use :func:`write_train_val_test_from_chunks`."""
    warnings.warn(
        "save_features_and_labels() is deprecated; use write_train_val_test_from_chunks()",
        DeprecationWarning,
        stacklevel=2,
    )
    return write_train_val_test_from_chunks(
        chunks_dirs,
        labels_list,
        save_dir,
        split_train_val_test=split_train_val_test,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        cleanup_chunks=cleanup_chunks,
        stratify_by=stratify_by,
        cleanup_spool_dirs=cleanup_spool_dirs,
    )


def concatenate_per_dataset_labels(
    labels_list: List[Union[np.ndarray, Dict[str, np.ndarray]]],
    *,
    memmap_dir: Optional[Path] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Stack labels from multiple datasets in list order (e.g. multiple shards of one model).

    For dict outputs (multi-head), every dict must share the same keys; arrays are
    concatenated along axis 0.

    memmap_dir
        If set, write concatenated outputs as ``.npy`` memory-mapped files under this
        directory (numeric dtypes only) and return read-only memmaps. Use when stacking
        large trial-heavy heads to avoid a full in-RAM ``np.concatenate``.
    """
    if not labels_list:
        raise ValueError("labels_list is empty")
    first = labels_list[0]
    if isinstance(first, dict):
        keys_ref = set(first.keys())
        dicts: List[Dict[str, np.ndarray]] = [first]
        for lab in labels_list[1:]:
            if not isinstance(lab, dict) or set(lab.keys()) != keys_ref:
                raise ValueError("All label dicts must have the same keys")
            dicts.append(lab)
        keys = list(first.keys())
        if memmap_dir is None:
            return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}

        memmap_dir = Path(memmap_dir)
        if memmap_dir.exists():
            shutil.rmtree(memmap_dir, ignore_errors=True)
        memmap_dir.mkdir(parents=True, exist_ok=True)
        out_dict: Dict[str, np.ndarray] = {}
        for k in keys:
            dt = dicts[0][k].dtype
            tail = dicts[0][k].shape[1:]
            total_n = sum(int(d[k].shape[0]) for d in dicts)
            if not _dtype_allows_memmap(dt):
                out_dict[k] = np.concatenate([d[k] for d in dicts], axis=0)
                continue
            out_path = memmap_dir / f"concat_{_head_name_for_memmap_file(k)}.npy"
            mm = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dt, shape=(total_n,) + tail
            )
            off = 0
            for d in dicts:
                block = d[k]
                n = int(block.shape[0])
                mm[off : off + n] = np.asarray(block)
                off += n
                del block
            del mm
            gc.collect()
            out_dict[k] = np.load(out_path, mmap_mode="r", allow_pickle=True)
        return out_dict

    arrays: List[np.ndarray] = [first]
    for lab in labels_list[1:]:
        if not isinstance(lab, np.ndarray):
            raise ValueError("Cannot mix ndarray and dict label entries in labels_list")
        arrays.append(lab)
    if memmap_dir is None:
        return np.concatenate(arrays, axis=0)

    memmap_dir = Path(memmap_dir)
    if memmap_dir.exists():
        shutil.rmtree(memmap_dir, ignore_errors=True)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    dt = arrays[0].dtype
    tail = arrays[0].shape[1:]
    total_n = sum(int(a.shape[0]) for a in arrays)
    if not _dtype_allows_memmap(dt):
        return np.concatenate(arrays, axis=0)
    out_path = memmap_dir / "concat_labels.npy"
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=dt, shape=(total_n,) + tail)
    off = 0
    for a in arrays:
        n = int(a.shape[0])
        mm[off : off + n] = a
        off += n
    del mm
    gc.collect()
    return np.load(out_path, mmap_mode="r", allow_pickle=True)


def subset_labels_by_split_indices(
    labels_full: Union[np.ndarray, Dict[str, np.ndarray]],
    split_indices: Dict[str, np.ndarray],
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Subset concatenated labels using split_indices from split_info.pkl
    (keys 'train', 'val', 'test' -> index arrays into the global agent order).
    """
    for k in ("train", "val", "test"):
        if k not in split_indices:
            raise ValueError(f"split_indices must contain '{k}'")
    out: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}
    if isinstance(labels_full, dict):
        for split_name in ("train", "val", "test"):
            idx = split_indices[split_name]
            out[split_name] = {head: arr[idx] for head, arr in labels_full.items()}
    else:
        for split_name in ("train", "val", "test"):
            idx = split_indices[split_name]
            out[split_name] = labels_full[idx]
    return out


def _save_labels_dir_streaming(
    labels_full: Dict[str, np.ndarray],
    split_indices: Dict[str, np.ndarray],
    save_dir: Path,
    *,
    agent_block: int = 1024,
) -> Dict[str, Path]:
    """
    Memory-safe label saver for multi-head dict labels.

    Writes:
      save_dir/labels_{train,val,test}_dir/<HEAD>.npy

    Avoids large intermediate arrays from fancy indexing (e.g. arr[idx]).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for k in ("train", "val", "test"):
        if k not in split_indices:
            raise ValueError(f"split_indices must contain '{k}'")

    any_head = next(iter(labels_full.keys()))
    n_total = int(labels_full[any_head].shape[0])

    maps: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        idx = np.asarray(split_indices[split], dtype=np.int64)
        m = np.full(n_total, -1, dtype=np.int64)
        m[idx] = np.arange(idx.shape[0], dtype=np.int64)
        maps[split] = m

    out_dirs: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        out_dir = save_dir / f"labels_{split}_dir"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_FORMAT.txt").write_text("per-key .npy labels directory\n")
        out_dirs[split] = out_dir

    for head, arr in labels_full.items():
        if int(arr.shape[0]) != n_total:
            raise ValueError(
                f"Head {head!r} has n={int(arr.shape[0])}, expected n_total={n_total}"
            )
        tail = arr.shape[1:]
        dt = arr.dtype

        out_paths: Dict[str, Path] = {}
        for split in ("train", "val", "test"):
            out_path = out_dirs[split] / f"{head}.npy"
            out_paths[split] = out_path
            if out_path.exists():
                continue
            n_split = int(len(split_indices[split]))
            mm = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dt, shape=(n_split,) + tail
            )
            del mm

        for start in range(0, n_total, agent_block):
            end = min(start + agent_block, n_total)
            block = np.asarray(arr[start:end])
            if block.shape[0] == 0:
                continue
            for split in ("train", "val", "test"):
                m = maps[split][start:end]
                keep = m >= 0
                if not np.any(keep):
                    continue
                mm = np.lib.format.open_memmap(out_paths[split], mode="r+")
                mm[m[keep]] = block[keep]
                del mm
            del block
        gc.collect()

    return {s: out_dirs[s] for s in ("train", "val", "test")}


def save_train_val_test_labels(
    labels_by_split: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    save_dir: Path,
    *,
    dict_format: str = "npz",
) -> Dict[str, Path]:
    """
    Write labels_train / labels_val / labels_test under save_dir.

    ndarray  -> labels_{split}.npy
    dict     -> ``dict_format="npz"``: labels_{split}.npz (one array per output head)
            -> ``dict_format="dir"``: labels_{split}_dir/KEY.npy (one file per head; memmap-friendly)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        lab = labels_by_split[split]
        if isinstance(lab, dict):
            if dict_format == "npz":
                path = save_dir / f"labels_{split}.npz"
                np.savez_compressed(path, **cast(Dict[str, Any], lab))
            elif dict_format == "dir":
                out_dir = save_dir / f"labels_{split}_dir"
                out_dir.mkdir(parents=True, exist_ok=True)
                # marker for humans / debugging (write early so partial dirs are recognizable)
                (out_dir / "_FORMAT.txt").write_text("per-key .npy labels directory\n")
                for k, v in lab.items():
                    out_path = out_dir / f"{k}.npy"
                    # Resume-safe: if a prior run crashed mid-write, keep the already-saved keys.
                    # (Useful for Colab + Drive transient disconnects.)
                    if out_path.exists():
                        continue
                    np.save(out_path, v)
                path = out_dir
            else:
                raise ValueError(f"dict_format must be 'npz' or 'dir', got {dict_format!r}")
        else:
            path = save_dir / f"labels_{split}.npy"
            np.save(path, lab)
        paths[split] = path
    return paths


def random_train_val_test_indices(
    n_agents: int,
    *,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Train/val/test index arrays into a single contiguous agent ordering (0..n_agents-1).

    Same relative sizes as :func:`write_train_val_test_from_chunks` (test_size, val_size of remainder),
    but **without** stratification — suitable for one generative family (e.g. ruleQhyp-only).
    """
    from sklearn.model_selection import train_test_split

    if n_agents < 3:
        raise ValueError(f"Need at least 3 agents for a 3-way split, got {n_agents}")
    agent_indices = np.arange(n_agents, dtype=np.int64)
    idx_temp, idx_test = train_test_split(
        agent_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    val_size_adjusted = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True,
    )
    return {
        "train": np.asarray(idx_train, dtype=np.int64),
        "val": np.asarray(idx_val, dtype=np.int64),
        "test": np.asarray(idx_test, dtype=np.int64),
    }


def save_split_info_pickle(
    save_dir: Path,
    split_indices: Dict[str, np.ndarray],
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    datasets: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``split_info.pkl`` compatible with identification-style loaders (includes ``split_indices``)."""
    import pickle

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "n_train": len(split_indices["train"]),
        "n_val": len(split_indices["val"]),
        "n_test": len(split_indices["test"]),
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "n_datasets": len(datasets) if datasets is not None else 1,
        "datasets": datasets if datasets is not None else [],
        "split_indices": split_indices,
    }
    if extra:
        info.update(extra)
    path = save_dir / "split_info.pkl"
    with open(path, "wb") as f:
        pickle.dump(info, f)
    return path


