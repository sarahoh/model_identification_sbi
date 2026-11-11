"""
Convert ``all_rats.pkl`` into a padded NumPy feature array for the model-ID RNN.

Output
------
``X`` with shape ``(N_subjects, max_sessions, max_trials, 11)``.
Defaults come from ``simulation.constants`` / ``helpers.observation_grid``
(``MAX_SESSIONS - 1``, ``MAX_TRIALS_PER_SESSION``, pad ``-1.0``).

Input pickle (from ``convert_real_matfiles_to_pkl_csv.py``)
-----------------------------------------------------------
Top-level keys are subject ids. Each value looks like::

    {"_layout": "rats", "sessions": [session_0, session_1, ...]}

Trial vectors live under each session's ``"data"`` key
(override with ``--session-data-key``).

Features (11 columns)
---------------------
Same columns as simulated model-ID features after one-hot expansion:

    new_session,
    visual_0, visual_1,
    auditory_0, auditory_1,
    action_0, action_1,
    reward,
    prev_action_0, prev_action_1,
    prev_reward

(``visual`` / ``action`` / etc. are binary in the raw data; each becomes two
one-hot columns.)

NaNs are replaced before saving so padding matches what the RNN expects.
Unlike simulation (train/val/test splits), this writes **one** array over all
real subjects; each subject row matches a simulation agent’s layout.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.observation_grid import (  # noqa: E402
    DEFAULT_OBSERVATION_MASK_VALUE,
    finalize_observation_tensor,
)
from simulation.constants import MAX_SESSIONS, MAX_TRIALS_PER_SESSION  # noqa: E402

# Use same defaults as simulation feature extractor.
DEFAULT_MAX_SESSIONS = MAX_SESSIONS - 1
DEFAULT_MAX_TRIALS = MAX_TRIALS_PER_SESSION
DEFAULT_MASK_VALUE = float(DEFAULT_OBSERVATION_MASK_VALUE)
DEFAULT_INPUT_PKL = PROJECT_ROOT / "storage/real_data/pkl/all_rats.pkl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "storage/model_identification/real_features_model_identification"

# Column names for axis 3 of X (must stay in this order).
DEFAULT_FEATURE_NAMES = [
    "new_session",
    "visual_0",
    "visual_1",
    "auditory_0",
    "auditory_1",
    "action_0",
    "action_1",
    "reward",
    "prev_action_0",
    "prev_action_1",
    "prev_reward",
]


def _binary_one_hot(x: np.ndarray, *, name: str) -> np.ndarray:
    """Convert binary-coded vector (0/1 or 1/2) to one-hot shape (n, 2)."""
    v = np.squeeze(np.asarray(x))
    if v.ndim != 1:
        raise ValueError(f"Feature '{name}' must be 1D for one-hot conversion, got {v.shape}.")
    if np.issubdtype(v.dtype, np.floating):
        v = np.where(np.isnan(v), 0.0, v)
    v = v.astype(np.int64)
    u = np.unique(v)
    if np.all(np.isin(u, [0, 1])):
        idx = v
    elif np.all(np.isin(u, [1, 2])):
        idx = v - 1
    else:
        raise ValueError(
            f"Feature '{name}' must be binary coded as 0/1 or 1/2. Unique values: {u.tolist()}"
        )
    out = np.zeros((idx.shape[0], 2), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _to_1d_numeric(x: Any, *, name: str) -> np.ndarray:
    """Convert arbitrary array-like input into a 1D float32 vector."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError(f"Feature '{name}' is scalar; expected per-trial vector.")
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"Feature '{name}' must be 1D after squeeze; got shape {arr.shape}."
        )
    return arr.astype(np.float32, copy=False)


def _extract_trial_feature_matrix(
    session: Dict[str, Any],
    feature_names: Sequence[str],
    *,
    session_data_key: str = "data",
) -> np.ndarray:
    """
    Extract (n_trials, n_features) matrix for one session.
    """
    if session_data_key not in session:
        raise KeyError(f"Session missing '{session_data_key}' (set --session-data-key).")
    source = session[session_data_key]

    if not isinstance(source, dict):
        raise TypeError(f"Session '{session_data_key}' must be a dict, got {type(source).__name__}.")

    if list(feature_names) != DEFAULT_FEATURE_NAMES:
        raise ValueError(
            f"--feature-names must be the default 11-col layout, got {list(feature_names)!r}."
        )

    required = ["visual_cue", "auditory_cue", "action", "reward"]
    missing = [k for k in required if k not in source]
    if missing:
        raise KeyError(f"Missing raw keys {missing} under '{session_data_key}'.")

    visual_oh = _binary_one_hot(source["visual_cue"], name="visual_cue")
    auditory_oh = _binary_one_hot(source["auditory_cue"], name="auditory_cue")
    action_oh = _binary_one_hot(source["action"], name="action")
    reward = _to_1d_numeric(source["reward"], name="reward").reshape(-1, 1)
    n_trials = reward.shape[0]

    lengths = {
        "visual_cue": visual_oh.shape[0],
        "auditory_cue": auditory_oh.shape[0],
        "action": action_oh.shape[0],
        "reward": n_trials,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Trial length mismatch: {lengths}.")


    new_session = np.zeros((n_trials, 1), dtype=np.float32)
    if n_trials > 0:
        new_session[0, 0] = 1.0

    prev_action_oh = np.zeros_like(action_oh, dtype=np.float32)
    if n_trials > 1:
        prev_action_oh[1:] = action_oh[:-1]

    prev_reward = np.zeros((n_trials, 1), dtype=np.float32)
    if n_trials > 1:
        prev_reward[1:, 0] = reward[:-1, 0]

    # Same block order as get_modelID_features_from_dataframe(..., feature_names=[...]).
    return np.concatenate(
        [
            new_session,
            visual_oh,
            auditory_oh,
            action_oh,
            reward,
            prev_action_oh,
            prev_reward,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _truncate_trials(trial_mat: np.ndarray, max_trials: int, mode: str) -> np.ndarray:
    """Truncate trials to max_trials, either from the start or end."""
    n_trials = trial_mat.shape[0]
    if n_trials <= max_trials:
        return trial_mat
    if mode == "keep_start":
        return trial_mat[:max_trials]
    if mode == "keep_end":
        return trial_mat[-max_trials:]
    raise ValueError(f"Unknown trial truncation mode: {mode}")


def _recompute_default_derived_columns(trial_mat: np.ndarray) -> np.ndarray:
    """
    For DEFAULT_FEATURE_NAMES, recompute columns that are defined relative to the start
    of the *kept* window: new_session, prev_action_*, prev_reward.
    """
    if trial_mat.ndim != 2 or trial_mat.shape[1] != 11:
        raise ValueError(f"Expected (T,11) matrix for default schema, got {trial_mat.shape}.")
    if trial_mat.shape[0] == 0:
        return trial_mat
    out = np.asarray(trial_mat, dtype=np.float32).copy()
    n_trials = out.shape[0]
    # 0 new_session, 5:7 action one-hot, 7 reward, 8:10 prev_action one-hot, 10 prev_reward
    out[:, 0] = 0.0
    out[0, 0] = 1.0
    out[:, 8:10] = 0.0
    out[:, 10] = 0.0
    if n_trials > 1:
        out[1:, 8:10] = out[:-1, 5:7]
        out[1:, 10] = out[:-1, 7]
    return out


def _load_subject_blob(pkl_path: Path) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict):
        raise TypeError(f"Top-level pickle object must be dict, got {type(blob).__name__}")
    return blob


def _subject_ids(blob: Dict[str, Any]) -> List[str]:
    # Only keep subjects that look like expected entries.
    ids = []
    for sid, payload in blob.items():
        if isinstance(payload, dict) and "sessions" in payload:
            ids.append(str(sid))
    return sorted(ids)


def convert(
    pkl_path: Path,
    out_dir: Path,
    *,
    feature_names: Sequence[str],
    max_sessions: int,
    max_trials_per_session: int,
    mask_value: float,
    session_data_key: str,
    trial_truncation: str,
    session_truncation: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Main conversion routine."""
    blob = _load_subject_blob(pkl_path)
    subject_ids = _subject_ids(blob)
    if not subject_ids:
        raise ValueError("No subject entries with a 'sessions' list were found in pickle.")

    n_subjects = len(subject_ids)
    n_features = len(feature_names)

    X = np.full(
        (n_subjects, max_sessions, max_trials_per_session, n_features),
        fill_value=np.float32(mask_value),
        dtype=np.float32,
    )
    n_sessions_kept = np.zeros((n_subjects,), dtype=np.int32)
    n_trials_kept = np.zeros((n_subjects, max_sessions), dtype=np.int32)

    for i, sid in enumerate(subject_ids):
        sessions = blob[sid]["sessions"]
        if not isinstance(sessions, list):
            raise TypeError(f"{sid}: 'sessions' must be list, got {type(sessions).__name__}")

        # Session truncation for subjects with > max_sessions sessions.
        if len(sessions) > max_sessions:
            if session_truncation == "keep_start":
                sessions = sessions[:max_sessions]
            elif session_truncation == "keep_end":
                sessions = sessions[-max_sessions:]
            else:
                raise ValueError(f"Unknown session truncation mode: {session_truncation}")

        n_sessions_kept[i] = len(sessions)

        for s_idx, sess in enumerate(sessions):
            trial_mat = _extract_trial_feature_matrix(
                sess,
                feature_names,
                session_data_key=session_data_key,
            )  # (trials, features)

            trial_mat = _truncate_trials(
                trial_mat,
                max_trials=max_trials_per_session,
                mode=trial_truncation,
            )
            if list(feature_names) == DEFAULT_FEATURE_NAMES:
                trial_mat = _recompute_default_derived_columns(trial_mat)
            n_t = trial_mat.shape[0]

            # Fail if canonical new_session is violated.
            if list(feature_names) == DEFAULT_FEATURE_NAMES and n_t > 0:
                if trial_mat[0, 0] != 1.0:
                    raise ValueError(
                        f"{sid} session_index={s_idx}: expected new_session[0]=1.0, got {float(trial_mat[0,0])}. "
                        "This indicates upstream data schema drift; inspect the session dict under "
                        f"key '{session_data_key}'."
                    )
                if n_t > 1 and not np.all(trial_mat[1:, 0] == 0.0):
                    bad = np.where(trial_mat[1:, 0] != 0.0)[0]
                    bad_t = int(bad[0] + 1) if bad.size else -1
                    raise ValueError(
                        f"{sid} session_index={s_idx}: expected new_session[t]=0 for t>=1, "
                        f"but found new_session[{bad_t}]={float(trial_mat[bad_t,0])}."
                    )

            n_trials_kept[i, s_idx] = n_t
            X[i, s_idx, :n_t, :] = trial_mat

    X = finalize_observation_tensor(X, mask_value=float(mask_value))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Same (N, S, T, F) layout as get_modelID_features_from_dataframe chunk arrays; filenames differ from train/val/test.
    np.save(out_dir / "features_real.npy", X)
    np.save(out_dir / "n_sessions_real.npy", n_sessions_kept)
    np.save(out_dir / "n_trials_real.npy", n_trials_kept)
    np.save(out_dir / "subject_ids_real.npy", np.array(subject_ids, dtype=object), allow_pickle=True)

    manifest = {
        "input_pkl": str(pkl_path),
        "output_dir": str(out_dir),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "mask_value": float(mask_value),
        "feature_names": list(feature_names),
        "max_sessions": int(max_sessions),
        "max_trials_per_session": int(max_trials_per_session),
        "session_data_key": session_data_key,
        "trial_truncation": trial_truncation,
        "session_truncation": session_truncation,
        "n_subjects": n_subjects,
        "min_sessions_kept": int(n_sessions_kept.min()),
        "max_sessions_kept": int(n_sessions_kept.max()),
        "min_trials_kept_nonzero": int(n_trials_kept[n_trials_kept > 0].min()) if np.any(n_trials_kept > 0) else 0,
        "max_trials_kept": int(n_trials_kept.max()) if n_trials_kept.size > 0 else 0,
    }
    with open(out_dir / "manifest_real_npy.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return X, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all_rats.pkl to masked NumPy arrays (N,35,300,F)."
    )
    parser.add_argument(
        "--input-pkl",
        type=Path,
        default=DEFAULT_INPUT_PKL,
        help="Path to all_rats.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--feature-names",
        nargs="+",
        default=DEFAULT_FEATURE_NAMES,
        help="Manifest/output labels for the 11-d layout (default). Dict session data requires "
        "this exact list; use a 2D ndarray under --session-data-key for other widths.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help="Max sessions per subject.",
    )
    parser.add_argument(
        "--max-trials-per-session",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help="Max trials per session.",
    )
    parser.add_argument(
        "--mask-value",
        type=float,
        default=DEFAULT_MASK_VALUE,
        help="Mask fill value for missing sessions/trials.",
    )
    parser.add_argument(
        "--session-data-key",
        type=str,
        default="data",
        help="Session dict key containing trial-level data.",
    )
    parser.add_argument(
        "--trial-truncation",
        choices=["keep_start", "keep_end"],
        default="keep_end",
        help="If session has > max_trials, keep first or last trials.",
    )
    parser.add_argument(
        "--session-truncation",
        choices=["keep_start", "keep_end"],
        default="keep_start",
        help="If subject has > max_sessions, keep first or last sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_pkl = args.input_pkl.resolve()
    out_dir = args.out_dir.resolve()

    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pickle not found: {input_pkl}")

    X, manifest = convert(
        pkl_path=input_pkl,
        out_dir=out_dir,
        feature_names=args.feature_names,
        max_sessions=args.max_sessions,
        max_trials_per_session=args.max_trials_per_session,
        mask_value=args.mask_value,
        session_data_key=args.session_data_key,
        trial_truncation=args.trial_truncation,
        session_truncation=args.session_truncation,
    )

    print("=" * 70)
    print("Conversion complete")
    print("=" * 70)
    print(f"features_real.npy shape: {X.shape}, dtype={X.dtype}")
    print(f"Saved to: {out_dir}")
    print(f"Feature names ({len(manifest['feature_names'])}): {manifest['feature_names']}")
    print(f"Manifest: {out_dir / 'manifest_real_npy.json'}")


if __name__ == "__main__":
    main()

