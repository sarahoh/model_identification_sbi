"""
Convert MATLAB .mat files (real data) into pickle files.
Converts **rat** data only (human data will use a separate pipeline later).

Rats: ``multi_rule_rats/single_sessions`` — group by subject; filenames like
``Strat100_DARK_D1_final.mat``; all rats merged into one dict (subject id → sessions).

Default output root: ``storage/real_data/pkl`` (repo root) with:

- ``pkl/all_rats.pkl`` — dict keyed by rat subject id → ``{"_layout": "rats", "sessions": [...]}``
- ``pkl/manifest_rats.json`` (summary + subject list)

Path option args ``--matfiles_base`` / ``--out_root`` / ``--rats_root`` resolved from the repository root.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Rat single-session filenames: {Subject}_{COND}_D{day}_final.mat
_RAT_SINGLE_SESSION_RE = re.compile(
    r"^(?P<subject>.+)_(?P<cond>[A-Z0-9]+)_D(?P<day>\d+)_final\.mat$",
    re.IGNORECASE,
)

# Default CLI paths are relative to repo root (parent of ``scripts/``), not ``os.getcwd()`` —
# the debugger may set cwd to another folder (e.g. a venv workspace).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _is_mat_metadata_key(key: str) -> bool:
    return key.startswith("__") or key in {"__header__", "__version__", "__globals__"}


def _mat_to_python(obj: Any) -> Any:
    """
    Conversion of loaded MATLAB objects into basic Python types.
    """
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            # Keep numpy arrays as-is (pickle will preserve them).
            return obj
        return obj
    except Exception:
        return obj


# MATLAB v7.3 internal HDF5 groups (metadata fields).
_HDF5_MAT_SKIP_TOP_KEYS = frozenset({"#refs#", "#subsystem#"})


def load_mat_file(mat_path: Path) -> Dict[str, Any]:
    """
    Load MATLAB v7.3 ``.mat`` (HDF5) via h5py.
    Older ``-v7``/``-v6`` binary MAT-files are not supported.
    """
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "Loading v7.3 .mat files requires h5py. Install with: pip install h5py"
        ) from e

    def read_group(g: "h5py.Group") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in g.keys():
            obj = g[key]
            if isinstance(obj, h5py.Dataset):
                out[key] = np.array(obj[...])
            elif isinstance(obj, h5py.Group):
                out[key] = read_group(obj)
            else:
                out[key] = repr(obj)
        return out

    with h5py.File(mat_path, "r") as f:
        out: Dict[str, Any] = {}
        for k in f.keys():
            if k in _HDF5_MAT_SKIP_TOP_KEYS:
                continue
            obj = f[k]
            if isinstance(obj, h5py.Group):
                out[k] = read_group(obj)
            elif isinstance(obj, h5py.Dataset):
                out[k] = np.array(obj[...])
            else:
                out[k] = repr(obj)
        return {key: val for key, val in out.items() if not _is_mat_metadata_key(key)}


def discover_rat_single_sessions(
    single_sessions_dir: Path,
) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Group .mat files by rat subject id; each value is list of (day, path) sorted by day.
    """
    if not single_sessions_dir.is_dir():
        raise FileNotFoundError(f"Rat single_sessions dir not found: {single_sessions_dir}")

    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for p in sorted(single_sessions_dir.glob("*.mat")):
        m = _RAT_SINGLE_SESSION_RE.match(p.name)
        if not m:
            continue
        subject = m.group("subject")
        day = int(m.group("day"))
        groups.setdefault(subject, []).append((day, p))

    for subj in groups:
        groups[subj].sort(key=lambda t: t[0])
    return groups


def _build_subject_entry(
    layout: str,
    mat_dicts: List[Dict[str, Any]],
    meta_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Payload for one subject under the merged pickle: ``sessions`` plus ``_layout``."""
    sessions: List[Dict[str, Any]] = []
    for i, (mat, meta) in enumerate(zip(mat_dicts, meta_list)):
        sess: Dict[str, Any] = dict(mat)
        if "data" in sess and not isinstance(sess["data"], dict):
            raise TypeError(
                "Expected MATLAB session 'data' to be a dict of raw vectors "
                f"(got {type(sess['data']).__name__})."
            )
        sess["session_index"] = i
        sess["meta"] = dict(meta)
        sessions.append(sess)
    return {
        "_layout": layout,
        "sessions": sessions,
    }


def run_export_rats(
    mat_root: Path,
    out_dir: Path,
    variables: Optional[set[str]],
) -> List[Dict[str, Any]]:
    """
    Build ``all_rats.pkl``: top-level keys = subject ids, values = ``_layout`` + ``sessions``.
    """
    merged: Dict[str, Any] = {}
    pkl_stem = "all_rats"

    groups = discover_rat_single_sessions(mat_root)
    if not groups:
        raise FileNotFoundError(
            f"No rat single-session .mat files matched *_COND_D#_final.mat under {mat_root}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rats: {len(groups)} subjects under {mat_root} → {out_dir / (pkl_stem + '.pkl')}")

    for subject, day_paths in sorted(groups.items()):
        paths_in_order = [p for _, p in day_paths]
        meta_list: List[Dict[str, Any]] = []
        for day, p in day_paths:
            m = _RAT_SINGLE_SESSION_RE.match(p.name)
            cond = m.group("cond") if m else "?"
            meta_list.append({"path": str(p), "day": int(day), "condition": cond})

        print(
            f"  {subject}: {len(paths_in_order)} sessions "
            f"D{day_paths[0][0]}..D{day_paths[-1][0]}"
        )

        mat_dicts = [load_mat_file(p) for p in paths_in_order]
        if variables is not None:
            mat_dicts = [{k: v for k, v in seg.items() if k in variables} for seg in mat_dicts]

        merged[subject] = _build_subject_entry("rats", mat_dicts, meta_list)

    pkl_path = export_to_pkl(merged, out_dir, stem=pkl_stem)
    print(f"Wrote merged pickle ({len(merged)} subjects): {pkl_path}")

    manifest = [
        {
            "layout": "rats",
            "out_dir": str(out_dir),
            "pkl": str(pkl_path),
            "n_subjects": len(merged),
            "subject_ids": sorted(merged.keys()),
            "sessions_per_subject": {k: len(merged[k]["sessions"]) for k in sorted(merged.keys())},
        }
    ]
    return manifest


def export_to_pkl(
    mat_dict: Dict[str, Any],
    out_dir: Path,
    stem: str,
) -> Path:
    import pickle

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{stem}.pkl"

    # Recursively applies ``_mat_to_python`` to values; keeps nested dict / array structure.
    serializable = {k: _mat_to_python(v) for k, v in mat_dict.items()}
    with open(pkl_path, "wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rat .mat data (single_sessions, by day) to all_rats.pkl. "
            "(Human pipeline is separate; not implemented in this script.)"
        )
    )
    parser.add_argument(
        "--matfiles_base",
        type=Path,
        default=Path("storage/real_data/matfiles"),
        help="Base folder (relative to repo root unless absolute) with multi_rule_rats/.",
    )
    parser.add_argument(
        "--rats_root",
        type=Path,
        default=None,
        help="Rat single_sessions dir (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("storage/real_data/pkl"),
        help="Output folder (relative to repo root unless absolute); writes all_rats.pkl and manifest_rats.json.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of MATLAB top-level variable names to keep in each session dict.",
    )

    args = parser.parse_args()

    matfiles_base: Path = _resolve_repo_path(args.matfiles_base)
    rats_root = (
        _resolve_repo_path(args.rats_root)
        if args.rats_root is not None
        else (matfiles_base / "multi_rule_rats" / "single_sessions")
    )

    out_root: Path = _resolve_repo_path(args.out_root)
    variables = set(args.variables) if args.variables else None

    out_root.mkdir(parents=True, exist_ok=True)

    if not rats_root.exists():
        raise FileNotFoundError(f"Rat data path not found: {rats_root}")

    print("=" * 70)
    print(f"Rats  |  mat_root: {rats_root}")
    print(f"        out_dir: {out_root}")
    print("=" * 70)

    manifest = run_export_rats(
        mat_root=rats_root,
        out_dir=out_root,
        variables=variables,
    )
    manifest_path = out_root / "manifest_rats.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Wrote manifest to: {manifest_path}\n")


if __name__ == "__main__":
    main()


