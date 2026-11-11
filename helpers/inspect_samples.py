"""
Small utilities for inspecting saved feature/label tensors.

These helpers are meant for notebooks: they print a few agents/sessions/trials to quickly
sanity-check that padding/masking and ordering look sensible.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def build_index_plan(
    N: int,
    S: int,
    T: int,
    *,
    n_agents: int = 3,
    seed: int = 0,
) -> Dict[int, Dict[int, list[int]]]:
    """
    Build a deterministic "inspection plan" of indices:
    - choose `n_agents` unique agents
    - for each agent choose sessions [0, 1, random] (de-duped)
    - for each session choose trials [0, 1, random] (de-duped)
    """
    rng = np.random.default_rng(seed)
    agents = rng.choice(int(N), size=min(int(n_agents), int(N)), replace=False)

    plan: Dict[int, Dict[int, list[int]]] = {}
    for ag in agents:
        sess_base: list[int] = [0] if S > 0 else []
        if S > 1:
            sess_base.append(1)
        if S > 0:
            sess_base.append(int(rng.integers(0, int(S))))

        seen_s: set[int] = set()
        sess: list[int] = []
        for s in sess_base:
            if s not in seen_s:
                sess.append(int(s))
                seen_s.add(int(s))

        plan[int(ag)] = {}
        for s in sess:
            trial_base: list[int] = [0] if T > 0 else []
            if T > 1:
                trial_base.append(1)
            if T > 0:
                trial_base.append(int(rng.integers(0, int(T))))

            seen_t: set[int] = set()
            trials: list[int] = []
            for t in trial_base:
                if t not in seen_t:
                    trials.append(int(t))
                    seen_t.add(int(t))

            plan[int(ag)][int(s)] = trials

    return plan


def _format_value(v: Any, *, max_vector_elems: int = 12) -> Any:
    if np.ndim(v) == 0:
        return v
    flat = np.asarray(v).ravel()
    shown = flat[:max_vector_elems]
    if flat.size > max_vector_elems:
        shown = np.concatenate([shown, np.array(["..."], dtype=object)])
    return shown


def print_feature_samples(
    name: str,
    features: np.ndarray,
    plan: Mapping[int, Mapping[int, list[int]]],
    *,
    max_vector_elems: int = 12,
) -> None:
    """
    Print a few feature vectors from an array shaped (N, S, T, F).
    """
    a = np.asarray(features)
    print(f"\n[{name}]")
    print(f"  shape: {a.shape} dtype: {a.dtype}")

    if a.ndim != 4:
        raise ValueError(f"features must be 4D (N,S,T,F), got shape {a.shape}")

    for ag, sess_map in plan.items():
        print(f"  agent {ag}")
        for s, trials in sess_map.items():
            print(f"    session {s}")
            for t in trials:
                v = a[int(ag), int(s), int(t)]
                print(f"      trial {t}: {_format_value(v, max_vector_elems=max_vector_elems)}")


def print_label_samples(
    name: str,
    labels: np.ndarray,
    plan: Mapping[int, Mapping[int, list[int]]],
    *,
    max_vector_elems: int = 12,
) -> None:
    """
    Print a few label values. Supports:
    - (N,) and (N,P): printed once per agent
    - (N,S,T,...)   : printed per session/trial according to `plan`
    """
    a = np.asarray(labels)
    print(f"\n[{name}]")
    print(f"  shape: {a.shape} dtype: {a.dtype}")

    for ag, sess_map in plan.items():
        print(f"  agent {ag}")

        if a.ndim in (1, 2):
            v = a[int(ag)]
            shown = _format_value(v, max_vector_elems=max_vector_elems)
            if np.ndim(v) == 0:
                print(f"    value: {shown}")
            else:
                flat = np.asarray(v).ravel()
                print(f"    value: {shown[0] if flat.size == 1 else shown}")
            continue

        if a.ndim < 3:
            raise ValueError(f"labels must be 1D/2D or >=3D, got shape {a.shape}")

        for s, trials in sess_map.items():
            print(f"    session {s}")
            for t in trials:
                v = a[int(ag), int(s), int(t)]
                print(f"      trial {t}: {_format_value(v, max_vector_elems=max_vector_elems)}")

