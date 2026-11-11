"""
Label setup for ``extract_and_curate_labels`` and training-time parameter scaling.

**Config (no data):** ``build_output_head_config`` only builds dicts telling the extractor
what to read from metadata / parquet. Import and use:

    from helpers.label_setup import build_output_head_config

**Parameter scaling:** Ground truth from the extractor should stay in **simulator units**.
Per-parameter z-scoring (``StandardScaler``, fit on train) uses the functions below. 
Apply **after** train/val split.

Then:

    labels = extract_and_curate_labels(..., output_head_config=output_head_config, ...)

Notes
-----
- Presence heads use dotted metadata keys like ``variant_settings.defaults.<KEY>``.
  ``variant_settings`` is the ``variant_info`` dict from ``generate_variants_from_tree``.
- Presence names match feature-tree keys: ``LR``, ``Ex`` (ruleQhyp only for ``Ex``).
- **Reserved** trial latents: ``attention_state``, ``exploit_mode``, ``chosen_rule`` (special config).
- **Q-value** trial columns: names starting with ``Q_`` (e.g. ``Q_LEFT``, ``Q_s0_a0``), matching
  simulator parquet columns. Other names in ``latent_labels`` are rejected (no “everything else is Q”).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from simulation.constants import N_ACTIONS, N_STATES, RULES

# ---------------------------------------------------------------------------
# output_head_config builder
# ---------------------------------------------------------------------------

_RESERVED_LATENT = frozenset({"attention_state", "exploit_mode", "chosen_rule", "next_action"})
_RULEQHYP_ONLY_LATENTS = frozenset({"exploit_mode", "chosen_rule"})

# Presence heads (variant feature toggles) are extracted from
# ``variant_settings.defaults.<NAME>`` where defaults store fixed values when a feature is disabled
# and ``None`` when the corresponding parameter is free / feature enabled.
#
# Keep these aligned with your model-tree keys whenever possible (e.g. "Ft", "Fb", "E", "Ba", ...).
_RULEQHYP_PRESENCE_DEFAULTS = frozenset(
    {
        "An",
        "E",
        "Ft",
        "Fb",
        "Ws",   # defaults.Ws is None when Ws is free, 0.0 otherwise
        "Sr",   # defaults.Sr is None when Sr is free, 0.0 when disabled
        # LR and Ex are special-cased below
        # Ba, Sa, Sda: nested under E / Sa — use variant_settings + gating (below)
        # Sdr: nested under Sr — use variant_settings + gating (below)
        # Ls, Ws_Ls_linked, WsLs are special-cased below (custom transforms)
    }
)

# Heads that must not use a bare defaults.<key> lookup (key may be absent when parent is off).
_RULEQHYP_PRESENCE_NESTED = frozenset({"Ba", "Sa", "Sda", "Sdr"})


def _variant_defaults(vs: Any) -> dict:
    if isinstance(vs, dict):
        d = vs.get("defaults")
        if isinstance(d, dict):
            return d
    return {}


def _presence_e_child_free(vs: Any, child: str) -> float:
    """Ba / Sa: only applicable when epsilon (E) is enabled (``defaults.E is None``)."""
    d = _variant_defaults(vs)
    if d.get("E") is not None:
        return 0.0
    return float(d.get(child) is None)


def _presence_sda_free(vs: Any) -> float:
    """Sda free only when Sa is free and ``defaults.Sda`` is None (key may be absent if Sa off)."""
    d = _variant_defaults(vs)
    if d.get("E") is not None or d.get("Sa") is not None:
        return 0.0
    return float(d.get("Sda") is None)


def _presence_sdr_free(vs: Any) -> float:
    """Sdr free only when Sr is free and ``defaults.Sdr`` is None."""
    d = _variant_defaults(vs)
    if d.get("Sr") is not None:
        return 0.0
    return float(d.get("Sdr") is None)


def _parse_mode_from_description(
    description: Any,
    *,
    prefix: str,
    allowed: Sequence[str],
) -> str:
    """
    Parse a feature mode token from ``variant_settings.description``.

    Example description:
        "Features: addAn, ... WsLs_separate, Sr_disabled"

    Returns the suffix after ``prefix`` (e.g. "separate").
    """
    s = str(description)
    toks = [t.strip().strip(",") for t in s.split() if t.startswith(prefix)]
    if not toks:
        raise ValueError(f"Missing {prefix} token in variant_settings.description: {s!r}")
    mode = toks[0][len(prefix) :]
    if mode not in set(allowed):
        raise ValueError(
            f"Unexpected {prefix} mode {mode!r} (allowed {list(allowed)}). "
            f"Full description: {s!r}"
        )
    return mode


def _trial_column_head(column: str) -> Dict[str, Any]:
    """One trial-level column read from parquet (continuous or encoded)."""
    return {"label_column_trialdata": column, "return_one_hot": False}


def _is_q_value_column(name: str) -> bool:
    """True if ``name`` matches simulator Q-value column naming (``Q_...``)."""
    return name.startswith("Q_")


def _next_action_from_action_grid(action_grid: Any) -> np.ndarray:
    """
    Derive a (session, trial) ``next_action`` **binary** grid from an ``action`` grid.

    Input conventions:
    - ``action_grid`` is a padded session grid produced by ``_pad_agent_trial_columns_to_session_grid``.
    - Scalar action: shape (S, T) with float dtype + NaN padding (values typically 0/1).
    - One-hot action: shape (S, T, A) with float dtype + NaN padding.

    Output:
    - float32 grid shape (S, T) with values in {0.0, 1.0} where valid.
    - invalid entries set to NaN (so PyTorch can mask with isfinite + BCE-with-logits).
    """
    a = np.asarray(action_grid)
    if a.ndim not in (2, 3):
        raise ValueError(f"action_grid must be 2D (S,T) or 3D (S,T,A), got shape {a.shape}")

    if a.ndim == 3:
        trial_ok = np.isfinite(a).all(axis=-1)
        cls = np.nanargmax(a, axis=-1).astype(np.float32, copy=False)
    else:
        trial_ok = np.isfinite(a)
        # Actions are expected to be integer-coded (e.g. 0/1), but stored as float with NaNs.
        cls = a.astype(np.float32, copy=False)

    # Shift within each session: next_action[t] = action[t+1].
    next_ok = np.roll(trial_ok, shift=-1, axis=1)
    next_ok[:, -1] = False
    next_cls = np.roll(cls, shift=-1, axis=1)

    out = np.full(next_cls.shape, np.nan, dtype=np.float32)
    out[next_ok] = next_cls[next_ok].astype(np.float32, copy=False)
    return out

_DEFAULT_LATENT_RULEQHYP: tuple[str, ...] = (
    "attention_state",
    "exploit_mode",
    "chosen_rule",
) + tuple(f"Q_{r}" for r in RULES)

_DEFAULT_LATENT_QLEARN: tuple[str, ...] = ("attention_state",) + tuple(
    f"Q_s{s}_a{a}" for s in range(N_STATES) for a in range(N_ACTIONS)
)

# Public helper: use this when you want explicit control (e.g. defaults + extras).
def default_latent_labels(*, base_model: str) -> tuple[str, ...]:
    base_model = str(base_model)
    if base_model == "ruleQhyp":
        return _DEFAULT_LATENT_RULEQHYP
    if base_model == "Qlearn":
        return _DEFAULT_LATENT_QLEARN
    raise ValueError(f"Unsupported base_model: {base_model!r} (expected 'ruleQhyp' or 'Qlearn')")

# ``extract_params`` for ``params_all``; see ``data_prep.extract_and_curate_labels``.
ParamNamesSpec = Optional[Union[Literal["all", "free"], Sequence[str]]]


def _resolve_extract_params(param_names: ParamNamesSpec) -> Union[str, List[str], None]:
    """Map API argument to head ``extract_params`` (or None to omit ``params_all``)."""
    if param_names is None:
        return None
    if isinstance(param_names, str):
        if param_names not in ("all", "free"):
            raise ValueError(
                f"param_names string must be 'all' or 'free', got {param_names!r}. "
                "For a subset use a sequence, e.g. ('Ap', 'B')."
            )
        return param_names
    if len(param_names) == 0:
        return None
    out: List[str] = []
    for p in param_names:
        if not isinstance(p, str):
            raise TypeError(f"param_names entries must be str, got {type(p)}: {p!r}")
        out.append(p)
    return out


def build_output_head_config(
    *,
    base_model: str,
    param_names: ParamNamesSpec = "all",
    latent_labels: Optional[Sequence[str]] = None,
    presence_labels: Optional[Sequence[str]] = None,
    params_conditional_loss_on: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build an output_head_config dict for `extract_and_curate_labels`.

    Parameters
    ----------
    base_model
        "ruleQhyp" or "Qlearn".
    param_names
        Which parameters to put in the ``params_all`` head (passed through as ``extract_params``
        in ``extract_and_curate_labels``): ``"all"`` (default) = every key in each agent's
        ``params_dict``; ``"free"`` = fitted-only names from ``model_spec``; a non-empty
        sequence = that subset (names must exist per agent). ``None`` or ``()`` → no param head.
    latent_labels
        Trial-level targets. **Reserved:** ``attention_state``, ``exploit_mode``, ``chosen_rule``
        (ruleQhyp only for the last two). **Q columns:** any other name must start with ``Q_``
        (parquet column = head name). Unknown names raise. If ``None``, uses built-in defaults.
    presence_labels
        ``LR``, ``Ex`` (``Ex`` only added for ruleQhyp). If ``None``, defaults to
        ``("LR", "Ex")`` for ruleQhyp and ``("LR",)`` for Qlearn.
    params_conditional_loss_on
        Reserved for a future mapping from parameter name -> presence head name so ``L`` / ``R``
        (etc.) only contribute to loss when the corresponding feature is ON. **Not implemented:**
        a non-empty mapping raises ``NotImplementedError``. Pass ``None`` or ``{}`` until
        ``data_prep`` and ``paramRNN`` honor ``conditional_loss_on`` end-to-end.
    """
    base_model = str(base_model)
    if base_model not in ("ruleQhyp", "Qlearn"):
        raise ValueError(f"Unsupported base_model: {base_model!r} (expected 'ruleQhyp' or 'Qlearn')")

    head: Dict[str, Dict[str, Any]] = {}

    if latent_labels is None:
        latent_labels_t = _DEFAULT_LATENT_RULEQHYP if base_model == "ruleQhyp" else _DEFAULT_LATENT_QLEARN
    else:
        latent_labels_t = tuple(latent_labels)

    if presence_labels is None:
        presence_labels = ("LR", "Ex") if base_model == "ruleQhyp" else ("LR",)
    presence_labels = tuple(presence_labels)

    if base_model == "Qlearn":
        bad = set(latent_labels_t) & _RULEQHYP_ONLY_LATENTS
        if bad:
            raise ValueError(
                f"latent_labels {sorted(bad)} are ruleQhyp-only (base_model=Qlearn)."
            )

    unknown = tuple(
        x
        for x in latent_labels_t
        if x not in _RESERVED_LATENT and not _is_q_value_column(x)
    )
    if unknown:
        raise ValueError(
            "latent_labels must be reserved names "
            f"{sorted(_RESERVED_LATENT)} or Q-value columns (start with 'Q_'). "
            f"Unrecognized: {list(unknown)}"
        )
    qcols = tuple(x for x in latent_labels_t if x not in _RESERVED_LATENT and _is_q_value_column(x))

    if presence_labels:
        if "LR" in presence_labels:
            head["LR"] = {
                "label_key_metadata": "variant_settings.defaults.L",
                "return_one_hot": False,
                "label_transform_fn": lambda v: float(v is None),
            }

        if base_model == "ruleQhyp" and "Ex" in presence_labels:
            head["Ex"] = {
                "label_key_metadata": "variant_settings.defaults.Ex",
                "return_one_hot": False,
                "label_transform_fn": lambda v: float(v is None),
            }

        # Optional additional feature-presence heads for ruleQhyp.
        # Convention: 1.0 means "feature on / param free" (defaults value is None),
        # 0.0 means "feature off / param fixed" (defaults value is numeric).
        if base_model == "ruleQhyp":
            for name in presence_labels:
                if name in ("LR", "Ex"):
                    continue
                if name in _RULEQHYP_PRESENCE_NESTED:
                    continue
                if name not in _RULEQHYP_PRESENCE_DEFAULTS:
                    continue
                head[name] = {
                    "label_key_metadata": f"variant_settings.defaults.{name}",
                    "return_one_hot": False,
                    "label_transform_fn": lambda v: float(v is None),
                }

            if "Ba" in presence_labels:
                head["Ba"] = {
                    "label_key_metadata": "variant_settings",
                    "return_one_hot": False,
                    "label_transform_fn": lambda vs: _presence_e_child_free(vs, "Ba"),
                }
            if "Sa" in presence_labels:
                head["Sa"] = {
                    "label_key_metadata": "variant_settings",
                    "return_one_hot": False,
                    "label_transform_fn": lambda vs: _presence_e_child_free(vs, "Sa"),
                }
            if "Sda" in presence_labels:
                head["Sda"] = {
                    "label_key_metadata": "variant_settings",
                    "return_one_hot": False,
                    "label_transform_fn": _presence_sda_free,
                }
            if "Sdr" in presence_labels:
                head["Sdr"] = {
                    "label_key_metadata": "variant_settings",
                    "return_one_hot": False,
                    "label_transform_fn": _presence_sdr_free,
                }

            # WsLs hierarchical binary heads.
            # WsLs: present if any mode other than disabled (description contains WsLs_disabled iff off).
            if "WsLs" in presence_labels:
                head["WsLs"] = {
                    "label_key_metadata": "variant_settings.description",
                    "return_one_hot": False,
                    "label_transform_fn": lambda s: float("WsLs_disabled" not in str(s)),
                }
            # Ls: present if Ls is free (None) or linked to Ws ('Ws') — i.e. not fixed at 0.
            # NOTE: `v is not None` returns False when v=None, so the correct check is
            # `v is None or v == "Ws"` (covers both free and linked cases).
            if "Ls" in presence_labels:
                head["Ls"] = {
                    "label_key_metadata": "variant_settings.defaults.Ls",
                    "return_one_hot": False,
                    "label_transform_fn": lambda v: float(v is None or v == "Ws"),
                }
            # Ws_Ls_linked: Ls is constrained to equal Ws (Ls='Ws' in defaults).
            if "Ws_Ls_linked" in presence_labels:
                head["Ws_Ls_linked"] = {
                    "label_key_metadata": "variant_settings.defaults.Ls",
                    "return_one_hot": False,
                    "label_transform_fn": lambda v: float(v == "Ws"),
                }

    extract_params = _resolve_extract_params(param_names)
    if extract_params is not None:
        head["params_all"] = {
            "extract_params": extract_params,
            "return_one_hot": False,
            "return_param_names": True,
        }
        # TODO: wire conditional_loss_on through extract_and_curate_labels + paramRNN param loss.
        if params_conditional_loss_on is not None and len(params_conditional_loss_on) > 0:
            raise NotImplementedError(
                "params_conditional_loss_on / params_all['conditional_loss_on'] is not "
                "implemented yet (label extraction + paramRNN masking). "
                "Remove the argument or pass params_conditional_loss_on=None until supported."
            )

    if not latent_labels_t:
        return head

    if "attention_state" in latent_labels_t:
        head["attention_state"] = _trial_column_head("attention_state")

    if base_model == "ruleQhyp":
        if "exploit_mode" in latent_labels_t:
            head["exploit_mode"] = _trial_column_head("exploit_mode")
        if "chosen_rule" in latent_labels_t:
            head["chosen_rule"] = {
                **_trial_column_head("chosen_rule"),
                "category_names": list(RULES),
                "categorical_as_index": True,
            }

    if "next_action" in latent_labels_t:
        head["next_action"] = {
            **_trial_column_head("action"),
            # Transform per-agent padded action grid -> per-agent padded next_action grid.
            "label_transform_fn": _next_action_from_action_grid,
        }

    for col in qcols:
        head[col] = _trial_column_head(col)

    return head


# ---------------------------------------------------------------------------
# Per-parameter StandardScaler 
# ---------------------------------------------------------------------------

def fit_per_param_scalers(
    y: np.ndarray,
    param_names: Sequence[str],
) -> Dict[str, StandardScaler]:
    """Fit one StandardScaler per column. ``y`` shape ``(n_samples, n_params)``; names match columns."""
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got shape {y.shape}")
    if y.shape[1] != len(param_names):
        raise ValueError(f"y has {y.shape[1]} columns but {len(param_names)} param_names")
    scalers: Dict[str, StandardScaler] = {}
    for j, name in enumerate(param_names):
        s = StandardScaler()
        s.fit(y[:, j : j + 1])
        scalers[name] = s
    return scalers


def transform_per_param(
    y: np.ndarray,
    param_names: Sequence[str],
    scalers: Mapping[str, StandardScaler],
) -> np.ndarray:
    cols = [scalers[name].transform(y[:, j : j + 1]) for j, name in enumerate(param_names)]
    return np.concatenate(cols, axis=1)


def inverse_transform_per_param(
    y_std: np.ndarray,
    param_names: Sequence[str],
    scalers: Mapping[str, StandardScaler],
) -> np.ndarray:
    cols = [scalers[name].inverse_transform(y_std[:, j : j + 1]) for j, name in enumerate(param_names)]
    return np.concatenate(cols, axis=1)


def normalize_train_labels(
    name_to_labels: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, Dict[str, StandardScaler]]:
    """
    Sort names, then fit+transform each parameter column.
    """
    names = sorted(name_to_labels.keys())
    parts = []
    for n in names:
        a = np.asarray(name_to_labels[n], dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[1] != 1:
            raise ValueError(f"name_to_labels[{n!r}] must be (n_agent,) or (n_agent, 1), got {a.shape}")
        parts.append(a)
    y = np.concatenate(parts, axis=1)
    scalers = fit_per_param_scalers(y, names)
    y_std = transform_per_param(y, names, scalers)
    return y_std, scalers


def normalize_val_labels(
    name_to_labels: Mapping[str, np.ndarray],
    scalers: Mapping[str, StandardScaler],
) -> np.ndarray:
    """Transform with training scalers; names sorted like ``normalize_train_labels_tutorial``."""
    names = sorted(name_to_labels.keys())
    parts = []
    for n in names:
        a = np.asarray(name_to_labels[n], dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        parts.append(a)
    y = np.concatenate(parts, axis=1)
    return transform_per_param(y, names, scalers)
