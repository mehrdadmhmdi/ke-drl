"""Shared preprocessing helpers for the Expedia real-data example.

The raw Expedia tensors store low-cardinality variables as numeric codes.  For
kernel and linear models those codes should not be treated as ordered distances
unless that is intended, so this module fits a train-split one-hot encoder and
exports enough metadata to reproduce the exact state basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


DEFAULT_CATEGORICAL_STATE_COLS = (
    "srch_length_of_stay",
    "srch_room_count",
    "srch_saturday_night_bool",
    "random_bool",
    "prop_starrating",
    "comp_rate",
    "comp_inv",
)


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return [c.strip() for c in x.split(",") if c.strip()]


def _value_label(v: float) -> str:
    fv = float(v)
    if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
        return str(int(round(fv)))
    return f"{fv:.6g}".replace("-", "m").replace(".", "p")


def _is_integer_like(values: np.ndarray, tol: float = 1e-8) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.all(np.abs(finite - np.round(finite)) <= tol))


def resolve_categorical_state_cols(
    categorical_arg: Optional[str],
    state_names: Sequence[str],
    train_states: np.ndarray,
    max_auto_cardinality: int = 20,
) -> List[str]:
    """Resolve categorical state columns from "auto", "none", or a CSV list."""
    names = [str(c) for c in state_names]
    arg = "auto" if categorical_arg is None else str(categorical_arg).strip()
    if arg.lower() in {"", "none", "no", "false", "0"}:
        return []

    if arg.lower() != "auto":
        requested = parse_csv_list(arg) or []
        missing = [c for c in requested if c not in names]
        if missing:
            raise ValueError(
                f"Requested categorical state columns not found: {missing}. "
                f"Available state columns: {names}"
            )
        return requested

    x = np.asarray(train_states, dtype=np.float64)
    out: List[str] = []
    for j, name in enumerate(names):
        finite = x[:, j][np.isfinite(x[:, j])]
        if finite.size == 0:
            continue
        unique = np.unique(finite)
        hinted = name in DEFAULT_CATEGORICAL_STATE_COLS
        low_card_integer = unique.size <= int(max_auto_cardinality) and _is_integer_like(unique)
        if hinted and low_card_integer:
            out.append(name)
    return out


@dataclass
class StateEncoder:
    raw_state_names: List[str]
    categorical_state_names: List[str]
    numeric_state_names: List[str]
    categorical_levels: Dict[str, List[float]]
    encoded_state_names: List[str]
    one_hot: bool = True

    def transform(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"states must be 2D, got shape={x.shape}")
        if x.shape[1] != len(self.raw_state_names):
            raise ValueError(
                f"states has {x.shape[1]} columns but encoder expects "
                f"{len(self.raw_state_names)} raw columns"
            )

        cat = set(self.categorical_state_names)
        pieces: List[np.ndarray] = []
        for j, name in enumerate(self.raw_state_names):
            col = x[:, j]
            if self.one_hot and name in cat:
                levels = np.asarray(self.categorical_levels[name], dtype=np.float64)
                if levels.size == 0:
                    pieces.append(np.zeros((x.shape[0], 0), dtype=np.float64))
                    continue
                encoded = np.isclose(
                    col.reshape(-1, 1),
                    levels.reshape(1, -1),
                    rtol=0.0,
                    atol=1e-8,
                ).astype(np.float64)
                encoded[~np.isfinite(col), :] = 0.0
                pieces.append(encoded)
            else:
                pieces.append(col.reshape(-1, 1))

        if not pieces:
            return np.zeros((x.shape[0], 0), dtype=np.float64)
        return np.concatenate(pieces, axis=1).astype(np.float64, copy=False)

    def diagnostics(self, states: np.ndarray) -> Dict[str, object]:
        x = np.asarray(states, dtype=np.float64)
        out: Dict[str, object] = {
            "n_rows": int(x.shape[0]),
            "raw_state_dim": int(len(self.raw_state_names)),
            "encoded_state_dim": int(len(self.encoded_state_names)),
            "categorical_state_names": list(self.categorical_state_names),
            "one_hot": bool(self.one_hot),
            "unknown_by_column": {},
        }
        for name in self.categorical_state_names:
            j = self.raw_state_names.index(name)
            col = x[:, j]
            levels = np.asarray(self.categorical_levels.get(name, []), dtype=np.float64)
            finite = np.isfinite(col)
            if levels.size == 0:
                known = np.zeros_like(col, dtype=bool)
            else:
                known = np.any(
                    np.isclose(
                        col.reshape(-1, 1),
                        levels.reshape(1, -1),
                        rtol=0.0,
                        atol=1e-8,
                    ),
                    axis=1,
                )
            unknown = finite & (~known)
            out["unknown_by_column"][name] = {
                "known_levels": [_json_float(v) for v in levels.tolist()],
                "unknown_count": int(unknown.sum()),
                "unknown_fraction": float(unknown.mean()) if unknown.size else 0.0,
                "missing_count": int((~finite).sum()),
            }
        return out

    def to_metadata(self) -> Dict[str, object]:
        return {
            "raw_state_names": list(self.raw_state_names),
            "categorical_state_names": list(self.categorical_state_names),
            "numeric_state_names": list(self.numeric_state_names),
            "categorical_levels": {
                k: [_json_float(v) for v in vals]
                for k, vals in self.categorical_levels.items()
            },
            "encoded_state_names": list(self.encoded_state_names),
            "one_hot": bool(self.one_hot),
        }


def state_encoder_from_metadata(meta: Dict[str, object]) -> StateEncoder:
    raw_names = [str(c) for c in meta.get("raw_state_names", [])]
    cat_names = [str(c) for c in meta.get("categorical_state_names", [])]
    numeric_names = [str(c) for c in meta.get("numeric_state_names", [])]
    raw_levels = meta.get("categorical_levels", {}) or {}
    levels = {
        str(k): [float(v) for v in vals]
        for k, vals in raw_levels.items()
    }
    encoded_names = [str(c) for c in meta.get("encoded_state_names", [])]
    if not raw_names or not encoded_names:
        raise ValueError("State encoder metadata must contain raw_state_names and encoded_state_names.")
    return StateEncoder(
        raw_state_names=raw_names,
        categorical_state_names=cat_names,
        numeric_state_names=numeric_names,
        categorical_levels=levels,
        encoded_state_names=encoded_names,
        one_hot=bool(meta.get("one_hot", True)),
    )


def _json_float(v: float):
    fv = float(v)
    if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
        return int(round(fv))
    return fv


def fit_state_encoder(
    raw_state_names: Sequence[str],
    train_states: np.ndarray,
    categorical_state_cols: Optional[str] = "auto",
    one_hot: bool = True,
    max_auto_cardinality: int = 20,
) -> StateEncoder:
    names = [str(c) for c in raw_state_names]
    x = np.asarray(train_states, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(
            f"train_states must have shape (n,{len(names)}), got {x.shape}"
        )

    cat_names = resolve_categorical_state_cols(
        categorical_arg=categorical_state_cols,
        state_names=names,
        train_states=x,
        max_auto_cardinality=max_auto_cardinality,
    )
    cat_set = set(cat_names)
    numeric_names = [name for name in names if name not in cat_set]

    levels: Dict[str, List[float]] = {}
    encoded_names: List[str] = []
    for j, name in enumerate(names):
        if one_hot and name in cat_set:
            vals = x[:, j]
            vals = vals[np.isfinite(vals)]
            uniq = np.unique(vals.astype(np.float64))
            levels[name] = [float(v) for v in uniq.tolist()]
            encoded_names.extend([f"{name}__cat_{_value_label(v)}" for v in uniq])
        else:
            encoded_names.append(name)

    return StateEncoder(
        raw_state_names=names,
        categorical_state_names=cat_names if one_hot else [],
        numeric_state_names=numeric_names if one_hot else names,
        categorical_levels=levels,
        encoded_state_names=encoded_names,
        one_hot=bool(one_hot),
    )
