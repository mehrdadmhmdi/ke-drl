#!/usr/bin/env python3
"""
Train reward-specific offline IQL critics on raw logged data and distill each critic
into a plain linear-Gaussian policy object with no neural-network policy module.

Target policy form
------------------
A' | S=s ~ N( s @ Theta_mu + epsilon_mu,
              diag(exp(s @ Theta_sigma + epsilon_sigma)^2) )

In this script Theta_sigma = 0 and only epsilon_sigma is estimated from constant
residual standard deviations.

Artifacts per reward
--------------------
- iql_value_<reward>.d3
- linear_gaussian_policy_<reward>.npz
- linear_gaussian_policy_<reward>.json
- linear_gaussian_policy_<reward>_summary.json

Global artifacts
----------------
- train_stats_raw.json
- run_summary.json
- focus_matrix.json
- policy_difference_summary.json
- plots/*.png   (only overlay plots)

Notes
-----
- No policy neural-network classes are defined here.
- Integer-valued action handling happens only at inference time.
- The critic is trained with d3rlpy and is used only to select reward-focused
  target actions.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")
with open(os.devnull, "w") as fnull, contextlib.redirect_stderr(fnull):
    import d3rlpy

# Plot style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

DATA_COLOR = '#2E8B57'
REV_COLOR = '#13294B'
CLICK_COLOR = '#FF5F05'
AUX_COLOR = '#9467BD'


def pretty_series_label(name: str) -> str:
    s = str(name).strip().lower()
    if s == 'data':
        return 'data'
    if 'click' in s:
        return 'click-focused policy'
    if 'revenue' in s or 'sales' in s:
        return 'revenue-focused policy'
    return str(name).replace('_', ' ').title()


def series_color(name: str) -> str:
    s = str(name).strip().lower()
    if s == 'data':
        return DATA_COLOR
    if 'click' in s:
        return CLICK_COLOR
    if 'revenue' in s or 'sales' in s:
        return REV_COLOR
    return AUX_COLOR


def reward_suggests_click_like(name: str) -> bool:
    s = str(name).lower()
    return any(k in s for k in ['click', 'booking', 'count', 'visit', 'impression'])


# -----------------------------------------------------------------------------
# basic utilities
# -----------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> str:
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return str(device_str)


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return [c.strip() for c in x.split(",") if c.strip()]


def detect_key(d: dict, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def normalize_blob_payload(blob: dict) -> dict:
    if not isinstance(blob, dict):
        raise TypeError(f"Loaded blob must be a dict, got {type(blob)}")

    if any(k in blob for k in ["s0", "a0", "r0", "r", "s1"]):
        return blob

    if "data" in blob and isinstance(blob["data"], dict):
        out = dict(blob["data"])
        meta = blob.get("meta", {})
        if isinstance(meta, dict):
            for k, v in meta.items():
                if k not in out:
                    out[k] = v
        return out

    raise KeyError(
        "Blob is neither flat nor nested in {'data','meta'} format. "
        f"Top-level keys are: {list(blob.keys())}"
    )


def get_2d_tensor(blob: dict, key: str) -> torch.Tensor:
    if key not in blob:
        raise KeyError(f"Missing key '{key}'")
    x = blob[key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError(f"{key} must be 1D or 2D, got shape={tuple(x.shape)}")
    return x.float()


def default_names_for_blob_section(blob: dict, tensor_key: str, names_key: str) -> List[str]:
    x = get_2d_tensor(blob, tensor_key)
    names = blob.get(names_key, None)
    if isinstance(names, (list, tuple)):
        if len(names) != x.shape[1]:
            raise ValueError(
                f"{names_key} has length {len(names)} but {tensor_key} has {x.shape[1]} columns."
            )
        return [str(v) for v in names]
    return [f"{tensor_key}_{j}" for j in range(x.shape[1])]


def select_named_columns(
    blob: dict,
    tensor_key: str,
    names_key: str,
    wanted_cols: Optional[List[str]],
    role_name: str,
) -> Tuple[torch.Tensor, List[str]]:
    x = get_2d_tensor(blob, tensor_key)
    names = default_names_for_blob_section(blob, tensor_key, names_key)
    if wanted_cols is None:
        return x.float(), names

    name_to_idx = {name: j for j, name in enumerate(names)}
    missing = [c for c in wanted_cols if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Requested {role_name} cols not found: {missing}\n"
            f"Available {role_name} cols are: {sorted(names)}"
        )
    idx = [name_to_idx[c] for c in wanted_cols]
    return x[:, idx].float(), list(wanted_cols)


def make_terminals(blob: dict, n_rows: int, episode_length: Optional[int]) -> np.ndarray:
    done_key = detect_key(blob, ["done", "terminal", "terminals", "dones"])
    if done_key is not None:
        terminals = blob[done_key]
        if isinstance(terminals, torch.Tensor):
            terminals = terminals.detach().cpu().numpy()
        terminals = np.asarray(terminals).reshape(-1)
        if terminals.shape[0] != n_rows:
            raise ValueError(
                f"Terminal key '{done_key}' has length {terminals.shape[0]} but expected {n_rows}."
            )
        print(f"using terminal flags from '{done_key}'")
        return terminals.astype(np.bool_)

    if episode_length is not None and episode_length > 0:
        terminals = np.zeros(n_rows, dtype=np.bool_)
        ends = np.arange(episode_length - 1, n_rows, episode_length)
        terminals[ends] = True
        if n_rows > 0 and not terminals[-1]:
            terminals[-1] = True
        print(f"reconstructed terminal flags with episode_length={episode_length}")
        return terminals

    print("no terminal info found -> using 1-step episodes for all samples")
    return np.ones(n_rows, dtype=np.bool_)


def maybe_subsample_arrays(
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    terminals: Optional[np.ndarray],
    max_n: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    n = obs.shape[0]
    if max_n is None or max_n <= 0 or max_n >= n:
        return obs, act, rew, terminals
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    obs = obs[idx]
    act = act[idx]
    rew = rew[idx]
    if terminals is not None:
        terminals = terminals[idx]
        terminals[-1] = True
    return obs, act, rew, terminals


def resolve_blob_path(blob_arg: Optional[str], data_base: Optional[str], default_name: Optional[str] = None) -> Optional[Path]:
    blob_name = blob_arg if blob_arg is not None else default_name
    if blob_name is None:
        return None
    p = Path(blob_name)
    if p.is_absolute() or p.exists() or data_base is None:
        return p
    return Path(data_base) / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_blob(path: Path) -> dict:
    print(f"Loading {path}")
    obj = torch.load(path, map_location="cpu")
    return normalize_blob_payload(obj)


def maybe_load_split(
    path: Optional[Path],
    state_cols: Optional[List[str]],
    action_cols: Optional[List[str]],
    reward_cols: Optional[List[str]],
    max_n: Optional[int],
    seed: int,
    episode_length: Optional[int],
) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    blob = load_blob(path)
    s, state_names = select_named_columns(blob, "s0", "state_cols", state_cols, "state")
    a, action_names = select_named_columns(blob, "a0", "action_cols", action_cols, "action")
    reward_key = "r0" if "r0" in blob else "r"
    r, reward_names = select_named_columns(blob, reward_key, "reward_cols", reward_cols, "reward")
    terminals = make_terminals(blob, s.shape[0], episode_length)

    obs = s.detach().cpu().numpy().astype(np.float64)
    act = a.detach().cpu().numpy().astype(np.float64)
    rew = r.detach().cpu().numpy().astype(np.float64)
    obs, act, rew, terminals = maybe_subsample_arrays(obs, act, rew, terminals, max_n, seed)
    return {
        "obs": obs,
        "act": act,
        "rew": rew,
        "terminals": terminals,
        "state_names": state_names,
        "action_names": action_names,
        "reward_names": reward_names,
    }


# -----------------------------------------------------------------------------
# d3rlpy wrappers
# -----------------------------------------------------------------------------

def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    out = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            out[k] = v
    return out


def load_d3(path: str, device: str = "cpu"):
    if hasattr(d3rlpy, "load_learnable"):
        return d3rlpy.load_learnable(path, device=device)
    if hasattr(d3rlpy, "load_learner"):
        return d3rlpy.load_learner(path, device=device)
    if hasattr(d3rlpy, "load"):
        return d3rlpy.load(path)
    raise AttributeError("No compatible d3rlpy load function found.")


def make_dataset(obs: np.ndarray, act: np.ndarray, rew: np.ndarray, terminals: np.ndarray):
    from d3rlpy.dataset import MDPDataset
    obs32 = obs.astype(np.float32)
    act32 = act.astype(np.float32)
    rew32 = rew.astype(np.float32)
    try:
        return MDPDataset(observations=obs32, actions=act32, rewards=rew32, terminals=terminals)
    except TypeError:
        return MDPDataset(obs32, act32, rew32, terminals)


def make_iql_learner(
    device_str: str,
    gamma: float,
    batch_size: int,
    actor_lr: float,
    critic_lr: float,
    value_lr: float,
    expectile: float,
    weight_temp: float,
    max_weight: float,
):
    errors: List[str] = []

    try:
        from d3rlpy.algos import IQLConfig, IQL
        cfg_kwargs = {
            "gamma": gamma,
            "batch_size": batch_size,
            "actor_learning_rate": actor_lr,
            "critic_learning_rate": critic_lr,
            "value_learning_rate": value_lr,
            "expectile": expectile,
            "weight_temp": weight_temp,
            "max_weight": max_weight,
            "reward_scaler": None,
            "action_scaler": None,
            "observation_scaler": None,
            "scaler": None,
        }
        cfg = IQLConfig(**_filter_kwargs_for_callable(IQLConfig, cfg_kwargs))
        if hasattr(cfg, "create"):
            create_kwargs = _filter_kwargs_for_callable(cfg.create, {"device": device_str, "enable_ddp": False})
            return cfg.create(**create_kwargs)
        ctor_kwargs = _filter_kwargs_for_callable(IQL, {"config": cfg, "device": device_str, "enable_ddp": False})
        return IQL(**ctor_kwargs)
    except Exception as e:
        errors.append(f"IQLConfig path: {e}")

    try:
        from d3rlpy.algos import IQL
        direct_kwargs = {
            "gamma": gamma,
            "batch_size": batch_size,
            "actor_learning_rate": actor_lr,
            "critic_learning_rate": critic_lr,
            "value_learning_rate": value_lr,
            "expectile": expectile,
            "weight_temp": weight_temp,
            "max_weight": max_weight,
            "reward_scaler": None,
            "action_scaler": None,
            "observation_scaler": None,
            "scaler": None,
            "use_gpu": bool("cuda" in device_str),
            "device": device_str,
        }
        return IQL(**_filter_kwargs_for_callable(IQL, direct_kwargs))
    except Exception as e:
        errors.append(f"IQL direct path: {e}")

    raise RuntimeError(
        "Could not construct IQL learner from d3rlpy. Tried multiple APIs.\n - " + "\n - ".join(errors)
    )


def fit_algo(algo, dataset, n_steps: int, n_steps_per_epoch: int, experiment_name: str, quiet: bool = True):
    def _fit_once():
        kwargs = {
            "n_steps": n_steps,
            "n_steps_per_epoch": n_steps_per_epoch,
            "experiment_name": experiment_name,
            "with_timestamp": False,
            "show_progress": not quiet,
            "verbose": not quiet,
        }

        try:
            return algo.fit(dataset, **_filter_kwargs_for_callable(algo.fit, kwargs))
        except TypeError:
            pass

        try:
            return algo.fit(
                dataset,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                experiment_name=experiment_name,
                with_timestamp=False,
            )
        except TypeError:
            pass

        try:
            return algo.fit(
                dataset,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                experiment_name=experiment_name,
            )
        except TypeError:
            pass

        return algo.fit(dataset, n_steps, n_steps_per_epoch)

    if quiet:
        import logging
        old_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        old_env = {
            "TQDM_DISABLE": os.environ.get("TQDM_DISABLE"),
            "DISABLE_TQDM": os.environ.get("DISABLE_TQDM"),
        }
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["DISABLE_TQDM"] = "1"

        try:
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                return _fit_once()
        finally:
            logging.disable(old_disable)
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _fit_once()

def save_algo(algo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(algo, "save"):
        algo.save(str(path))
        return
    if hasattr(algo, "save_model"):
        algo.save_model(str(path))
        return
    raise AttributeError("This d3rlpy learner has no save/save_model method.")


def predict_values(algo, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    if not hasattr(algo, "predict_value"):
        raise AttributeError("This learner does not expose predict_value.")
    out = algo.predict_value(obs.astype(np.float32), act.astype(np.float32))
    return np.asarray(out, dtype=np.float64).reshape(-1)


# -----------------------------------------------------------------------------
# linear-Gaussian policy object and inference
# -----------------------------------------------------------------------------

@dataclass
class LinearGaussianpolicy:
    theta_mu: np.ndarray         # (d_s, d_a)
    epsilon_mu: np.ndarray       # (d_a,)
    theta_sigma: np.ndarray      # (d_s, d_a), zero here
    epsilon_sigma: np.ndarray    # (d_a,)
    action_lows: np.ndarray      # (d_a,)
    action_highs: np.ndarray     # (d_a,)
    action_names: List[str]
    state_names: List[str]
    reward_name: str
    integer_idx: Optional[int] = None
    integer_low: Optional[int] = None
    integer_high: Optional[int] = None
    integer_name: Optional[str] = None


def mean_action(policy: LinearGaussianpolicy, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return s @ policy.theta_mu + policy.epsilon_mu


def std_action(policy: LinearGaussianpolicy, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    log_std = s @ policy.theta_sigma + policy.epsilon_sigma
    return np.exp(log_std)


def _clip_and_round(policy: LinearGaussianpolicy, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    a = np.clip(a, policy.action_lows, policy.action_highs)
    if policy.integer_idx is not None:
        j = int(policy.integer_idx)
        a[..., j] = np.round(a[..., j])
        if policy.integer_low is not None:
            a[..., j] = np.maximum(a[..., j], policy.integer_low)
        if policy.integer_high is not None:
            a[..., j] = np.minimum(a[..., j], policy.integer_high)
    return a


def greedy_action(policy: LinearGaussianpolicy, s: np.ndarray) -> np.ndarray:
    return _clip_and_round(policy, mean_action(policy, s))


def sample_action(policy: LinearGaussianpolicy, s: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    mu = mean_action(policy, s)
    sd = std_action(policy, s)
    eps = rng.standard_normal(mu.shape)
    a = mu + eps * sd
    return _clip_and_round(policy, a)


def save_linear_gaussian_policy(policy: LinearGaussianpolicy, npz_path: Path, json_path: Path) -> None:
    ensure_dir(npz_path.parent)
    np.savez_compressed(
        npz_path,
        theta_mu=policy.theta_mu,
        epsilon_mu=policy.epsilon_mu,
        theta_sigma=policy.theta_sigma,
        epsilon_sigma=policy.epsilon_sigma,
        action_lows=policy.action_lows,
        action_highs=policy.action_highs,
        integer_idx=-1 if policy.integer_idx is None else int(policy.integer_idx),
        integer_low=-1 if policy.integer_low is None else int(policy.integer_low),
        integer_high=-1 if policy.integer_high is None else int(policy.integer_high),
    )
    meta = {
        "policy_type": "linear_gaussian",
        "reward_name": policy.reward_name,
        "state_names": policy.state_names,
        "action_names": policy.action_names,
        "state_dim": len(policy.state_names),
        "action_dim": len(policy.action_names),
        "integer_action_index": policy.integer_idx,
        "integer_action_low": policy.integer_low,
        "integer_action_high": policy.integer_high,
        "integer_action_name": policy.integer_name,
        "action_lows": policy.action_lows.tolist(),
        "action_highs": policy.action_highs.tolist(),
    }
    json_path.write_text(json.dumps(meta, indent=2))


def load_linear_gaussian_policy(npz_path: str | Path, json_path: str | Path) -> LinearGaussianpolicy:
    arr = np.load(npz_path)
    meta = json.loads(Path(json_path).read_text())

    def _maybe_none(v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        iv = int(v)
        return None if iv < 0 else iv

    return LinearGaussianpolicy(
        theta_mu=np.asarray(arr["theta_mu"], dtype=np.float64),
        epsilon_mu=np.asarray(arr["epsilon_mu"], dtype=np.float64),
        theta_sigma=np.asarray(arr["theta_sigma"], dtype=np.float64),
        epsilon_sigma=np.asarray(arr["epsilon_sigma"], dtype=np.float64),
        action_lows=np.asarray(arr["action_lows"], dtype=np.float64),
        action_highs=np.asarray(arr["action_highs"], dtype=np.float64),
        action_names=list(meta["action_names"]),
        state_names=list(meta["state_names"]),
        reward_name=str(meta["reward_name"]),
        integer_idx=_maybe_none(meta.get("integer_action_index")),
        integer_low=_maybe_none(meta.get("integer_action_low")),
        integer_high=_maybe_none(meta.get("integer_action_high")),
        integer_name=meta.get("integer_action_name", None),
    )


# -----------------------------------------------------------------------------
# action support and target selection
# -----------------------------------------------------------------------------

@dataclass
class ActionSpec:
    action_names: List[str]
    lows: np.ndarray
    highs: np.ndarray
    integer_idx: Optional[int]
    integer_low: Optional[int]
    integer_high: Optional[int]
    integer_name: Optional[str]


def infer_action_spec(
    action_names: List[str],
    act_train: np.ndarray,
    integer_action_col: Optional[str],
) -> ActionSpec:
    lows = np.nanmin(act_train, axis=0).astype(np.float64)
    highs = np.nanmax(act_train, axis=0).astype(np.float64)

    integer_idx: Optional[int] = None
    integer_low: Optional[int] = None
    integer_high: Optional[int] = None
    integer_name: Optional[str] = None

    if integer_action_col is not None:
        if integer_action_col not in action_names:
            raise ValueError(
                f"Requested integer action col '{integer_action_col}' not found. Available: {action_names}"
            )
        integer_idx = action_names.index(integer_action_col)
        integer_low = int(np.round(lows[integer_idx]))
        integer_high = int(np.round(highs[integer_idx]))
        integer_name = action_names[integer_idx]

    return ActionSpec(
        action_names=list(action_names),
        lows=lows,
        highs=highs,
        integer_idx=integer_idx,
        integer_low=integer_low,
        integer_high=integer_high,
        integer_name=integer_name,
    )


def build_support_bank(
    obs_train: np.ndarray,
    act_train: np.ndarray,
    pool_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = obs_train.shape[0]
    if pool_size <= 0 or pool_size >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=pool_size, replace=False)
        idx.sort()
    return obs_train[idx], act_train[idx], idx


def build_reward_focused_bank(
    obs_train: np.ndarray,
    act_train: np.ndarray,
    reward_values: np.ndarray,
    reward_name: str,
    seed: int,
    standard_pool_size: int,
    click_pool_size: int,
    click_min_reward: float,
    click_quantile: float,
    min_bank_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    reward_values = np.asarray(reward_values, dtype=np.float64).reshape(-1)
    if reward_suggests_click_like(reward_name):
        mask = reward_values >= float(click_min_reward)
        strategy = f'reward>={float(click_min_reward):.3g}'
        if int(mask.sum()) < int(min_bank_size):
            qthr = float(np.quantile(reward_values, float(click_quantile)))
            mask = reward_values >= qthr
            strategy = f'reward>=q{float(click_quantile):.2f}({qthr:.4f})'
        if int(mask.sum()) == 0:
            mask = np.ones_like(reward_values, dtype=bool)
            strategy = 'fallback_all_rows'
        obs_pool = obs_train[mask]
        act_pool = act_train[mask]
        pool_size = click_pool_size
        bank_type = 'click_enriched'
    else:
        obs_pool = obs_train
        act_pool = act_train
        pool_size = standard_pool_size
        strategy = 'standard_all_rows'
        bank_type = 'standard'

    n = obs_pool.shape[0]
    if pool_size <= 0 or pool_size >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=pool_size, replace=False)
        idx.sort()
    bank_states = obs_pool[idx]
    bank_actions = act_pool[idx]
    info = {
        'bank_type': bank_type,
        'selection_strategy': strategy,
        'bank_n_before_subsample': int(n),
        'bank_n_after_subsample': int(bank_states.shape[0]),
    }
    return bank_states, bank_actions, info


def _nearest_and_random_candidates(
    obs_batch: np.ndarray,
    bank_states: np.ndarray,
    neighbor_k: int,
    random_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    b = obs_batch.shape[0]
    n_bank = bank_states.shape[0]
    neighbor_k = min(max(0, int(neighbor_k)), n_bank)
    random_k = min(max(0, int(random_k)), n_bank)

    dist = np.sum((obs_batch[:, None, :] - bank_states[None, :, :]) ** 2, axis=2)
    if neighbor_k > 0:
        nn_idx = np.argpartition(dist, kth=max(neighbor_k - 1, 0), axis=1)[:, :neighbor_k]
    else:
        nn_idx = np.empty((b, 0), dtype=np.int64)

    if random_k > 0:
        rand_idx = np.stack([rng.choice(n_bank, size=random_k, replace=False) for _ in range(b)], axis=0)
    else:
        rand_idx = np.empty((b, 0), dtype=np.int64)

    all_idx = np.concatenate([nn_idx, rand_idx], axis=1)
    out = []
    max_len = 0
    for row in all_idx:
        uniq = np.unique(row.astype(np.int64))
        out.append(uniq)
        max_len = max(max_len, int(uniq.size))
    padded = np.full((b, max_len), -1, dtype=np.int64)
    for i, arr in enumerate(out):
        padded[i, : arr.size] = arr
    return padded


def build_critic_targets_best(
    algo,
    obs: np.ndarray,
    bank_states: np.ndarray,
    bank_actions: np.ndarray,
    neighbor_k: int,
    random_k: int,
    batch_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    d_a = bank_actions.shape[1]

    target_best_action = np.zeros((n, d_a), dtype=np.float64)
    target_qmax = np.zeros(n, dtype=np.float64)
    target_qsecond = np.zeros(n, dtype=np.float64)
    target_margin = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_batch = obs[start:end]
        idx_mat = _nearest_and_random_candidates(obs_batch, bank_states, neighbor_k, random_k, rng)
        b = obs_batch.shape[0]

        flat_obs = []
        flat_act = []
        group_sizes = []
        for i in range(b):
            valid = idx_mat[i][idx_mat[i] >= 0]
            if valid.size == 0:
                valid = np.array([rng.integers(0, bank_actions.shape[0])], dtype=np.int64)
            cand_act = bank_actions[valid]
            flat_obs.append(np.repeat(obs_batch[i:i + 1], repeats=cand_act.shape[0], axis=0))
            flat_act.append(cand_act)
            group_sizes.append(cand_act.shape[0])

        obs_rep = np.concatenate(flat_obs, axis=0)
        act_rep = np.concatenate(flat_act, axis=0)
        q_rep = predict_values(algo, obs_rep, act_rep)

        offset = 0
        for i in range(b):
            m = group_sizes[i]
            q = q_rep[offset:offset + m]
            cand = act_rep[offset:offset + m]
            offset += m

            best_j = int(np.argmax(q))
            target_best_action[start + i] = cand[best_j]
            target_qmax[start + i] = float(q[best_j])
            if m >= 2:
                qs = np.sort(q)
                second = float(qs[-2])
            else:
                second = float(q[best_j])
            target_qsecond[start + i] = second
            target_margin[start + i] = float(q[best_j] - second)

    return {
        "target_best_action": target_best_action,
        "target_qmax": target_qmax,
        "target_qsecond": target_qsecond,
        "target_margin": target_margin,
    }


# -----------------------------------------------------------------------------
# closed-form linear-Gaussian fit
# -----------------------------------------------------------------------------

def _apply_integer_override(best_actions: np.ndarray, spec: ActionSpec) -> np.ndarray:
    out = np.asarray(best_actions, dtype=np.float64).copy()
    if spec.integer_idx is not None:
        j = int(spec.integer_idx)
        out[:, j] = np.round(out[:, j])
        if spec.integer_low is not None:
            out[:, j] = np.maximum(out[:, j], spec.integer_low)
        if spec.integer_high is not None:
            out[:, j] = np.minimum(out[:, j], spec.integer_high)
    return out


def fit_linear_gaussian_policy(
    X: np.ndarray,
    Y: np.ndarray,
    reward_name: str,
    state_names: List[str],
    action_names: List[str],
    spec: ActionSpec,
    ridge: float,
    min_std: float,
    max_std: float,
    sample_weight: Optional[np.ndarray] = None,
    std_mode: str = "heteroskedastic",
    std_ridge: Optional[float] = None,
    std_floor: float = 1e-6,
    std_blend: float = 0.85,
) -> Tuple[LinearGaussianpolicy, Dict[str, object]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d_s = X.shape
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}")

    X1 = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)

    if sample_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"sample_weight must have length {n}, got {w.shape[0]}")
        w = np.maximum(w, 1e-8)
        w = w / max(float(np.mean(w)), 1e-12)

    sw = np.sqrt(w)[:, None]

    # Mean-action regression.
    reg_mu = ridge * np.eye(d_s + 1, dtype=np.float64)
    reg_mu[-1, -1] = 0.0
    Xw = X1 * sw
    Yw = Y * sw
    B_mu = np.linalg.solve(Xw.T @ Xw + reg_mu, Xw.T @ Yw)

    theta_mu = B_mu[:-1, :]
    epsilon_mu = B_mu[-1, :]

    Yhat = X @ theta_mu + epsilon_mu
    resid = Y - Yhat

    # Old constant residual scale.
    global_sigma = np.sqrt(np.average(resid ** 2, axis=0, weights=w))
    global_sigma = np.clip(global_sigma, float(min_std), float(max_std))

    std_mode = str(std_mode).lower()
    if std_mode in ["constant", "residual_constant", "old"]:
        theta_sigma = np.zeros_like(theta_mu)
        epsilon_sigma = np.log(global_sigma)
        sigma_train = np.tile(global_sigma.reshape(1, -1), (n, 1))
        std_fit_mse = 0.0

    elif std_mode in ["heteroskedastic", "state_dependent", "linear"]:
        # State-dependent variance model:
        # log sigma(s) is fitted from stabilized residual magnitudes.
        # The global scale anchor avoids zero residuals producing degenerate std.
        std_blend = float(np.clip(std_blend, 0.0, 1.0))
        local_sigma = np.sqrt(resid ** 2 + float(std_floor) ** 2)
        sigma_target = np.sqrt(
            std_blend * local_sigma ** 2
            + (1.0 - std_blend) * global_sigma.reshape(1, -1) ** 2
        )
        sigma_target = np.clip(sigma_target, float(min_std), float(max_std))
        log_sigma_target = np.log(sigma_target)

        if std_ridge is None:
            std_ridge = ridge
        reg_std = float(std_ridge) * np.eye(d_s + 1, dtype=np.float64)
        reg_std[-1, -1] = 0.0

        B_std = np.linalg.solve(
            Xw.T @ Xw + reg_std,
            Xw.T @ (log_sigma_target * sw),
        )

        theta_sigma = B_std[:-1, :]
        epsilon_sigma = B_std[-1, :]

        log_sigma_fit = X @ theta_sigma + epsilon_sigma
        log_sigma_fit = np.clip(log_sigma_fit, np.log(float(min_std)), np.log(float(max_std)))
        sigma_train = np.exp(log_sigma_fit)
        std_fit_mse = float(np.mean((log_sigma_target - log_sigma_fit) ** 2))

    else:
        raise ValueError(
            f"Unknown std_mode={std_mode}. Use 'constant' or 'heteroskedastic'."
        )

    policy = LinearGaussianpolicy(
        theta_mu=theta_mu,
        epsilon_mu=epsilon_mu,
        theta_sigma=theta_sigma,
        epsilon_sigma=epsilon_sigma,
        action_lows=spec.lows.copy(),
        action_highs=spec.highs.copy(),
        action_names=list(action_names),
        state_names=list(state_names),
        reward_name=str(reward_name),
        integer_idx=spec.integer_idx,
        integer_low=spec.integer_low,
        integer_high=spec.integer_high,
        integer_name=spec.integer_name,
    )

    greedy_train = greedy_action(policy, X)
    mu_train = mean_action(policy, X)

    fit_meta = {
        "ridge": float(ridge),
        "std_mode": std_mode,
        "std_ridge": None if std_ridge is None else float(std_ridge),
        "std_floor": float(std_floor),
        "std_blend": float(std_blend),
        "theta_mu_norm_fro": float(np.linalg.norm(theta_mu)),
        "theta_sigma_norm_fro": float(np.linalg.norm(theta_sigma)),
        "epsilon_mu": epsilon_mu.tolist(),
        "epsilon_sigma": epsilon_sigma.tolist(),
        "train_mu_mse": float(np.mean((mu_train - Y) ** 2)),
        "train_greedy_mse": float(np.mean((greedy_train - Y) ** 2)),
        "residual_std_constant": global_sigma.tolist(),
        "train_std_mean": np.mean(sigma_train, axis=0).tolist(),
        "train_std_min": np.min(sigma_train, axis=0).tolist(),
        "train_std_max": np.max(sigma_train, axis=0).tolist(),
        "log_std_fit_mse": float(std_fit_mse),
        "sample_weight_mean": float(np.mean(w)),
        "sample_weight_max": float(np.max(w)),
    }
    return policy, fit_meta


# -----------------------------------------------------------------------------
# plotting: overlay only
# -----------------------------------------------------------------------------

def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _shared_bins(arrays: Sequence[np.ndarray], n_bins: int = 60) -> np.ndarray:
    finite = [_finite_1d(a) for a in arrays]
    finite = [a for a in finite if a.size > 0]
    if not finite:
        return np.linspace(0.0, 1.0, n_bins + 1)
    both = np.concatenate(finite)
    lo = np.min(both)
    hi = np.max(both)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        q_lo, q_hi = np.percentile(both, [0.5, 99.5]) if both.size >= 20 else (lo, hi)
        if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
            lo, hi = q_lo, q_hi
        return np.linspace(lo, hi, n_bins + 1)
    center = float(lo if np.isfinite(lo) else 0.0)
    width = max(1.0, abs(center) * 0.05 + 1e-3)
    return np.array([center - width, center, center + width], dtype=np.float64)


def _pmf(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(support, dtype=np.float64)
    rounded = np.round(vals).astype(int)
    support_int = np.round(np.asarray(support, dtype=np.float64)).astype(int)
    counts = np.array([(rounded == s).mean() for s in support_int], dtype=np.float64)
    return counts

ACTION_LABEL_MAP = {
    "avg_price_per_night": "Avg price per night",
    "Total promotions": "Total promotions",
    "std_price_usd": "Standard deviation of price",
    "total_promotions": "Total promotions",
}

def pretty_action_label(name: str) -> str:
    return ACTION_LABEL_MAP.get(str(name), str(name).replace("_", " ").title())

def save_overlay_action_distribution_plots(
    action_names: List[str],
    logged_actions: np.ndarray,
    greedy_actions_by_policy: Dict[str, np.ndarray],
    integer_idx: Optional[int],
    out_path: Path,
) -> None:
    n_actions = len(action_names)
    fig, axes = plt.subplots(1, n_actions, figsize=(5.5 * n_actions, 4.2))
    if n_actions == 1:
        axes = [axes]

    policy_names = list(greedy_actions_by_policy.keys())

    for j, ax in enumerate(axes):
        arrays = [logged_actions[:, j]] + [greedy_actions_by_policy[p][:, j] for p in policy_names]
        if integer_idx is not None and j == integer_idx:
            allv = np.concatenate([np.round(_finite_1d(a)) for a in arrays if _finite_1d(a).size > 0]) if arrays else np.array([0.0])
            if allv.size == 0:
                support = np.arange(0, 2)
            else:
                lo = int(np.min(allv))
                hi = int(np.max(allv))
                support = np.arange(lo, hi + 1)
            ax.step(support, _pmf(logged_actions[:, j], support), where='mid', color=series_color('data'), linewidth=2.3, label=pretty_series_label('data'))
            for p in policy_names:
                ax.step(support, _pmf(greedy_actions_by_policy[p][:, j], support), where='mid',
                        color=series_color(p), linewidth=2.3, label=pretty_series_label(p))
            ax.set_ylabel('Probability')
        else:
            bins = _shared_bins(arrays, n_bins=60)
            hist_data, edges = np.histogram(_finite_1d(logged_actions[:, j]), bins=bins, density=True)
            mids = 0.5 * (edges[:-1] + edges[1:])
            ax.step(mids, hist_data, where='mid', color=series_color('data'), linewidth=2.3, label=pretty_series_label('data'))
            for p in policy_names:
                hist_pol, _ = np.histogram(_finite_1d(greedy_actions_by_policy[p][:, j]), bins=bins, density=True)
                ax.step(mids, hist_pol, where='mid', color=series_color(p), linewidth=2.3, label=pretty_series_label(p))
            ax.set_ylabel('Density')
        ax.set_xlabel(pretty_action_label(action_names[j]))
        ax.legend(frameon=True)

    fig.suptitle('Greedy action overlays: data vs learned policies')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=700, bbox_inches='tight')
    plt.close(fig)



def save_overlay_action_mean_profile_plot(
    action_names: List[str],
    train_action_mean: np.ndarray,
    train_action_std: np.ndarray,
    logged_eval_action_mean: np.ndarray,
    greedy_action_means_by_policy: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    action_names = list(action_names)
    train_action_mean = np.asarray(train_action_mean, dtype=np.float64).reshape(-1)
    train_action_std = np.asarray(train_action_std, dtype=np.float64).reshape(-1)
    logged_eval_action_mean = np.asarray(logged_eval_action_mean, dtype=np.float64).reshape(-1)
    sd = np.maximum(train_action_std, 1e-8)

    x = np.arange(len(action_names))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    data_z = (logged_eval_action_mean - train_action_mean) / sd
    ax.plot(x, data_z, marker='o', linewidth=2.8, markersize=8, color=series_color('data'), label=pretty_series_label('data'))

    for p, mu in greedy_action_means_by_policy.items():
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        z = (mu - train_action_mean) / sd
        ax.plot(x, z, marker='o', linewidth=2.8, markersize=8, color=series_color(p), label=pretty_series_label(p))

    ax.axhline(0.0, linestyle='--', linewidth=1.5, color='gray', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_action_label(x) for x in action_names], ha='right')
    ax.set_ylabel('Action mean shift (z-score relative to train actions)')
    ax.set_title('Greedy Action Mean Profiles')
    ax.legend(frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=700, bbox_inches='tight')
    plt.close(fig)


def save_overlay_top2_scatter_plot(
    action_names: List[str],
    greedy_actions_by_policy: Dict[str, np.ndarray],
    out_path: Path,
) -> Optional[Dict[str, object]]:
    policy_names = list(greedy_actions_by_policy.keys())
    if len(policy_names) < 2:
        return None

    A, B = policy_names[0], policy_names[1]
    XA = np.asarray(greedy_actions_by_policy[A], dtype=np.float64)
    XB = np.asarray(greedy_actions_by_policy[B], dtype=np.float64)
    if XA.ndim != 2 or XB.ndim != 2 or XA.shape[1] != XB.shape[1] or XA.shape[1] < 2:
        return None

    mean_abs_diff = np.mean(np.abs(XA - XB), axis=0)
    top2 = np.argsort(-mean_abs_diff)[:2]
    if top2.size < 2:
        return None
    i, j = int(top2[0]), int(top2[1])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(XA[:, i], XA[:, j], s=12, alpha=0.28, color=series_color(A), label=pretty_series_label(A))
    ax.scatter(XB[:, i], XB[:, j], s=12, alpha=0.28, color=series_color(B), label=pretty_series_label(B))
    ax.set_xlabel(pretty_action_label(action_names[i]))
    ax.set_ylabel(pretty_action_label(action_names[j]))
    ax.set_title('Greedy Actions on the Two Most Separated Action Coordinates')
    ax.legend(frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=700, bbox_inches='tight')
    plt.close(fig)

    return {
        'top2_action_indices': [i, j],
        'top2_action_names': [action_names[i], action_names[j]],
        'mean_abs_diff_all_dims': mean_abs_diff.tolist(),
        'policy_order': [A, B],
    }

# -----------------------------------------------------------------------------
# evaluation summaries
# -----------------------------------------------------------------------------

def summarize_action_shifts(
    action_names: List[str],
    logged_actions: np.ndarray,
    greedy_actions: np.ndarray,
    sampled_actions: np.ndarray,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    logged_mean = np.mean(logged_actions, axis=0)
    greedy_mean = np.mean(greedy_actions, axis=0)
    sampled_mean = np.mean(sampled_actions, axis=0)
    for j, name in enumerate(action_names):
        out[name] = {
            "logged_mean": float(logged_mean[j]),
            "greedy_policy_mean": float(greedy_mean[j]),
            "sampled_policy_mean": float(sampled_mean[j]),
            "delta_greedy_minus_logged": float(greedy_mean[j] - logged_mean[j]),
            "delta_sampled_minus_logged": float(sampled_mean[j] - logged_mean[j]),
        }
    return out


def train_one_reward(
    reward_idx: int,
    reward_name: str,
    split_train: Dict[str, np.ndarray],
    split_test: Optional[Dict[str, np.ndarray]],
    args,
    out_dir: Path,
) -> Dict[str, object]:
    obs_train = split_train['obs']
    act_train = split_train['act']
    rew_train_raw = split_train['rew'][:, reward_idx].reshape(-1)
    terminals_train = split_train['terminals']
    state_names = split_train['state_names']
    action_names = split_train['action_names']

    click_like = reward_suggests_click_like(reward_name)
    rew_train_for_critic = np.asarray(rew_train_raw, dtype=np.float64).copy()
    if click_like:
        rew_train_for_critic = args.click_reward_scale * rew_train_for_critic
        policy_steps = int(args.click_policy_steps if args.click_policy_steps > 0 else args.policy_steps)
        candidate_neighbor_k = int(args.click_candidate_neighbor_k)
        candidate_random_k = int(args.click_candidate_random_k)
    else:
        policy_steps = int(args.policy_steps)
        candidate_neighbor_k = int(args.candidate_neighbor_k)
        candidate_random_k = int(args.candidate_random_k)

    dataset = make_dataset(obs_train, act_train, rew_train_for_critic, terminals_train)
    algo = make_iql_learner(
        device_str=args.device,
        gamma=args.gamma,
        batch_size=args.policy_batch,
        actor_lr=args.iql_actor_lr,
        critic_lr=args.iql_critic_lr,
        value_lr=args.iql_value_lr,
        expectile=args.expectile,
        weight_temp=args.weight_temp,
        max_weight=args.max_weight,
    )
    fit_algo(
        algo,
        dataset,
        n_steps=policy_steps,
        n_steps_per_epoch=args.policy_steps_per_epoch,
        experiment_name=f'iql_{reward_name}',
        quiet=bool(args.quiet_d3rlpy),
    )

    value_path = out_dir / f'iql_value_{reward_name}.d3'
    save_algo(algo, value_path)

    spec = infer_action_spec(action_names, act_train, args.integer_action_col)
    bank_states, bank_actions, bank_info = build_reward_focused_bank(
        obs_train=obs_train,
        act_train=act_train,
        reward_values=rew_train_raw,
        reward_name=reward_name,
        seed=args.seed + 17 * reward_idx,
        standard_pool_size=args.candidate_pool_size,
        click_pool_size=args.click_candidate_pool_size,
        click_min_reward=args.click_bank_min_reward,
        click_quantile=args.click_bank_quantile,
        min_bank_size=args.click_bank_min_size,
    )

    actor_obs = obs_train
    actor_reward = rew_train_raw.copy()
    if args.actor_target_n > 0 and args.actor_target_n < obs_train.shape[0]:
        rng = np.random.default_rng(args.seed + 123 * (reward_idx + 1))
        idx = rng.choice(obs_train.shape[0], size=args.actor_target_n, replace=False)
        idx.sort()
        actor_obs = obs_train[idx]
        actor_reward = rew_train_raw[idx]

    critic_targets = build_critic_targets_best(
        algo=algo,
        obs=actor_obs,
        bank_states=bank_states,
        bank_actions=bank_actions,
        neighbor_k=candidate_neighbor_k,
        random_k=candidate_random_k,
        batch_size=args.target_batch_size,
        seed=args.seed + reward_idx,
    )

    target_best_action = _apply_integer_override(critic_targets['target_best_action'], spec)

    sample_weight = None
    if click_like:
        sample_weight = 1.0 + float(args.click_weight_boost) * np.maximum(actor_reward, 0.0)

    policy, fit_meta = fit_linear_gaussian_policy(
        X=actor_obs,
        Y=target_best_action,
        reward_name=reward_name,
        state_names=state_names,
        action_names=action_names,
        spec=spec,
        ridge=args.linear_ridge,
        min_std=args.min_policy_std,
        max_std=args.max_policy_std,
        sample_weight=sample_weight,
        std_mode=args.policy_std_mode,
        std_ridge=args.policy_std_ridge,
        std_floor=args.policy_std_floor,
        std_blend=args.policy_std_blend,
    )

    policy_npz = out_dir / f'linear_gaussian_policy_{reward_name}.npz'
    policy_json = out_dir / f'linear_gaussian_policy_{reward_name}.json'
    save_linear_gaussian_policy(policy, policy_npz, policy_json)

    eval_obs = split_test['obs'] if split_test is not None else actor_obs[: min(5000, actor_obs.shape[0])]
    eval_logged_actions = split_test['act'] if split_test is not None else act_train[: eval_obs.shape[0]]

    rng = np.random.default_rng(args.seed + 991 * (reward_idx + 1))
    greedy_actions = greedy_action(policy, eval_obs)
    sampled_actions = sample_action(policy, eval_obs, rng=rng)
    mu_eval = mean_action(policy, eval_obs)
    sd_eval = std_action(policy, eval_obs)

    summary = {
        'reward_index': int(reward_idx),
        'reward_name': reward_name,
        'value_model_path': str(value_path),
        'linear_policy_npz': str(policy_npz),
        'linear_policy_json': str(policy_json),
        'state_names': state_names,
        'action_names': action_names,
        'integer_action_name': spec.integer_name,
        'integer_action_index': spec.integer_idx,
        'integer_action_low': spec.integer_low,
        'integer_action_high': spec.integer_high,
        'action_lows': spec.lows.tolist(),
        'action_highs': spec.highs.tolist(),
        'theta_mu': policy.theta_mu.tolist(),
        'epsilon_mu': policy.epsilon_mu.tolist(),
        'theta_sigma': policy.theta_sigma.tolist(),
        'epsilon_sigma': policy.epsilon_sigma.tolist(),
        'target_qmax_mean': float(np.mean(critic_targets['target_qmax'])),
        'target_margin_mean': float(np.mean(critic_targets['target_margin'])),
        'fit': fit_meta,
        'eval_greedy_action_mean': np.mean(greedy_actions, axis=0).tolist(),
        'eval_sampled_action_mean': np.mean(sampled_actions, axis=0).tolist(),
        'eval_mu_mean': np.mean(mu_eval, axis=0).tolist(),
        'eval_sd_mean': np.mean(sd_eval, axis=0).tolist(),
        'value_metrics': {},
        'action_shift_summary': summarize_action_shifts(
            action_names=action_names,
            logged_actions=eval_logged_actions,
            greedy_actions=greedy_actions,
            sampled_actions=sampled_actions,
        ),
        'reward_is_click_like': bool(click_like),
        'critic_reward_scale_used': float(args.click_reward_scale if click_like else 1.0),
        'candidate_neighbor_k_used': int(candidate_neighbor_k),
        'candidate_random_k_used': int(candidate_random_k),
        'bank_info': bank_info,
    }
    (out_dir / f'linear_gaussian_policy_{reward_name}_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# -----------------------------------------------------------------------------
# full post-training evaluation so nothing is missing / broken
# -----------------------------------------------------------------------------

def _eval_split_for_posthoc(split_train: Dict[str, np.ndarray], split_test: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if split_test is not None:
        return split_test
    n = min(5000, split_train["obs"].shape[0])
    return {
        "obs": split_train["obs"][:n],
        "act": split_train["act"][:n],
        "rew": split_train["rew"][:n],
        "terminals": split_train["terminals"][:n],
        "state_names": split_train["state_names"],
        "action_names": split_train["action_names"],
        "reward_names": split_train["reward_names"],
    }


def recompute_full_cross_metrics(
    out_dir: Path,
    reward_names: List[str],
    split_train: Dict[str, np.ndarray],
    split_eval: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    critics = {
        reward_name: load_d3(str(out_dir / f"iql_value_{reward_name}.d3"), device="cpu")
        for reward_name in reward_names
    }
    policies = {
        reward_name: load_linear_gaussian_policy(
            out_dir / f"linear_gaussian_policy_{reward_name}.npz",
            out_dir / f"linear_gaussian_policy_{reward_name}.json",
        )
        for reward_name in reward_names
    }

    X_eval = np.asarray(split_eval["obs"], dtype=np.float64)
    A_eval = np.asarray(split_eval["act"], dtype=np.float64)

    policy_greedy = {name: greedy_action(pol, X_eval) for name, pol in policies.items()}
    rngs = {name: np.random.default_rng(seed + 1207 * (i + 1)) for i, name in enumerate(reward_names)}
    policy_sampled = {name: sample_action(policies[name], X_eval, rng=rngs[name]) for name in reward_names}

    matrix = {}
    for pol_name in reward_names:
        row = {}
        greedy = policy_greedy[pol_name]
        sampled = policy_sampled[pol_name]
        for reward_name in reward_names:
            learner = critics[reward_name]
            row[f"Q_{reward_name}_on_policy_greedy_mean"] = float(predict_values(learner, X_eval, greedy).mean())
            row[f"Q_{reward_name}_on_policy_sampled_mean"] = float(predict_values(learner, X_eval, sampled).mean())
            row[f"Q_{reward_name}_on_data_mean"] = float(predict_values(learner, X_eval, A_eval).mean())
        matrix[pol_name] = row

    return matrix, policy_greedy


# -----------------------------------------------------------------------------
# focus matrix and summaries
# -----------------------------------------------------------------------------

def build_focus_matrix(full_cross_metrics: Dict[str, Dict[str, float]], reward_names: List[str]) -> Dict[str, object]:
    matrix: Dict[str, Dict[str, float]] = {}
    for policy_name in reward_names:
        matrix[policy_name] = {}
        row = full_cross_metrics[policy_name]
        for reward_name in reward_names:
            matrix[policy_name][reward_name] = float(row[f"Q_{reward_name}_on_policy_greedy_mean"])

    dominance = {}
    if len(reward_names) == 2:
        a, b = reward_names
        dominance = {
            f"{a}_focus_gap_on_{a}": float(matrix[a][a] - matrix[b][a]),
            f"{b}_focus_gap_on_{b}": float(matrix[b][b] - matrix[a][b]),
            f"{a}_minus_{b}_on_{b}": float(matrix[a][b] - matrix[b][b]),
            f"{b}_minus_{a}_on_{a}": float(matrix[b][a] - matrix[a][a]),
        }
    return {
        "policy_reward_value_matrix": matrix,
        "dominance_gaps": dominance,
    }


def print_focus_matrix(focus: Dict[str, object]) -> None:
    matrix = focus.get("policy_reward_value_matrix", {})
    gaps = focus.get("dominance_gaps", {})
    if not matrix:
        return
    policy_names = list(matrix.keys())
    reward_names = list(next(iter(matrix.values())).keys()) if policy_names else []

    print("\n" + "=" * 84)
    print("TWO-POLICY DIFFERENCE SUMMARY")
    print("=" * 84)
    print("policy-vs-reward critic matrix (bigger means that policy scores higher under that reward critic):")
    header = ["policy"] + reward_names
    print(" | ".join(f"{h:>24s}" for h in header))
    for p in policy_names:
        row = [p] + [f"{float(matrix[p][r]):.6f}" for r in reward_names]
        print(" | ".join(f"{v:>24s}" for v in row))

    if gaps:
        print("\nFocus gaps:")
        for k, v in gaps.items():
            print(f"  {k:35s} {float(v): .6f}")
    print("=" * 84)


def compute_policy_difference_summary(
    reward_names: List[str],
    policy_greedy: Dict[str, np.ndarray],
    action_names: List[str],
) -> Dict[str, object]:
    if len(reward_names) != 2:
        return {}
    a, b = reward_names
    gA = np.asarray(policy_greedy[a], dtype=np.float64)
    gB = np.asarray(policy_greedy[b], dtype=np.float64)
    delta = gA - gB
    l2 = np.linalg.norm(delta, axis=1)
    per_dim_abs = np.mean(np.abs(delta), axis=0)

    out = {
        "policy_A": a,
        "policy_B": b,
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
        "max_l2": float(np.max(l2)),
        "disagree_rate_maxabs_gt_1e6": float(np.mean(np.max(np.abs(delta), axis=1) > 1e-6)),
        "per_dim_mean_abs_diff": {name: float(val) for name, val in zip(action_names, per_dim_abs)},
        "policy_A_action_mean": {name: float(val) for name, val in zip(action_names, np.mean(gA, axis=0))},
        "policy_B_action_mean": {name: float(val) for name, val in zip(action_names, np.mean(gB, axis=0))},
    }

    print("\nGreedy-action distance on the evaluation split:")
    print(f"  mean ||a_A-a_B||_2        = {out['mean_l2']:.6f}")
    print(f"  median ||a_A-a_B||_2      = {out['median_l2']:.6f}")
    print(f"  max ||a_A-a_B||_2         = {out['max_l2']:.6f}")
    print(f"  disagree rate max|Δ|>1e-6 = {out['disagree_rate_maxabs_gt_1e6']:.6f}")
    print("  per-dimension mean |Δ|:")
    for name, val in out["per_dim_mean_abs_diff"].items():
        print(f"    {name:24s} {val: .6f}")

    return out


def print_artifact_paths(out_dir: Path, reward_names: List[str], plot_paths: Dict[str, str]) -> None:
    print("\nSaved artifact paths:")
    print(f"  {out_dir / 'train_stats_raw.json'}")
    print(f"  {out_dir / 'run_summary.json'}")
    print(f"  {out_dir / 'focus_matrix.json'}")
    print(f"  {out_dir / 'policy_difference_summary.json'}")
    for reward_name in reward_names:
        print(f"  {out_dir / f'iql_value_{reward_name}.d3'}")
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}.npz'}")
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}.json'}")
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}_summary.json'}")
    if plot_paths:
        for _, path in plot_paths.items():
            print(f"  {path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reward-specific plain linear-Gaussian policy fitting from raw behavior data.")

    p.add_argument("--data_base", "--data-base", default="./data")
    p.add_argument("--ckpt_dir", "--ckpt-dir", default="./checkpoints")
    p.add_argument("--train_blob", "--train-blob", default="expedia_train_timeindexed.pt")
    p.add_argument("--test_blob", "--test-blob", default=None)
    p.add_argument("--max_train", "--max-train", type=int, default=300000)
    p.add_argument("--max_test", "--max-test", type=int, default=20000)
    p.add_argument("--episode_length", "--episode-length", type=int, default=None)

    p.add_argument("--state_cols", "--state-cols", default=None)
    p.add_argument("--action_cols", "--action-cols", default=None)
    p.add_argument("--reward_cols", "--reward-cols", default=None)
    p.add_argument("--integer_action_col", "--integer-action-col", default="total_promotions")
    p.add_argument("--reward_indices", "--reward-indices", nargs="*", type=int, default=None)

    p.add_argument("--policy_steps", "--policy-steps", type=int, default=80000)
    p.add_argument("--policy_steps_per_epoch", "--policy-steps-per-epoch", type=int, default=5000)
    p.add_argument("--policy_batch", "--policy-batch", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--iql_actor_lr", "--iql-actor-lr", type=float, default=3e-4)
    p.add_argument("--iql_critic_lr", "--iql-critic-lr", type=float, default=3e-4)
    p.add_argument("--iql_value_lr", "--iql-value-lr", type=float, default=3e-4)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--weight_temp", "--weight-temp", type=float, default=3.0)
    p.add_argument("--max_weight", "--max-weight", type=float, default=100.0)
    p.add_argument("--quiet_d3rlpy", "--quiet-d3rlpy",dest="quiet_d3rlpy",action="store_true",default=True,
                   help="Suppress d3rlpy epoch/progress output. Default: True.")
    p.add_argument("--show_d3rlpy_progress", "--show-d3rlpy-progress",dest="quiet_d3rlpy",action="store_false",
                   help="Show d3rlpy epoch/progress output.")

    p.add_argument("--candidate_pool_size", "--candidate-pool-size", type=int, default=4096)
    p.add_argument("--candidate_neighbor_k", "--candidate-neighbor-k", type=int, default=64)
    p.add_argument("--candidate_random_k", "--candidate-random-k", type=int, default=32)
    p.add_argument("--target_batch_size", "--target-batch-size", type=int, default=256)
    p.add_argument("--actor_target_n", "--actor-target-n", type=int, default=60000)

    p.add_argument("--linear_ridge", "--linear-ridge", type=float, default=1e-3)
    p.add_argument("--min_policy_std", "--min-policy-std", type=float, default=0.05)
    p.add_argument("--max_policy_std", "--max-policy-std", type=float, default=25.0)
    p.add_argument("--policy_std_mode", "--policy-std-mode", type=str, default="heteroskedastic",
                   choices=["constant", "residual_constant", "old", "heteroskedastic", "state_dependent", "linear"])
    p.add_argument("--policy_std_ridge", "--policy-std-ridge", type=float, default=1e-2)
    p.add_argument("--policy_std_floor", "--policy-std-floor", type=float, default=1e-6)
    p.add_argument("--policy_std_blend", "--policy-std-blend", type=float, default=0.85)

    p.add_argument('--click_policy_steps', '--click-policy-steps', type=int, default=20000)
    p.add_argument('--click_candidate_pool_size', '--click-candidate-pool-size', type=int, default=16384)
    p.add_argument('--click_candidate_neighbor_k', '--click-candidate-neighbor-k', type=int, default=128)
    p.add_argument('--click_candidate_random_k', '--click-candidate-random-k', type=int, default=64)
    p.add_argument('--click_bank_min_reward', '--click-bank-min-reward', type=float, default=1.0)
    p.add_argument('--click_bank_quantile', '--click-bank-quantile', type=float, default=0.80)
    p.add_argument('--click_bank_min_size', '--click-bank-min-size', type=int, default=2000)
    p.add_argument('--click_weight_boost', '--click-weight-boost', type=float, default=8.0)
    p.add_argument('--click_reward_scale', '--click-reward-scale', type=float, default=8.0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_plots", "--save-plots", action="store_true")
    return p


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main(args) -> None:
    args.device = resolve_device(args.device)
    set_seeds(args.seed)

    state_cols = parse_csv_list(args.state_cols)
    action_cols = parse_csv_list(args.action_cols)
    reward_cols = parse_csv_list(args.reward_cols)

    train_path = resolve_blob_path(args.train_blob, args.data_base)
    test_path = resolve_blob_path(args.test_blob, args.data_base) if args.test_blob is not None else None
    out_dir = Path(args.ckpt_dir)
    ensure_dir(out_dir)

    split_train = maybe_load_split(
        path=train_path,
        state_cols=state_cols,
        action_cols=action_cols,
        reward_cols=reward_cols,
        max_n=args.max_train,
        seed=args.seed,
        episode_length=args.episode_length,
    )
    split_test = maybe_load_split(
        path=test_path,
        state_cols=state_cols,
        action_cols=action_cols,
        reward_cols=reward_cols,
        max_n=args.max_test,
        seed=args.seed + 1,
        episode_length=args.episode_length,
    ) if test_path is not None else None

    if split_train is None:
        raise ValueError("Training split could not be loaded.")

    print("\nUsing raw arrays exactly as loaded from the blob.")
    print("Train states :", split_train["obs"].shape)
    print("Train actions:", split_train["act"].shape)
    print("Train rewards:", split_train["rew"].shape)
    print("State cols   :", split_train["state_names"])
    print("Action cols  :", split_train["action_names"])
    print("Reward cols  :", split_train["reward_names"])

    if args.reward_indices is None or len(args.reward_indices) == 0:
        reward_indices = list(range(split_train["rew"].shape[1]))
    else:
        reward_indices = list(args.reward_indices)

    bad = [j for j in reward_indices if j < 0 or j >= split_train["rew"].shape[1]]
    if bad:
        raise ValueError(f"Invalid reward indices {bad}. Available range: 0..{split_train['rew'].shape[1] - 1}")

    train_stats = {
        "state_mean": np.mean(split_train["obs"], axis=0).tolist(),
        "state_std": np.std(split_train["obs"], axis=0).tolist(),
        "action_mean": np.mean(split_train["act"], axis=0).tolist(),
        "action_std": np.std(split_train["act"], axis=0).tolist(),
        "reward_mean": np.mean(split_train["rew"], axis=0).tolist(),
        "reward_std": np.std(split_train["rew"], axis=0).tolist(),
        "state_names": split_train["state_names"],
        "action_names": split_train["action_names"],
        "reward_names": split_train["reward_names"],
    }
    (out_dir / "train_stats_raw.json").write_text(json.dumps(train_stats, indent=2))

    results = []
    for ridx in reward_indices:
        reward_name = str(split_train["reward_names"][ridx]).replace("/", "_")
        meta = train_one_reward(
            reward_idx=ridx,
            reward_name=reward_name,
            split_train=split_train,
            split_test=split_test,
            args=args,
            out_dir=out_dir,
        )
        results.append(meta)

    saved_reward_names = [str(split_train["reward_names"][j]).replace("/", "_") for j in reward_indices]
    split_eval = _eval_split_for_posthoc(split_train, split_test)

    full_cross_metrics, policy_greedy = recompute_full_cross_metrics(
        out_dir=out_dir,
        reward_names=saved_reward_names,
        split_train=split_train,
        split_eval=split_eval,
        seed=args.seed,
    )

    # update per-policy summaries with the complete cross-metrics
    for row in results:
        reward_name = row["reward_name"]
        row["value_metrics"] = full_cross_metrics[reward_name]
        summary_path = out_dir / f"linear_gaussian_policy_{reward_name}_summary.json"
        summary_path.write_text(json.dumps(row, indent=2))

    focus = build_focus_matrix(full_cross_metrics=full_cross_metrics, reward_names=saved_reward_names)
    (out_dir / "focus_matrix.json").write_text(json.dumps(focus, indent=2))

    diff_summary = compute_policy_difference_summary(
        reward_names=saved_reward_names,
        policy_greedy=policy_greedy,
        action_names=split_train["action_names"],
    )
    (out_dir / "policy_difference_summary.json").write_text(json.dumps(diff_summary, indent=2))

    plot_paths = {}
    if bool(args.save_plots):
        plots_dir = out_dir / "plots"
        ensure_dir(plots_dir)

        logged_eval_actions = np.asarray(split_eval["act"], dtype=np.float64)
        save_overlay_action_distribution_plots(
            action_names=split_train["action_names"],
            logged_actions=logged_eval_actions,
            greedy_actions_by_policy=policy_greedy,
            integer_idx=infer_action_spec(split_train["action_names"], split_train["act"], args.integer_action_col).integer_idx,
            out_path=plots_dir / "overlay_action_distributions_greedy.png",
        )
        plot_paths["overlay_action_distributions_greedy"] = str(plots_dir / "overlay_action_distributions_greedy.png")

        greedy_means = {name: np.mean(policy_greedy[name], axis=0) for name in saved_reward_names}
        save_overlay_action_mean_profile_plot(
            action_names=split_train["action_names"],
            train_action_mean=np.mean(split_train["act"], axis=0),
            train_action_std=np.std(split_train["act"], axis=0),
            logged_eval_action_mean=np.mean(logged_eval_actions, axis=0),
            greedy_action_means_by_policy=greedy_means,
            out_path=plots_dir / "overlay_action_mean_profile_zscore.png",
        )
        plot_paths["overlay_action_mean_profile_zscore"] = str(plots_dir / "overlay_action_mean_profile_zscore.png")

        scatter_info = save_overlay_top2_scatter_plot(
            action_names=split_train["action_names"],
            greedy_actions_by_policy=policy_greedy,
            out_path=plots_dir / "overlay_top2_action_scatter.png",
        )
        if scatter_info is not None:
            plot_paths["overlay_top2_action_scatter"] = str(plots_dir / "overlay_top2_action_scatter.png")
            (plots_dir / "overlay_top2_action_scatter_info.json").write_text(json.dumps(scatter_info, indent=2))
            plot_paths["overlay_top2_action_scatter_info"] = str(plots_dir / "overlay_top2_action_scatter_info.json")

    run_summary = {
        "seed": args.seed,
        "device": args.device,
        "train_blob": str(train_path),
        "test_blob": str(test_path) if test_path is not None else None,
        "state_names": split_train["state_names"],
        "action_names": split_train["action_names"],
        "reward_names": split_train["reward_names"],
        "results": results,
        "focus_matrix_path": str(out_dir / "focus_matrix.json"),
        "policy_difference_summary_path": str(out_dir / "policy_difference_summary.json"),
        "plot_paths": plot_paths,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    print_focus_matrix(focus)
    print_artifact_paths(out_dir, saved_reward_names, plot_paths)
    print(f"\nFinished. Artifacts saved in: {out_dir}")


# -----------------------------------------------------------------------------
# public helpers
# -----------------------------------------------------------------------------

def get_linear_policy_greedy_actions(policy_npz: str, policy_json: str, states: np.ndarray) -> np.ndarray:
    policy = load_linear_gaussian_policy(policy_npz, policy_json)
    return greedy_action(policy, states)


def sample_linear_policy_actions(policy_npz: str, policy_json: str, states: np.ndarray, seed: int = 123) -> np.ndarray:
    policy = load_linear_gaussian_policy(policy_npz, policy_json)
    rng = np.random.default_rng(seed)
    return sample_action(policy, states, rng=rng)


def get_linear_policy_params(policy_npz: str, policy_json: str) -> Dict[str, np.ndarray]:
    policy = load_linear_gaussian_policy(policy_npz, policy_json)
    return {
        "theta_mu": policy.theta_mu,
        "epsilon_mu": policy.epsilon_mu,
        "theta_sigma": policy.theta_sigma,
        "epsilon_sigma": policy.epsilon_sigma,
        "action_lows": policy.action_lows,
        "action_highs": policy.action_highs,
        "integer_idx": None if policy.integer_idx is None else np.array([policy.integer_idx]),
        "integer_low": None if policy.integer_low is None else np.array([policy.integer_low]),
        "integer_high": None if policy.integer_high is None else np.array([policy.integer_high]),
    }


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
