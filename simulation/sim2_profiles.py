from __future__ import annotations

from copy import deepcopy
from typing import Any


POLICY_CODE_TO_NAME = {
    "U": "uniform",
    "G": "gaussian",
    "L": "logistic",
}

UG_CENTER = [0.1, -0.1, 0.15, -0.45, 0.0]
UL_CENTER = [0.08, -0.12, 0.16, -0.42, -0.02]
GAUSSIAN_CENTER = [0.12, -0.08, 0.12, -0.35, 0.04]
LOGISTIC_CENTER = [0.05, -0.12, 0.18, -0.4, -0.03]


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def gaussian_policy(theta_mean: list[float], epsilon_mean: float, *, log_std: float) -> dict[str, Any]:
    return {
        "name": "gaussian",
        "theta_mean": theta_mean,
        "theta_std": [0.0, 0.0, 0.0, 0.0, 0.0],
        "epsilon_mean": [epsilon_mean],
        "epsilon_std": [log_std],
    }


def logistic_policy(theta_loc: list[float], epsilon_loc: float, *, log_scale: float) -> dict[str, Any]:
    return {
        "name": "logistic",
        "theta_loc": theta_loc,
        "theta_scale": [0.0, 0.0, 0.0, 0.0, 0.0],
        "epsilon_loc": [epsilon_loc],
        "epsilon_scale": [log_scale],
    }


def uniform_centered_policy(theta_center: list[float], epsilon_center: float, *, half_width: float) -> dict[str, Any]:
    return {
        "name": "uniform",
        "theta_lower": theta_center,
        "theta_upper": theta_center,
        "epsilon_lower": [epsilon_center - half_width],
        "epsilon_upper": [epsilon_center + half_width],
    }


POLICY_PAIR_CONFIGS: dict[str, dict[str, Any]] = {
    "UG": {
        "uniform": uniform_centered_policy(UG_CENTER, 0.025, half_width=0.75),
        "gaussian": gaussian_policy(UG_CENTER, 0.025, log_std=-2.6),
    },
    "UL": {
        "uniform": uniform_centered_policy(UL_CENTER, 0.03, half_width=0.75),
        "logistic": logistic_policy(UL_CENTER, 0.03, log_scale=-2.8),
    },
    "GU": {
        "gaussian": gaussian_policy(GAUSSIAN_CENTER, 0.02, log_std=-1.6),
        "uniform": uniform_centered_policy(GAUSSIAN_CENTER, 0.02, half_width=0.18),
    },
    "GL": {
        "gaussian": gaussian_policy(GAUSSIAN_CENTER, 0.02, log_std=-1.6),
        "logistic": logistic_policy(GAUSSIAN_CENTER, 0.02, log_scale=-2.8),
    },
    "LU": {
        "logistic": logistic_policy(LOGISTIC_CENTER, 0.03, log_scale=-1.8),
        "uniform": uniform_centered_policy(LOGISTIC_CENTER, 0.03, half_width=0.18),
    },
    "LG": {
        "logistic": logistic_policy(LOGISTIC_CENTER, 0.03, log_scale=-1.8),
        "gaussian": gaussian_policy(LOGISTIC_CENTER, 0.03, log_std=-2.7),
    },
}


def supported_policy_pairs() -> list[str]:
    return [
        f"{behavior}{target}"
        for behavior in POLICY_CODE_TO_NAME
        for target in POLICY_CODE_TO_NAME
        if behavior != target
    ]


def validate_policy_pair(pair: str) -> str:
    pair = pair.strip().upper()
    if len(pair) != 2 or any(code not in POLICY_CODE_TO_NAME for code in pair):
        allowed = ", ".join(supported_policy_pairs())
        raise ValueError(f"SIM2_POLICY_PAIR must be one of {allowed}; got {pair!r}.")
    if pair[0] == pair[1]:
        allowed = ", ".join(supported_policy_pairs())
        raise ValueError(f"SIM2_POLICY_PAIR must use different behavior and target policies ({allowed}); got {pair!r}.")
    return pair


def scenario_policies(pair: str) -> tuple[str, str]:
    pair = validate_policy_pair(pair)
    return POLICY_CODE_TO_NAME[pair[0]], POLICY_CODE_TO_NAME[pair[1]]


def scenario_policy_config(pair: str) -> dict[str, Any]:
    pair = validate_policy_pair(pair)
    behavior_policy, target_policy = scenario_policies(pair)
    cfg = {
        "Behvaioral_policy": behavior_policy,
        "evaluation_Target_policy": target_policy,
    }
    cfg.update(deepcopy(POLICY_PAIR_CONFIGS[pair]))
    return cfg


def apply_policy_pair_overrides(
    params: dict[str, Any],
    pair: str,
    *,
    apply_profile: bool = True,
) -> dict[str, Any]:
    """Apply the pair's policy block and optional numeric YAML profile in place."""
    pair = validate_policy_pair(pair)
    behavior_policy, target_policy = scenario_policies(pair)
    params["policy"] = scenario_policy_config(pair)
    profile = deepcopy((params.get("policy_pair_overrides") or {}).get(pair, {}))
    if apply_profile and profile:
        deep_update(params, profile)
    sim2_meta = params.setdefault("simulation_2", {})
    sim2_meta.update(
        {
            "policy_pair": pair,
            "behavior_policy": behavior_policy,
            "target_policy": target_policy,
            "pair_profile_applied": bool(apply_profile and profile),
        }
    )
    return profile
