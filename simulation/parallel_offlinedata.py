from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_workers(n_rep: int, params: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        return max(1, min(n_rep, requested))
    cfg_workers = params.get("offline_data_workers")
    if cfg_workers is not None:
        return max(1, min(n_rep, _as_int(cfg_workers, 1)))
    env_workers = os.environ.get("OFFLINE_DATA_WORKERS")
    if env_workers:
        return max(1, min(n_rep, _as_int(env_workers, 1)))
    slurm_workers = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_workers:
        return max(1, min(n_rep, _as_int(slurm_workers, 1)))
    return max(1, min(n_rep, os.cpu_count() or 1))


def _tail(path: Path, n_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def _run_one(rep_id: int, *, python: str, validate: bool) -> Path:
    env = os.environ.copy()
    env["SLURM_ARRAY_TASK_ID"] = str(rep_id)
    env["OFFLINE_DATA_ID"] = str(rep_id)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    log_path = Path("logs") / f"offline_data_{rep_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    commands = [[python, "main_offlinedata.py"]]
    if validate:
        commands.append([python, "validate_sim_config.py", "--params", "params.yaml", "--data", f"data/offline_data_{rep_id}.pt"])

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"offline replicate {rep_id}\n")
        log.flush()
        for command in commands:
            log.write("RUN: " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"offline replicate {rep_id} failed with exit code {proc.returncode}; "
                    f"see {log_path}"
                )
    return log_path


def run_parallel_offline_data(
    n_rep: int,
    *,
    workers: int,
    python: str | None = None,
    validate: bool = True,
) -> None:
    python = python or sys.executable
    print(f"Parallel offline data generation: n_rep={n_rep}, workers={workers}, validate={validate}", flush=True)
    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, rep_id, python=python, validate=validate): rep_id
            for rep_id in range(n_rep)
        }
        for future in as_completed(futures):
            rep_id = futures[future]
            try:
                log_path = future.result()
                print(f"offline replicate {rep_id} done; log={log_path}", flush=True)
            except BaseException as exc:
                failures.append(exc)
                log_path = Path("logs") / f"offline_data_{rep_id}.log"
                print(str(exc), flush=True)
                tail = _tail(log_path)
                if tail:
                    print(f"--- tail {log_path} ---\n{tail}\n--- end tail ---", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} offline data job(s) failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    n_rep = int((params.get("experiment") or {}).get("num_replicates", 1))
    workers = _default_workers(n_rep, params, args.workers)
    run_parallel_offline_data(n_rep, workers=workers, validate=not args.no_validate)


if __name__ == "__main__":
    main()
