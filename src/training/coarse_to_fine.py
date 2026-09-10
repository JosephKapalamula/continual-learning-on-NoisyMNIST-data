"""Fast coarse-to-fine tuning for NoisyMNIST (Karpathy-style + successive halving).

Strategy (fastest practical on CPU, exact on GPU):
  1. COARSE (cheap fidelity): random/log-uniform sample a wide range on
     short streams (e.g. 200 train / 256 valid). Rank by digit-present
     clean MSE (primary), clean MSE (secondary), stability (finite).
  2. ZOOM (Karpathy): take interval between best neighbours, sample denser
     inside it. E.g. best=1e-2 among [1e-3,1e-2,1e-1] -> fine=[3e-3,6e-3,1e-2,2e-2,3e-2].
  3. SUCCESSIVE HALVING on train-steps as resource: r_min -> r_max with
     factor eta=3, keep top 1/3 each rung. Saves ~70% vs full grid.
  4. FREEZE + multi-seed confirm on fresh streams, report mean+-std.

Why random over grid (Bergstra & Bengio 2012): only 1-2 hyperparams matter;
random covers more distinct values per dimension for same budget.
Why halving (Jamieson & Talwalkar 2016 / Li et al. ASHA 2018): bad configs
reveal themselves early; don't waste 1000 steps on them.

All trials reuse FIXED pre-generated experiences so every candidate sees
identical digits / background / target noise / order.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from src.models import ModelConfig

from .tuning import (
    NetworkIDBDTrialConfig,
    SGDTrialConfig,
    run_network_idbd_trial,
    run_sgd_trial,
)


# ---------------------------------------------------------------- ranking
def rank_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by digit-present clean MSE, then clean MSE. Finite-only on top."""
    work = df.copy()
    work["_finite_rank"] = (~work["finite"].astype(bool)).astype(int)
    return work.sort_values(
        by=["_finite_rank", "digit_present_clean_target_mse", "clean_target_mse"],
        ignore_index=True,
    ).drop(columns=["_finite_rank"])


# ---------------------------------------------------------------- SGD
def sgd_coarse_grid() -> list[float]:
    """Wide log-spaced coarse grid covering 4 orders of magnitude."""
    return [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def karpathy_zoom(best: float, coarse: Sequence[float]) -> list[float]:
    """Take range between best neighbours, sample denser inside.

    Example: coarse=[1e-3,1e-2,1e-1], best=1e-2
      -> neighbours 1e-3..1e-1 -> fine logspace of 5 points.
    Edge best -> extend one half-decade outward.
    """
    coarse_sorted = sorted(coarse)
    idx = min(range(len(coarse_sorted)), key=lambda i: abs(math.log10(coarse_sorted[i]) - math.log10(best)))
    lo = coarse_sorted[idx - 1] if idx > 0 else best / 3.16
    hi = coarse_sorted[idx + 1] if idx < len(coarse_sorted) - 1 else best * 3.16
    return sorted(float(v) for v in np.geomspace(lo, hi, 5))


def tune_sgd_coarse_to_fine(
    model_config: ModelConfig,
    training_experiences,
    validation_experiences,
    initialization_seed: int,
    training_seed: int,
    validation_seed: int,
    coarse: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Run coarse sweep then Karpathy zoom. Returns (coarse_df, fine_df, fine_grid)."""
    coarse = list(coarse) if coarse is not None else sgd_coarse_grid()
    coarse_rows = []
    for lr in coarse:
        cfg = SGDTrialConfig(
            learning_rate=float(lr),
            initialization_seed=initialization_seed,
            training_seed=training_seed,
            validation_seed=validation_seed,
            train_steps=len(training_experiences),
            validation_steps=len(validation_experiences),
        )
        coarse_rows.append(run_sgd_trial(model_config, training_experiences, validation_experiences, cfg))
    coarse_df = rank_trials(pd.DataFrame(coarse_rows))
    best_lr = float(coarse_df.iloc[0]["learning_rate"])
    fine_grid = karpathy_zoom(best_lr, coarse)
    fine_rows = []
    for lr in fine_grid:
        cfg = SGDTrialConfig(
            learning_rate=float(lr),
            initialization_seed=initialization_seed,
            training_seed=training_seed,
            validation_seed=validation_seed,
            train_steps=len(training_experiences),
            validation_steps=len(validation_experiences),
        )
        fine_rows.append(run_sgd_trial(model_config, training_experiences, validation_experiences, cfg))
    fine_df = rank_trials(pd.DataFrame(fine_rows))
    return coarse_df, fine_df, fine_grid


# ------------------------------------------------------- successive halving
def successive_halving_schedule(r_min: int, r_max: int, eta: int = 3) -> list[int]:
    """Rungs e.g. 100 -> 300 -> 900 for r_min=100, r_max=1000, eta=3."""
    rungs = [r_min]
    while rungs[-1] * eta < r_max:
        rungs.append(rungs[-1] * eta)
    if rungs[-1] != r_max:
        rungs.append(r_max)
    return rungs


def tune_sgd_successive_halving(
    model_config: ModelConfig,
    full_training,
    validation_experiences,
    learning_rates: Sequence[float],
    initialization_seed: int,
    training_seed: int,
    validation_seed: int,
    r_min: int = 100,
    eta: int = 3,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Keep top 1/eta configs each rung, growing train prefix. Returns (final, per_rung)."""
    r_max = len(full_training)
    rungs = successive_halving_schedule(r_min, r_max, eta)
    candidates = [float(v) for v in learning_rates]
    per_rung: dict[int, pd.DataFrame] = {}
    for rung in rungs:
        prefix = full_training[:rung]
        rows = []
        for lr in candidates:
            cfg = SGDTrialConfig(
                learning_rate=lr,
                initialization_seed=initialization_seed,
                training_seed=training_seed,
                validation_seed=validation_seed,
                train_steps=rung,
                validation_steps=len(validation_experiences),
            )
            rows.append(run_sgd_trial(model_config, prefix, validation_experiences, cfg))
        df = rank_trials(pd.DataFrame(rows))
        per_rung[rung] = df
        keep = max(1, len(candidates) // eta)
        candidates = [float(v) for v in df.iloc[:keep]["learning_rate"].tolist()]
        if len(candidates) == 1 and rung != rungs[-1]:
            # champion gets remaining rungs alone; still record them
            pass
    return per_rung[rungs[-1]], per_rung


# ------------------------------------------------------- NetworkIDBD random
def sample_network_idbd_configs(
    n: int,
    rng: np.random.Generator,
    initialization_seed: int,
    training_seed: int,
    validation_seed: int,
    train_steps: int,
    validation_steps: int,
) -> list[NetworkIDBDTrialConfig]:
    """Log-uniform random search over the 4 most sensitive dims.

    meta_lr in [1e-2, 3e-1], alpha0 in [1e-8, 1e-5],
    eta in [3e-2, 3e-1], tau in [1e3, 1e5]. decay fixed 0.9995 pilot.
    """
    cfgs = []
    for _ in range(n):
        meta_lr = float(10 ** rng.uniform(-2, math.log10(0.3)))
        alpha0 = float(10 ** rng.uniform(-8, -5))
        eta = float(10 ** rng.uniform(math.log10(0.03), math.log10(0.3)))
        tau = float(10 ** rng.uniform(3, 5))
        cfgs.append(
            NetworkIDBDTrialConfig(
                meta_lr=meta_lr,
                initial_beta=float(math.log(alpha0)),
                decay=0.9995,
                beta_min=float(math.log(1e-8)),
                beta_max=float(math.log(1e4)),
                initialization_seed=initialization_seed,
                training_seed=training_seed,
                validation_seed=validation_seed,
                train_steps=train_steps,
                validation_steps=validation_steps,
                eta=eta,
                tau=tau,
            )
        )
    return cfgs


def tune_network_idbd_random(
    model_config: ModelConfig,
    training_experiences,
    validation_experiences,
    n_configs: int = 8,
    seed: int = 0,
    initialization_seed: int = 20260202,
    training_seed: int = 314159,
    validation_seed: int = 271828,
) -> pd.DataFrame:
    """Random-search n_configs on fixed streams, ranked multi-metric."""
    rng = np.random.default_rng(seed)
    cfgs = sample_network_idbd_configs(
        n, rng, initialization_seed, training_seed, validation_seed,
        len(training_experiences), len(validation_experiences),
    )
    rows = [run_network_idbd_trial(model_config, training_experiences, validation_experiences, c) for c in cfgs]
    # attach sampled alpha0 for readability
    df = pd.DataFrame(rows)
    df["alpha0"] = np.exp(df["initial_beta"].to_numpy())
    return rank_trials(df)


def trial_dict(cfg) -> dict:
    """Readable dict for professor tables."""
    return asdict(cfg)
