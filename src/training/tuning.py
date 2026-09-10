"""Leakage-free hyperparameter search helpers for online NoisyMNIST SGD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd

from src.data import Experience
from src.models import ModelConfig, NoisyMNISTMLP
from src.optim import NetworkIDBD
from src.utils import seed_everything

from .online_sgd import (
    LossConvention,
    evaluate_model,
    make_vanilla_sgd,
    online_network_idbd_update,
    online_sgd_update,
)


@dataclass(frozen=True)
class SGDTrialConfig:
    """One SGD trial configuration."""

    learning_rate: float
    initialization_seed: int
    training_seed: int
    validation_seed: int
    train_steps: int
    validation_steps: int


@dataclass(frozen=True)
class NetworkIDBDTrialConfig:
    """One experimental neural-IDBD trial configuration."""

    meta_lr: float
    initial_beta: float
    decay: float
    beta_min: float
    beta_max: float
    initialization_seed: int
    training_seed: int
    validation_seed: int
    train_steps: int
    validation_steps: int
    eta: float = 0.1
    tau: float = 1e4


def run_sgd_trial(
    model_config: ModelConfig,
    training_experiences: Sequence[Experience],
    validation_experiences: Sequence[Experience],
    trial_config: SGDTrialConfig,
) -> dict[str, float | int | bool]:
    """Run one deterministic trial on fixed, already-generated experiences."""

    if trial_config.train_steps != len(training_experiences):
        raise ValueError("train_steps must equal the training experience count.")
    if trial_config.validation_steps != len(validation_experiences):
        raise ValueError("validation_steps must equal the validation experience count.")

    seed_everything(trial_config.initialization_seed)
    model = NoisyMNISTMLP(model_config)
    optimizer = make_vanilla_sgd(model, trial_config.learning_rate)

    for experience in training_experiences:
        online_sgd_update(
            model,
            optimizer,
            experience,
            LossConvention.MEAN_SQUARED_ERROR,
            compute_gradient_norm=False,
            check_parameter_finiteness=True,
        )

    metrics = evaluate_model(
        model,
        validation_experiences,
        trial_config.validation_steps,
        expected_pool_name="validation",
    )
    return {
        **asdict(trial_config),
        "noisy_target_mse": metrics.noisy_target_mse,
        "clean_target_mse": metrics.clean_target_mse,
        "digit_present_count": metrics.digit_present_count,
        "digit_present_clean_target_mse": metrics.digit_present_clean_target_mse,
        "digit_absent_clean_target_mse": metrics.digit_absent_clean_target_mse,
        "mean_prediction": metrics.mean_prediction,
        "finite": metrics.finite,
    }


def tune_sgd_learning_rates(
    model_config: ModelConfig,
    training_experiences: Sequence[Experience],
    validation_experiences: Sequence[Experience],
    learning_rates: Sequence[float],
    initialization_seed: int,
    training_seed: int,
    validation_seed: int,
) -> pd.DataFrame:
    """Evaluate learning rates against one paired validation protocol."""

    if not learning_rates:
        raise ValueError("learning_rates must contain at least one value.")
    trial_rows = []
    for learning_rate in learning_rates:
        trial_config = SGDTrialConfig(
            learning_rate=float(learning_rate),
            initialization_seed=initialization_seed,
            training_seed=training_seed,
            validation_seed=validation_seed,
            train_steps=len(training_experiences),
            validation_steps=len(validation_experiences),
        )
        trial_rows.append(
            run_sgd_trial(
                model_config,
                training_experiences,
                validation_experiences,
                trial_config,
            )
        )
    return pd.DataFrame(trial_rows).sort_values(
        by=["digit_present_clean_target_mse", "clean_target_mse"],
        ignore_index=True,
    )


def run_network_idbd_trial(
    model_config: ModelConfig,
    training_experiences: Sequence[Experience],
    validation_experiences: Sequence[Experience],
    trial_config: NetworkIDBDTrialConfig,
) -> dict[str, float | int | bool]:
    """Run one deterministic trial of the experimental neural-IDBD candidate."""

    if trial_config.train_steps != len(training_experiences):
        raise ValueError("train_steps must equal the training experience count.")
    if trial_config.validation_steps != len(validation_experiences):
        raise ValueError("validation_steps must equal the validation experience count.")

    seed_everything(trial_config.initialization_seed)
    model = NoisyMNISTMLP(model_config)
    optimizer = NetworkIDBD(
        model.parameters(),
        meta_lr=trial_config.meta_lr,
        initial_beta=trial_config.initial_beta,
        decay=trial_config.decay,
        beta_min=trial_config.beta_min,
        beta_max=trial_config.beta_max,
        eta=trial_config.eta,
        tau=trial_config.tau,
    )

    for experience in training_experiences:
        online_network_idbd_update(
            model,
            optimizer,
            experience,
            LossConvention.MEAN_SQUARED_ERROR,
            check_parameter_finiteness=True,
        )

    metrics = evaluate_model(
        model,
        validation_experiences,
        trial_config.validation_steps,
        expected_pool_name="validation",
    )
    return {
        **asdict(trial_config),
        "noisy_target_mse": metrics.noisy_target_mse,
        "clean_target_mse": metrics.clean_target_mse,
        "digit_present_count": metrics.digit_present_count,
        "digit_present_clean_target_mse": metrics.digit_present_clean_target_mse,
        "digit_absent_clean_target_mse": metrics.digit_absent_clean_target_mse,
        "mean_prediction": metrics.mean_prediction,
        "finite": metrics.finite,
    }


def tune_network_idbd(
    model_config: ModelConfig,
    training_experiences: Sequence[Experience],
    validation_experiences: Sequence[Experience],
    trial_configs: Sequence[NetworkIDBDTrialConfig],
) -> pd.DataFrame:
    """Evaluate experimental neural-IDBD configurations on paired streams."""

    if not trial_configs:
        raise ValueError("trial_configs must contain at least one configuration.")
    rows = [
        run_network_idbd_trial(
            model_config,
            training_experiences,
            validation_experiences,
            trial_config,
        )
        for trial_config in trial_configs
    ]
    return pd.DataFrame(rows).sort_values(
        by=["digit_present_clean_target_mse", "clean_target_mse"],
        ignore_index=True,
    )