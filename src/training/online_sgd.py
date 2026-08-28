"""Transparent batch-size-one SGD for the NoisyMNIST reproduction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Iterable

import torch
from torch import Tensor, nn

from src.data import Experience
from src.models import NoisyMNISTMLP


class LossConvention(str, Enum):
    """Explicit scalar squared-error conventions.

    The official note uses the phrase "sample mean squared prediction error" in
    its linear example but does not publish an explicit NoisyMNIST loss equation.
    Keeping the convention explicit prevents an unnoticed factor-of-two change.
    """

    MEAN_SQUARED_ERROR = "mean_squared_error"
    HALF_SQUARED_ERROR = "half_squared_error"


@dataclass(frozen=True)
class OnlineStepMetrics:
    """Detached scalar diagnostics from one immediate online update."""

    stream_step: int
    prediction: float
    noisy_target: float
    clean_target: float
    loss: float
    noisy_squared_error: float
    clean_squared_error: float
    hidden_active_fraction: float
    gradient_l2_norm: float | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Mean metrics from a fixed finite stream without parameter updates."""

    num_steps: int
    noisy_target_mse: float
    clean_target_mse: float
    mean_prediction: float
    finite: bool


def prediction_loss(
    prediction: Tensor,
    target: Tensor,
    convention: LossConvention,
) -> Tensor:
    """Return an explicit scalar prediction loss."""

    if prediction.shape != () or target.shape != ():
        raise ValueError("Online prediction and target must both be scalar tensors.")
    squared_error = (prediction - target).square()
    if convention is LossConvention.MEAN_SQUARED_ERROR:
        return squared_error
    if convention is LossConvention.HALF_SQUARED_ERROR:
        return 0.5 * squared_error
    raise ValueError(f"Unsupported loss convention: {convention}")


def make_vanilla_sgd(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.SGD:
    """Create source-compatible SGD with no momentum, decay, or foreach state."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=False,
    )


def _gradient_l2_norm(model: nn.Module) -> float:
    """Compute a finite global gradient norm without flattening parameters."""

    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter_norm = float(torch.linalg.vector_norm(parameter.grad).detach().cpu())
            squared_norm += parameter_norm**2
    return math.sqrt(squared_norm)


def online_sgd_update(
    model: NoisyMNISTMLP,
    optimizer: torch.optim.SGD,
    experience: Experience,
    loss_convention: LossConvention,
    compute_gradient_norm: bool = False,
    check_parameter_finiteness: bool = False,
) -> OnlineStepMetrics:
    """Consume one experience and apply exactly one immediate SGD update."""

    x_flat = experience["x_flat"]
    noisy_target = experience["noisy_target"]
    clean_target = experience["clean_target"]
    if x_flat.shape != (model.config.input_size,):
        raise ValueError("Expected one flattened NoisyMNIST experience.")
    if experience["source_partition"] != "official_train":
        raise ValueError("Online development updates must not consume official test data.")
    if experience["pool_name"] != "training":
        raise ValueError("Weight updates must consume the isolated training pool only.")

    optimizer.zero_grad(set_to_none=True)
    forward_result = model.forward_with_activations(x_flat)
    loss = prediction_loss(forward_result.prediction, noisy_target, loss_convention)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss before SGD update.")
    loss.backward()

    gradient_norm = _gradient_l2_norm(model) if compute_gradient_norm else None
    if gradient_norm is not None and not math.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite gradient norm.")

    with torch.no_grad():
        prediction_value = float(forward_result.prediction.detach().cpu())
        noisy_target_value = float(noisy_target.detach().cpu())
        clean_target_value = float(clean_target.detach().cpu())
        hidden_active_fraction = float(
            (forward_result.hidden_activations.detach() > 0)
            .to(dtype=torch.float32)
            .mean()
            .cpu()
        )

    optimizer.step()
    if check_parameter_finiteness:
        for parameter in model.parameters():
            parameter_norm = torch.linalg.vector_norm(parameter.detach())
            if not bool(torch.isfinite(parameter_norm)):
                raise FloatingPointError("Non-finite parameter after SGD update.")

    noisy_squared_error = (prediction_value - noisy_target_value) ** 2
    clean_squared_error = (prediction_value - clean_target_value) ** 2
    return OnlineStepMetrics(
        stream_step=experience["stream_step"],
        prediction=prediction_value,
        noisy_target=noisy_target_value,
        clean_target=clean_target_value,
        loss=float(loss.detach().cpu()),
        noisy_squared_error=noisy_squared_error,
        clean_squared_error=clean_squared_error,
        hidden_active_fraction=hidden_active_fraction,
        gradient_l2_norm=gradient_norm,
    )


def evaluate_model(
    model: NoisyMNISTMLP,
    experiences: Iterable[Experience],
    num_steps: int,
    expected_pool_name: str,
) -> EvaluationMetrics:
    """Evaluate a fixed finite stream without gradients or weight updates."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    was_training = model.training
    noisy_error_sum = 0.0
    clean_error_sum = 0.0
    prediction_sum = 0.0
    finite = True
    observed_steps = 0

    model.eval()
    try:
        with torch.inference_mode():
            # islice prevents a longer/indefinite iterator from being advanced by
            # one unmeasured experience at the evaluation boundary.
            for experience in islice(experiences, num_steps):
                if experience["source_partition"] != "official_train":
                    raise ValueError(
                        "Pre-freeze evaluation must not consume official test data."
                    )
                if experience["pool_name"] != expected_pool_name:
                    raise ValueError(
                        f"Expected evaluation pool {expected_pool_name!r}, "
                        f"received {experience['pool_name']!r}."
                    )
                prediction = model(experience["x_flat"])
                if prediction.shape != ():
                    raise ValueError(
                        "Expected one scalar prediction per evaluation experience."
                    )
                prediction_value = float(prediction.cpu())
                noisy_target_value = float(experience["noisy_target"].cpu())
                clean_target_value = float(experience["clean_target"].cpu())
                noisy_error_sum += (prediction_value - noisy_target_value) ** 2
                clean_error_sum += (prediction_value - clean_target_value) ** 2
                prediction_sum += prediction_value
                finite = finite and math.isfinite(prediction_value)
                observed_steps += 1
    finally:
        model.train(was_training)
    if observed_steps != num_steps:
        raise ValueError(
            f"Expected {num_steps} evaluation experiences, received {observed_steps}."
        )
    return EvaluationMetrics(
        num_steps=observed_steps,
        noisy_target_mse=noisy_error_sum / observed_steps,
        clean_target_mse=clean_error_sum / observed_steps,
        mean_prediction=prediction_sum / observed_steps,
        finite=finite,
    )
