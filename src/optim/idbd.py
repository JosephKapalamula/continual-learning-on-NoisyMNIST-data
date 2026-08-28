"""Canonical linear Incremental Delta-Bar-Delta (IDBD).

This module implements the linear squared-error algorithm in Sutton (1992),
also restated as Algorithm 4 by Degris et al. (2024). It is deliberately not a
neural-network optimizer and must not be presented as Oak's unpublished
neural method.

Primary sources:
    Sutton (1992): https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf
    Degris et al. (2024): https://arxiv.org/html/2401.17401v1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]
_MAX_SAFE_DELTA_FOR_SQUARING = math.sqrt(np.finfo(np.float64).max)


@dataclass(frozen=True)
class IDBDConfig:
    """Configuration for canonical linear IDBD.

    ``initial_step_size`` and ``meta_step_size`` are experiment protocol
    values. Sutton's optional beta floor and beta-increment clipping are not
    applied: they were described as practical safeguards, not as part of the
    empirical canonical update used in the paper.
    """

    num_features: int
    initial_step_size: float
    meta_step_size: float
    dtype: np.dtype[Any] = np.dtype(np.float64)

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        if isinstance(self.num_features, bool) or not isinstance(
            self.num_features, (int, np.integer)
        ):
            raise TypeError("num_features must be an integer.")
        object.__setattr__(self, "num_features", int(self.num_features))
        if self.num_features <= 0:
            raise ValueError("num_features must be positive.")
        if not math.isfinite(self.initial_step_size) or self.initial_step_size <= 0:
            raise ValueError("initial_step_size must be positive and finite.")
        if not math.isfinite(self.meta_step_size) or self.meta_step_size <= 0:
            raise ValueError("meta_step_size must be positive and finite.")
        if dtype.kind != "f":
            raise TypeError("IDBD state requires a floating-point NumPy dtype.")


@dataclass(frozen=True)
class LinearSGDConfig:
    """Configuration for the paired vanilla linear-SGD reference."""

    num_features: int
    step_size: float
    dtype: np.dtype[Any] = np.dtype(np.float64)

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        if isinstance(self.num_features, bool) or not isinstance(
            self.num_features, (int, np.integer)
        ):
            raise TypeError("num_features must be an integer.")
        object.__setattr__(self, "num_features", int(self.num_features))
        if self.num_features <= 0:
            raise ValueError("num_features must be positive.")
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be positive and finite.")
        if dtype.kind != "f":
            raise TypeError("Linear SGD state requires a floating-point NumPy dtype.")


@dataclass(frozen=True)
class LinearUpdate:
    """Scalar diagnostics from one prequential predict-then-update step."""

    step: int
    prediction: float
    target: float
    delta: float
    squared_error: float
    weight_change_l2: float


def _validate_input(x: ArrayLike, num_features: int) -> NDArray[Any]:
    """Return a finite one-dimensional numeric view without needless casting."""

    x_array = np.asarray(x)
    if x_array.shape != (num_features,):
        raise ValueError(
            f"Expected x shape ({num_features},), received {x_array.shape}."
        )
    if x_array.dtype.kind not in "bufi":
        raise TypeError("x must contain boolean, integer, or floating-point values.")
    if x_array.dtype.kind == "f" and not bool(np.isfinite(x_array).all()):
        raise FloatingPointError("x contains a non-finite value.")
    return x_array


def _validate_target(target: float | np.floating[Any]) -> float:
    """Return a finite scalar target."""

    target_value = float(target)
    if not math.isfinite(target_value):
        raise FloatingPointError("target must be finite.")
    return target_value


def _validate_active_indices(
    active_indices: ArrayLike,
    num_features: int,
) -> NDArray[np.intp]:
    """Validate sorted unique indices for an exact binary sparse update."""

    indices = np.asarray(active_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise TypeError("active_indices must be a one-dimensional integer array.")
    indices = indices.astype(np.intp, copy=False)
    if indices.size:
        if indices[0] < 0 or indices[-1] >= num_features:
            raise IndexError("active_indices contains an out-of-range coordinate.")
        if indices.size > 1 and not bool(np.all(np.diff(indices) > 0)):
            raise ValueError("active_indices must be sorted and unique.")
    return indices


class CanonicalLinearIDBD:
    """Canonical per-feature IDBD for one linear prediction unit.

    Update ordering is part of the algorithmic contract:

    1. prediction and delta use old weights;
    2. new beta uses old h;
    3. new alpha is exp(new beta);
    4. new weights and h use that new alpha;
    5. the h recurrence uses old h on its right-hand side.

    Coordinates where ``x_i == 0`` are skipped as an exact optimization: all
    four canonical updates leave those coordinates unchanged.
    """

    def __init__(
        self,
        config: IDBDConfig,
        initial_weights: Optional[ArrayLike] = None,
    ) -> None:
        self.config = config
        if initial_weights is None:
            initial = np.zeros(config.num_features, dtype=config.dtype)
        else:
            initial = np.asarray(initial_weights, dtype=config.dtype)
            if initial.shape != (config.num_features,):
                raise ValueError("initial_weights has the wrong shape.")
            if not bool(np.isfinite(initial).all()):
                raise FloatingPointError("initial_weights contains a non-finite value.")
            initial = initial.copy()
        self._initial_weights = initial
        self.reset()

    def reset(self) -> None:
        """Restore initial weights, uniform alpha, zero trace, and step zero."""

        self.weights = self._initial_weights.copy()
        initial_beta = math.log(self.config.initial_step_size)
        self.beta = np.full(
            self.config.num_features,
            initial_beta,
            dtype=self.config.dtype,
        )
        # Construct alpha from beta so the canonical identity alpha=exp(beta)
        # is exact in the stored dtype, including after checkpoint restoration.
        self.alpha = np.exp(self.beta)
        if not bool(np.all(np.isfinite(self.alpha))) or not bool(
            np.all(self.alpha > 0.0)
        ):
            raise FloatingPointError(
                "initial_step_size is not representable as a finite positive alpha "
                "in the configured dtype."
            )
        self.h = np.zeros(self.config.num_features, dtype=self.config.dtype)
        self.step_index = 0

    def predict(self, x: ArrayLike) -> float:
        """Return ``w.T @ x`` without changing state."""

        x_array = _validate_input(x, self.config.num_features)
        prediction = float(np.dot(self.weights, x_array))
        if not math.isfinite(prediction):
            raise FloatingPointError("Prediction is non-finite.")
        return prediction

    def step(self, x: ArrayLike, target: float | np.floating[Any]) -> LinearUpdate:
        """Predict, then apply one exact canonical IDBD update."""

        x_array = _validate_input(x, self.config.num_features)
        active_indices = np.flatnonzero(x_array)
        active_x = x_array[active_indices].astype(self.config.dtype, copy=False)
        return self._step_active_values(active_indices, active_x, target)

    def step_binary_active(
        self,
        active_indices: ArrayLike,
        target: float | np.floating[Any],
    ) -> LinearUpdate:
        """Apply the same update from active indices of a binary feature vector.

        This is an algebraically exact performance path, not an approximation.
        Every omitted coordinate has ``x_i=0`` and is unchanged by all canonical
        recurrences.
        """

        indices = _validate_active_indices(active_indices, self.config.num_features)
        active_x = np.ones(indices.size, dtype=self.config.dtype)
        return self._step_active_values(indices, active_x, target)

    def _step_active_values(
        self,
        active_indices: NDArray[np.intp],
        active_x: FloatArray,
        target: float | np.floating[Any],
    ) -> LinearUpdate:
        """Commit one update from validated nonzero coordinates and values."""

        target_value = _validate_target(target)
        prediction = float(np.dot(self.weights[active_indices], active_x))
        delta = target_value - prediction
        if not math.isfinite(delta):
            raise FloatingPointError("Prediction error is non-finite.")
        if abs(delta) > _MAX_SAFE_DELTA_FOR_SQUARING:
            raise FloatingPointError("Squared prediction error would overflow.")
        squared_error = delta * delta

        weight_change_l2 = 0.0
        if active_indices.size:
            old_h = self.h[active_indices]
            old_weights = self.weights[active_indices]

            with np.errstate(over="raise", invalid="raise"):
                beta_next = (
                    self.beta[active_indices]
                    + self.config.meta_step_size * delta * active_x * old_h
                )
                alpha_next = np.exp(beta_next)
                weight_change = alpha_next * delta * active_x
                weights_next = old_weights + weight_change
                trace_decay = np.maximum(
                    1.0 - alpha_next * np.square(active_x),
                    0.0,
                )
                h_next = old_h * trace_decay + weight_change

            if not all(
                bool(np.isfinite(values).all())
                for values in (beta_next, alpha_next, weights_next, h_next)
            ):
                raise FloatingPointError("IDBD produced non-finite adaptive state.")
            if not bool(np.all(alpha_next > 0.0)):
                raise FloatingPointError("IDBD alpha underflowed to a non-positive value.")
            weight_change_l2 = float(np.linalg.vector_norm(weight_change))
            if not math.isfinite(weight_change_l2):
                raise FloatingPointError("IDBD weight-change norm overflowed.")

            # Commit only after every next-state tensor has been validated.
            self.beta[active_indices] = beta_next
            self.alpha[active_indices] = alpha_next
            self.weights[active_indices] = weights_next
            self.h[active_indices] = h_next

        metrics = LinearUpdate(
            step=self.step_index,
            prediction=prediction,
            target=target_value,
            delta=delta,
            squared_error=squared_error,
            weight_change_l2=weight_change_l2,
        )
        self.step_index += 1
        return metrics

    def state_is_finite(self) -> bool:
        """Return whether all weights and adaptive state values are finite."""

        return bool(np.all(self.alpha > 0.0)) and all(
            bool(np.isfinite(values).all())
            for values in (self.weights, self.beta, self.alpha, self.h)
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a deep copy of exact optimizer state."""

        return {
            "initial_weights": self._initial_weights.copy(),
            "weights": self.weights.copy(),
            "beta": self.beta.copy(),
            "alpha": self.alpha.copy(),
            "h": self.h.copy(),
            "step_index": self.step_index,
            "num_features": self.config.num_features,
            "initial_step_size": self.config.initial_step_size,
            "meta_step_size": self.config.meta_step_size,
            "dtype": self.config.dtype.str,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state after validating dimensions, dtype, and alpha/beta consistency."""

        raw_num_features = state["num_features"]
        if isinstance(raw_num_features, bool) or not isinstance(
            raw_num_features, (int, np.integer)
        ):
            raise TypeError("State num_features must be an integer.")
        if int(raw_num_features) != self.config.num_features:
            raise ValueError("State num_features does not match this IDBD instance.")
        if float(state["initial_step_size"]) != self.config.initial_step_size:
            raise ValueError("State initial_step_size does not match this IDBD instance.")
        if float(state["meta_step_size"]) != self.config.meta_step_size:
            raise ValueError("State meta_step_size does not match this IDBD instance.")
        if np.dtype(state["dtype"]) != self.config.dtype:
            raise TypeError("State dtype does not match this IDBD instance.")

        restored: dict[str, FloatArray] = {}
        for name in ("initial_weights", "weights", "beta", "alpha", "h"):
            values = np.asarray(state[name])
            if values.shape != (self.config.num_features,):
                raise ValueError(f"State array {name!r} has the wrong shape.")
            if values.dtype != self.config.dtype:
                raise TypeError(f"State array {name!r} has the wrong dtype.")
            if not bool(np.isfinite(values).all()):
                raise FloatingPointError(f"State array {name!r} is non-finite.")
            restored[name] = values.copy()

        expected_alpha = np.exp(restored["beta"])
        if not np.array_equal(expected_alpha, restored["alpha"]):
            raise ValueError("Restored alpha must equal exp(beta) exactly.")
        if not bool(np.all(restored["alpha"] > 0.0)):
            raise ValueError("Restored alpha must be strictly positive.")
        raw_step_index = state["step_index"]
        if isinstance(raw_step_index, bool) or not isinstance(
            raw_step_index, (int, np.integer)
        ):
            raise TypeError("step_index must be an integer.")
        step_index = int(raw_step_index)
        if step_index < 0:
            raise ValueError("step_index cannot be negative.")

        self._initial_weights = restored["initial_weights"]
        self.weights = restored["weights"]
        self.beta = restored["beta"]
        self.alpha = restored["alpha"]
        self.h = restored["h"]
        self.step_index = step_index


class VanillaLinearSGD:
    """Minimal online linear SGD reference using the canonical delta convention."""

    def __init__(
        self,
        config: LinearSGDConfig,
        initial_weights: Optional[ArrayLike] = None,
    ) -> None:
        self.config = config
        if initial_weights is None:
            initial = np.zeros(config.num_features, dtype=config.dtype)
        else:
            initial = np.asarray(initial_weights, dtype=config.dtype)
            if initial.shape != (config.num_features,):
                raise ValueError("initial_weights has the wrong shape.")
            if not bool(np.isfinite(initial).all()):
                raise FloatingPointError("initial_weights contains a non-finite value.")
            initial = initial.copy()
        self._initial_weights = initial
        self.reset()

    def reset(self) -> None:
        """Restore initial weights and step zero."""

        self.weights = self._initial_weights.copy()
        self.step_index = 0

    def predict(self, x: ArrayLike) -> float:
        """Return ``w.T @ x`` without changing state."""

        x_array = _validate_input(x, self.config.num_features)
        prediction = float(np.dot(self.weights, x_array))
        if not math.isfinite(prediction):
            raise FloatingPointError("Prediction is non-finite.")
        return prediction

    def step(self, x: ArrayLike, target: float | np.floating[Any]) -> LinearUpdate:
        """Predict, then apply one immediate delta-rule update."""

        x_array = _validate_input(x, self.config.num_features)
        active_indices = np.flatnonzero(x_array)
        active_x = x_array[active_indices].astype(self.config.dtype, copy=False)
        return self._step_active_values(active_indices, active_x, target)

    def step_binary_active(
        self,
        active_indices: ArrayLike,
        target: float | np.floating[Any],
    ) -> LinearUpdate:
        """Apply the exact binary-feature update from sorted active indices."""

        indices = _validate_active_indices(active_indices, self.config.num_features)
        active_x = np.ones(indices.size, dtype=self.config.dtype)
        return self._step_active_values(indices, active_x, target)

    def _step_active_values(
        self,
        active_indices: NDArray[np.intp],
        active_x: FloatArray,
        target: float | np.floating[Any],
    ) -> LinearUpdate:
        """Commit one SGD update from validated nonzero coordinates and values."""

        target_value = _validate_target(target)
        prediction = float(np.dot(self.weights[active_indices], active_x))
        delta = target_value - prediction
        if not math.isfinite(delta):
            raise FloatingPointError("Prediction error is non-finite.")
        if abs(delta) > _MAX_SAFE_DELTA_FOR_SQUARING:
            raise FloatingPointError("Squared prediction error would overflow.")
        squared_error = delta * delta

        weight_change_l2 = 0.0
        if active_indices.size:
            weight_change = self.config.step_size * delta * active_x
            weights_next = self.weights[active_indices] + weight_change
            if not bool(np.isfinite(weights_next).all()):
                raise FloatingPointError("Linear SGD produced non-finite weights.")
            weight_change_l2 = float(np.linalg.vector_norm(weight_change))
            if not math.isfinite(weight_change_l2):
                raise FloatingPointError("Linear SGD weight-change norm overflowed.")
            self.weights[active_indices] = weights_next

        metrics = LinearUpdate(
            step=self.step_index,
            prediction=prediction,
            target=target_value,
            delta=delta,
            squared_error=squared_error,
            weight_change_l2=weight_change_l2,
        )
        self.step_index += 1
        return metrics

    def state_is_finite(self) -> bool:
        """Return whether all weights are finite."""

        return bool(np.isfinite(self.weights).all())
