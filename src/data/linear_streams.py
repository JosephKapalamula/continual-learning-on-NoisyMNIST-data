"""Deterministic online streams for canonical linear IDBD verification."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LinearStreamConfig:
    """Configuration for one useful and many irrelevant Bernoulli features."""

    num_features: int
    useful_probability: float
    irrelevant_probability: float
    useful_weight: float
    target_noise_variance: float
    seed: int
    chunk_size: int = 512
    dtype: np.dtype[Any] = np.dtype(np.float64)

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        if isinstance(self.num_features, bool) or not isinstance(
            self.num_features, (int, np.integer)
        ):
            raise TypeError("num_features must be an integer.")
        object.__setattr__(self, "num_features", int(self.num_features))
        if self.num_features < 2:
            raise ValueError("The verification stream needs at least two features.")
        for name, value in (
            ("useful_probability", self.useful_probability),
            ("irrelevant_probability", self.irrelevant_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not math.isfinite(self.useful_weight):
            raise ValueError("useful_weight must be finite.")
        if self.target_noise_variance < 0.0 or not math.isfinite(
            self.target_noise_variance
        ):
            raise ValueError("target_noise_variance must be non-negative and finite.")
        if isinstance(self.chunk_size, bool) or not isinstance(
            self.chunk_size, (int, np.integer)
        ):
            raise TypeError("chunk_size must be an integer.")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")
        if dtype.kind != "f":
            raise TypeError("Stream targets require a floating-point NumPy dtype.")

    @property
    def target_noise_std(self) -> float:
        """Standard deviation corresponding to the configured variance."""

        return math.sqrt(self.target_noise_variance)


@dataclass(frozen=True)
class LinearExperience:
    """One online linear-prediction experience."""

    x: NDArray[np.bool_]
    clean_target: float
    noisy_target: float
    gaussian_noise: float
    stream_step: int


class BernoulliLinearStream(Iterator[LinearExperience]):
    """Private-RNG, resettable stream with bounded chunked generation.

    Feature zero is useful. Every remaining feature is independently sampled
    and independent of the target. Chunking reduces RNG call overhead but does
    not change the online predict-then-update protocol.
    """

    def __init__(self, config: LinearStreamConfig) -> None:
        self.config = config
        self._constructor_seed = int(config.seed)
        self.reset()

    @property
    def stream_step(self) -> int:
        """Zero-based index of the next experience."""

        return self._stream_step

    def reset(self, seed: int | None = None) -> None:
        """Reset the private generator and discard any buffered experiences."""

        if seed is None:
            active_seed = self._constructor_seed
        else:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise TypeError("reset seed must be an integer.")
            active_seed = int(seed)
            if active_seed < 0:
                raise ValueError("reset seed must be non-negative.")
        self._active_seed = active_seed
        feature_seed, noise_seed = np.random.SeedSequence(active_seed).spawn(2)
        self._feature_rng = np.random.default_rng(feature_seed)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._stream_step = 0
        self._buffer_position = 0
        self._feature_buffer = np.empty((0, self.config.num_features), dtype=np.bool_)
        self._noise_buffer = np.empty(0, dtype=self.config.dtype)

    def __iter__(self) -> "BernoulliLinearStream":
        return self

    def _fill_buffer(self) -> None:
        cfg = self.config
        features = self._feature_rng.random((cfg.chunk_size, cfg.num_features))
        features = features < cfg.irrelevant_probability
        features[:, 0] = (
            self._feature_rng.random(cfg.chunk_size) < cfg.useful_probability
        )
        noise = self._noise_rng.normal(
            loc=0.0,
            scale=cfg.target_noise_std,
            size=cfg.chunk_size,
        ).astype(cfg.dtype, copy=False)
        self._feature_buffer = features
        self._noise_buffer = noise
        self._buffer_position = 0

    def __next__(self) -> LinearExperience:
        if self._buffer_position >= len(self._noise_buffer):
            self._fill_buffer()

        position = self._buffer_position
        x = self._feature_buffer[position]
        gaussian_noise = float(self._noise_buffer[position])
        clean_target = self.config.useful_weight * float(x[0])
        noisy_target = clean_target + gaussian_noise
        experience = LinearExperience(
            # Never expose a mutable view into the stream's checkpoint buffer.
            x=x.copy(),
            clean_target=clean_target,
            noisy_target=noisy_target,
            gaussian_noise=gaussian_noise,
            stream_step=self._stream_step,
        )
        self._buffer_position += 1
        self._stream_step += 1
        return experience

    def state_dict(self) -> dict[str, Any]:
        """Return an exact checkpoint, including unused buffered experiences."""

        return {
            "constructor_seed": self._constructor_seed,
            "active_seed": self._active_seed,
            "stream_step": self._stream_step,
            "feature_rng_state": copy.deepcopy(self._feature_rng.bit_generator.state),
            "noise_rng_state": copy.deepcopy(self._noise_rng.bit_generator.state),
            "buffer_position": self._buffer_position,
            "feature_buffer": self._feature_buffer.copy(),
            "noise_buffer": self._noise_buffer.copy(),
            "num_features": self.config.num_features,
            "useful_probability": self.config.useful_probability,
            "irrelevant_probability": self.config.irrelevant_probability,
            "useful_weight": self.config.useful_weight,
            "target_noise_variance": self.config.target_noise_variance,
            "chunk_size": self.config.chunk_size,
            "dtype": self.config.dtype.str,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint after validating stream compatibility."""

        constructor_seed = state["constructor_seed"]
        if isinstance(constructor_seed, bool) or not isinstance(
            constructor_seed, (int, np.integer)
        ):
            raise TypeError("Checkpoint constructor_seed must be an integer.")
        if int(constructor_seed) != self._constructor_seed:
            raise ValueError("Checkpoint constructor seed does not match this stream.")
        raw_num_features = state["num_features"]
        if isinstance(raw_num_features, bool) or not isinstance(
            raw_num_features, (int, np.integer)
        ):
            raise TypeError("Checkpoint num_features must be an integer.")
        if int(raw_num_features) != self.config.num_features:
            raise ValueError("Checkpoint feature count does not match this stream.")
        for name in (
            "useful_probability",
            "irrelevant_probability",
            "useful_weight",
            "target_noise_variance",
        ):
            if float(state[name]) != float(getattr(self.config, name)):
                raise ValueError(f"Checkpoint {name} does not match this stream.")
        raw_chunk_size = state["chunk_size"]
        if isinstance(raw_chunk_size, bool) or not isinstance(
            raw_chunk_size, (int, np.integer)
        ):
            raise TypeError("Checkpoint chunk_size must be an integer.")
        if int(raw_chunk_size) != self.config.chunk_size:
            raise ValueError("Checkpoint chunk size does not match this stream.")
        if np.dtype(state["dtype"]) != self.config.dtype:
            raise TypeError("Checkpoint dtype does not match this stream.")

        feature_buffer = np.asarray(state["feature_buffer"])
        noise_buffer = np.asarray(state["noise_buffer"])
        if feature_buffer.dtype != np.dtype(np.bool_):
            raise TypeError("Checkpoint feature buffer must have boolean dtype.")
        if noise_buffer.dtype != self.config.dtype:
            raise TypeError("Checkpoint noise buffer has the wrong dtype.")
        if feature_buffer.ndim != 2 or feature_buffer.shape[1] != self.config.num_features:
            raise ValueError("Checkpoint feature buffer has the wrong shape.")
        if noise_buffer.shape != (feature_buffer.shape[0],):
            raise ValueError("Checkpoint noise buffer has the wrong shape.")
        if feature_buffer.shape[0] not in (0, self.config.chunk_size):
            raise ValueError("Checkpoint buffer length is inconsistent with chunk_size.")
        if not bool(np.isfinite(noise_buffer).all()):
            raise FloatingPointError("Checkpoint noise buffer contains non-finite values.")

        for name in ("active_seed", "stream_step", "buffer_position"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"Checkpoint {name} must be an integer.")
            if int(value) < 0:
                raise ValueError(f"Checkpoint {name} must be non-negative.")
        active_seed = int(state["active_seed"])
        stream_step = int(state["stream_step"])
        buffer_position = int(state["buffer_position"])
        if not 0 <= buffer_position <= len(noise_buffer):
            raise ValueError("Checkpoint buffer position is out of range.")
        if stream_step == 0:
            if len(noise_buffer) != 0 or buffer_position != 0:
                raise ValueError("A step-zero checkpoint must have an empty buffer.")
        else:
            expected_position = ((stream_step - 1) % self.config.chunk_size) + 1
            if (
                len(noise_buffer) != self.config.chunk_size
                or buffer_position != expected_position
            ):
                raise ValueError(
                    "Checkpoint step and buffer position are inconsistent."
                )

        # Validate RNG payloads on temporary generators so a failure cannot leave
        # this stream partially restored.
        feature_rng = np.random.default_rng()
        feature_rng.bit_generator.state = copy.deepcopy(state["feature_rng_state"])
        noise_rng = np.random.default_rng()
        noise_rng.bit_generator.state = copy.deepcopy(state["noise_rng_state"])

        self._active_seed = active_seed
        self._stream_step = stream_step
        self._feature_rng = feature_rng
        self._noise_rng = noise_rng
        self._buffer_position = buffer_position
        self._feature_buffer = feature_buffer.copy()
        self._noise_buffer = noise_buffer.copy()
