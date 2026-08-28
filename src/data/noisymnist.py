"""Importable form of the validated Phase 1 NoisyMNIST protocol.

The stochastic draw order and constants intentionally match the frozen executed
artifact 01_noisymnist_dataset_and_protocol.ipynb. Test loading is kept
behind a separate factory so development and validation code need not construct
or inspect the official test partition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, TypedDict

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated Phase 1 constants and reproducibility choices."""

    # PAPER_DEFINED
    canvas_size: int = 64
    digit_size: int = 28
    center_start: int = 18
    center_stop: int = 46
    digit_probability: float = 0.10
    background_probability: float = 0.01
    odd_clean_target: float = 1.0
    even_clean_target: float = -1.0
    absent_clean_target: float = 0.0
    target_noise_mean: float = 0.0
    target_noise_variance: float = 5.0
    target_noise_std: float = math.sqrt(5.0)

    # REPRODUCTION_PROTOCOL
    train_size: int = 55_000
    validation_size: int = 5_000
    split_seed: int = 20_260_201
    train_stream_seed: int = 314_159
    validation_stream_seed: int = 271_828
    test_stream_seed: int = 161_803
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_mnist: bool = False

    def __post_init__(self) -> None:
        if self.canvas_size != 64 or self.digit_size != 28:
            raise ValueError("The faithful protocol requires 64x64 and 28x28 geometry.")
        if self.center_start != 18 or self.center_stop != 46:
            raise ValueError("The faithful center is exactly rows/columns 18:46.")
        if self.center_stop - self.center_start != self.digit_size:
            raise ValueError("Center extent must equal digit_size.")
        if not (0.0 <= self.digit_probability <= 1.0):
            raise ValueError("digit_probability must be in [0, 1].")
        if not (0.0 <= self.background_probability <= 1.0):
            raise ValueError("background_probability must be in [0, 1].")
        if not math.isclose(
            self.target_noise_std**2,
            self.target_noise_variance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("target_noise_std must equal sqrt(target_noise_variance).")
        if self.dtype is not torch.float32:
            raise TypeError("NoisyMNIST stream outputs are required to be float32.")
        if self.normalize_mnist:
            raise ValueError("The faithful protocol preserves ToTensor values in [0, 1].")

    @property
    def num_pixels(self) -> int:
        """Total flattened input width."""

        return self.canvas_size**2

    @property
    def num_outer_pixels(self) -> int:
        """Number of Bernoulli distractor pixels."""

        return self.num_pixels - self.digit_size**2


class Experience(TypedDict):
    """One batch-size-one online NoisyMNIST experience."""

    x: Tensor
    x_flat: Tensor
    noisy_target: Tensor
    clean_target: Tensor
    gaussian_noise: Tensor
    digit_present: bool
    digit_label: Optional[int]
    mnist_index: Optional[int]
    source_partition: str
    pool_name: str
    stream_step: int


class MNISTPool:
    """Named, index-preserving view into one official MNIST partition."""

    def __init__(
        self,
        dataset: MNIST,
        indices: Sequence[int],
        name: str,
        source_partition: str,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.index_set = frozenset(self.indices)
        self.name = name
        self.source_partition = source_partition

        if not self.indices:
            raise ValueError(f"{name} cannot be empty.")
        if len(self.index_set) != len(self.indices):
            raise ValueError(f"{name} contains duplicate original indices.")
        if min(self.indices) < 0 or max(self.indices) >= len(dataset):
            raise IndexError(f"{name} contains an out-of-range original index.")

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, pool_position: int) -> tuple[Tensor, int, int]:
        """Return image, label, and original official-partition index."""

        if not 0 <= pool_position < len(self):
            raise IndexError("pool_position is out of range.")
        original_index = self.indices[pool_position]
        image, label = self.dataset[original_index]
        if image.shape != (1, 28, 28) or image.dtype != torch.float32:
            raise AssertionError("MNIST must yield float32 [1, 28, 28] tensors.")
        return image, int(label), original_index


class NoisyMNISTStream(Iterator[Experience]):
    """Indefinite online stream with a private CPU random generator."""

    def __init__(
        self,
        pool: MNISTPool,
        config: ExperimentConfig,
        seed: int,
    ) -> None:
        if len(pool) == 0:
            raise ValueError("The digit pool must not be empty.")
        self.pool = pool
        self.config = config
        self._constructor_seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._device = torch.device(config.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was configured but is not available.")
        self.reset()

    @property
    def seed(self) -> int:
        """Seed used by the most recent reset."""

        return self._active_seed

    @property
    def stream_step(self) -> int:
        """Zero-based index of the next experience."""

        return self._stream_step

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset to step zero using the constructor seed or an explicit seed."""

        active_seed = self._constructor_seed if seed is None else int(seed)
        self._generator.manual_seed(active_seed)
        self._active_seed = active_seed
        self._stream_step = 0

    def state_dict(self) -> dict[str, Any]:
        """Return exact private RNG and position state for checkpointing."""

        return {
            "constructor_seed": self._constructor_seed,
            "active_seed": self._active_seed,
            "stream_step": self._stream_step,
            "generator_state": self._generator.get_state().clone(),
            "pool_name": self.pool.name,
            "source_partition": self.pool.source_partition,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact stream state after validating pool provenance."""

        if int(state["constructor_seed"]) != self._constructor_seed:
            raise ValueError("Checkpoint constructor seed does not match this stream.")
        if state["pool_name"] != self.pool.name:
            raise ValueError("Checkpoint pool name does not match this stream.")
        if state["source_partition"] != self.pool.source_partition:
            raise ValueError("Checkpoint source partition does not match this stream.")
        self._active_seed = int(state["active_seed"])
        self._stream_step = int(state["stream_step"])
        self._generator.set_state(state["generator_state"].clone())

    def __iter__(self) -> "NoisyMNISTStream":
        return self

    def __next__(self) -> Experience:
        cfg = self.config
        center = slice(cfg.center_start, cfg.center_stop)
        step = self._stream_step

        # Preserve the validated Phase 1 draw order exactly.
        digit_present = bool(
            torch.rand((), generator=self._generator).item() < cfg.digit_probability
        )
        canvas_cpu = (
            torch.rand(
                (1, cfg.canvas_size, cfg.canvas_size),
                generator=self._generator,
            )
            < cfg.background_probability
        ).to(dtype=cfg.dtype)
        canvas_cpu[:, center, center] = 0.0

        digit_label: Optional[int] = None
        mnist_index: Optional[int] = None
        clean_value = cfg.absent_clean_target
        if digit_present:
            pool_position = int(
                torch.randint(
                    low=0,
                    high=len(self.pool),
                    size=(1,),
                    generator=self._generator,
                ).item()
            )
            digit_image, digit_label, mnist_index = self.pool.get(pool_position)
            canvas_cpu[:, center, center] = digit_image
            clean_value = (
                cfg.odd_clean_target
                if digit_label % 2 == 1
                else cfg.even_clean_target
            )

        clean_target_cpu = torch.tensor(clean_value, dtype=cfg.dtype)
        gaussian_noise_cpu = (
            torch.randn((), generator=self._generator, dtype=cfg.dtype)
            * cfg.target_noise_std
            + cfg.target_noise_mean
        )
        noisy_target_cpu = clean_target_cpu + gaussian_noise_cpu

        x = canvas_cpu.to(device=self._device)
        clean_target = clean_target_cpu.to(device=self._device)
        gaussian_noise = gaussian_noise_cpu.to(device=self._device)
        noisy_target = noisy_target_cpu.to(device=self._device)

        self._stream_step += 1
        return Experience(
            x=x,
            x_flat=x.reshape(cfg.num_pixels),
            noisy_target=noisy_target,
            clean_target=clean_target,
            gaussian_noise=gaussian_noise,
            digit_present=digit_present,
            digit_label=digit_label,
            mnist_index=mnist_index,
            source_partition=self.pool.source_partition,
            pool_name=self.pool.name,
            stream_step=step,
        )


@dataclass(frozen=True)
class TrainValidationPartitions:
    """Official-training-derived pools and their original indices."""

    dataset: MNIST
    training_pool: MNISTPool
    validation_pool: MNISTPool
    train_indices: np.ndarray
    validation_indices: np.ndarray


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray | Tensor,
    validation_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split indices with proportional class allocation."""

    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if not 0 < validation_size < len(labels_array):
        raise ValueError("validation_size must be between zero and dataset size.")

    classes, counts = np.unique(labels_array, return_counts=True)
    exact_allocations = counts * (validation_size / len(labels_array))
    allocations = np.floor(exact_allocations).astype(np.int64)
    remainder = validation_size - int(allocations.sum())
    fractional_parts = exact_allocations - allocations
    largest_remainders = np.argsort(-fractional_parts, kind="stable")[:remainder]
    allocations[largest_remainders] += 1

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for digit_class, class_validation_size in zip(classes, allocations, strict=True):
        class_indices = np.flatnonzero(labels_array == digit_class)
        class_indices = rng.permutation(class_indices)
        validation_parts.append(class_indices[:class_validation_size])
        train_parts.append(class_indices[class_validation_size:])

    train_indices = np.concatenate(train_parts).astype(np.int64, copy=False)
    validation_indices = np.concatenate(validation_parts).astype(np.int64, copy=False)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def make_train_validation_partitions(
    data_root: str | Path,
    config: ExperimentConfig,
    download: bool = False,
) -> TrainValidationPartitions:
    """Load only official training data and construct isolated train/validation pools."""

    dataset = MNIST(
        root=Path(data_root),
        train=True,
        download=download,
        transform=transforms.ToTensor(),
    )
    labels = dataset.targets.cpu().numpy()
    train_indices, validation_indices = stratified_split_indices(
        labels,
        validation_size=config.validation_size,
        seed=config.split_seed,
    )
    training_pool = MNISTPool(
        dataset,
        train_indices,
        name="training",
        source_partition="official_train",
    )
    validation_pool = MNISTPool(
        dataset,
        validation_indices,
        name="validation",
        source_partition="official_train",
    )
    return TrainValidationPartitions(
        dataset=dataset,
        training_pool=training_pool,
        validation_pool=validation_pool,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def make_test_pool(
    data_root: str | Path,
    config: ExperimentConfig,
    download: bool = False,
) -> MNISTPool:
    """Explicitly load the embargoed official test pool for final evaluation only."""

    del config  # Kept in the signature to make protocol use explicit.
    dataset = MNIST(
        root=Path(data_root),
        train=False,
        download=download,
        transform=transforms.ToTensor(),
    )
    return MNISTPool(
        dataset,
        range(len(dataset)),
        name="test",
        source_partition="official_test",
    )


def make_fixed_stream(
    pool: MNISTPool,
    num_steps: int,
    seed: int,
    config: ExperimentConfig,
) -> Iterator[Experience]:
    """Yield a deterministic finite sequence from a fresh stream."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    stream = NoisyMNISTStream(pool, config, seed)
    for _ in range(num_steps):
        yield next(stream)


def experiences_equal(left: Experience, right: Experience) -> bool:
    """Compare every tensor and metadata field bit-for-bit."""

    tensor_keys = (
        "x",
        "x_flat",
        "noisy_target",
        "clean_target",
        "gaussian_noise",
    )
    metadata_keys = (
        "digit_present",
        "digit_label",
        "mnist_index",
        "source_partition",
        "pool_name",
        "stream_step",
    )
    return all(torch.equal(left[key], right[key]) for key in tensor_keys) and all(
        left[key] == right[key] for key in metadata_keys
    )


def sequences_equal(
    left: Sequence[Experience],
    right: Sequence[Experience],
) -> bool:
    """Compare finite experience sequences exactly."""

    return len(left) == len(right) and all(
        experiences_equal(a, b) for a, b in zip(left, right, strict=True)
    )
