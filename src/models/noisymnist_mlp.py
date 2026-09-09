"""Faithful NoisyMNIST network architecture and initialization.

The Oak Lab source specifies a 4,096-input network with 10,000 ReLU hidden
units and one prediction unit. Input-to-hidden weights are uniform in
(-0.0001, 0.0001), and hidden-to-output weights are zero. Bias behavior is
not specified, so the reproduction default disables biases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the faithful NoisyMNIST multilayer perceptron."""

    # PAPER_DEFINED
    input_size: int = 4_096
    hidden_size: int = 10_000
    output_size: int = 1
    input_hidden_low: float = -0.0001
    input_hidden_high: float = 0.0001

    # REPRODUCTION_PROTOCOL. Enabling biases is ABLATION_OR_EXTENSION because
    # the source does not specify whether biases exist or how they are initialized.
    initialization_seed: int = 2_026_020_2
    use_bias: bool = False
    bias_initial_value: Optional[float] = None
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Fail loudly when the faithful architecture contract is violated."""

        if self.input_size != 4_096:
            raise ValueError("The faithful input size is 4,096.")
        if self.hidden_size != 10_000:
            raise ValueError("The faithful hidden size is 10,000.")
        if self.output_size != 1:
            raise ValueError("The faithful network has one scalar prediction unit.")
        if self.input_hidden_low != -0.0001 or self.input_hidden_high != 0.0001:
            raise ValueError(
                "The faithful input-hidden initialization interval is exactly "
                "(-0.0001, 0.0001)."
            )
        if self.dtype is not torch.float32:
            raise TypeError("The faithful Phase 2 implementation uses torch.float32.")
        if self.use_bias and self.bias_initial_value is None:
            raise ValueError(
                "Biases are source-unspecified. An explicit bias_initial_value is "
                "required when use_bias=True."
            )
        if not self.use_bias and self.bias_initial_value is not None:
            raise ValueError("bias_initial_value must be None when use_bias=False.")


@dataclass(frozen=True)
class ForwardPass:
    """Prediction and hidden activations from one forward pass."""

    prediction: Tensor
    hidden_activations: Tensor


class NoisyMNISTMLP(nn.Module):
    """Readable 4,096 -> 10,000 ReLU -> 1 network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        target_device = torch.device(config.device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        cuda_devices: list[int] = []
        if target_device.type == "cuda":
            cuda_devices = [
                target_device.index
                if target_device.index is not None
                else torch.cuda.current_device()
            ]

        # Keep constructor-time randomness isolated because the layers are
        # immediately overwritten by the source-defined initialization below.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            self.network = nn.Sequential(
                nn.Linear(
                    config.input_size,
                    config.hidden_size,
                    bias=config.use_bias,
                    device=target_device,
                    dtype=config.dtype,
                ),
                nn.ReLU(),
                nn.Linear(
                    config.hidden_size,
                    config.output_size,
                    bias=config.use_bias,
                    device=target_device,
                    dtype=config.dtype,
                ),
            )

        self._last_hidden_activations: Optional[Tensor] = None
        self.network[1].register_forward_hook(self._capture_hidden_activations)
        self.reset_parameters_from_config(config.initialization_seed)

    def _capture_hidden_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Capture ReLU output for diagnostics during the current forward pass."""

        self._last_hidden_activations = output

    @property
    def input_to_hidden(self) -> nn.Linear:
        """Return the input-to-hidden layer from the sequential network."""

        return self.network[0]

    @property
    def activation(self) -> nn.ReLU:
        """Return the hidden activation layer from the sequential network."""

        return self.network[1]

    @property
    def hidden_to_output(self) -> nn.Linear:
        """Return the hidden-to-output layer from the sequential network."""

        return self.network[2]

    @property
    def first_layer_weights(self) -> nn.Parameter:
        """Direct access to the [10,000, 4,096] input-hidden weights."""

        return self.input_to_hidden.weight

    @property
    def second_layer_weights(self) -> nn.Parameter:
        """Direct access to the [1, 10,000] hidden-output weights."""

        return self.hidden_to_output.weight

    @property
    def first_layer_bias(self) -> Optional[nn.Parameter]:
        """Input-hidden bias, or None under the faithful default."""

        return self.input_to_hidden.bias

    @property
    def second_layer_bias(self) -> Optional[nn.Parameter]:
        """Hidden-output bias, or None under the faithful default."""

        return self.hidden_to_output.bias

    def reset_parameters_from_config(self, seed: Optional[int] = None) -> None:
        """Reset source-defined weights and any explicitly enabled extension biases.

        The two weight initializations are PAPER_DEFINED. If ``use_bias=True``,
        filling the biases is an ABLATION_OR_EXTENSION rather than a paper claim.
        """

        active_seed = self.config.initialization_seed if seed is None else int(seed)
        parameter_device = self.first_layer_weights.device
        generator = torch.Generator(device=parameter_device)
        generator.manual_seed(active_seed)

        with torch.no_grad():
            self.first_layer_weights.uniform_(
                self.config.input_hidden_low,
                self.config.input_hidden_high,
                generator=generator,
            )
            self.second_layer_weights.zero_()
            if self.config.use_bias:
                # ABLATION_OR_EXTENSION: bias behavior is unresolved in the source.
                assert self.config.bias_initial_value is not None
                assert self.first_layer_bias is not None
                assert self.second_layer_bias is not None
                self.first_layer_bias.fill_(self.config.bias_initial_value)
                self.second_layer_bias.fill_(self.config.bias_initial_value)

    def forward_with_activations(self, x: Tensor) -> ForwardPass:
        """Return predictions and the hidden activations captured by the hook."""

        if x.ndim not in (1, 2):
            raise ValueError("Expected input shape [4096] or [batch, 4096].")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected final input dimension {self.config.input_size}, "
                f"received {x.shape[-1]}."
            )
        if x.dtype != self.config.dtype:
            raise TypeError(f"Expected dtype {self.config.dtype}, received {x.dtype}.")
        if x.device != self.first_layer_weights.device:
            raise ValueError(
                f"Input is on {x.device}; model is on {self.first_layer_weights.device}."
            )

        self._last_hidden_activations = None
        prediction = self.network(x).squeeze(-1)
        hidden_activations = self._last_hidden_activations
        if hidden_activations is None:
            raise RuntimeError("The hidden-activation hook did not run.")
        return ForwardPass(
            prediction=prediction,
            hidden_activations=hidden_activations,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return one scalar per input experience."""

        return self.forward_with_activations(x).prediction

    def parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_bytes(self) -> int:
        """Return bytes occupied by trainable parameters."""

        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )
