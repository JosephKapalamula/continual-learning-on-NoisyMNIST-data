"""Engineered neural-IDBD candidate from the reverse-engineering note.

The Oak Lab article does not publish NetworkIDBD source code. This module
implements the equations and additions described in the attached reverse-
engineering article and must therefore be reported as an approximation.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer


class NetworkIDBD(Optimizer):
    """Per-parameter neural IDBD with scale normalization and step control."""

    def __init__(
        self,
        params,
        meta_lr: float = 0.1,
        initial_beta: float = math.log(1e-6),
        decay: float = 0.9995,
        eps: float = 1e-8,
        beta_min: float = math.log(1e-8),
        beta_max: float = math.log(1e4),
        eta: float = 0.1,
        tau: float = 1e4,
    ) -> None:
        defaults = {
            "meta_lr": meta_lr,
            "initial_beta": initial_beta,
            "decay": decay,
            "eps": eps,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "eta": eta,
            "tau": tau,
        }
        if not math.isfinite(meta_lr) or meta_lr <= 0.0:
            raise ValueError("meta_lr must be positive and finite.")
        if not math.isfinite(initial_beta):
            raise ValueError("initial_beta must be finite.")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1).")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")
        if not math.isfinite(beta_min) or not math.isfinite(beta_max):
            raise ValueError("beta bounds must be finite.")
        if beta_min >= beta_max:
            raise ValueError("beta_min must be less than beta_max.")
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be positive and finite.")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive and finite.")

        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                state["beta"] = torch.full_like(parameter, initial_beta)
                state["h"] = torch.zeros_like(parameter)
                state["v"] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(
        self,
        phi: Mapping[torch.nn.Parameter, Tensor],
        delta: Tensor,
        closure=None,
    ):
        """Apply one engineered neural-IDBD update.

        ``phi[p]`` must equal ``d(prediction) / d(p)`` for the current
        experience. The update follows the attached article's full algorithm,
        including normalizer adaptation, active-parameter beta decay, and the
        global effective-step overshoot bound.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not torch.is_tensor(delta) or delta.numel() != 1:
            raise ValueError("delta must be a scalar tensor.")
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("delta must be finite.")

        effective_step = None
        updates: list[tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]] = []
        eta_value = None

        for group in self.param_groups:
            theta = group["meta_lr"]
            decay = group["decay"]
            eps = group["eps"]
            beta_min = group["beta_min"]
            beta_max = group["beta_max"]
            eta = group["eta"]
            tau = group["tau"]
            log_decay = math.log(decay)
            if eta_value is None:
                eta_value = eta
            elif eta_value != eta:
                raise ValueError("All parameter groups must use the same eta.")

            for parameter in group["params"]:
                if parameter not in phi:
                    continue
                phi_parameter = phi[parameter].detach()
                if phi_parameter.shape != parameter.shape:
                    raise ValueError("Each phi tensor must match its parameter shape.")
                if phi_parameter.device != parameter.device:
                    raise ValueError("Each phi tensor must share its parameter device.")
                if not bool(torch.isfinite(phi_parameter).all()):
                    raise FloatingPointError("phi must be finite.")

                state = self.state[parameter]
                beta = state["beta"]
                h = state["h"]
                v = state["v"]
                alpha_old = torch.exp(beta)
                meta_gradient = delta * phi_parameter * h
                absolute_meta = meta_gradient.abs()
                v_next = torch.maximum(
                    absolute_meta,
                    v + (alpha_old * phi_parameter.square() / tau)
                    * (absolute_meta - v),
                )
                beta_next = beta + theta * meta_gradient / v_next.clamp_min(eps)
                beta_next = beta_next + log_decay * phi_parameter.ne(0).to(beta.dtype)
                beta_next = beta_next.clamp(beta_min, beta_max)
                alpha_next = torch.exp(beta_next)

                if effective_step is None:
                    effective_step = torch.zeros(
                        (), device=parameter.device, dtype=parameter.dtype
                    )
                effective_step = effective_step + (
                    alpha_next * phi_parameter.square()
                ).sum()
                updates.append(
                    (parameter, state, phi_parameter, beta_next, alpha_next, v_next)
                )

        if effective_step is None or not bool(torch.isfinite(effective_step)):
            raise FloatingPointError("NetworkIDBD effective step is invalid.")
        if effective_step.item() == 0.0:
            step_scale = torch.ones_like(effective_step)
        else:
            step_scale = torch.minimum(
                torch.ones_like(effective_step),
                torch.as_tensor(
                    eta_value,
                    device=effective_step.device,
                    dtype=effective_step.dtype,
                )
                / effective_step,
            )

        for parameter, state, phi_parameter, beta_next, alpha_next, v_next in updates:
            h = state["h"]
            parameter.add_(step_scale * alpha_next * delta * phi_parameter)
            h.copy_(
                torch.maximum(
                    h * (1.0 - step_scale * alpha_next * phi_parameter.square()),
                    torch.zeros_like(h),
                )
                + step_scale * alpha_next * delta * phi_parameter
            )
            state["beta"].copy_(beta_next)
            state["v"].copy_(v_next)
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("NetworkIDBD produced a non-finite parameter.")
            if not bool(torch.isfinite(h).all()):
                raise FloatingPointError("NetworkIDBD produced a non-finite trace.")

        return loss
