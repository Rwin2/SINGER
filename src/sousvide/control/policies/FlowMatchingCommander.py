"""
Flow Matching Commander for SINGER.

Extends CommandSV to support flow matching:
- Accepts noisy action + flow time as additional inputs
- Predicts velocity field (same dim as action)
- Inference via Euler ODE integration

Inspired by:
- CS224R HW1 flow matching implementation
- π₀ (Physical Intelligence, 2024)
- DreamZero (NVIDIA, 2026) — Gaussian timestep weighting

Compatible with existing SINGER pipeline: same state/objective/vision inputs,
just adds (noisy_action, time) and changes loss from MSE to flow matching.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import sousvide.control.policies.BaseNetworks as bn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for flow matching timestep t ∈ [0, 1]."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestep tensor, shape (B,) or scalar
        Returns:
            embedding: shape (B, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class CommandSV_FlowMatching(nn.Module):
    """
    Flow Matching variant of CommandSV.

    Input: concat(tx_com, obj_com, z_par, z_img, noisy_action, time_emb)
    Output: velocity field, same shape as action

    The noisy_action and time_emb are additional inputs compared to CommandSV.
    This allows the model to predict the velocity v(x_t, t | state) that
    pushes noisy actions toward clean expert actions.
    """

    def __init__(self,
                 state: List[int],
                 objective: List[int],
                 fparam_size: int,
                 fimage_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 time_embed_dim: int = 32,
                 active_end: bool = False,
                 dropout: float = 0.2):

        super().__init__()

        self.ix_tx = np.array(state)
        self.ix_obj = np.array(objective)
        self.output_size = output_size
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: state + objective + history + vision + noisy_action + time_embed
        input_size = (len(state) + len(objective) + fparam_size + fimage_size
                      + output_size + time_embed_dim)

        self.network = bn.SimpleMLP(input_size, hidden_sizes, output_size,
                                     active_end=active_end, dropout=dropout)

    def forward(self, tx: torch.Tensor, obj: torch.Tensor,
                zpar: torch.Tensor, zimg: torch.Tensor,
                noisy_action: Optional[torch.Tensor] = None,
                flow_time: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.

        When noisy_action and flow_time are None (inference without flow),
        falls back to standard CommandSV behavior (predicts from state only).

        Args:
            tx:            State tensor (B, 8)
            obj:           Objective tensor (B, 3)
            zpar:          History encoder output (B, 8)
            zimg:          Vision features (B, 128)
            noisy_action:  Noisy action for flow matching (B, output_size)
            flow_time:     Flow matching timestep (B,) in [0, 1]

        Returns:
            velocity:  Predicted velocity field (B, output_size)
            None
        """
        if noisy_action is None:
            # Fallback: standard forward (for compatibility or single-step)
            noisy_action = torch.zeros(tx.shape[0], self.output_size,
                                        device=tx.device)
            flow_time = torch.ones(tx.shape[0], device=tx.device)

        t_emb = self.time_embed(flow_time)
        xcm = torch.cat((tx, obj, zpar, zimg, noisy_action, t_emb), -1)
        velocity = self.network(xcm)

        return velocity, None


class FlowMatchingSchedule:
    """
    Conditional Optimal-Transport Flow Matching schedule for SINGER.

    Adapted from CS224R HW1 FlowMatchingSchedule:
    - interpolate(): build x_t and target velocity for training
    - sample(): Euler ODE integration for inference

    Supports Gaussian timestep weighting (DreamZero-inspired).
    """

    def __init__(self, action_dim: int = 4, device: str = 'cpu',
                 num_steps: int = 10, gaussian_weighting: bool = False):
        self.action_dim = action_dim
        self.device = device
        self.num_steps = num_steps
        self.gaussian_weighting = gaussian_weighting

    def interpolate(self, x1: torch.Tensor, t: torch.Tensor):
        """
        Build noisy sample x_t and target velocity for training.

        Args:
            x1: clean action data, shape (B, action_dim)
            t:  timesteps in [0, 1], shape (B,)

        Returns:
            (x_t, velocity): both shape (B, action_dim)
        """
        noise = torch.randn_like(x1)
        t_expand = t.unsqueeze(1)  # (B, 1) for broadcasting
        x_t = t_expand * x1 + (1 - t_expand) * noise
        velocity = x1 - noise
        return x_t, velocity

    def sample_timesteps(self, batch_size: int, device: torch.device):
        """Sample timesteps, optionally with Gaussian weighting (DreamZero)."""
        if self.gaussian_weighting:
            # DreamZero: Gaussian centered at 0.5, more weight on intermediate t
            t = torch.randn(batch_size, device=device) * 0.25 + 0.5
            t = t.clamp(0.001, 0.999)
        else:
            t = torch.rand(batch_size, device=device)
        return t

    def compute_loss_weights(self, t: torch.Tensor):
        """Compute per-sample loss weights (DreamZero Gaussian weighting)."""
        if self.gaussian_weighting:
            w = torch.exp(-2 * ((t - 0.5) / 1.0) ** 2)
            w = w / w.mean()  # normalize
            return w
        return torch.ones_like(t)

    @torch.no_grad()
    def sample(self, model_fn, state_inputs: tuple, device: torch.device):
        """
        Generate samples by integrating the learned velocity field.

        Args:
            model_fn:      callable that takes (*state_inputs, noisy_action, t) → velocity
            state_inputs:  tuple of (tx, obj, zpar, zimg) tensors
            device:        torch device

        Returns:
            Sampled actions, shape (B, action_dim)
        """
        B = state_inputs[0].shape[0]
        a_t = torch.randn(B, self.action_dim, device=device)

        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = torch.full((B,), i * dt, device=device)
            velocity, _ = model_fn(*state_inputs, a_t, t)
            a_t = a_t + velocity * dt

        return a_t


class FlowMatchingCommanderWrapper(nn.Module):
    """
    Wraps SVNet with a flow matching Commander replacement.

    Runs HistoryEncoder + VisionMLP (frozen) from the base SVNet,
    then concatenates state + noisy_action + time_embed through a new MLP.

    Compatible with SINGER Pilot interface (delegates get_data, get_network, etc.).
    """

    def __init__(self, svnet_model, output_size, num_flow_steps=10,
                 time_embed_dim=32, hidden_sizes=[256, 128], dropout=0.2):
        super().__init__()
        self.svnet = svnet_model
        self.output_size = output_size
        self.num_flow_steps = num_flow_steps
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Commander input = 147 (state features) + output_size (noisy action) + time_embed_dim
        input_size = 147 + output_size + time_embed_dim
        self.fm_commander = bn.SimpleMLP(input_size, hidden_sizes, output_size,
                                          active_end=False, dropout=dropout)

        # Delegate SVNet attributes for Pilot compatibility
        self.Nz = svnet_model.Nz
        self.network = svnet_model.network
        self.get_data = svnet_model.get_data
        self.get_network = svnet_model.get_network
        self.get_commander_inputs = svnet_model.get_commander_inputs
        self.extract_inputs = svnet_model.extract_inputs

    def forward_flow(self, tx_com, obj_com, dxu_par, img_vis, tx_vis,
                     noisy_action, flow_time):
        """Flow matching forward: predict velocity given noisy action + time."""
        with torch.no_grad():
            _, z_par = self.svnet.network["HistoryEncoder"](dxu_par)
            y_vis, _ = self.svnet.network["VisionMLP"](img_vis, tx_vis)

        t_emb = self.time_embed(flow_time)
        xcm = torch.cat((tx_com, obj_com, z_par, y_vis, noisy_action, t_emb), -1)
        velocity = self.fm_commander(xcm)
        return velocity

    def forward_sample(self, tx_com, obj_com, dxu_par, img_vis, tx_vis,
                       num_steps=None):
        """Euler ODE sampling for inference."""
        if num_steps is None:
            num_steps = self.num_flow_steps
        with torch.no_grad():
            _, z_par = self.svnet.network["HistoryEncoder"](dxu_par)
            y_vis, _ = self.svnet.network["VisionMLP"](img_vis, tx_vis)

        B = tx_com.shape[0]
        device = tx_com.device
        a_t = torch.randn(B, self.output_size, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_emb = self.time_embed(t)
            xcm = torch.cat((tx_com, obj_com, z_par, y_vis, a_t, t_emb), -1)
            velocity = self.fm_commander(xcm)
            a_t = a_t + velocity * dt

        return a_t

    def forward(self, tx_com, obj_com, dxu_par, img_vis, tx_vis):
        """SVNet-compatible forward: uses flow matching sampling for inference."""
        return self.forward_sample(tx_com, obj_com, dxu_par, img_vis, tx_vis), None
