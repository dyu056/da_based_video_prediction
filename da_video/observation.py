from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SparseObservationOperator:
    def __init__(
        self,
        kind: str = "full",
        image_size: int = 64,
        observe_every: int = 2,
        noise_std: float = 0.02,
        mask_fraction: float = 0.25,
        downsample_size: int = 16,
        seed: int = 42,
    ) -> None:
        if kind not in {"full", "masked", "lowres"}:
            raise ValueError(f"Unsupported observation kind: {kind}")
        self.kind = kind
        self.image_size = image_size
        self.observe_every = observe_every
        self.noise_std = noise_std
        self.mask_fraction = mask_fraction
        self.downsample_size = downsample_size
        self.seed = seed

        self._mask_cpu: torch.Tensor | None = None
        if self.kind == "masked":
            total_pixels = image_size * image_size
            observed_pixels = max(1, int(math.ceil(mask_fraction * total_pixels)))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            indices = torch.randperm(total_pixels, generator=generator)[:observed_pixels]
            mask = torch.zeros(total_pixels, dtype=torch.bool)
            mask[indices] = True
            self._mask_cpu = mask.reshape(1, 1, image_size, image_size)

    def has_observation(self, step_index: int) -> bool:
        return self.observe_every > 0 and (step_index + 1) % self.observe_every == 0

    def _mask(self, device: torch.device) -> torch.Tensor:
        if self._mask_cpu is None:
            raise RuntimeError("Mask requested for a non-masked observation operator.")
        return self._mask_cpu.to(device=device)

    def project(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 4:
            raise ValueError(f"Expected frames with shape [batch, channels, height, width], got {tuple(frames.shape)}")

        if self.kind == "full":
            observed = frames
        elif self.kind == "masked":
            mask = self._mask(frames.device)
            observed = frames.masked_select(mask.expand(frames.shape[0], -1, -1, -1)).reshape(frames.shape[0], -1)
            return observed
        else:
            observed = F.interpolate(
                frames,
                size=(self.downsample_size, self.downsample_size),
                mode="bilinear",
                align_corners=False,
            )
        return observed.reshape(frames.shape[0], -1)

    def observe(self, frames: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        observation = self.project(frames)
        if add_noise and self.noise_std > 0.0:
            observation = observation + self.noise_std * torch.randn_like(observation)
        return observation

    def noise_variance(self, observation_dim: int, device: torch.device) -> torch.Tensor:
        variance = self.noise_std**2
        return torch.full((observation_dim,), variance, device=device)
