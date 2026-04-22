from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LatentForecastState:
    previous_latent: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor


def _group_count(num_channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if num_channels % candidate == 0:
            return candidate
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class FrameEncoder(nn.Module):
    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(inplace=True),
            ResidualConvBlock(32),
            nn.Conv2d(32, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(latent_channels), latent_channels),
            nn.SiLU(inplace=True),
            ResidualConvBlock(latent_channels),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.features(frames)


class FrameDecoder(nn.Module):
    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            ResidualConvBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(inplace=True),
            ResidualConvBlock(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(16), 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        combined = torch.cat([inputs, hidden], dim=1)
        gate_values = self.gates(combined)
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gate_values, chunks=4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)

        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class OpenLoopVideoPredictor(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        latent_channels = latent_dim
        hidden_channels = hidden_dim

        self.encoder = FrameEncoder(latent_channels=latent_channels)
        self.decoder = FrameDecoder(latent_channels=latent_channels)
        self.recurrent_cell = ConvLSTMCell(
            input_channels=latent_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )
        self.latent_head = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        leading_shape = frames.shape[:-3]
        flat_frames = frames.reshape(-1, *frames.shape[-3:])
        latents = self.encoder(flat_frames)
        return latents.reshape(*leading_shape, *latents.shape[1:])

    def decode_latents(
        self,
        latents: torch.Tensor,
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        leading_shape = latents.shape[:-3]
        flat_latents = latents.reshape(-1, *latents.shape[-3:])
        frames = self.decoder(flat_latents)
        if apply_sigmoid:
            frames = torch.sigmoid(frames)
        return frames.reshape(*leading_shape, *frames.shape[1:])

    def initialize_state(
        self,
        context_frames: torch.Tensor,
    ) -> tuple[LatentForecastState, dict[str, torch.Tensor]]:
        batch_size, context_steps = context_frames.shape[:2]
        context_latents = self.encode_frames(context_frames)
        latent_channels, latent_height, latent_width = context_latents.shape[2:]
        reconstructed_context_logits = self.decode_latents(context_latents, apply_sigmoid=False)
        reconstructed_context = torch.sigmoid(reconstructed_context_logits)

        hidden = context_latents.new_zeros(batch_size, self.recurrent_cell.hidden_channels, latent_height, latent_width)
        cell = context_latents.new_zeros(batch_size, self.recurrent_cell.hidden_channels, latent_height, latent_width)

        for t in range(context_steps):
            hidden, cell = self.recurrent_cell(context_latents[:, t], (hidden, cell))

        state = LatentForecastState(
            previous_latent=context_latents[:, -1],
            hidden=hidden,
            cell=cell,
        )
        aux = {
            "recon_context": reconstructed_context,
            "recon_context_logits": reconstructed_context_logits,
            "context_latents": context_latents,
        }
        return state, aux

    def forecast_step(
        self,
        state: LatentForecastState,
    ) -> tuple[LatentForecastState, dict[str, torch.Tensor]]:
        hidden, cell = self.recurrent_cell(state.previous_latent, (state.hidden, state.cell))
        predicted_latent = state.previous_latent + self.latent_head(hidden)
        predicted_frame_logits = self.decode_latents(predicted_latent, apply_sigmoid=False)
        predicted_frame = torch.sigmoid(predicted_frame_logits)
        next_state = LatentForecastState(
            previous_latent=predicted_latent,
            hidden=hidden,
            cell=cell,
        )
        return next_state, {
            "pred_latent": predicted_latent,
            "pred_frame_logits": predicted_frame_logits,
            "pred_frame": predicted_frame,
        }

    def forward(
        self,
        context_frames: torch.Tensor,
        pred_steps: int,
        future_frames: torch.Tensor | None = None,
        teacher_force_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        state, aux = self.initialize_state(context_frames)
        teacher_force_latents = None
        if future_frames is not None and teacher_force_ratio > 0.0:
            teacher_force_latents = self.encode_frames(future_frames)
        pred_latents = []
        pred_frames = []
        pred_frame_logits = []
        for step_idx in range(pred_steps):
            state, step_outputs = self.forecast_step(state)
            pred_latents.append(step_outputs["pred_latent"])
            pred_frame_logits.append(step_outputs["pred_frame_logits"])
            pred_frames.append(step_outputs["pred_frame"])

            if teacher_force_latents is not None and step_idx < pred_steps - 1:
                teacher_mask = (
                    torch.rand(state.previous_latent.shape[0], 1, 1, 1, device=state.previous_latent.device)
                    < teacher_force_ratio
                )
                state = LatentForecastState(
                    previous_latent=torch.where(
                        teacher_mask,
                        teacher_force_latents[:, step_idx],
                        state.previous_latent,
                    ),
                    hidden=state.hidden,
                    cell=state.cell,
                )

        return {
            "pred_frames": torch.stack(pred_frames, dim=1),
            "pred_frame_logits": torch.stack(pred_frame_logits, dim=1),
            "pred_latents": torch.stack(pred_latents, dim=1),
            "recon_context": aux["recon_context"],
            "recon_context_logits": aux["recon_context_logits"],
            "context_latents": aux["context_latents"],
        }


def _stride_generator(depth: int, reverse: bool = False) -> list[int]:
    strides = [1, 2] * 10
    selected = strides[:depth]
    if reverse:
        return list(reversed(selected))
    return selected


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        transpose: bool = False,
        act_norm: bool = False,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act_norm:
            x = self.act(self.norm(x))
        return x


class ConvSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        if stride == 1:
            transpose = False
        self.block = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        act_norm: bool = False,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act_norm:
            x = self.act(self.norm(x))
        return x


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        incep_ker: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList(
            [
                GroupConv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    groups=groups,
                    act_norm=True,
                )
                for kernel in incep_ker
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = y + layer(x)
        return y


class SimVPEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, depth: int) -> None:
        super().__init__()
        strides = _stride_generator(depth)
        layers = [ConvSC(in_channels, hidden_channels, stride=strides[0])]
        layers.extend(ConvSC(hidden_channels, hidden_channels, stride=s) for s in strides[1:])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.layers[0](x)
        latent = skip
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, skip


class SimVPDecoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        strides = _stride_generator(depth, reverse=True)
        layers = [ConvSC(hidden_channels, hidden_channels, stride=s, transpose=True) for s in strides[:-1]]
        layers.append(ConvSC(2 * hidden_channels, hidden_channels, stride=strides[-1], transpose=True))
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.readout(x))


class MidXNet(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        depth: int,
        incep_ker: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.depth = depth
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, depth):
            enc_layers.append(
                Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
            )

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, depth - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
            )
        dec_layers.append(
            Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups)
        )

        self.enc_layers = nn.ModuleList(enc_layers)
        self.dec_layers = nn.ModuleList(dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape
        x = x.reshape(batch_size, timesteps * channels, height, width)

        skips: list[torch.Tensor] = []
        z = x
        for index, layer in enumerate(self.enc_layers):
            z = layer(z)
            if index < self.depth - 1:
                skips.append(z)

        z = self.dec_layers[0](z)
        for index in range(1, self.depth):
            z = self.dec_layers[index](torch.cat([z, skips[-index]], dim=1))

        return z.reshape(batch_size, timesteps, channels, height, width)


class SimVPPredictor(nn.Module):
    """Compact SimVP-style baseline adapted for Moving MNIST 10->10 prediction."""

    def __init__(
        self,
        shape_in: tuple[int, int, int, int],
        hid_s: int = 32,
        hid_t: int = 128,
        n_s: int = 4,
        n_t: int = 4,
        incep_ker: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        timesteps, channels, _, _ = shape_in
        self.timesteps = timesteps
        self.encoder = SimVPEncoder(channels, hid_s, n_s)
        self.mid = MidXNet(timesteps * hid_s, hid_t, n_t, incep_ker=incep_ker, groups=groups)
        self.decoder = SimVPDecoder(hid_s, channels, n_s)

    def forward(self, context_frames: torch.Tensor, pred_steps: int) -> dict[str, torch.Tensor]:
        if pred_steps != self.timesteps:
            raise ValueError(
                f"SimVPPredictor expects pred_steps == context length ({self.timesteps}), got {pred_steps}."
            )

        batch_size, context_steps, channels, height, width = context_frames.shape
        flat_context = context_frames.reshape(batch_size * context_steps, channels, height, width)

        embed, skip = self.encoder(flat_context)
        hidden_channels, hidden_height, hidden_width = embed.shape[1:]
        embed = embed.reshape(batch_size, context_steps, hidden_channels, hidden_height, hidden_width)

        hidden = self.mid(embed).reshape(batch_size * context_steps, hidden_channels, hidden_height, hidden_width)
        pred_frames = self.decoder(hidden, skip).reshape(batch_size, context_steps, channels, height, width)
        return {"pred_frames": pred_frames}
