from __future__ import annotations

import torch

from da_video.models import LatentForecastState


def clone_state(state: LatentForecastState) -> LatentForecastState:
    return LatentForecastState(
        previous_latent=state.previous_latent.clone(),
        hidden=state.hidden.clone(),
        cell=state.cell.clone(),
    )


def mean_state(state: LatentForecastState) -> LatentForecastState:
    return LatentForecastState(
        previous_latent=state.previous_latent.mean(dim=0, keepdim=True),
        hidden=state.hidden.mean(dim=0, keepdim=True),
        cell=state.cell.mean(dim=0, keepdim=True),
    )


def sample_state_ensemble(
    state: LatentForecastState,
    ensemble_size: int,
    latent_noise_std: float,
    hidden_noise_std: float,
    cell_noise_std: float,
) -> LatentForecastState:
    if state.previous_latent.shape[0] != 1:
        raise ValueError("sample_state_ensemble currently expects a single-sample state.")

    def expand_with_noise(tensor: torch.Tensor, noise_std: float) -> torch.Tensor:
        expanded = tensor.expand(ensemble_size, *tensor.shape[1:]).clone()
        if noise_std > 0.0:
            expanded = expanded + noise_std * torch.randn_like(expanded)
        return expanded

    return LatentForecastState(
        previous_latent=expand_with_noise(state.previous_latent, latent_noise_std),
        hidden=expand_with_noise(state.hidden, hidden_noise_std),
        cell=expand_with_noise(state.cell, cell_noise_std),
    )


def _state_components(state: LatentForecastState) -> list[torch.Tensor]:
    return [state.previous_latent, state.hidden, state.cell]


def state_to_matrix(state: LatentForecastState) -> torch.Tensor:
    pieces = [tensor.reshape(tensor.shape[0], -1) for tensor in _state_components(state)]
    return torch.cat(pieces, dim=1)


def matrix_to_state(matrix: torch.Tensor, template_state: LatentForecastState) -> LatentForecastState:
    batch_size = matrix.shape[0]
    rebuilt: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, tensor in vars(template_state).items():
        component_size = tensor[0].numel()
        component = matrix[:, cursor : cursor + component_size].reshape(batch_size, *tensor.shape[1:])
        rebuilt[name] = component
        cursor += component_size
    return LatentForecastState(**rebuilt)


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    original_device = matrix.device
    original_dtype = matrix.dtype
    matrix_cpu = matrix.detach().to(device="cpu", dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_cpu)
    clipped = torch.clamp(eigenvalues, min=0.0)
    sqrt_matrix = eigenvectors @ torch.diag(torch.sqrt(clipped)) @ eigenvectors.transpose(-1, -2)
    return sqrt_matrix.to(device=original_device, dtype=original_dtype)


def etkf_update(
    forecast_state: LatentForecastState,
    forecast_observations: torch.Tensor,
    observation: torch.Tensor,
    observation_variance: torch.Tensor,
) -> tuple[LatentForecastState, dict[str, torch.Tensor]]:
    state_matrix = state_to_matrix(forecast_state)
    ensemble_size = state_matrix.shape[0]

    x_mean = state_matrix.mean(dim=0, keepdim=True)
    y_mean = forecast_observations.mean(dim=0, keepdim=True)
    x_anomalies = (state_matrix - x_mean).transpose(0, 1)
    y_anomalies = (forecast_observations - y_mean).transpose(0, 1)

    precision = torch.reciprocal(observation_variance.clamp_min(1e-8))
    weighted_y = y_anomalies * precision[:, None]
    system_matrix = (ensemble_size - 1) * torch.eye(
        ensemble_size,
        device=state_matrix.device,
        dtype=state_matrix.dtype,
    ) + y_anomalies.transpose(0, 1) @ weighted_y

    posterior_weights = torch.linalg.solve(
        system_matrix,
        y_anomalies.transpose(0, 1) @ ((observation - y_mean.squeeze(0)) * precision),
    )
    transform = _matrix_sqrt_psd((ensemble_size - 1) * torch.linalg.inv(system_matrix))

    analysis_mean = x_mean + (x_anomalies @ posterior_weights.unsqueeze(-1)).transpose(0, 1)
    analysis_anomalies = x_anomalies @ transform
    analysis_matrix = analysis_mean + analysis_anomalies.transpose(0, 1)
    analysis_state = matrix_to_state(analysis_matrix, forecast_state)

    prior_var = state_matrix.var(dim=0, unbiased=False)
    post_var = analysis_matrix.var(dim=0, unbiased=False)
    diagnostics = {
        "innovation_norm": torch.linalg.norm(observation - y_mean.squeeze(0)) / observation.numel() ** 0.5,
        "forecast_spread": torch.sqrt(prior_var.mean().clamp_min(1e-8)),
        "analysis_spread": torch.sqrt(post_var.mean().clamp_min(1e-8)),
        "forecast_mean": x_mean.squeeze(0),
        "analysis_mean": analysis_mean.squeeze(0),
        "forecast_var": prior_var,
        "analysis_var": post_var,
    }
    return analysis_state, diagnostics
