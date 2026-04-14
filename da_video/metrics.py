from __future__ import annotations

import math

import torch


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def pattern_correlation(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_flat = prediction.reshape(prediction.shape[0], prediction.shape[1], -1)
    target_flat = target.reshape(target.shape[0], target.shape[1], -1)

    pred_centered = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=-1, keepdim=True)

    numerator = torch.sum(pred_centered * target_centered, dim=-1)
    denominator = torch.sqrt(
        torch.sum(pred_centered**2, dim=-1) * torch.sum(target_centered**2, dim=-1) + eps
    )
    return (numerator / denominator).mean()


def gaussian_entropy_diag(variance: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    safe_variance = variance.clamp_min(eps)
    return 0.5 * torch.mean(torch.log(2.0 * math.pi * math.e * safe_variance))


def gaussian_relative_entropy_diag(
    mean_p: torch.Tensor,
    variance_p: torch.Tensor,
    mean_q: torch.Tensor,
    variance_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    safe_var_p = variance_p.clamp_min(eps)
    safe_var_q = variance_q.clamp_min(eps)
    mean_delta_sq = (mean_p - mean_q) ** 2
    kl_terms = torch.log(safe_var_q / safe_var_p) + (safe_var_p + mean_delta_sq) / safe_var_q - 1.0
    return 0.5 * torch.mean(kl_terms)


def gaussian_mutual_information_diag(
    prior_variance: torch.Tensor,
    posterior_variance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    safe_prior = prior_variance.clamp_min(eps)
    safe_posterior = posterior_variance.clamp_min(eps)
    return 0.5 * torch.mean(torch.clamp(torch.log(safe_prior / safe_posterior), min=0.0))
