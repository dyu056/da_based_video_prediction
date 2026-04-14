from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from da_video.assimilation import etkf_update, mean_state, sample_state_ensemble, state_to_matrix
from da_video.data import MovingMNISTDataset
from da_video.metrics import (
    gaussian_entropy_diag,
    gaussian_mutual_information_diag,
    gaussian_relative_entropy_diag,
    pattern_correlation,
    rmse,
)
from da_video.models import OpenLoopVideoPredictor
from da_video.observation import SparseObservationOperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a first latent ETKF experiment on Moving MNIST.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/da_experiment")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--observation-kind", type=str, default="full", choices=["full", "masked", "lowres"])
    parser.add_argument("--observe-every", type=int, default=2)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--observation-noise", type=float, default=0.02)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--latent-noise", type=float, default=0.03)
    parser.add_argument("--hidden-noise", type=float, default=0.01)
    parser.add_argument("--cell-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        return torch.device("mps")
    if device_name == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(
    data_root: str,
    split: str,
    num_sequences: int,
    seq_len: int,
    image_size: int,
    num_digits: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = MovingMNISTDataset(
        root=data_root,
        split=split,
        num_sequences=num_sequences,
        seq_len=seq_len,
        image_size=image_size,
        num_digits=num_digits,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def split_sequence(batch: torch.Tensor, context_frames: int, pred_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    context = batch[:, :context_frames]
    future = batch[:, context_frames : context_frames + pred_frames]
    return context, future


def load_openloop_model(checkpoint_path: Path, device: torch.device) -> tuple[OpenLoopVideoPredictor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    model = OpenLoopVideoPredictor(
        latent_dim=train_args.get("latent_dim", 64),
        hidden_dim=train_args.get("hidden_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, train_args


@torch.no_grad()
def run_openloop_rollout(model: OpenLoopVideoPredictor, context: torch.Tensor, pred_frames: int) -> torch.Tensor:
    return model(context, pred_steps=pred_frames)["pred_frames"]


@torch.no_grad()
def run_assimilated_rollout(
    model: OpenLoopVideoPredictor,
    context: torch.Tensor,
    future: torch.Tensor,
    operator: SparseObservationOperator,
    ensemble_size: int,
    latent_noise: float,
    hidden_noise: float,
    cell_noise: float,
) -> dict[str, torch.Tensor | float]:
    base_state, _ = model.initialize_state(context)
    ensemble_state = sample_state_ensemble(
        base_state,
        ensemble_size=ensemble_size,
        latent_noise_std=latent_noise,
        hidden_noise_std=hidden_noise,
        cell_noise_std=cell_noise,
    )

    pred_frames = []
    entropy_values = []
    relative_entropy_values = []
    mutual_information_values = []
    innovation_values = []
    spread_values = []
    observed_steps = 0

    for step_idx in range(future.shape[1]):
        forecast_state, step_outputs = model.forecast_step(ensemble_state)
        forecast_matrix = state_to_matrix(forecast_state)
        forecast_var = forecast_matrix.var(dim=0, unbiased=False)
        entropy_values.append(float(gaussian_entropy_diag(forecast_var).cpu()))
        spread_values.append(float(torch.sqrt(forecast_var.mean().clamp_min(1e-8)).cpu()))

        current_state = forecast_state
        current_frame = step_outputs["pred_frame"].mean(dim=0, keepdim=True)

        if operator.has_observation(step_idx):
            observation = operator.observe(future[:, step_idx], add_noise=True).squeeze(0)
            forecast_observations = operator.project(step_outputs["pred_frame"])
            analysis_state, diagnostics = etkf_update(
                forecast_state=forecast_state,
                forecast_observations=forecast_observations,
                observation=observation,
                observation_variance=operator.noise_variance(
                    forecast_observations.shape[-1],
                    device=forecast_observations.device,
                ),
            )
            current_state = analysis_state
            current_frame = model.decode_latents(mean_state(analysis_state).previous_latent)
            relative_entropy_values.append(
                float(
                    gaussian_relative_entropy_diag(
                        mean_p=diagnostics["analysis_mean"],
                        variance_p=diagnostics["analysis_var"],
                        mean_q=diagnostics["forecast_mean"],
                        variance_q=diagnostics["forecast_var"],
                    ).cpu()
                )
            )
            mutual_information_values.append(
                float(
                    gaussian_mutual_information_diag(
                        prior_variance=diagnostics["forecast_var"],
                        posterior_variance=diagnostics["analysis_var"],
                    ).cpu()
                )
            )
            innovation_values.append(float(diagnostics["innovation_norm"].cpu()))
            observed_steps += 1

        pred_frames.append(current_frame)
        ensemble_state = current_state

    stacked_predictions = torch.stack(pred_frames, dim=1)
    default_zero = [0.0]
    return {
        "pred_frames": stacked_predictions,
        "entropy": float(np.mean(entropy_values or default_zero)),
        "relative_entropy": float(np.mean(relative_entropy_values or default_zero)),
        "mutual_information": float(np.mean(mutual_information_values or default_zero)),
        "innovation_norm": float(np.mean(innovation_values or default_zero)),
        "spread": float(np.mean(spread_values or default_zero)),
        "observed_steps": float(observed_steps),
    }


def aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}


def save_comparison_grid(
    context: torch.Tensor,
    future: torch.Tensor,
    openloop_pred: torch.Tensor,
    da_pred: torch.Tensor,
    operator: SparseObservationOperator,
    output_path: Path,
) -> None:
    pred_frames = future.shape[1]
    figure, axes = plt.subplots(4, pred_frames, figsize=(1.8 * pred_frames, 7.0))
    for frame_idx in range(pred_frames):
        context_idx = max(0, context.shape[1] - pred_frames + frame_idx)
        axes[0, frame_idx].imshow(context[0, context_idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, frame_idx].set_title(f"Context {frame_idx + 1}")
        axes[1, frame_idx].imshow(future[0, frame_idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, frame_idx].set_title(f"Target {frame_idx + 1}")
        axes[2, frame_idx].imshow(openloop_pred[0, frame_idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, frame_idx].set_title(f"Open {frame_idx + 1}")
        label = f"DA {frame_idx + 1}"
        if operator.has_observation(frame_idx):
            label += "*"
        axes[3, frame_idx].imshow(da_pred[0, frame_idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[3, frame_idx].set_title(label)
        for row in range(4):
            axes[row, frame_idx].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, train_args = load_openloop_model(checkpoint_path, device=device)
    context_frames = train_args.get("context_frames", 10)
    pred_frames = train_args.get("pred_frames", 10)
    seq_len = train_args.get("seq_len", context_frames + pred_frames)
    image_size = train_args.get("image_size", 64)
    num_digits = train_args.get("num_digits", 2)

    loader = make_loader(
        data_root=args.data_root,
        split=args.split,
        num_sequences=args.eval_sequences,
        seq_len=seq_len,
        image_size=image_size,
        num_digits=num_digits,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    operator = SparseObservationOperator(
        kind=args.observation_kind,
        image_size=image_size,
        observe_every=args.observe_every,
        noise_std=args.observation_noise,
        mask_fraction=args.mask_fraction,
        downsample_size=args.downsample_size,
        seed=args.seed,
    )

    openloop_metrics = []
    da_metrics = []
    sample_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    openloop_runtime = 0.0
    da_runtime = 0.0

    for batch in loader:
        batch = batch.to(device=device, dtype=torch.float32)
        context, future = split_sequence(batch, context_frames=context_frames, pred_frames=pred_frames)
        batch_size = context.shape[0]

        for sample_idx in range(batch_size):
            context_sample = context[sample_idx : sample_idx + 1]
            future_sample = future[sample_idx : sample_idx + 1]

            start = time.perf_counter()
            openloop_pred = run_openloop_rollout(model, context_sample, pred_frames=pred_frames)
            openloop_runtime += time.perf_counter() - start

            start = time.perf_counter()
            da_result = run_assimilated_rollout(
                model=model,
                context=context_sample,
                future=future_sample,
                operator=operator,
                ensemble_size=args.ensemble_size,
                latent_noise=args.latent_noise,
                hidden_noise=args.hidden_noise,
                cell_noise=args.cell_noise,
            )
            da_runtime += time.perf_counter() - start

            openloop_metrics.append(
                {
                    "rmse": float(rmse(openloop_pred, future_sample).cpu()),
                    "pcc": float(pattern_correlation(openloop_pred, future_sample).cpu()),
                }
            )
            da_metrics.append(
                {
                    "rmse": float(rmse(da_result["pred_frames"], future_sample).cpu()),
                    "pcc": float(pattern_correlation(da_result["pred_frames"], future_sample).cpu()),
                    "entropy": float(da_result["entropy"]),
                    "relative_entropy": float(da_result["relative_entropy"]),
                    "mutual_information": float(da_result["mutual_information"]),
                    "innovation_norm": float(da_result["innovation_norm"]),
                    "spread": float(da_result["spread"]),
                    "observed_steps": float(da_result["observed_steps"]),
                }
            )

            if sample_payload is None:
                sample_payload = (
                    context_sample.detach().cpu(),
                    future_sample.detach().cpu(),
                    openloop_pred.detach().cpu(),
                    da_result["pred_frames"].detach().cpu(),
                )

    openloop_summary = aggregate(openloop_metrics)
    da_summary = aggregate(da_metrics)
    comparison = {
        "rmse_delta": da_summary["rmse"] - openloop_summary["rmse"],
        "pcc_delta": da_summary["pcc"] - openloop_summary["pcc"],
        "runtime_ratio": da_runtime / max(openloop_runtime, 1e-8),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "config": vars(args),
        "checkpoint": str(checkpoint_path),
        "openloop": openloop_summary,
        "da": da_summary,
        "comparison": comparison,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if sample_payload is not None:
        save_comparison_grid(
            context=sample_payload[0],
            future=sample_payload[1],
            openloop_pred=sample_payload[2],
            da_pred=sample_payload[3],
            operator=operator,
            output_path=output_dir / "comparison_predictions.png",
        )

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Open-loop summary: {openloop_summary}")
    print(f"DA summary: {da_summary}")
    print(f"Comparison: {comparison}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
