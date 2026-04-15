from __future__ import annotations

import argparse
import json
import random
import sys
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
from da_video.metrics import pattern_correlation, rmse
from da_video.models import OpenLoopVideoPredictor
from da_video.observation import SparseObservationOperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate open-loop and latent ETKF over a series of host checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/da_epoch_curve")
    parser.add_argument("--data-root", type=str, default="data")
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
    return batch[:, :context_frames], batch[:, context_frames : context_frames + pred_frames]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[OpenLoopVideoPredictor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint["args"]
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
) -> torch.Tensor:
    base_state, _ = model.initialize_state(context)
    ensemble_state = sample_state_ensemble(
        base_state,
        ensemble_size=ensemble_size,
        latent_noise_std=latent_noise,
        hidden_noise_std=hidden_noise,
        cell_noise_std=cell_noise,
    )
    pred_frames = []
    for step_idx in range(future.shape[1]):
        forecast_state, step_outputs = model.forecast_step(ensemble_state)
        current_state = forecast_state
        current_frame = step_outputs["pred_frame"].mean(dim=0, keepdim=True)
        if operator.has_observation(step_idx):
            observation = operator.observe(future[:, step_idx], add_noise=True).squeeze(0)
            forecast_observations = operator.project(step_outputs["pred_frame"])
            analysis_state, _ = etkf_update(
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
        pred_frames.append(current_frame)
        ensemble_state = current_state
    return torch.stack(pred_frames, dim=1)


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    operator: SparseObservationOperator,
) -> dict[str, float]:
    model, train_args = load_model(checkpoint_path, device=device)
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

    open_metrics = []
    da_metrics = []
    for batch in loader:
        batch = batch.to(device=device, dtype=torch.float32)
        context, future = split_sequence(batch, context_frames=context_frames, pred_frames=pred_frames)
        for sample_idx in range(context.shape[0]):
            context_sample = context[sample_idx : sample_idx + 1]
            future_sample = future[sample_idx : sample_idx + 1]
            open_pred = run_openloop_rollout(model, context_sample, pred_frames=pred_frames)
            da_pred = run_assimilated_rollout(
                model=model,
                context=context_sample,
                future=future_sample,
                operator=operator,
                ensemble_size=args.ensemble_size,
                latent_noise=args.latent_noise,
                hidden_noise=args.hidden_noise,
                cell_noise=args.cell_noise,
            )
            open_metrics.append(
                {
                    "rmse": float(rmse(open_pred, future_sample).cpu()),
                    "pcc": float(pattern_correlation(open_pred, future_sample).cpu()),
                }
            )
            da_metrics.append(
                {
                    "rmse": float(rmse(da_pred, future_sample).cpu()),
                    "pcc": float(pattern_correlation(da_pred, future_sample).cpu()),
                }
            )

    open_summary = aggregate(open_metrics)
    da_summary = aggregate(da_metrics)
    epoch_number = int(checkpoint_path.stem.split("_")[-1])
    return {
        "epoch": epoch_number,
        "openloop_rmse": open_summary["rmse"],
        "openloop_pcc": open_summary["pcc"],
        "da_rmse": da_summary["rmse"],
        "da_pcc": da_summary["pcc"],
    }


def plot_curves(records: list[dict[str, float]], output_path: Path) -> None:
    epochs = [record["epoch"] for record in records]
    open_rmse = [record["openloop_rmse"] for record in records]
    da_rmse = [record["da_rmse"] for record in records]
    open_pcc = [record["openloop_pcc"] for record in records]
    da_pcc = [record["da_pcc"] for record in records]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    axes[0].plot(epochs, open_rmse, color="#7aa6c2", linewidth=2.5, marker="o", label="Host rollout (no assimilation)")
    axes[0].plot(epochs, da_rmse, color="#d98f5f", linewidth=2.5, marker="s", label="Same host + latent ETKF")
    axes[0].set_title("RMSE vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(epochs, open_pcc, color="#7aa6c2", linewidth=2.5, marker="o", label="Host rollout (no assimilation)")
    axes[1].plot(epochs, da_pcc, color="#d98f5f", linewidth=2.5, marker="s", label="Same host + latent ETKF")
    axes[1].set_title("PCC vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PCC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    figure.suptitle("True Open-Loop vs Latent ETKF Evolution Across Host Checkpoints", fontsize=15, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found under {checkpoint_dir}")

    operator = SparseObservationOperator(
        kind=args.observation_kind,
        image_size=64,
        observe_every=args.observe_every,
        noise_std=args.observation_noise,
        mask_fraction=args.mask_fraction,
        downsample_size=args.downsample_size,
        seed=args.seed,
    )

    records = []
    for checkpoint_path in checkpoints:
        record = evaluate_checkpoint(checkpoint_path, args=args, device=device, operator=operator)
        records.append(record)
        print(
            f"Epoch {record['epoch']:03d} | "
            f"open_rmse={record['openloop_rmse']:.4f} da_rmse={record['da_rmse']:.4f} "
            f"open_pcc={record['openloop_pcc']:.4f} da_pcc={record['da_pcc']:.4f}"
        )

    records.sort(key=lambda item: item["epoch"])
    metrics_path = output_dir / "da_epoch_curve_metrics.json"
    metrics_path.write_text(json.dumps({"config": vars(args), "records": records}, indent=2), encoding="utf-8")
    plot_curves(records, output_dir / "da_epoch_curve.png")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved plot to {output_dir / 'da_epoch_curve.png'}")


if __name__ == "__main__":
    main()
