from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from da_video.data import MovingMNISTDataset
from da_video.metrics import pattern_correlation, rmse
from da_video.models import OpenLoopVideoPredictor, SimVPPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Moving MNIST video prediction baselines.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--model", type=str, default="openloop", choices=["openloop", "simvp"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-sequences", type=int, default=4000)
    parser.add_argument("--val-sequences", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--context-frames", type=int, default=10)
    parser.add_argument("--pred-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-digits", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--simvp-hid-s", type=int, default=32)
    parser.add_argument("--simvp-hid-t", type=int, default=128)
    parser.add_argument("--simvp-n-s", type=int, default=4)
    parser.add_argument("--simvp-n-t", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--future-recon-weight", type=float, default=0.1)
    parser.add_argument("--latent-weight", type=float, default=0.5)
    parser.add_argument("--foreground-weight", type=float, default=6.0)
    parser.add_argument("--save-every-epoch", action="store_true")
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


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = MovingMNISTDataset(
        root=args.data_root,
        split="train",
        num_sequences=args.train_sequences,
        seq_len=args.seq_len,
        image_size=args.image_size,
        num_digits=args.num_digits,
        seed=args.seed,
    )
    val_dataset = MovingMNISTDataset(
        root=args.data_root,
        split="val",
        num_sequences=args.val_sequences,
        seq_len=args.seq_len,
        image_size=args.image_size,
        num_digits=args.num_digits,
        seed=args.seed,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": resolve_device(args.device).type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def split_sequence(batch: torch.Tensor, context_frames: int, pred_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    context = batch[:, :context_frames]
    future = batch[:, context_frames : context_frames + pred_frames]
    return context, future


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "simvp":
        return SimVPPredictor(
            shape_in=(args.context_frames, 1, args.image_size, args.image_size),
            hid_s=args.simvp_hid_s,
            hid_t=args.simvp_hid_t,
            n_s=args.simvp_n_s,
            n_t=args.simvp_n_t,
        )
    return OpenLoopVideoPredictor(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)


def compute_losses(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    context: torch.Tensor,
    future: torch.Tensor,
    recon_weight: float,
    future_recon_weight: float,
    latent_weight: float,
    foreground_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, pred_steps, channels, height, width = future.shape
    pixel_weights = 1.0 + foreground_weight * future
    pred_loss = torch.mean(pixel_weights * (outputs["pred_frames"] - future) ** 2)
    total_loss = pred_loss

    recon_loss = future.new_tensor(0.0)
    latent_loss = future.new_tensor(0.0)
    future_recon_loss = future.new_tensor(0.0)

    if "recon_context" in outputs and recon_weight > 0.0:
        recon_loss = F.mse_loss(outputs["recon_context"], context)
        total_loss = total_loss + recon_weight * recon_loss

    if "pred_latents" in outputs and latent_weight > 0.0 and hasattr(model, "encoder") and hasattr(model, "decoder"):
        future_flat = future.reshape(batch_size * pred_steps, channels, height, width)
        with torch.no_grad():
            target_future_latents = model.encoder(future_flat)
            latent_channels, latent_height, latent_width = target_future_latents.shape[1:]
            target_future_latents = target_future_latents.reshape(
                batch_size,
                pred_steps,
                latent_channels,
                latent_height,
                latent_width,
            )

        latent_loss = F.mse_loss(outputs["pred_latents"], target_future_latents)
        total_loss = total_loss + latent_weight * latent_loss

        if future_recon_weight > 0.0:
            future_reconstruction = model.decoder(
                target_future_latents.reshape(batch_size * pred_steps, latent_channels, latent_height, latent_width)
            )
            future_reconstruction = future_reconstruction.reshape(batch_size, pred_steps, channels, height, width)
            future_recon_loss = F.mse_loss(future_reconstruction, future)
            total_loss = total_loss + future_recon_weight * future_recon_loss

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "future_recon_loss": float(future_recon_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "rmse": float(rmse(outputs["pred_frames"], future).detach().cpu()),
        "pcc": float(pattern_correlation(outputs["pred_frames"], future).detach().cpu()),
    }
    return total_loss, metrics


def aggregate_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    context_frames: int,
    pred_frames: int,
    recon_weight: float,
    future_recon_weight: float,
    latent_weight: float,
    foreground_weight: float,
) -> dict[str, float]:
    model.eval()
    collected = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            context, future = split_sequence(batch, context_frames=context_frames, pred_frames=pred_frames)
            outputs = model(context, pred_steps=pred_frames)
            _, metrics = compute_losses(
                model,
                outputs,
                context=context,
                future=future,
                recon_weight=recon_weight,
                future_recon_weight=future_recon_weight,
                latent_weight=latent_weight,
                foreground_weight=foreground_weight,
            )
            collected.append(metrics)
    return aggregate_metric_dict(collected)


def save_prediction_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    context_frames: int,
    pred_frames: int,
    output_path: Path,
) -> None:
    model.eval()
    batch = next(iter(loader)).to(device=device, dtype=torch.float32)
    context, future = split_sequence(batch, context_frames=context_frames, pred_frames=pred_frames)
    with torch.no_grad():
        outputs = model(context, pred_steps=pred_frames)

    sample_index = 0
    figure, axes = plt.subplots(3, pred_frames, figsize=(1.8 * pred_frames, 5.5))
    for frame_idx in range(pred_frames):
        axes[0, frame_idx].imshow(
            context[sample_index, max(0, context_frames - pred_frames + frame_idx), 0].detach().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, frame_idx].set_title(f"Context {frame_idx + 1}")
        axes[1, frame_idx].imshow(
            future[sample_index, frame_idx, 0].detach().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, frame_idx].set_title(f"Target {frame_idx + 1}")
        axes[2, frame_idx].imshow(
            outputs["pred_frames"][sample_index, frame_idx, 0].detach().cpu(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[2, frame_idx].set_title(f"Pred {frame_idx + 1}")
        for row in range(3):
            axes[row, frame_idx].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader = make_dataloaders(args)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"
    epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
    if args.save_every_epoch:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training model={args.model} on device={device} "
        f"(num_workers={args.num_workers}) -> {output_dir}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metrics = []

        for batch in train_loader:
            batch = batch.to(device=device, dtype=torch.float32)
            context, future = split_sequence(batch, context_frames=args.context_frames, pred_frames=args.pred_frames)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(context, pred_steps=args.pred_frames)
            loss, metrics = compute_losses(
                model,
                outputs,
                context=context,
                future=future,
                recon_weight=args.recon_weight,
                future_recon_weight=args.future_recon_weight,
                latent_weight=args.latent_weight,
                foreground_weight=args.foreground_weight,
            )
            loss.backward()
            optimizer.step()
            train_metrics.append(metrics)

        train_summary = aggregate_metric_dict(train_metrics)
        val_summary = evaluate(
            model,
            val_loader,
            device=device,
            context_frames=args.context_frames,
            pred_frames=args.pred_frames,
            recon_weight=args.recon_weight,
            future_recon_weight=args.future_recon_weight,
            latent_weight=args.latent_weight,
            foreground_weight=args.foreground_weight,
        )
        epoch_summary = {"epoch": epoch, "train": train_summary, "val": val_summary}
        history.append(epoch_summary)

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        }

        if val_summary["loss"] < best_val_loss:
            best_val_loss = val_summary["loss"]
            torch.save(checkpoint_payload, best_checkpoint_path)

        if args.save_every_epoch:
            torch.save(checkpoint_payload, epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_summary['loss']:.4f} val_loss={val_summary['loss']:.4f} "
            f"val_rmse={val_summary['rmse']:.4f} val_pcc={val_summary['pcc']:.4f}"
        )

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "history": history}, handle, indent=2)

    save_prediction_grid(
        model,
        val_loader,
        device=device,
        context_frames=args.context_frames,
        pred_frames=args.pred_frames,
        output_path=output_dir / "sample_predictions.png",
    )

    print(f"Saved checkpoint to {best_checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    if args.save_every_epoch:
        print(f"Saved per-epoch checkpoints to {epoch_checkpoint_dir}")


if __name__ == "__main__":
    main()
