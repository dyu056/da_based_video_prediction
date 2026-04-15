from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a summary figure for the finished server-side DA result.")
    parser.add_argument(
        "--host-metrics",
        type=str,
        default="outputs/server_da_result/da_host_openloop_v1_metrics.json",
    )
    parser.add_argument(
        "--da-metrics",
        type=str,
        default="outputs/server_da_result/da_eval_full_every2_v1_metrics.json",
    )
    parser.add_argument(
        "--comparison-image",
        type=str,
        default="outputs/server_da_result/da_eval_full_every2_v1_comparison_predictions.png",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/server_da_result/da_server_result_summary.png",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default="outputs/server_da_result/openloop_vs_da_comparison.png",
    )
    parser.add_argument(
        "--epoch-reference-output",
        type=str,
        default="outputs/server_da_result/openloop_epoch_vs_da_reference.png",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    host_metrics = load_json(args.host_metrics)
    da_metrics = load_json(args.da_metrics)

    history = host_metrics["history"]
    epochs = [entry["epoch"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    val_rmse = [entry["val"]["rmse"] for entry in history]
    val_pcc = [entry["val"]["pcc"] for entry in history]

    openloop = da_metrics["openloop"]
    da = da_metrics["da"]
    comparison = da_metrics["comparison"]

    figure = plt.figure(figsize=(15, 10))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.28, wspace=0.22)

    ax_loss = figure.add_subplot(grid[0, 0])
    ax_loss.plot(epochs, val_loss, color="#1f4e79", linewidth=2.5)
    ax_loss.set_title("Host Model Training: Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Loss")
    ax_loss.grid(alpha=0.25)

    best_epoch = int(np.argmin(val_loss))
    ax_loss.scatter([epochs[best_epoch]], [val_loss[best_epoch]], color="#c0392b", zorder=5)
    ax_loss.annotate(
        f"best epoch {epochs[best_epoch]}\nloss={val_loss[best_epoch]:.4f}",
        (epochs[best_epoch], val_loss[best_epoch]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc"},
    )

    ax_metric = figure.add_subplot(grid[0, 1])
    ax_metric.plot(epochs, val_rmse, color="#2e8b57", linewidth=2.5, label="Val RMSE")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Val RMSE", color="#2e8b57")
    ax_metric.tick_params(axis="y", labelcolor="#2e8b57")
    ax_metric.grid(alpha=0.25)

    ax_metric_2 = ax_metric.twinx()
    ax_metric_2.plot(epochs, val_pcc, color="#8b1e3f", linewidth=2.5, label="Val PCC")
    ax_metric_2.set_ylabel("Val PCC", color="#8b1e3f")
    ax_metric_2.tick_params(axis="y", labelcolor="#8b1e3f")
    ax_metric.set_title("Host Model Training: Validation RMSE / PCC")

    ax_bar = figure.add_subplot(grid[1, 0])
    labels = ["RMSE", "PCC"]
    x = np.arange(len(labels))
    width = 0.34
    open_values = [openloop["rmse"], openloop["pcc"]]
    da_values = [da["rmse"], da["pcc"]]
    bars_1 = ax_bar.bar(
        x - width / 2,
        open_values,
        width,
        label="Host rollout (no assimilation)",
        color="#7aa6c2",
    )
    bars_2 = ax_bar.bar(
        x + width / 2,
        da_values,
        width,
        label="Same host + latent ETKF",
        color="#d98f5f",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title("Finished Server Evaluation: Same Host, Different Test-Time Inference")
    ax_bar.legend(frameon=False)
    ax_bar.grid(axis="y", alpha=0.25)

    for bars in (bars_1, bars_2):
        for bar in bars:
            ax_bar.annotate(
                f"{bar.get_height():.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )

    summary_text = (
        f"RMSE delta: {comparison['rmse_delta']:+.4f}\n"
        f"PCC delta: {comparison['pcc_delta']:+.4f}\n"
        f"Runtime ratio: {comparison['runtime_ratio']:.2f}x\n"
        f"Observed steps: {da['observed_steps']:.0f}\n"
        f"Innovation norm: {da['innovation_norm']:.4f}\n"
        f"Obs: full frame every 2 steps"
    )
    ax_bar.text(
        1.08,
        0.96,
        summary_text,
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f8f8f8", "ec": "#cccccc"},
    )

    ax_img = figure.add_subplot(grid[1, 1])
    comparison_image = mpimg.imread(args.comparison_image)
    ax_img.imshow(comparison_image)
    ax_img.set_title("Prediction Grid: Host Rollout vs Host + Latent ETKF")
    ax_img.axis("off")

    figure.suptitle(
        "L40 Server Result: Open-Loop Host Rollout vs Same Host with Latent ETKF",
        fontsize=16,
        y=0.98,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    comparison_figure = plt.figure(figsize=(12, 5.8))
    comparison_grid = comparison_figure.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.28)

    comp_ax = comparison_figure.add_subplot(comparison_grid[0, 0])
    metric_labels = ["RMSE", "PCC"]
    x = np.arange(len(metric_labels))
    width = 0.35
    comp_open = [openloop["rmse"], openloop["pcc"]]
    comp_da = [da["rmse"], da["pcc"]]
    comp_bars_1 = comp_ax.bar(
        x - width / 2,
        comp_open,
        width,
        label="Host rollout (no assimilation)",
        color="#7aa6c2",
    )
    comp_bars_2 = comp_ax.bar(
        x + width / 2,
        comp_da,
        width,
        label="Same host + latent ETKF",
        color="#d98f5f",
    )
    comp_ax.set_xticks(x)
    comp_ax.set_xticklabels(metric_labels)
    comp_ax.set_ylabel("Metric value")
    comp_ax.set_title("Final Evaluation on 128 Validation Sequences")
    comp_ax.grid(axis="y", alpha=0.25)
    comp_ax.legend(frameon=False, loc="upper left")

    for bars in (comp_bars_1, comp_bars_2):
        for bar in bars:
            comp_ax.annotate(
                f"{bar.get_height():.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )

    delta_ax = comparison_figure.add_subplot(comparison_grid[0, 1])
    delta_labels = ["RMSE delta", "PCC delta"]
    delta_values = [comparison["rmse_delta"], comparison["pcc_delta"]]
    delta_colors = ["#2e8b57" if value < 0 else "#8b1e3f" for value in delta_values]
    y_pos = np.arange(len(delta_labels))
    delta_ax.barh(y_pos, delta_values, color=delta_colors, alpha=0.85)
    delta_ax.axvline(0.0, color="#333333", linewidth=1.0)
    delta_ax.set_yticks(y_pos)
    delta_ax.set_yticklabels(delta_labels)
    delta_ax.set_title("Change from Open-Loop")
    delta_ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(delta_values):
        delta_ax.annotate(
            f"{value:+.4f}",
            (value, idx),
            ha="left" if value >= 0 else "right",
            va="center",
            xytext=(6 if value >= 0 else -6, 0),
            textcoords="offset points",
            fontsize=10,
        )

    comparison_figure.text(
        0.69,
        0.18,
        (
            f"Runtime ratio: {comparison['runtime_ratio']:.2f}x\n"
            f"Observed steps: {da['observed_steps']:.0f}\n"
            f"Observation: full frame every 2 steps\n"
            f"Innovation norm: {da['innovation_norm']:.4f}"
        ),
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f8f8f8", "ec": "#cccccc"},
    )
    comparison_figure.suptitle(
        "Open-Loop vs Latent ETKF on the Same Trained Host Model",
        fontsize=15,
        y=0.98,
    )
    comparison_output_path = Path(args.comparison_output)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_figure.savefig(comparison_output_path, dpi=180, bbox_inches="tight")
    plt.close(comparison_figure)

    epoch_reference_figure, epoch_axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    epoch_axes[0].plot(epochs, val_rmse, color="#1f4e79", linewidth=2.5, label="Open-loop host per epoch")
    epoch_axes[0].axhline(
        openloop["rmse"],
        color="#7aa6c2",
        linestyle="--",
        linewidth=2.0,
        label="Open-loop final eval",
    )
    epoch_axes[0].axhline(
        da["rmse"],
        color="#d98f5f",
        linestyle="-.",
        linewidth=2.0,
        label="Host + latent ETKF final eval",
    )
    epoch_axes[0].set_title("RMSE Across Host Training")
    epoch_axes[0].set_xlabel("Epoch")
    epoch_axes[0].set_ylabel("RMSE")
    epoch_axes[0].grid(alpha=0.25)
    epoch_axes[0].legend(frameon=False, fontsize=9)

    epoch_axes[1].plot(epochs, val_pcc, color="#2e8b57", linewidth=2.5, label="Open-loop host per epoch")
    epoch_axes[1].axhline(
        openloop["pcc"],
        color="#7aa6c2",
        linestyle="--",
        linewidth=2.0,
        label="Open-loop final eval",
    )
    epoch_axes[1].axhline(
        da["pcc"],
        color="#d98f5f",
        linestyle="-.",
        linewidth=2.0,
        label="Host + latent ETKF final eval",
    )
    epoch_axes[1].set_title("PCC Across Host Training")
    epoch_axes[1].set_xlabel("Epoch")
    epoch_axes[1].set_ylabel("PCC")
    epoch_axes[1].grid(alpha=0.25)
    epoch_axes[1].legend(frameon=False, fontsize=9)

    epoch_reference_figure.suptitle(
        "Open-Loop Epoch Curves with Final DA Reference Lines",
        fontsize=15,
        y=1.02,
    )
    epoch_reference_figure.text(
        0.5,
        -0.02,
        "DA here is a test-time inference method, so these are reference lines rather than a separately trained DA epoch curve.",
        ha="center",
        fontsize=10,
    )
    epoch_reference_output_path = Path(args.epoch_reference_output)
    epoch_reference_output_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_reference_figure.savefig(epoch_reference_output_path, dpi=180, bbox_inches="tight")
    plt.close(epoch_reference_figure)
    print(f"Saved figure to {output_path}")
    print(f"Saved comparison figure to {comparison_output_path}")
    print(f"Saved epoch reference figure to {epoch_reference_output_path}")


if __name__ == "__main__":
    main()
