from __future__ import annotations

import argparse
import json
from pathlib import Path


PANEL_WIDTH = 420
PANEL_HEIGHT = 260
MARGIN_LEFT = 56
MARGIN_RIGHT = 20
MARGIN_TOP = 34
MARGIN_BOTTOM = 42
PANEL_GAP = 24
SVG_WIDTH = PANEL_WIDTH * 3 + PANEL_GAP * 2
SVG_HEIGHT = PANEL_HEIGHT + 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/validation curves from a baseline metrics.json file.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json produced by train_baseline.py")
    parser.add_argument("--output", type=str, required=True, help="Path to the output SVG figure")
    parser.add_argument("--title", type=str, default="Training Curves", help="Figure title")
    return parser.parse_args()


def polyline_points(values: list[float], x0: float, y0: float, width: float, height: float) -> str:
    if len(values) == 1:
        return f"{x0 + width / 2:.2f},{y0 + height / 2:.2f}"

    vmin = min(values)
    vmax = max(values)
    vrange = max(vmax - vmin, 1e-8)
    points = []
    for idx, value in enumerate(values):
        x = x0 + idx * width / (len(values) - 1)
        y = y0 + height - (value - vmin) * height / vrange
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def draw_axis(values: list[float], x0: float, y0: float, width: float, height: float, title: str, ylabel: str) -> str:
    if values:
        vmin = min(values)
        vmax = max(values)
    else:
        vmin = 0.0
        vmax = 1.0
    vrange = max(vmax - vmin, 1e-8)
    axis = []
    axis.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" fill="white" stroke="#cccccc" stroke-width="1"/>')
    for tick_idx in range(5):
        tick_y = y0 + tick_idx * height / 4
        tick_value = vmax - tick_idx * vrange / 4
        axis.append(f'<line x1="{x0:.2f}" y1="{tick_y:.2f}" x2="{x0 + width:.2f}" y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        axis.append(
            f'<text x="{x0 - 8:.2f}" y="{tick_y + 4:.2f}" font-size="11" text-anchor="end" fill="#374151">{tick_value:.3f}</text>'
        )
    axis.append(f'<text x="{x0 + width / 2:.2f}" y="{y0 - 10:.2f}" font-size="16" text-anchor="middle" fill="#111827">{title}</text>')
    axis.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 + height + 30:.2f}" font-size="12" text-anchor="middle" fill="#374151">Epoch</text>'
    )
    axis.append(
        f'<text x="{x0 - 42:.2f}" y="{y0 + height / 2:.2f}" font-size="12" text-anchor="middle" fill="#374151" transform="rotate(-90 {x0 - 42:.2f} {y0 + height / 2:.2f})">{ylabel}</text>'
    )
    return "\n".join(axis)


def draw_series(values: list[float], x0: float, y0: float, width: float, height: float, color: str) -> str:
    points = polyline_points(values, x0, y0, width, height)
    circles = []
    if values:
        vmin = min(values)
        vmax = max(values)
        vrange = max(vmax - vmin, 1e-8)
        for idx, value in enumerate(values):
            x = x0 + idx * width / max(len(values) - 1, 1)
            y = y0 + height - (value - vmin) * height / vrange
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="{color}"/>')
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>\n' + "\n".join(circles)


def legend(x: float, y: float) -> str:
    return "\n".join(
        [
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 18:.2f}" y2="{y:.2f}" stroke="#2d6a8e" stroke-width="2.6"/>',
            f'<text x="{x + 24:.2f}" y="{y + 4:.2f}" font-size="12" fill="#374151">Train</text>',
            f'<line x1="{x + 90:.2f}" y1="{y:.2f}" x2="{x + 108:.2f}" y2="{y:.2f}" stroke="#c85c3a" stroke-width="2.6"/>',
            f'<text x="{x + 114:.2f}" y="{y + 4:.2f}" font-size="12" fill="#374151">Validation</text>',
        ]
    )


def panel_svg(
    train_values: list[float],
    val_values: list[float],
    panel_index: int,
    title: str,
    ylabel: str,
) -> str:
    panel_x = panel_index * (PANEL_WIDTH + PANEL_GAP)
    plot_x = panel_x + MARGIN_LEFT
    plot_y = MARGIN_TOP + 24
    plot_width = PANEL_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = PANEL_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    combined = train_values + val_values
    axis = draw_axis(combined, plot_x, plot_y, plot_width, plot_height, title, ylabel)
    train_svg = draw_series(train_values, plot_x, plot_y, plot_width, plot_height, "#2d6a8e")
    val_svg = draw_series(val_values, plot_x, plot_y, plot_width, plot_height, "#c85c3a")
    return "\n".join([axis, train_svg, val_svg])


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_path = Path(args.output)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload["history"]

    train_loss = [record["train"]["loss"] for record in history]
    val_loss = [record["val"]["loss"] for record in history]
    train_rmse = [record["train"]["rmse"] for record in history]
    val_rmse = [record["val"]["rmse"] for record in history]
    train_pcc = [record["train"]["pcc"] for record in history]
    val_pcc = [record["val"]["pcc"] for record in history]

    sections = [
        panel_svg(train_loss, val_loss, 0, "Loss vs Epoch", "Loss"),
        panel_svg(train_rmse, val_rmse, 1, "RMSE vs Epoch", "RMSE"),
        panel_svg(train_pcc, val_pcc, 2, "PCC vs Epoch", "PCC"),
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
<rect width="100%" height="100%" fill="white"/>
<text x="{SVG_WIDTH / 2:.2f}" y="28" font-size="20" text-anchor="middle" fill="#111827">{args.title}</text>
{legend(18, 54)}
{"".join(sections)}
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
