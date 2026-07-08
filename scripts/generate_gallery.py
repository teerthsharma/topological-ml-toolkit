from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import topoml


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def vr_filtration() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [0.4, 0.9],
            [2.0, 0.3],
            [2.4, 1.0],
        ],
        dtype=float,
    )
    radii = [0.35, 0.85, 1.35]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, radius in zip(axes, radii):
        ax.scatter(points[:, 0], points[:, 1], s=42, color="#2457a6")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) <= radius:
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color="#e07a5f")
        ax.set_title(f"radius = {radius}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Vietoris-Rips filtration")
    save(fig, "vr_filtration.svg")


def barcode_and_betti() -> None:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)
    finite = diagram.finite_pairs(0)

    fig, ax = plt.subplots(figsize=(6, 2.7))
    for row, pair in enumerate(finite):
        ax.hlines(row, pair.birth, pair.death, color="#2457a6", linewidth=4)
        ax.scatter([pair.birth, pair.death], [row, row], color="#e07a5f", zorder=3)
    ax.set_xlabel("radius")
    ax.set_ylabel("feature")
    ax.set_title("H0 barcode")
    save(fig, "barcode.svg")

    radii = np.linspace(0, 6, 128)
    betti = [diagram.betti_at(float(radius)).beta0 for radius in radii]
    fig, ax = plt.subplots(figsize=(6, 2.7))
    ax.plot(radii, betti, color="#2457a6", linewidth=3)
    ax.set_xlabel("radius")
    ax.set_ylabel("beta0")
    ax.set_title("Betti curve")
    save(fig, "betti_curve.svg")


def delay_embedding() -> None:
    t = np.linspace(0, 4 * math.pi, 128)
    signal = np.sin(t)
    cloud = topoml.time_delay_embedding(signal, dimension=2, tau=8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(t, signal, color="#2457a6")
    axes[0].set_title("signal")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("value")
    axes[1].scatter(cloud[:, 0], cloud[:, 1], s=12, color="#e07a5f")
    axes[1].set_title("delay embedding")
    axes[1].set_xlabel("x(t)")
    axes[1].set_ylabel("x(t + tau)")
    axes[1].set_aspect("equal")
    save(fig, "time_delay_embedding.svg")


def main() -> None:
    vr_filtration()
    barcode_and_betti()
    delay_embedding()


if __name__ == "__main__":
    main()

