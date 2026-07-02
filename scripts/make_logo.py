#!/usr/bin/env python3
"""Generate the Foveance brand assets (SVG + PNG) into assets/.

The mark is a *foveated gaze*: a bright focal core surrounded by concentric rings of dots that
grow smaller and fainter toward the periphery (high fidelity at the fovea, low at the edge), with
a leading-edge arc that reads as looking forward. That is the thesis in one glyph: spend fidelity
where attention will land next. Run:  python scripts/make_logo.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.normpath(os.path.join(HERE, "..", "assets"))
os.makedirs(OUT, exist_ok=True)

INDIGO = "#4F46E5"
MID = "#6D5DE3"
VIOLET = "#7C3AED"
AMBER = "#F59E0B"
INK = "#0F172A"
PAPER = "#F8FAFC"
GRAY = "#64748B"

# (radius, n_dots, dot_radius, color, alpha) -- dense+bright at centre, sparse+faint outside.
RINGS = [
    (0.00, 1, 0.082, AMBER, 1.00),
    (0.165, 6, 0.046, INDIGO, 1.00),
    (0.305, 12, 0.032, MID, 0.74),
    (0.445, 20, 0.022, VIOLET, 0.42),
]


def draw_mark(ax, cx: float = 0.5, cy: float = 0.5, s: float = 1.0) -> None:
    """Draw the foveated-gaze mark centred at (cx, cy), scaled by s."""
    # faint outer lens ring
    ax.add_patch(Circle((cx, cy), 0.5 * s, fill=False, ec=INDIGO, alpha=0.16, lw=2.2 * s, zorder=2))
    # foveated dot field
    for r, n, dr, col, al in RINGS:
        if n == 1:
            ax.add_patch(Circle((cx, cy), dr * s, color=col, alpha=al, zorder=6))
            continue
        for i in range(n):
            a = 2 * np.pi * i / n + r * 6.0  # per-ring phase offset for an organic feel
            x = cx + r * s * np.cos(a)
            y = cy + r * s * np.sin(a)
            ax.add_patch(Circle((x, y), dr * s, color=col, alpha=al, zorder=5))
    # focal halo + leading-edge "looking forward" arc
    ax.add_patch(Circle((cx, cy), 0.122 * s, fill=False, ec=AMBER, alpha=0.55, lw=1.6 * s, zorder=6))
    d = s
    ax.add_patch(Arc((cx, cy), 1.04 * d, 1.04 * d, angle=0, theta1=-34, theta2=34,
                     ec=AMBER, lw=3.4 * s, alpha=0.95, zorder=7, capstyle="round"))


def _save(fig, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), transparent=True,
                    bbox_inches="tight", pad_inches=0.04, dpi=220)
    plt.close(fig)


def make_mark() -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_mark(ax)
    _save(fig, "logo-mark")


def make_lockup(name: str, text_color: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_mark(ax, cx=1.1, cy=1.1, s=1.7)
    ax.text(2.25, 1.30, "Foveance", fontsize=46, fontweight="bold", color=text_color,
            va="center", ha="left", family="DejaVu Sans")
    ax.text(2.30, 0.62, "ANTICIPATORY  CONTEXT  ALLOCATION", fontsize=13.5, color=GRAY,
            va="center", ha="left", family="DejaVu Sans")
    _save(fig, name)


if __name__ == "__main__":
    make_mark()
    make_lockup("logo", INK)        # for light backgrounds
    make_lockup("logo-dark", PAPER)  # for dark backgrounds (README dark mode)
    print("wrote assets/logo-mark.{svg,png}, logo.{svg,png}, logo-dark.{svg,png}")
