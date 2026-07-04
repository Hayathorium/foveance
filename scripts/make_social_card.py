#!/usr/bin/env python3
"""Render assets/social-card.png — the 1280x640 image GitHub/LinkedIn/Twitter/Slack show when the
repo link is shared. Matches the logo palette. Requires Pillow: pip install pillow."""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "social-card.png"))

W, H = 1280, 640
BG = (11, 14, 26)
INK = (233, 236, 245)
DIM = (150, 156, 180)
AMBER = (245, 158, 11)
INDIGO = (129, 140, 248)
GREEN = (74, 222, 128)


def font(size, bold=True):
    names = (["arialbd.ttf", "seguisb.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def foveated_icon(d, cx, cy, r):
    """The logo's foveated-gaze motif: concentric rings of dots, dense amber core."""
    rng = 0
    for ring in range(1, 6):
        rad = r * ring / 5.0
        n = 6 * ring
        for k in range(n):
            a = 2 * math.pi * k / n + ring * 0.4
            x = cx + rad * math.cos(a)
            y = cy + rad * math.sin(a)
            dot = max(2, int(7 - ring))
            shade = INDIGO if (rng + k) % 3 else (168, 156, 240)
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=shade)
        rng += 1
    d.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6], outline=(60, 66, 110), width=3)
    d.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], outline=AMBER, width=4)
    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=AMBER)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # subtle amber arc accent, top-right
    d.arc([W - 260, -160, W + 160, 260], start=110, end=210, fill=AMBER, width=10)

    foveated_icon(d, 150, 150, 78)
    d.text((250, 108), "Foveance", font=font(58), fill=INK)
    d.text((252, 178), "ANTICIPATORY  CONTEXT  ALLOCATION", font=font(20), fill=DIM)

    d.text((80, 268), "Cut your LLM token bill", font=font(72), fill=INK)
    d.text((80, 344), "60%+", font=font(72), fill=AMBER)
    d.text((250, 350), "— same code, same answers.", font=font(52), fill=INK)

    d.text((80, 452), "Auto-compresses agent context for Claude Code, Codex, OpenAI,",
           font=font(28, bold=False), fill=DIM)
    d.text((80, 490), "Anthropic & Ollama. One command. Open source.",
           font=font(28, bold=False), fill=DIM)

    # install pill
    d.rounded_rectangle([80, 552, 470, 604], radius=12, fill=(22, 26, 48), outline=(60, 66, 110))
    d.text((104, 566), "$ pip install foveance", font=font(26), fill=GREEN)
    d.text((520, 566), "github.com/aimaghsoodi/foveance", font=font(24, bold=False), fill=INDIGO)

    img.save(OUT)
    print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
