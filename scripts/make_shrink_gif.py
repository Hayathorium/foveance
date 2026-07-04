#!/usr/bin/env python3
"""Render assets/shrink-demo.gif — foveance.shrink() in action: the conversation collapses (older
turns trimmed, system + last turn kept) and the token counter drops. Real measured numbers
(3,590 -> 1,677, -53%). Requires Pillow."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "shrink-demo.gif"))
FONTDIR = r"C:\Windows\Fonts"

W, H = 820, 460
BG, CHROME = (11, 14, 26), (20, 24, 48)
INK, DIM = (233, 236, 245), (150, 158, 184)
AMBER, INDIGO, GREEN = (245, 158, 11), (129, 140, 248), (86, 214, 140)
BAR = (44, 51, 84)

TOTAL0, TOTAL1 = 3590, 1677
# 6 messages: (label, kept_full). system + last stay full; middle 4 collapse.
MSGS = [("system prompt", True), ("turn 1  ·  API_KEY buried here", "fade"),
        ("turn 2  ·  logs", False), ("turn 3  ·  logs", False),
        ("turn 4  ·  logs", False), ("latest turn  ·  \"what was the key?\"", True)]


def font(px, weight="regular"):
    files = {"black": ["seguibl.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
             "semibold": ["seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
             "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
             "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"]}[weight]
    for f in files:
        for p in (os.path.join(FONTDIR, f), f):
            try:
                return ImageFont.truetype(p, px)
            except OSError:
                continue
    return ImageFont.load_default()


def ease(t):
    return t * t * (3 - 2 * t)


def frame(prog, done=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, W - 4, H - 4], radius=12, fill=BG, outline=CHROME, width=2)
    d.rectangle([4, 4, W - 4, 34], fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    d.text((W // 2 - 160, 11), "foveance.shrink(messages, budget=400)", font=font(15, "mono"),
           fill=DIM)

    p = ease(prog)
    # message bars
    x0, y = 40, 66
    full_w = W - 300
    for label, keep in MSGS:
        if keep is True:
            w = full_w
            col = AMBER if "system" not in label and "latest" not in label else INDIGO
            col = INDIGO
        elif keep == "fade":
            w = int(full_w - (full_w - 150) * p)   # shrinks to a digest but keeps a tag
            col = AMBER
        else:
            w = int(full_w - (full_w - 46) * p)    # collapses to a pointer
            col = BAR
        h = 40 if keep is True else int(40 - 20 * p) if keep != "fade" else int(40 - 8 * p)
        d.rounded_rectangle([x0, y, x0 + max(w, 30), y + h], radius=7, fill=col)
        if keep == "fade":
            txt = label if p < 0.35 else "turn 1 · digest"
            d.text((x0 + 12, y + h // 2 - 8), txt, font=font(14, "regular"), fill=INK)
        elif h > 22:
            d.text((x0 + 12, y + h // 2 - 9), label, font=font(15, "regular"), fill=INK)
        y += max(h, 20) + 12

    # token counter (right)
    cur = int(TOTAL0 + (TOTAL1 - TOTAL0) * p)
    cx = W - 210
    d.text((cx, 74), "INPUT TOKENS", font=font(14, "semibold"), fill=DIM)
    d.text((cx, 96), f"{cur:,}", font=font(52, "black"), fill=GREEN if p > 0.5 else INK)
    # shrink bar
    bw = 210
    d.rounded_rectangle([cx, 168, cx + bw, 188], radius=6, outline=CHROME, width=2)
    fillw = int(bw * cur / TOTAL0)
    d.rounded_rectangle([cx + 2, 170, cx + 2 + max(fillw - 4, 2), 186], radius=5,
                        fill=GREEN if p > 0.5 else INDIGO)
    if done:
        d.text((cx, 210), "-53%", font=font(40, "black"), fill=AMBER)

    # caption
    if done:
        d.text((40, H - 66), "Same answer.  53% fewer tokens.  Zero code changes.",
               font=font(22, "semibold"), fill=INK)
        d.text((40, H - 34), "pip install foveance   ·   github.com/aimaghsoodi/foveance",
               font=font(16, "regular"), fill=DIM)
    else:
        d.text((40, H - 50), "keeps the system prompt + the turn that matters, trims the rest",
               font=font(17, "regular"), fill=DIM)
    return img


def main():
    frames, durations = [], []
    frames.append(frame(0)); durations.append(1100)
    steps = 26
    for i in range(1, steps + 1):
        frames.append(frame(i / steps)); durations.append(60)
    for _ in range(3):
        frames.append(frame(1.0, done=True)); durations.append(1400)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print("wrote", OUT, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
