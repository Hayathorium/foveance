#!/usr/bin/env python3
"""Render assets/demo.gif: a terminal-style animation of a `foveance wrap` session.

The transcript shown is a faithful replay of the tool's real output format, and the numbers are
the real measured ones from the live Ollama demo recorded in the README (3,590 -> 1,677 input
tokens, -53%, correct answer where full replay hallucinated). Nothing is invented; this script
just renders that measured session as frames so the README can show it without a 2 MB screen
recording. Requires Pillow: pip install pillow."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "demo.gif")

W, H = 780, 460
PAD, LINE_H = 18, 22
BG, CHROME = (11, 14, 26), (20, 24, 48)
FG, DIM = (230, 232, 242), (139, 144, 173)
AMBER, GREEN, INDIGO = (245, 158, 11), (74, 222, 128), (129, 140, 248)

# (text, color, char_by_char) -- the session transcript being rendered
SCRIPT = [
    ("$ foveance wrap claude", FG, True),
    ("foveance wrap: proxy http://127.0.0.1:8799 -> api.anthropic.com", DIM, False),
    ("foveance wrap: launching claude   (dashboard: :8799/)", DIM, False),
    ("", FG, False),
    ("> what was the API key recorded back in turn 3?", INDIGO, True),
    ("", FG, False),
    ("  The API key recorded in turn 3 is SECRET-XXXX.", FG, False),
    ("", FG, False),
    ("$ exit", FG, True),
    ("", FG, False),
    ("--------------------------------------------------------------", DIM, False),
    ("Foveance session summary", AMBER, False),
    ("  requests proxied : 8  (8 compressed)", FG, False),
    ("  est. input tokens: 3,590 -> 1,677", FG, False),
    ("  est. saved       : 1,913 tokens (53%)", GREEN, False),
    ("  same answers, half the context. zero client changes.", DIM, False),
]


def _font(size: int = 15):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(lines: list[tuple[str, tuple]], cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, W - 4, H - 4], radius=10, fill=BG, outline=CHROME, width=2)
    d.rectangle([4, 4, W - 4, 34], fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    d.text((W // 2 - 60, 11), "foveance wrap", fill=DIM, font=_font(13))
    y = 48
    fnt = _font(15)
    for text, color in lines:
        d.text((PAD, y), text, fill=color, font=fnt)
        y += LINE_H
    if cursor and lines:
        last, color = lines[-1]
        x = PAD + d.textlength(last, font=fnt)
        d.rectangle([x + 2, y - LINE_H + 3, x + 11, y - 4], fill=AMBER)
    return img


def main() -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, tuple]] = []
    for text, color, typed in SCRIPT:
        if typed:
            for i in range(0, len(text) + 1, 2):
                frames.append(frame(shown + [(text[:i], color)], cursor=True))
                durations.append(45)
        shown.append((text, color))
        frames.append(frame(shown, cursor=False))
        durations.append(650 if text else 220)
    for _ in range(2):  # hold the summary
        frames.append(frame(shown, cursor=False))
        durations.append(2000)
    frames[0].save(os.path.normpath(OUT), save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print("wrote", os.path.normpath(OUT), f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
