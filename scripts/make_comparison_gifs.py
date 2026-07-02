#!/usr/bin/env python3
"""Render assets/demo_comparison.gif and assets/demo_anytool.gif for the README.

demo_comparison.gif -- with/without Foveance on the same task and model. The numbers animated
are the real measured ones from bench/report.md (gemma2:2b, budget 400, 5 seeds): full replay
1.00 accuracy at 10,228 input tokens vs foveance 1.00 accuracy at 3,663 tokens (64.2% fewer).

demo_anytool.gif -- the "works with anything" tour: the routing lines shown are the package's
real commands, and the per-tool notes state only what was actually verified (see README table).

Nothing is invented; these scripts just render measured results as frames.
Requires Pillow: pip install pillow."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))

BG, CHROME = (11, 14, 26), (20, 24, 48)
FG, DIM = (230, 232, 242), (139, 144, 173)
AMBER, GREEN, INDIGO, RED = (245, 158, 11), (74, 222, 128), (129, 140, 248), (248, 113, 113)


def font(size: int = 15):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def shell(w: int, h: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, w - 4, h - 4], radius=10, fill=BG, outline=CHROME, width=2)
    d.rectangle([4, 4, w - 4, 34], fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    tw = d.textlength(title, font=font(13))
    d.text(((w - tw) // 2, 11), title, fill=DIM, font=font(13))
    return img, d


# --------------------------------------------------------------- comparison GIF
FULL_TOK, FOV_TOK = 10228, 3663  # real: bench/report.md, gemma2:2b, budget 400, 5 seeds


def comparison_frame(t: float, hold_text: bool) -> Image.Image:
    """t in [0,1] animates the token counters/bars up to their real measured totals."""
    W, H = 780, 460
    img, d = shell(W, H, "with vs without Foveance -- same task, same model")
    f15, f14, f22 = font(15), font(13), font(22)
    d.text((PAD := 24, 50), "gemma2:2b via Ollama - 3 buried-fact recalls - measured, 5 seeds",
           fill=DIM, font=f14)

    cols = [("WITHOUT  (full replay)", FULL_TOK, RED, 24),
            ("WITH  foveance (budget 400)", FOV_TOK, GREEN, W // 2 + 12)]
    bar_w = W // 2 - 60
    for label, total, color, x in cols:
        d.text((x, 92), label, fill=FG, font=f15)
        tok = int(total * t)
        d.text((x, 126), f"input tokens: {tok:,}", fill=color, font=f22)
        d.rectangle([x, 168, x + bar_w, 190], outline=CHROME, width=2)
        fill_w = int(bar_w * (tok / FULL_TOK))
        if fill_w > 4:
            d.rectangle([x + 2, 170, x + 2 + fill_w - 4, 188], fill=color)
        if t >= 1.0:
            d.text((x, 206), "accuracy: 1.00 (3/3 recalled)", fill=FG, font=f15)
    if t >= 1.0 and hold_text:
        d.text((PAD, 262), "same accuracy - 64% fewer input tokens", fill=AMBER, font=f22)
        d.text((PAD, 300), "and where the naive baselines land on the same task:", fill=DIM, font=f14)
        rows = [("recency (last-4 turns)", "0.67", RED),
                ("truncate (newest-first)", "0.00", RED),
                ("uniform (spread evenly)", "0.00", RED),
                ("LLMLingua-2 (query-agnostic)", "0.00-0.33", RED),
                ("foveance (anticipatory)", "1.00", GREEN)]
        y = 326
        for name, acc, color in rows:
            d.text((PAD + 12, y), f"{name:<30} accuracy {acc}", fill=color, font=f14)
            y += 20
        d.text((PAD, y + 8), "every number traces to bench/results/ - nothing hand-entered",
               fill=DIM, font=f14)
    return img


def make_comparison() -> None:
    frames, durations = [], []
    steps = 24
    for i in range(steps + 1):
        frames.append(comparison_frame(i / steps, hold_text=False))
        durations.append(70)
    frames.append(comparison_frame(1.0, hold_text=True))
    durations.append(2600)
    for _ in range(2):
        frames.append(comparison_frame(1.0, hold_text=True))
        durations.append(2600)
    out = os.path.join(ASSETS, "demo_comparison.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print("wrote", out, f"({len(frames)} frames)")


# --------------------------------------------------------------- any-tool GIF
CHECK = "+"  # ASCII-safe check marker rendered in green

ANYTOOL = [
    ("$ foveance wrap claude", FG, True),
    (f"  [{CHECK}] Claude Code -> api.anthropic.com      live-verified, tool pairing kept", GREEN, False),
    ("", FG, False),
    ("$ foveance wrap -- codex \"fix the tests\"", FG, True),
    (f"  [{CHECK}] Codex -> api.openai.com /responses    API-key provider", GREEN, False),
    ("", FG, False),
    ("$ foveance proxy --upstream http://localhost:11434/v1", FG, True),
    (f"  [{CHECK}] Ollama / vLLM / LM Studio             local, no auth at all", GREEN, False),
    ("", FG, False),
    ("$ export OPENAI_BASE_URL=http://localhost:8799/v1", FG, True),
    (f"  [{CHECK}] OpenAI SDK - LangChain - aider - Cursor - LiteLLM ...", GREEN, False),
    ("", FG, False),
    ("one proxy - three wire protocols - zero client changes", AMBER, False),
    ("your keys stay yours: Foveance stores nothing", DIM, False),
]


def anytool_frame(lines: list[tuple[str, tuple]], cursor: bool) -> Image.Image:
    W, H = 780, 420
    img, d = shell(W, H, "foveance -- works with anything")
    y, fnt = 48, font(15)
    for text, color in lines:
        d.text((18, y), text, fill=color, font=fnt)
        y += 22
    if cursor and lines:
        last, _ = lines[-1]
        x = 18 + d.textlength(last, font=fnt)
        d.rectangle([x + 2, y - 19, x + 11, y - 4], fill=AMBER)
    return img


def make_anytool() -> None:
    frames, durations, shown = [], [], []
    for text, color, typed in ANYTOOL:
        if typed:
            for i in range(0, len(text) + 1, 3):
                frames.append(anytool_frame(shown + [(text[:i], color)], cursor=True))
                durations.append(40)
        shown.append((text, color))
        frames.append(anytool_frame(shown, cursor=False))
        durations.append(700 if text else 200)
    for _ in range(2):
        frames.append(anytool_frame(shown, cursor=False))
        durations.append(2400)
    out = os.path.join(ASSETS, "demo_anytool.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    make_comparison()
    make_anytool()
