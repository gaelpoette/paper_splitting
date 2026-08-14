#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the "Lyapunov-control motivation" figure(s) used in the Introduction
of template.tex, to highlight the effect of Lyapunov stabilization (LC) on
top of full-batch optimizers.

For the same full-batch optimizers as in make_speth_motivation_figure.py
(GD, Momentum, Adam, RMSProp, Lion -- AdamW is excluded, see below), this
script assembles a grid comparing, on the PolyAllStiff (poly5) benchmark:
  - the full-batch sensitivity map with a fixed, hand-tuned learning rate
    (same "<optimizer>/ref_figure.png" source as the Speth-motivation figure),
  - the full-batch sensitivity map of its LC (Lyapunov Control) counterpart,
    i.e. the same optimizer with the learning rate chosen automatically at
    every iteration via the discrete Lyapunov/Armijo inequality, starting
    from an arbitrary initial learning rate ("<LC_optimizer>/ref_figure.png").

AdamW is excluded because the LC_adamW run in onn/BNR/POLY5 is marked SKIP
(no ref_figure.png/ref_distribution.json available at the time this script
was written).

Source data: pre-rendered ``ref_figure.png`` sensitivity maps produced by the
onn/BNR/POLY5 benchmark runner:
    <BASE>/<optimizer>/ref_figure.png       (full-batch, fixed eta)
    <BASE>/LC_<optimizer>/ref_figure.png    (full-batch, LC-adaptive eta)

As for the Speth-motivation figure, the five rows are split across two
figures so that each one comfortably fits a single journal page (see
make_speth_motivation_figure.py for the same rationale).

Usage:
    python3 make_lc_motivation_figure.py

Requires: Pillow (PIL). Writes:
    lc_motivation_poly5_a.png   (GD, Momentum, Adam)
    lc_motivation_poly5_b.png   (RMSProp, Lion)
into this directory.
"""
import os

from PIL import Image, ImageChops, ImageDraw, ImageFont

# Root of the onn BNR/POLY5 benchmark outputs.
BASE = "/media/dick/HD_1_To/onn/BNR/POLY5"

# Output directory: this script's own directory (chapitre5_img/).
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Column variants: full-batch (fixed eta) vs. LC (adaptive eta). Note the LC
# variant is a *prefix* ("LC_<opt>"), unlike the Speth-motivation figure
# where the mini-batch variants are *suffixes* ("<opt>_minibatch/_speth").
COLUMN_PREFIXES = ["", "LC_"]
COL_LABELS = [
    "Full-batch (fixed $\\eta$)",
    "LC (Lyapunov-controlled, adaptive $\\eta$)",
]

# The two row splits (AdamW excluded: LC_adamW is marked SKIP in the source data).
FIGURES = [
    (["GD", "momentum", "adam"], ["GD", "Momentum", "Adam"], "lc_motivation_poly5_a.png"),
    (["rms", "lion"], ["RMSProp", "Lion"], "lc_motivation_poly5_b.png"),
]

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 26
PAD = 6          # gap (px) between cells
LABEL_COL_W = 46  # width (px) reserved for the rotated row labels
LABEL_ROW_H = 40  # height (px) reserved for the column headers


def autocrop(im, bg=(255, 255, 255)):
    """Crop out the uniform-background border around the actual plot content."""
    rgb = im.convert("RGB")
    bg_im = Image.new("RGB", rgb.size, bg)
    diff = ImageChops.difference(rgb, bg_im)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def build_figure(rows, row_labels, out_name):
    font_col = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    font_row = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Load and autocrop every (row, column) sensitivity map.
    imgs = {}
    for r in rows:
        for prefix in COLUMN_PREFIXES:
            folder = f"{prefix}{r}"
            path = os.path.join(BASE, folder, "ref_figure.png")
            im = Image.open(path).convert("RGB")
            imgs[(r, prefix)] = autocrop(im)

    cell_w = max(im.width for im in imgs.values())
    cell_h = max(im.height for im in imgs.values())

    n_rows, n_cols = len(rows), len(COLUMN_PREFIXES)
    W = LABEL_COL_W + n_cols * cell_w + (n_cols + 1) * PAD
    H = LABEL_ROW_H + n_rows * cell_h + (n_rows + 1) * PAD

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column headers.
    for j, clabel in enumerate(COL_LABELS):
        x0 = LABEL_COL_W + PAD + j * (cell_w + PAD)
        bbox = draw.textbbox((0, 0), clabel, font=font_col)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (cell_w - tw) / 2, 6), clabel, fill=(0, 0, 0), font=font_col)

    # Cells + rotated row labels.
    for i, (r, rlabel) in enumerate(zip(rows, row_labels)):
        y0 = LABEL_ROW_H + PAD + i * (cell_h + PAD)
        for j, prefix in enumerate(COLUMN_PREFIXES):
            x0 = LABEL_COL_W + PAD + j * (cell_w + PAD)
            im = imgs[(r, prefix)]
            ox = x0 + (cell_w - im.width) // 2
            oy = y0 + (cell_h - im.height) // 2
            canvas.paste(im, (ox, oy))
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(0, 0, 0), width=1)

        txt_im = Image.new("RGBA", (cell_h, LABEL_COL_W), (255, 255, 255, 0))
        tdraw = ImageDraw.Draw(txt_im)
        bbox = tdraw.textbbox((0, 0), rlabel, font=font_row)
        tw = bbox[2] - bbox[0]
        tdraw.text(((cell_h - tw) / 2, 0), rlabel, fill=(0, 0, 0), font=font_row)
        txt_im = txt_im.rotate(90, expand=True)
        canvas.paste(txt_im, (0, y0 + (cell_h - txt_im.height) // 2), txt_im)

    out_path = os.path.join(OUT_DIR, out_name)
    canvas.save(out_path, dpi=(200, 200))
    print(f"saved {out_path}  size={canvas.size}  aspect(h/w)={H / W:.3f}")


if __name__ == "__main__":
    for rows, row_labels, out_name in FIGURES:
        build_figure(rows, row_labels, out_name)
