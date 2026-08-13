#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the "Speth-balancing motivation" figures used in the Introduction
of template.tex (Figure~\\ref{fig_speth_motivation}).

For a handful of full-batch optimizers (GD, Momentum, Adam, RMSProp, Lion,
AdamW), this script assembles a grid comparing, on the PolyAllStiff (poly5)
benchmark and for a fixed, shared constant learning rate:
  - the full-batch sensitivity map,
  - a naive/unbalanced mini-batch (m=2) sensitivity map,
  - the extended Speth-balanced (eS-) mini-batch (m=2) sensitivity map.

Source data: pre-rendered ``ref_figure.png`` sensitivity maps produced by the
onn/BNR/POLY5 benchmark runner, one subdirectory per (optimizer, variant):
    <BASE>/<optimizer>/ref_figure.png             (full-batch)
    <BASE>/<optimizer>_minibatch/ref_figure.png    (unbalanced, m=2)
    <BASE>/<optimizer>_speth/ref_figure.png        (Speth-balanced, m=2)

Because six rows in a single grid produce a figure far too tall to fit on a
journal page (aspect ratio ~2, vs. ~1 for three rows), the six optimizers are
split into two 3-row figures, following the same convention already used for
the sensitivity maps of the companion paper [partI] (its own Figures 1-2 for
benchmark #2 split ~7 optimizers over two figures of comparable page-filling
size). Each ``ref_figure.png`` is tightly autocropped (its white margins
removed) before being laid out on a canvas, to keep the composite as compact
as possible.

Usage:
    python3 make_speth_motivation_figure.py

Requires: Pillow (PIL). Writes:
    speth_motivation_poly5_a.png   (GD, Momentum, Adam)
    speth_motivation_poly5_b.png   (RMSProp, Lion, AdamW)
into this directory.
"""
import os

from PIL import Image, ImageChops, ImageDraw, ImageFont

# Root of the onn BNR/POLY5 benchmark outputs (ref_figure.png + ref_distribution.json
# per <optimizer>[_minibatch|_speth] subdirectory).
BASE = "/media/dick/HD_1_To/onn/BNR/POLY5"

# Output directory: this script's own directory (chapitre5_img/).
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Column variants and their labels, shared by both figures.
VARIANTS = ["", "_minibatch", "_speth"]
COL_LABELS = [
    "Full-batch",
    "Mini-batch (unbalanced, m=2)",
    "Speth-balanced (eS-, m=2)",
]

# The two 3-row splits (see docstring for why six rows are split in two).
FIGURES = [
    (["GD", "momentum", "adam"], ["GD", "Momentum", "Adam"], "speth_motivation_poly5_a.png"),
    (["rms", "lion", "adamW"], ["RMSProp", "Lion", "AdamW"], "speth_motivation_poly5_b.png"),
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

    # Load and autocrop every (row, variant) sensitivity map.
    imgs = {}
    for r in rows:
        for v in VARIANTS:
            path = os.path.join(BASE, f"{r}{v}", "ref_figure.png")
            im = Image.open(path).convert("RGB")
            imgs[(r, v)] = autocrop(im)

    cell_w = max(im.width for im in imgs.values())
    cell_h = max(im.height for im in imgs.values())

    n_rows, n_cols = len(rows), len(VARIANTS)
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
        for j, v in enumerate(VARIANTS):
            x0 = LABEL_COL_W + PAD + j * (cell_w + PAD)
            im = imgs[(r, v)]
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
