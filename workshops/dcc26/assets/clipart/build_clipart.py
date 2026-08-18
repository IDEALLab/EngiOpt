"""Build minimalistic clipart-style diagrams of a VAE and a GAN.

Run:
    python3 workshops/dcc26/assets/clipart/build_clipart.py

Outputs:
    workshops/dcc26/assets/clipart/vae.png
    workshops/dcc26/assets/clipart/gan.png
    workshops/dcc26/assets/clipart/vae_latent_only.png   # just the 3D surface piece

Style:
- Flat-fill trapezoid "blocks" for the network components (encoder / decoder /
  generator / discriminator), no gradients, no outlines.
- The VAE latent space is a real 3D surface plotted with a colorful colormap.
- Input / output samples are real handwritten digits (sklearn's 8x8 digits set,
  the canonical "what VAEs / GANs learn to model" demo).
- Clean sans-serif labels, math italic for variable names.
- Figures are deliberately compact (close to square) so they drop into slides
  without taking the whole row.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection='3d'
from sklearn.datasets import load_digits

# Repository: workshops/dcc26/assets/clipart/ — this file lives here.
OUT = Path(__file__).resolve().parent

# ---- palette ----
ENC = "#3678c6"      # blue, encoder & generator
DEC = "#e07d3a"      # warm orange, decoder
DISC = "#7a5db8"     # purple, discriminator
LATENT_LABEL = "#5d3a9b"
TEXT = "#1d2530"
GREY = "#8a8f97"
NOISE_BG = "#f1ecfb"

LATENT_CMAP = "plasma"   # try 'viridis', 'turbo', 'magma' for a different mood


# ---------- helpers ----------

_DIGITS = load_digits()


def digit_sample(target, idx=0):
    """Return the idx-th example of digit `target` from sklearn's 8x8 corpus."""
    matches = np.where(_DIGITS.target == target)[0]
    return _DIGITS.images[matches[idx]].astype(float)


def blur(sample, *, center_weight=0.62):
    """Center-weighted 3x3 blur — softens the digit but preserves its shape.

    A plain mean blur destroys an 8x8 digit because each pixel averages over
    9 neighbours, smearing the strokes. Weighting the centre pixel keeps the
    digit recognisable while still adding a 'reconstruction is approximate' feel.
    """
    h, w = sample.shape
    out = np.zeros_like(sample, dtype=float)
    edge_weight = (1.0 - center_weight) / 8.0
    for i in range(h):
        for j in range(w):
            total = sample[i, j] * center_weight
            weight_sum = center_weight
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        total += sample[ni, nj] * edge_weight
                        weight_sum += edge_weight
            out[i, j] = total / weight_sum
    return out


def noise(sample, *, seed=0, sigma=0.18):
    """Per-pixel Gaussian noise, scaled to the sample's dynamic range."""
    rng = np.random.default_rng(seed)
    scale = sigma * sample.max()
    out = sample + rng.normal(0, scale, size=sample.shape)
    return np.clip(out, 0, sample.max())


def digit_image(ax, x0, y0, w, h, sample, *, border=GREY, alpha=1.0):
    """Render an 8x8 grayscale digit (or any 2D array) as a small icon."""
    n_r, n_c = sample.shape
    cell_w = w / n_c
    cell_h = h / n_r
    max_v = float(sample.max()) if sample.max() > 0 else 1.0
    for i in range(n_r):
        for j in range(n_c):
            v = sample[i, j] / max_v
            color = plt.cm.Greys(0.05 + 0.82 * v)
            ax.add_patch(patches.Rectangle(
                (x0 + j * cell_w, y0 + (n_r - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=color, edgecolor="none", alpha=alpha,
            ))
    ax.add_patch(patches.Rectangle((x0, y0), w, h,
                                   facecolor="none", edgecolor=border, linewidth=1.0))


def block(ax, points, color, *, label=None, label_color="white", label_size=13,
          sub_label=None, sub_label_color=None):
    """Trapezoidal block with a centered bold label and optional caption below."""
    poly = patches.Polygon(points, facecolor=color, edgecolor="none")
    ax.add_patch(poly)
    if label is not None:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        ax.text(cx, cy, label, fontsize=label_size, ha="center", va="center",
                color=label_color, fontweight="bold")
        if sub_label is not None:
            ax.text(cx, min(p[1] for p in points) - 0.35, sub_label,
                    fontsize=10, ha="center",
                    color=sub_label_color or color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, *, color=GREY, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def add_latent_surface(fig, *, left, bottom, width, height,
                       cmap=LATENT_CMAP, view=(28, -55)):
    """Add a 3D surface inset positioned via figure-normalized coordinates."""
    ax3d = fig.add_axes([left, bottom, width, height], projection="3d")
    X, Y = np.meshgrid(np.linspace(-2.2, 2.2, 60), np.linspace(-2.2, 2.2, 60))
    Z = (0.55 * np.sin(X * 1.1) * np.cos(Y * 1.1)
         + 0.40 * np.cos(X * 2.0 + 0.8)
         + 0.30 * np.sin(Y * 1.6 - 0.3))
    ax3d.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none",
                      antialiased=True, rcount=60, ccount=60)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_axis_off()
    ax3d.view_init(elev=view[0], azim=view[1])
    # transparent panes / no background
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.set_visible(False)
    ax3d.patch.set_alpha(0.0)
    return ax3d


# ---------- VAE diagram ----------

def build_vae(out_path: Path):
    # Tighter canvas, content packed to fill it.
    fig = plt.figure(figsize=(10.0, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    W, H = 10.0, 5.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    img_size = 1.55
    img_y = 1.45  # centered vertically: 1.45 + 1.55/2 = 2.225, axis mid = 2.5

    # --- input digit ---
    digit_in = digit_sample(3, idx=0)
    digit_image(ax, 0.20, img_y, img_size, img_size, digit_in)
    ax.text(0.20 + img_size / 2, img_y - 0.40, r"$x$",
            fontsize=18, ha="center", color=TEXT, style="italic")
    ax.text(0.20 + img_size / 2, img_y - 0.78, "input",
            fontsize=9.5, ha="center", color=GREY)

    cy = img_y + img_size / 2  # vertical center of the pipeline
    arrow(ax, 1.78, cy, 2.10, cy)

    # --- encoder (tall, fills most of the vertical space) ---
    enc_x0, enc_x1 = 2.15, 3.85
    block(ax,
          [(enc_x0, 0.40), (enc_x0, 4.60), (enc_x1, 3.65), (enc_x1, 1.35)],
          ENC, label="Encoder", label_size=13)

    arrow(ax, 3.90, cy, 4.20, cy)

    # --- latent surface (large, fills full height) ---
    lat_x0, lat_x1 = 4.20, 5.90
    lat_y0, lat_y1 = 0.05, 4.95
    add_latent_surface(fig,
                       left=lat_x0 / W, bottom=lat_y0 / H,
                       width=(lat_x1 - lat_x0) / W,
                       height=(lat_y1 - lat_y0) / H)
    ax.text((lat_x0 + lat_x1) / 2, 0.18, r"latent space  $z$",
            fontsize=11.5, ha="center", color=LATENT_LABEL, fontweight="bold")

    arrow(ax, 5.90, cy, 6.20, cy)

    # --- decoder ---
    dec_x0, dec_x1 = 6.25, 7.95
    block(ax,
          [(dec_x0, 1.35), (dec_x0, 3.65), (dec_x1, 4.60), (dec_x1, 0.40)],
          DEC, label="Decoder", label_size=13)

    arrow(ax, 8.00, cy, 8.30, cy)

    # --- reconstruction ---
    digit_out = blur(digit_in, center_weight=0.62)
    digit_image(ax, 8.35, img_y, img_size, img_size, digit_out)
    ax.text(8.35 + img_size / 2, img_y - 0.40, r"$\hat{x}$",
            fontsize=18, ha="center", color=TEXT, style="italic")
    ax.text(8.35 + img_size / 2, img_y - 0.78, "reconstruction",
            fontsize=9.5, ha="center", color=GREY)

    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------- VAE latent-surface only (handy as a standalone piece) ----------

def build_latent_only(out_path: Path):
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    add_latent_surface(fig, left=0.00, bottom=0.10, width=1.00, height=0.86)
    fig.text(0.5, 0.04, r"latent space  $z$", fontsize=15,
             ha="center", color=LATENT_LABEL, fontweight="bold")
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------- GAN diagram ----------

def build_gan(out_path: Path):
    # Tighter packing: smaller canvas, content fills it.
    fig = plt.figure(figsize=(10.0, 6.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    W, H = 10.0, 6.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    img_size = 1.30

    # --- noise z (dashed circle of dots) — top-left ---
    z_cx, z_cy, z_r = 0.95, 4.70, 0.75
    ax.add_patch(patches.Circle((z_cx, z_cy), z_r,
                                facecolor=NOISE_BG, edgecolor=DISC,
                                linewidth=1.4, linestyle="--"))
    rng = np.random.default_rng(3)
    n_dots = 26
    radii = rng.random(n_dots) ** 0.5 * z_r * 0.78
    angles = rng.random(n_dots) * 2 * np.pi
    for r_, a_ in zip(radii, angles):
        ax.add_patch(patches.Circle((z_cx + r_ * np.cos(a_),
                                     z_cy + r_ * np.sin(a_)),
                                    0.065, facecolor=DISC, edgecolor="none"))
    ax.text(z_cx, z_cy - z_r - 0.30, r"$z$", fontsize=16,
            ha="center", style="italic", color=TEXT)
    ax.text(z_cx, z_cy - z_r - 0.65, "noise", fontsize=9.5,
            ha="center", color=GREY)

    # arrow z -> G
    arrow(ax, z_cx + z_r + 0.05, z_cy, 2.05, z_cy)

    # --- generator (top branch) ---
    g_x0, g_x1 = 2.10, 3.80
    block(ax,
          [(g_x0, 3.95), (g_x0, 5.40), (g_x1, 5.75), (g_x1, 3.60)],
          ENC, label="G", label_size=20,
          sub_label="generator", sub_label_color=ENC)

    arrow(ax, 3.85, z_cy, 4.20, z_cy)

    # --- fake sample (digit-shaped, lightly noisy) ---
    digit_fake = digit_sample(7, idx=2)
    digit_fake = noise(digit_fake, seed=21, sigma=0.18)
    fake_x = 4.25
    digit_image(ax, fake_x, z_cy - img_size / 2, img_size, img_size, digit_fake)
    ax.text(fake_x + img_size / 2, z_cy + img_size / 2 + 0.20, r"fake  $\hat{x}$",
            fontsize=12, ha="center", color=TEXT, style="italic")

    # --- real sample (clean digit 7) ---
    real_y_center = 1.30
    digit_real = digit_sample(7, idx=1)
    digit_image(ax, fake_x, real_y_center - img_size / 2, img_size, img_size, digit_real)
    ax.text(fake_x + img_size / 2, real_y_center - img_size / 2 - 0.30, r"real  $x$",
            fontsize=12, ha="center", color=TEXT, style="italic")
    ax.text(fake_x + img_size / 2, real_y_center - img_size / 2 - 0.62,
            "from the dataset", fontsize=9, ha="center", color=GREY)

    # arrows from fake/real into D
    arrow(ax, fake_x + img_size + 0.05, z_cy - img_size / 2 + 0.20, 6.80, 3.85)
    arrow(ax, fake_x + img_size + 0.05, real_y_center + img_size / 2 - 0.20, 6.80, 2.30)

    # --- discriminator ---
    d_x0, d_x1 = 6.85, 8.60
    block(ax,
          [(d_x0, 1.30), (d_x0, 4.80), (d_x1, 3.95), (d_x1, 2.15)],
          DISC, label="D", label_size=20)

    # discriminator sub-label below D
    ax.text((d_x0 + d_x1) / 2, 0.85, "discriminator",
            fontsize=10, ha="center", color=DISC, fontweight="bold")

    # arrow D -> verdict
    arrow(ax, d_x1 + 0.05, 3.05, d_x1 + 0.35, 3.05)

    # verdict
    ax.text(d_x1 + 0.45, 3.22, "real  /  fake?",
            fontsize=12.5, ha="left", color=TEXT, fontweight="bold")
    ax.text(d_x1 + 0.45, 2.85, "verdict",
            fontsize=9.5, ha="left", color=GREY)

    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------- main ----------

if __name__ == "__main__":
    build_vae(OUT / "vae.png")
    build_latent_only(OUT / "vae_latent_only.png")
    build_gan(OUT / "gan.png")
