"""Low-level plotting helpers used by the Plotter mixin."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy import units as u
from astropy.wcs import WCS

_LABEL_BOX = dict(facecolor="wheat", alpha=0.8, edgecolor="black")


# ---------------------------------------------------------------------------
# Panel decorators
# ---------------------------------------------------------------------------

def add_colorbar(fig, img, ax, label: str, fontsize: int = 10,
                 shrink: float = 0.9, pad: float = 0.04):
    cbar = fig.colorbar(img, ax=ax, shrink=shrink, pad=pad)
    cbar.set_label(label, fontsize=fontsize)
    return cbar


def add_label_box(ax, text: str, fontsize: int = 10) -> None:
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top", bbox=_LABEL_BOX)


def add_r500_circle(ax, cx: float, cy: float, theta500: u.Quantity) -> None:
    """Overlay an r500 circle and label on a WCS axis."""
    pixscale = abs(ax.wcs.pixel_scale_matrix[1, 1]) * u.deg
    r_pix = (theta500.to(u.deg) / pixscale).value
    ax.add_patch(Circle((cx, cy), r_pix, ec="k", fc="none", ls="--"))
    ax.text(
        cx + 0.65 * r_pix, cy - 0.65 * r_pix,
        r"r$_{500,c}$", color="k",
        rotation=45, ha="center", va="center",
    )


def add_beam_patch(ax, wcs: WCS, fwhm_arcsec: float = 5.0,
                   position: tuple[float, float] = (30, 30)) -> None:
    """Overlay a hatched beam circle (bottom-left by default)."""
    pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600  # arcsec/pixel
    r_pix = fwhm_arcsec / (2.0 * pixel_scale)
    ax.add_patch(Circle(position, r_pix, transform=ax.transData,
                        edgecolor="gray", facecolor="none",
                        hatch="////", linewidth=1.5))


def configure_wcs_ticks(
    ax, size: int = 10, show_ra: bool = True, show_dec: bool = True
) -> None:
    ax.coords[0].set_ticklabel(size=size)
    ax.coords[1].set_ticklabel(size=size)
    ax.coords[0].set_axislabel("")   # clear default "pos.eq.ra" etc.
    ax.coords[1].set_axislabel("")
    if not show_ra:
        ax.coords[0].set_ticklabel_visible(False)
    if not show_dec:
        ax.coords[1].set_ticklabel_visible(False)


def hide_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Multi-panel figure builders
# ---------------------------------------------------------------------------

def make_map_grid(
    nrows: int, ncols: int, wcs: WCS, figsize: tuple, dpi: int = 300
) -> tuple:
    """Create a WCS grid with a dedicated colorbar axis beside each panel."""
    width_ratios = [10, 0.5, 2.5] * ncols
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs  = fig.add_gridspec(
        nrows, ncols * 3,
        width_ratios=width_ratios,
        wspace=0.05, hspace=0.15,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            ax  = fig.add_subplot(gs[r, c * 3],     projection=wcs)
            cax = fig.add_subplot(gs[r, c * 3 + 1])
            # c * 3 + 2 → spacer, intentionally left as empty space
            axes[r, c] = (ax, cax)
    return fig, axes


def make_grid_figure(nrows: int, ncols: int, wcs: WCS, figsize: tuple, dpi: int = 300):
    """Create a (nrows × ncols) grid of WCS axes."""
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi,
        subplot_kw={"projection": wcs},
    )
    return fig, axes.flatten()
