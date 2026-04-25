"""Matplotlib style configuration and shared colormaps."""
from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pathlib import Path
_PLOT_DIR         = Path(__file__).parent
_STYLE_FILE       = str(_PLOT_DIR / "thesis.mplstyle")
_PLANCK_CMAP_FILE = str(_PLOT_DIR / "Planck_Parchment_RGB.txt")


def setup_style() -> None:
    """Apply the shared matplotlib style sheet."""
    plt.rcParams["axes.axisbelow"] = False
    plt.rc("font", size=7, family="Helvetica")
    plt.style.use(_STYLE_FILE)


def get_planck_cmap() -> ListedColormap:
    """Load the Planck Parchment colormap."""
    rgb = np.loadtxt(_PLANCK_CMAP_FILE) / 255.0
    return ListedColormap(rgb, name="planck")


def get_inferno_cmap():
    """Return a copy of 'inferno' with bad pixels shown in white."""
    cmap = cm.get_cmap("inferno").copy()
    cmap.set_bad(color="white")
    return cmap
