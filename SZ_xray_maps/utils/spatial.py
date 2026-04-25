"""Spatial utility functions: pixel masks, angular radius maps, and Gaussian smoothing."""
from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord


def circle_mask(image: np.ndarray, xc: float, yc: float, radius: float) -> np.ndarray:
    """Return a boolean mask that is true inside a circle."""
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < radius


def ellipse_mask(
    image: np.ndarray,
    xc: float,
    yc: float,
    a: float,
    b: float,
    angle: float = 0.0,
) -> np.ndarray:
    """Return a boolean mask that is true inside an ellipse."""
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]
    dx = x - xc
    dy = y - yc
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot =  cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0


def radius_map_arcsec(
    wcs: WCS,
    shape: tuple[int, int],
    center_pixel: tuple[float, float] | None = None,
) -> u.Quantity:
    """Return angular separations from a reference pixel in arcsec."""
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]

    if center_pixel is None:
        xc, yc = wcs.wcs.crpix[0] - 1, wcs.wcs.crpix[1] - 1
    else:
        xc, yc = center_pixel

    sky   = pixel_to_skycoord(x, y, wcs)
    sky_c = pixel_to_skycoord(xc, yc, wcs)
    return sky.separation(sky_c).to(u.arcsec)


# ---------------------------------------------------------------------------
# Gaussian smoothing with variance propagation
# ---------------------------------------------------------------------------

def gaussian_smooth(data: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Smooth *data* with a 2D Gaussian kernel of *sigma_pix* pixels.

    Returns *data* unchanged when *sigma_pix* <= 0.
    """
    if sigma_pix <= 0:
        return data
    return convolve(data, Gaussian2DKernel(sigma_pix))


def gaussian_smooth_with_var(
    data: np.ndarray,
    var: np.ndarray,
    sigma_pix: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth *data* and propagate *var* through the same Gaussian kernel.

    For the linear combination  s_j = Σᵢ gᵢ xᵢ  the propagated variance is

        Var(s_j) = Σᵢ gᵢ² Var(xᵢ)

    i.e. the variance map is convolved with the element-wise squared kernel
    (not re-normalised).  NaN entries in *var* are treated as zero variance so
    that masked / bad pixels contribute nothing to neighbouring pixels.

    Returns ``(smoothed_data, smoothed_var)``.
    Returns *(data, var)* unchanged when *sigma_pix* <= 0.
    """
    if sigma_pix <= 0:
        return data, var
    kernel = Gaussian2DKernel(sigma_pix)
    smoothed = convolve(data, kernel)
    var_clean = np.where(np.isfinite(var), var, 0.0)
    var_smoothed = convolve(var_clean, kernel.array ** 2)
    return smoothed, var_smoothed
