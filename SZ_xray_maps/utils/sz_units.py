"""Low-level SZ / CMB unit-conversion functions."""
from __future__ import annotations

import numpy as np
import astropy.constants as const
from astropy import units as u

_TCMB = 2.7255  # K


def getx(freq: float) -> float:
    """Adimensional frequency x = h*nu / (k_B * T_CMB)."""
    return (const.h * freq * u.Hz / const.k_B / (_TCMB * u.K)).to(
        u.dimensionless_unscaled
    ).value


def getJynorm() -> float:
    """Return the CMB surface-brightness normalisation in Jy/sr."""
    return (2e26 * (const.k_B * _TCMB * u.K) ** 3 / (const.h * const.c) ** 2).value


def compton_to_jy_per_pix(freq: float, cdelt1: float, cdelt2: float) -> float:
    """Compton-y → Jy/pixel at *freq* [Hz]. cdelt1/2 in deg."""
    x = getx(freq)
    factor = getJynorm()
    factor *= (-4.0 + x / np.tanh(0.5 * x))
    factor *= (x**4) * np.exp(x) / np.expm1(x) ** 2
    factor *= np.abs(cdelt1 * cdelt2) * (np.pi / 180.0) ** 2
    return factor


def jy_beam_to_jy_pix(cdelt1: float, cdelt2: float, bmaj: float, bmin: float) -> float:
    """Jy/beam → Jy/pixel. All inputs in deg."""
    return np.abs(cdelt1 * cdelt2) * (4 * np.log(2)) / (np.pi * bmaj * bmin)


def kcmb_to_k_bright(freq: float) -> float:
    """K_CMB → K_bright conversion factor at *freq* [Hz]."""
    x = getx(freq)
    return np.exp(x) * (x / np.expm1(x)) ** 2


def kcmb_to_jy_per_pix(freq: float, cdelt1: float, cdelt2: float) -> float:
    """K_CMB → Jy/pixel at *freq* [Hz]. cdelt1/2 in deg."""
    x = getx(freq)
    factor = getJynorm() / _TCMB
    factor *= (x**4) * np.exp(x) / np.expm1(x) ** 2
    factor *= np.abs(cdelt1 * cdelt2) * (np.pi / 180.0) ** 2
    return factor


def k_bright_to_jy_per_pix(freq: float, cdelt1: float, cdelt2: float) -> float:
    """K_bright → Jy/pixel at *freq* [Hz]. cdelt1/2 in deg."""
    return kcmb_to_jy_per_pix(freq, cdelt1, cdelt2) / kcmb_to_k_bright(freq)


def k_bright_to_jy_per_beam(
    freq: float, cdelt1: float, cdelt2: float, bmaj: float, bmin: float
) -> float:
    """K_bright → Jy/beam at *freq* [Hz]. All angular inputs in deg."""
    return k_bright_to_jy_per_pix(freq, cdelt1, cdelt2) / jy_beam_to_jy_pix(cdelt1, cdelt2, bmaj, bmin)
