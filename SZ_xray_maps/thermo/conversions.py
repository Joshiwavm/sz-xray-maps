"""Cluster thermodynamic conversions: SZ/X-ray observables → physical quantities."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import interpolate
from astropy import units as u
import astropy.constants as const

_DATA_DIR    = Path(__file__).parent.parent / "data"
_BIN_DIR     = _DATA_DIR / "cooling_function"

_NE_NH_RATIO = 1.2

# Regex to extract energy edges from per-bin filenames, e.g.
# cooling_function_J0459_0.37solar_0.400-0.413keV.npz
_BIN_PATTERN = re.compile(r"(\d+\.\d+)-(\d+\.\d+)keV\.npz$")


def _build_cooling_fn(
    metallicity: float,
    energy_range: tuple[float, float],
) -> interpolate.interp1d:
    """Return a T [K] → Λ(T) [erg cm³ s⁻¹] interpolator for *energy_range* [keV].

    Sums per-bin files whose energy ranges overlap *energy_range*.
    Each file stores Λ already integrated over its bin — no ΔE factor needed.
    """
    emin, emax   = energy_range
    temperature  = None
    total_power  = None
    glob_pat     = f"cooling_function_*{metallicity:.2f}solar*.npz"

    for fpath in sorted(_BIN_DIR.glob(glob_pat)):
        m = _BIN_PATTERN.search(fpath.name)
        if not m:
            continue
        e_lo, e_hi = float(m.group(1)), float(m.group(2))
        if e_lo >= emax or e_hi <= emin:
            continue
        with np.load(fpath) as f:
            if temperature is None:
                temperature = f["temperature"].copy()
                total_power = f["power_tot"].copy()
            else:
                total_power += f["power_tot"]

    if total_power is None:
        raise FileNotFoundError(
            f"No per-bin cooling-function files found for metallicity={metallicity:.2f} "
            f"solar in energy range {energy_range} keV under {_BIN_DIR}."
        )
    return interpolate.interp1d(temperature, total_power, kind="linear")


def _cooling_lambda(
    kT: u.Quantity,
    metallicity: float,
    energy_range: tuple[float, float],
) -> u.Quantity:
    """Evaluate the cooling function at ``kT`` with explicit astropy units."""
    lam_fn = _build_cooling_fn(metallicity, energy_range)
    T_K = (kT / const.k_B).to(u.K).value
    return lam_fn(T_K) * u.erg * u.cm**3 / u.s
    

class ClusterPhysics:
    """Static thermodynamic conversions for a galaxy cluster ICM."""

    @staticmethod
    def pressure(compton_y: np.ndarray, l_eff: u.Quantity) -> u.Quantity:
        """Compton-y + l_eff → projected electron pressure [keV cm⁻³]."""
        return (
            const.m_e * const.c**2 / const.sigma_T * compton_y / l_eff
        ).to(u.keV / u.cm**3)

    @staticmethod
    def density(
        counts_map: np.ndarray,
        ecf: u.Quantity,
        z: float,
        l_eff: u.Quantity,
        kT: u.Quantity,
        pixel_area_sr: u.Quantity,
        metallicity: float = 0.37,
        energy_range: tuple[float, float] | None = None,
    ) -> u.Quantity:
        """X-ray counts/s map → projected electron density [cm⁻³].

        Converts counts/s → surface brightness [erg/cm²/s/sr] via ECF and
        pixel solid angle, then inverts:
            ne² = 1.2 · 4π sr · (1+z)⁴ · S_X / (Λ · l_eff)
        """
        S_X = counts_map * u.ct / u.s * ecf / pixel_area_sr  # erg/cm²/s/sr
        lam = _cooling_lambda(kT, metallicity, energy_range)
        n_e_sq = _NE_NH_RATIO * 4 * np.pi * (1 + z)**4 * S_X / lam  * u.sr

        with np.errstate(invalid="ignore"):  # negative pixels → NaN, expected
            return np.sqrt(n_e_sq / l_eff).to(u.cm**-3)

    @staticmethod
    def temperature(P_e: u.Quantity, n_e: u.Quantity) -> u.Quantity:
        """P_e / n_e → projected electron temperature [keV]."""
        return (P_e / n_e).to(u.keV)

    @staticmethod
    def entropy(T_e: u.Quantity, n_e: u.Quantity) -> u.Quantity:
        """T_e / n_e^(2/3) → projected electron entropy [keV cm²]."""
        return (T_e / n_e ** (2 / 3)).to(u.keV * u.cm**2)

    @staticmethod
    def calibration_factor(
        compton_y: np.ndarray,
        l_eff: u.Quantity,
        ref_map: np.ndarray,
        ref_ecf: u.Quantity,
        ref_energy_range: tuple[float, float],
        ref_pixel_area_sr: u.Quantity,
        tgt_map: np.ndarray,
        tgt_ecf: u.Quantity,
        tgt_energy_range: tuple[float, float],
        tgt_pixel_area_sr: u.Quantity,
        z: float,
        kT: u.Quantity,
        mask: np.ndarray,
        metallicity: float = 0.37,
    ) -> float:
        """Multiplicative factor to rescale target density so median T matches reference.

        factor = median(T_ref) / median(T_tgt)

        Applying ``n_tgt / factor`` in the thermo maps yields:
            T_calibrated = P_e / (n_tgt / factor) = T_tgt × factor → median = T_ref
        """
        P_e   = ClusterPhysics.pressure(compton_y, l_eff)
        n_ref = ClusterPhysics.density(ref_map, ref_ecf, z, l_eff, kT, ref_pixel_area_sr,
                                       metallicity=metallicity, energy_range=ref_energy_range)
        n_tgt = ClusterPhysics.density(tgt_map, tgt_ecf, z, l_eff, kT, tgt_pixel_area_sr,
                                       metallicity=metallicity, energy_range=tgt_energy_range)
        T_ref = ClusterPhysics.temperature(P_e, n_ref)[mask].value
        T_tgt = ClusterPhysics.temperature(P_e, n_tgt)[mask].value
        T_ref = T_ref[np.isfinite(T_ref)]
        T_tgt = T_tgt[np.isfinite(T_tgt)]
        return float(np.nanmedian(T_ref) / np.nanmedian(T_tgt))
