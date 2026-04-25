"""Effective line-of-sight depth: uniform fitting and beta-model map."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from .conversions import ClusterPhysics


class LeffSolver:
    """Helpers for uniform and beta-model line-of-sight depths."""

    @staticmethod
    def beta_model(
        radius: u.Quantity,
        z: float,
        rc_arcsec: float = 19.5,
        beta: float = 0.82,
    ) -> u.Quantity:
        """Projected l_eff [kpc] from a spherical beta model."""
        rc  = rc_arcsec * u.arcsec
        r   = radius.to(u.arcsec)
        d_A = cosmo.angular_diameter_distance(z)

        leff = (d_A * rc.to(u.rad).value).to(u.kpc) / 2 * np.pi**0.5 * (1 + (r / rc) ** 2) ** (-0.5)
        leff *= gamma(3 * beta / 2 - 0.5) ** 2 * gamma(3 * beta)
        leff /= gamma(3 * beta / 2) ** 2 * gamma(3 * beta - 0.5)
        return leff

    @staticmethod
    def _median_temperature(
        l_eff_kpc: float,
        compton_y: np.ndarray,
        xray_map: np.ndarray,
        xray_ecf: u.Quantity,
        pixel_area_sr: u.Quantity,
        mask: np.ndarray,
        z: float,
        kT: u.Quantity,
        metallicity: float,
        energy_range: tuple[float, float] | None,
    ) -> float:
        """Median projected T_e [keV] within *mask* at a trial l_eff."""
        l_eff = l_eff_kpc * u.kpc
        P_e = ClusterPhysics.pressure(compton_y, l_eff)
        n_e = ClusterPhysics.density(
            xray_map, xray_ecf, z, l_eff, kT, pixel_area_sr,
            metallicity=metallicity, energy_range=energy_range,
        )
        T_e = ClusterPhysics.temperature(P_e, n_e)
        return float(np.nanmedian(T_e[mask].value))

    @staticmethod
    def solve_uniform(
        T_target: float,
        compton_y: np.ndarray,
        xray_map: np.ndarray,
        xray_ecf: u.Quantity,
        pixel_area_sr: u.Quantity,
        mask: np.ndarray,
        z: float,
        kT: u.Quantity,
        metallicity: float = 0.37,
        energy_range: tuple[float, float] | None = None,
        lmin: float = 1.0,
        lmax: float = 5000.0,
    ) -> float:
        """Find the uniform l_eff [kpc] that reproduces *T_target* [keV]."""
        f = lambda l: LeffSolver._median_temperature(
            l, compton_y, xray_map, xray_ecf, pixel_area_sr, mask, z, kT, metallicity, energy_range,
        ) - T_target
        return brentq(f, lmin, lmax)

    @staticmethod
    def monte_carlo(
        T_mean: float,
        T_std: float,
        N: int,
        compton_y: np.ndarray,
        xray_map: np.ndarray,
        xray_ecf: u.Quantity,
        pixel_area_sr: u.Quantity,
        mask: np.ndarray,
        z: float,
        kT: u.Quantity,
        metallicity: float = 0.37,
        energy_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample temperatures and solve for a uniform ``l_eff`` for each draw."""
        if seed is not None:
            np.random.seed(seed)
        T_samples = np.random.normal(T_mean, T_std, N)
        return np.array([
            LeffSolver.solve_uniform(
                T, compton_y, xray_map, xray_ecf, pixel_area_sr, mask, z, kT,
                metallicity=metallicity, energy_range=energy_range,
            )
            for T in T_samples
        ])
