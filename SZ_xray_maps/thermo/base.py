"""ThermoProcessor mixin: l_eff estimation, cross-calibration, and thermo maps."""
from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from .conversions import ClusterPhysics
from .leff import LeffSolver
from ..utils.spatial import circle_mask, radius_map_arcsec


class ThermoProcessor:
    """Mixin that adds thermodynamical map computation to the Manager."""

    def __init__(self) -> None:
        self.l_eff:         u.Quantity | None = None
        self.l_eff_samples: np.ndarray | None = None
        self.l_eff_map:     u.Quantity | None = None
        self.calib_mask:    np.ndarray | None = None
        self.calib_factors: dict[str, float] = {}
        self.thermo:        dict = {}
        self.snr:           dict = {}

    def _r500(self) -> u.Quantity:
        return (
            (3 * self.M500 / (4 * np.pi * 500 * cosmo.critical_density(self.z))) ** (1 / 3)
        ).to(u.kpc)

    # ------------------------------------------------------------------
    # Compton-y
    # ------------------------------------------------------------------

    def compute_compton_y(self, sz_label: str) -> "Manager":
        """Convert a loaded SZ map to Compton-y, stored in ``sz_maps[sz_label]``."""
        entry  = self.sz_maps[sz_label]
        y_full = entry["instrument"].to_compton_y(entry["data_full"], entry["header"])
        y_sub  = self._handler.subimage(y_full)
        entry["compton_y_full"] = y_full
        entry["compton_y"]      = y_sub
        return self

    # ------------------------------------------------------------------
    # Effective line-of-sight depth
    # ------------------------------------------------------------------

    def _make_calib_mask(self, radius_arcsec: float, shape: tuple) -> np.ndarray:
        """Circular mask of *radius_arcsec* centred on the subimage centre."""
        pixel_scale = abs(self._handler.wcs_sub.wcs.cdelt[1]) * 3600.0  # arcsec/pix
        ny, nx = shape
        return circle_mask(np.empty(shape), nx // 2, ny // 2, radius_arcsec / pixel_scale)

    def compute_leff_uniform(
        self,
        sz_label: str,
        xray_label: str,
        radius_arcsec: float = 10.0,
        N: int = 300,
        seed: int | None = None,
    ) -> "Manager":
        """Estimate a uniform l_eff via Monte Carlo temperature sampling."""

        sz_entry = self.sz_maps[sz_label]
        xr_entry = self.xray_maps[xray_label]
        mask = self._make_calib_mask(radius_arcsec, sz_entry["compton_y"].shape)
        self.calib_mask = mask

        xray_data = xr_entry["data"] - (xr_entry["background"] or 0.0)

        samples = LeffSolver.monte_carlo(
            T_mean=self.kT.value, T_std=self.kT_std.value, N=N,
            compton_y=sz_entry["compton_y"],
            xray_map=xray_data,
            xray_ecf=xr_entry["instrument"].ecf,
            pixel_area_sr=xr_entry.get("pixel_area_sr", self.pixel_area_sr),
            mask=mask,
            z=self.z,
            kT=self.kT,
            cluster_tag=self.cluster_tag,
            metallicity=self.metallicity,
            energy_range=xr_entry.get("energy_range"),
            seed=seed,
        )

        self.l_eff_samples = samples
        l_mean = round(float(np.mean(samples)))
        l_std  = round(float(np.std(samples)))
        self.l_eff = l_mean * u.kpc
        r500       = self._r500()
        ratio      = float((r500 / self.l_eff).decompose().value)
        ratio_std  = ratio * (l_std / l_mean)
        C          = float(np.sqrt(ratio))
        C_std      = 0.5 * ratio_std / C
        print(
            f"l_eff = {l_mean} ± {l_std} kpc  |  "
            f"r500,c = {r500.value:.0f} kpc  |  "
            f"r500,c / l_eff = {ratio:.3f} ± {ratio_std:.3f}  |  "
            f"C = {C:.3f} ± {C_std:.3f}"
        )
        return self

    def compute_leff_map(
        self,
        rc_arcsec: float = 19.5,
        beta: float = 0.82,
    ) -> "Manager":
        """Build a pixel-by-pixel l_eff map from a projected beta model."""
        rs = radius_map_arcsec(self._handler.wcs_sub, self._handler.subimage_shape)
        self.l_eff_map = LeffSolver.beta_model(rs, z=self.z, rc_arcsec=rc_arcsec, beta=beta)
        ny, nx = self.l_eff_map.shape
        l_central = self.l_eff_map[ny // 2, nx // 2].to(u.kpc)
        r500  = self._r500()
        ratio = (r500 / l_central).decompose().value
        print(
            f"l_eff (centre) = {l_central.value:.0f} kpc  |  "
            f"r500,c = {r500.value:.0f} kpc  |  "
            f"r500,c / l_eff = {ratio:.3f}  |  "
            f"C = {np.sqrt(ratio):.3f}"
        )
        return self

    # ------------------------------------------------------------------
    # Cross-calibration
    # ------------------------------------------------------------------

    def calibrate_xray(
        self,
        ref_label: str,
        target_label: str,
        sz_label: str,
        radius_arcsec: float = 10.0,
        l_eff: u.Quantity | None = None,
    ) -> "Manager":
        """Cross-calibrate *target_label* against *ref_label* by matching median temperatures.

        Computes T = P_e(SZ) / n_e for both instruments within the calibration mask,
        then stores ``calib_factors[target_label] = median(T_ref) / median(T_target)``.
        In ``compute_thermo``, Chandra density is divided by this factor so that
        T_calibrated = T_target * factor → median(T_target_calibrated) = median(T_ref).
        """
        l_eff  = l_eff if l_eff is not None else (self.l_eff if self.l_eff is not None else self.l_eff_map)
        kT_val = self.kT

        ref = self.xray_maps[ref_label]
        tgt = self.xray_maps[target_label]
        self.calib_mask = self._make_calib_mask(radius_arcsec, ref["data"].shape)
        mask = self.calib_mask

        compton_y = self.sz_maps[sz_label]["compton_y"]
        factor = ClusterPhysics.calibration_factor(
            compton_y=compton_y,
            l_eff=l_eff,
            ref_map=ref["data"] - (ref["background"] or 0.0),
            ref_ecf=ref["instrument"].ecf,
            ref_energy_range=ref["energy_range"],
            ref_pixel_area_sr=ref.get("pixel_area_sr", self.pixel_area_sr),
            tgt_map=tgt["data"] - (tgt["background"] or 0.0),
            tgt_ecf=tgt["instrument"].ecf,
            tgt_energy_range=tgt["energy_range"],
            tgt_pixel_area_sr=tgt.get("pixel_area_sr", self.pixel_area_sr),
            z=self.z,
            kT=kT_val,
            mask=mask,
            cluster_tag=self.cluster_tag,
            metallicity=self.metallicity,
        )
        self.calib_factors[target_label] = factor
        print(
            f"Calib factor ({ref_label} → {target_label}):  {factor:.4f}  "
            f"(n_{target_label} ÷ {factor:.4f}  →  T_{target_label} × {factor:.4f})"
        )
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self, sz_label: str = "band3") -> None:
        """Print a summary of key thermo results within the calibration mask."""
        mask = self.calib_mask

        snr = self.sz_maps[sz_label]["snr"]
        print(f"S/N {sz_label} peak:  {np.nanmin(snr):.0f} σ")

        r500 = self._r500()
        if self.l_eff is not None:
            ratio = (r500 / self.l_eff).decompose().value
            print(
                f"l_eff (uniform) = {self.l_eff.value:.0f} kpc  |  "
                f"r500 = {r500.value:.0f} kpc  |  "
                f"r500/l_eff = {ratio:.3f}  |  C = {np.sqrt(ratio):.3f}"
            )
        if self.l_eff_map is not None:
            ny, nx = self.l_eff_map.shape
            l_cen = self.l_eff_map[ny // 2, nx // 2].to(u.kpc)
            ratio  = (r500 / l_cen).decompose().value
            print(
                f"l_eff (beta-model centre) = {l_cen.value:.0f} kpc  |  "
                f"r500 = {r500.value:.0f} kpc  |  "
                f"r500/l_eff = {ratio:.3f}  |  C = {np.sqrt(ratio):.3f}"
            )
        if self.calib_factors:
            for lbl, f in self.calib_factors.items():
                print(f"calib_factor[{lbl}] = {f:.4f}")

        for lbl, th in self.thermo.items():
            T = th["T_e"][mask]
            n = th["n_e"][mask]
            print(
                f"{lbl:12s}  "
                f"T_e median = {np.nanmedian(T.value):.1f} keV  "
                f"T_e peak = {np.nanmax(T.value):.1f} keV  "
                f"n_e median = {np.nanmedian(n.value)*1e3:.2f} ×10⁻³ cm⁻³"
            )

    # ------------------------------------------------------------------
    # Thermodynamical maps
    # ------------------------------------------------------------------

    def compute_thermo(
        self,
        sz_label: str,
        xray_labels: list[str],
        use_leff_map: bool = False,
    ) -> "Manager":
        """Compute projected P, n, T, K maps. Results in ``self.thermo[label]``."""
        l_eff = self.l_eff_map if use_leff_map else self.l_eff

        compton_y = self.sz_maps[sz_label]["compton_y"]
        P_e       = ClusterPhysics.pressure(compton_y, l_eff)

        for label in xray_labels:
            entry = self.xray_maps[label]
            calib = self.calib_factors.get(label, 1.0)

            n_e = ClusterPhysics.density(
                entry["data"] - (entry["background"] or 0.0),
                entry["instrument"].ecf, self.z, l_eff, self.kT,
                entry.get("pixel_area_sr", self.pixel_area_sr),
                cluster_tag=self.cluster_tag,
                metallicity=self.metallicity,
                energy_range=entry.get("energy_range"),
            ) / calib

            T_e = ClusterPhysics.temperature(P_e, n_e)
            K_e = ClusterPhysics.entropy(T_e, n_e)

            self.thermo[label] = {"P_e": P_e, "n_e": n_e, "T_e": T_e, "K_e": K_e}
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_thermo_fits(
        self,
        output_dir: str,
        xray_labels: list[str] | None = None,
    ) -> None:
        """Write thermodynamical maps to FITS files with subimage WCS headers."""
        import os
        from astropy.io import fits

        os.makedirs(output_dir, exist_ok=True)
        wcs_hdr = self._handler.wcs_sub.to_header()

        _bunits = {
            "P_e": "keV cm**-3",
            "n_e": "cm**-3",
            "T_e": "keV",
            "K_e": "keV cm**2",
        }

        labels = xray_labels or list(self.thermo.keys())
        for lbl in labels:
            th = self.thermo[lbl]
            for qty, bunit in _bunits.items():
                data = th[qty].value.astype(np.float32)
                hdr  = wcs_hdr.copy()
                hdr["BUNIT"]    = bunit
                hdr["INSTRUME"] = lbl
                hdr["QUANTITY"] = qty
                hdr["CLUSTER"]  = self.cluster_tag
                hdu  = fits.PrimaryHDU(data=data, header=hdr)
                path = os.path.join(output_dir, f"{self.cluster_tag}_{lbl}_{qty}.fits")
                hdu.writeto(path, overwrite=True)
                print(f"Saved {path}")
