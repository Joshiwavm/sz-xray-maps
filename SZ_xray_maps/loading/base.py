"""Loader mixin: adds SZ and X-ray map loading to the Manager."""
from __future__ import annotations

import numpy as np
from astropy import units as u

from ..utils.spatial import gaussian_smooth, gaussian_smooth_with_var


class Loader:
    """Mixin that provides :meth:`add_sz` and :meth:`add_xray` to the Manager."""

    def __init__(self) -> None:
        self.sz_maps:   dict = {}
        self.xray_maps: dict = {}

    def _update_pixel_area_sr(self) -> None:
        cdelt = self._handler.wcs_sub.wcs.cdelt
        self.pixel_area_sr = np.abs(cdelt[0] * cdelt[1]) * (np.pi / 180.0) ** 2 * u.sr

    # ------------------------------------------------------------------
    # SZ maps
    # ------------------------------------------------------------------

    def add_sz(
        self,
        path: str,
        instrument,
        label: str | None = None,
        rms: float | None = None,
    ) -> "Manager":
        """Load an SZ FITS map and reproject it to the current reference grid."""
        from .fits_handler import FitsHandler

        label = label or instrument.name

        if self._handler is None:
            self._handler = FitsHandler(path, self._size_arcmin, self._center)
            self._update_pixel_area_sr()

        data, header, wcs = self._handler.load(path)
        data_jy     = instrument.to_jy_beam(data, header)
        reproj_full = self._handler.reproject_to_reference(data_jy, wcs)
        sub         = self._handler.subimage(reproj_full, scale=1e3)  # → mJy/beam

        rms_val = rms if rms is not None else header.get("rms", np.nan) * 1e3

        self.sz_maps[label] = {
            "_path":          path,
            "_raw_data":      data_jy,
            "_raw_wcs":       wcs,
            "data_full":      reproj_full,
            "data":           sub,
            "header":         header,
            "instrument":     instrument,
            "rms":            rms_val,
            "snr":            sub / rms_val,
            "compton_y_full": None,
            "compton_y":      None,
        }
        return self

    # ------------------------------------------------------------------
    # X-ray maps
    # ------------------------------------------------------------------

    def add_xray(
        self,
        path: str,
        instrument,
        label: str | None = None,
        smooth_fwhm_arcsec: float = 6.0,
        mask: np.ndarray | None = None,
        expmap_path: str | None = None,
        sig_path: str | None = None,
        energy_range: tuple[float, float] | None = None,
    ) -> "Manager":
        """Load, smooth, and reproject an X-ray FITS map onto the subimage grid."""
        label = label or instrument.name

        data, header, wcs = self._handler.load(path)
        raw_data = instrument.preprocess(data)
        reproj   = self._smooth_and_reproject(raw_data, header, wcs, smooth_fwhm_arcsec)

        expmap  = self._load_ancillary(expmap_path) if expmap_path else None
        sig_map = self._load_ancillary(sig_path)    if sig_path    else None

        noise_map = self._compute_xray_noise_map(
            raw_data, header, wcs, smooth_fwhm_arcsec, expmap_path
        )

        # Native pixel solid angle — must be used in S_X = rate*ECF/pixel_area so that
        # the per-pixel count rate (in the native X-ray pixel frame) gives the correct
        # surface brightness even after reproject_interp (which preserves values, not SB).
        native_pixel_area_sr = (
            abs(header.get("CDELT1", 0)) * abs(header.get("CDELT2", 0))
            * (np.pi / 180.0) ** 2
        ) * u.sr

        self.xray_maps[label] = {
            "_path":               path,
            "_raw_data":           raw_data,
            "_raw_wcs":            wcs,
            "_smooth_fwhm_arcsec": smooth_fwhm_arcsec,
            "_expmap_path":        expmap_path,
            "data":                reproj,
            "header":              header,
            "instrument":          instrument,
            "mask":                mask,
            "background":          None,
            "expmap":              expmap,
            "sig_map":             sig_map,
            "noise":               noise_map,
            "energy_range":        energy_range,
            "pixel_area_sr":       native_pixel_area_sr,
        }
        return self

    # ------------------------------------------------------------------
    # Internal re-projection helpers (used by choose_reference_wcs)
    # ------------------------------------------------------------------

    def _reproject_sz_entry(self, entry: dict) -> None:
        reproj_full = self._handler.reproject_to_reference(
            entry["_raw_data"], entry["_raw_wcs"]
        )
        sub = self._handler.subimage(reproj_full, scale=1e3)
        entry["data_full"]      = reproj_full
        entry["data"]           = sub
        entry["snr"]            = sub / entry["rms"]
        entry["compton_y_full"] = None
        entry["compton_y"]      = None

    def _reproject_xray_entry(self, entry: dict) -> None:
        reproj = self._smooth_and_reproject(
            entry["_raw_data"], entry["header"],
            entry["_raw_wcs"], entry["_smooth_fwhm_arcsec"],
        )
        entry["data"] = reproj
        entry["noise"] = self._compute_xray_noise_map(
            entry["_raw_data"], entry["header"], entry["_raw_wcs"],
            entry["_smooth_fwhm_arcsec"], entry["_expmap_path"],
        )

    def _load_ancillary(self, path: str) -> np.ndarray:
        """Load and reproject an ancillary map onto the subimage grid."""
        data, _, wcs = self._handler.load(path)
        return self._handler.reproject_to_sub(data, wcs)

    def _smooth_and_reproject(
        self, data: np.ndarray, header, wcs, fwhm_arcsec: float
    ) -> np.ndarray:
        if fwhm_arcsec > 0:
            sigma_pix = fwhm_arcsec / 2.355 / abs(header["CDELT2"] * 3600)
            data = gaussian_smooth(data, sigma_pix)
        return self._handler.reproject_to_sub(data, wcs)

    def _compute_xray_noise_map(
        self,
        raw_data: np.ndarray,
        header,
        wcs,
        fwhm_arcsec: float,
        expmap_path: str | None,
    ) -> np.ndarray | None:
        """Compute the noise map for a smoothed X-ray map.

        Poisson variance is estimated at native resolution, propagated through
        the same Gaussian smoothing kernel (Var(Σ gᵢXᵢ) = Σ gᵢ² Var(Xᵢ)),
        then reprojected onto the subimage grid.  Background uncertainty is
        added in quadrature later by compute_noise() once compute_background()
        has been called.
        """
        # --- raw per-pixel Poisson variance ---
        if expmap_path is not None:
            raw_expmap, expmap_hdr, raw_expmap_wcs = self._handler.load(expmap_path)
            if raw_expmap.shape != raw_data.shape:
                # Check whether the pixel scales differ by an integer factor.
                # If so, upsample the expmap by pixel replication — exact and
                # avoids interpolation artefacts.
                data_cdelt   = abs(header.get("CDELT2", 1.0))
                expmap_cdelt = abs(expmap_hdr.get("CDELT2", 1.0))
                ratio        = expmap_cdelt / data_cdelt
                int_factor   = int(round(ratio))
                if int_factor > 1 and abs(ratio - int_factor) < 0.01:
                    raw_expmap = np.repeat(
                        np.repeat(raw_expmap, int_factor, axis=0), int_factor, axis=1
                    )
                    # Trim to raw_data shape in case of rounding at the edges
                    raw_expmap = raw_expmap[: raw_data.shape[0], : raw_data.shape[1]]
                else:
                    from reproject import reproject_interp
                    raw_expmap, _ = reproject_interp(
                        (raw_expmap, raw_expmap_wcs), wcs, shape_out=raw_data.shape
                    )
            with np.errstate(invalid="ignore", divide="ignore"):
                var_raw = np.abs(raw_data) / np.where(raw_expmap > 0, raw_expmap, np.nan)
        else:
            T_exp = header.get("EXPOSURE", 0.0)
            if T_exp and T_exp > 0:
                var_raw = np.abs(raw_data) / T_exp
            else:
                return None

        # --- propagate variance through smoothing kernel ---
        if fwhm_arcsec > 0:
            sigma_pix = fwhm_arcsec / 2.355 / abs(header["CDELT2"] * 3600)
            _, var_smoothed = gaussian_smooth_with_var(raw_data, var_raw, sigma_pix)
        else:
            var_smoothed = var_raw

        # --- reproject variance and return noise (sqrt) ---
        var_reproj = self._handler.reproject_to_sub(var_smoothed, wcs)
        with np.errstate(invalid="ignore"):
            return np.sqrt(var_reproj)

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def compute_background(self, xray_label: str, mask: np.ndarray) -> "Manager":
        """Set the cluster mask and compute background from unmasked pixels (True = cluster)."""
        entry = self.xray_maps[xray_label]
        bkg_pixels = entry["data"][~mask]
        entry["mask"]           = mask
        entry["background"]     = float(np.nanmean(bkg_pixels))
        entry["background_std"] = float(np.nanstd(bkg_pixels))
        return self
