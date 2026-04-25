"""FitsHandler: generic FITS loading, reprojection, and subimage extraction."""
from __future__ import annotations

import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_interp


class FitsHandler:
    """Load FITS maps and reproject them onto a shared reference grid."""

    def __init__(
        self,
        reference_map: str,
        size_arcmin: float = 2.0,
        center: tuple[float, float] | None = None,
    ) -> None:
        self.size_arcmin = size_arcmin

        hdu = fits.open(reference_map)
        self._ref_data = self._to_2d(hdu[0].data)
        self._ref_hdr  = hdu[0].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            self.wcs_ref = WCS(self._ref_hdr, naxis=2)

        ny, nx = self._ref_data.shape

        # --- centre pixel ---
        if center is not None:
            ra, dec = center
            pix = self.wcs_ref.all_world2pix([[ra, dec]], 0)[0]
            self._cx = int(round(float(pix[0])))
            self._cy = int(round(float(pix[1])))
        else:
            self._cy, self._cx = ny // 2, nx // 2

        # --- half-width in pixels from arcmin + pixel scale ---
        pixel_scale_deg    = abs(self._ref_hdr["CDELT2"])
        pixel_scale_arcmin = pixel_scale_deg * 60.0
        self.zoom = max(1, int(round(0.5 * size_arcmin / pixel_scale_arcmin)))

        # --- subimage WCS sliced to match the data cutout exactly ---
        z = self.zoom
        self.wcs_sub = self.wcs_ref[
            self._cy - z : self._cy + z,
            self._cx - z : self._cx + z,
        ]

    @property
    def subimage_shape(self) -> tuple[int, int]:
        return (self.zoom * 2, self.zoom * 2)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, path: str) -> tuple[np.ndarray, fits.Header, WCS]:
        """Load a FITS file and return ``(2D data, header, 2D WCS)``."""
        hdu    = fits.open(path)
        data   = self._to_2d(hdu[0].data)
        header = hdu[0].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header, naxis=2)
        return data, header, wcs

    # ------------------------------------------------------------------
    # Reprojection
    # ------------------------------------------------------------------

    def reproject_to_reference(self, data: np.ndarray, wcs: WCS) -> np.ndarray:
        """Reproject *data* onto the full reference grid."""
        reproj, _ = reproject_interp(
            (data, wcs), self.wcs_ref, shape_out=self._ref_data.shape
        )
        return reproj

    def reproject_to_sub(self, data: np.ndarray, wcs: WCS) -> np.ndarray:
        """Reproject *data* directly onto the zoomed subimage grid."""
        reproj, _ = reproject_interp(
            (data, wcs), self.wcs_sub, shape_out=self.subimage_shape
        )
        return reproj

    # ------------------------------------------------------------------
    # Subimage
    # ------------------------------------------------------------------

    def subimage(self, data: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Extract the zoomed cutout aligned with ``wcs_sub``."""
        cy, cx, z = self._cy, self._cx, self.zoom
        return scale * data[cy - z : cy + z, cx - z : cx + z]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d(data: np.ndarray) -> np.ndarray:
        """Squeeze a FITS cube to 2-D."""
        if data.ndim == 2:
            return data
        if data.ndim == 4:
            return data[0, 0]
        if data.ndim == 3:
            return data[0]
        return data.squeeze()
