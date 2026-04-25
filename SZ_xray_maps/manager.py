"""Manager: central orchestrator for SZ + X-ray map analysis."""
from __future__ import annotations

import numpy as np
from astropy import units as u

from .loading.base import Loader
from .thermo.base import ThermoProcessor
from .thermo.error_propagation import ErrorPropagator
from .plot.base import Plotter


class Manager(Loader, ThermoProcessor, ErrorPropagator, Plotter):
    """High-level interface for loading, computing, and plotting SZ/X-ray products."""

    def __init__(
        self,
        size_arcmin: float = 2.0,
        center: tuple[float, float] | None = None,
        z: float | None = None,
        M500: u.Quantity | None = None,
        kT: u.Quantity | None = None,
        kT_std: u.Quantity | None = None,
        metallicity: float = 0.37,
    ) -> None:

        self._handler      = None   # set on first add_sz call
        self._size_arcmin  = size_arcmin
        self._center       = center
        self.z             = z
        self.M500          = M500
        self.pixel_area_sr = None
        self.kT            = kT
        self.kT_std        = kT_std
        self.metallicity   = metallicity

        Loader.__init__(self)
        ThermoProcessor.__init__(self)
        Plotter.__init__(self)

    # ------------------------------------------------------------------
    # Reference WCS
    # ------------------------------------------------------------------

    def choose_reference_wcs(self, label: str) -> "Manager":
        """Switch the reference WCS to a loaded map and reproject cached data."""
        from .loading.fits_handler import FitsHandler

        all_maps = {**self.sz_maps, **self.xray_maps}
        if label not in all_maps:
            raise KeyError(
                f"No map with label '{label}'. "
                f"SZ: {list(self.sz_maps)}  X-ray: {list(self.xray_maps)}"
            )

        self._handler = FitsHandler(all_maps[label]["_path"], self._size_arcmin, self._center)
        self._update_pixel_area_sr()

        for entry in self.sz_maps.values():
            self._reproject_sz_entry(entry)

        for entry in self.xray_maps.values():
            self._reproject_xray_entry(entry)

        print(f"Reference WCS → '{label}'.  "
              f"Compton-y maps invalidated; call compute_compton_y() again.")
        return self

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable overview of the current state."""
        _sep = "-" * 40
        print(_sep)
        print(f"  z       : {self.z}")
        print(f"  M500    : {self.M500}")
        if self._handler:
            h = self._handler
            ctr = f"({h._cx}, {h._cy}) px"
            if self._center:
                ctr = f"RA={self._center[0]:.4f}, Dec={self._center[1]:.4f}"
            zoom_info = (f"{h.size_arcmin:.2f} arcmin  →  ±{h.zoom} px  "
                         f"({h.subimage_shape[0]}×{h.subimage_shape[1]})  centre={ctr}")
        else:
            zoom_info = f"{self._size_arcmin:.2f} arcmin  (reference WCS not set yet)"
        print(f"  zoom    : {zoom_info}")
        print(_sep)
        print("  SZ maps:")
        for lbl, entry in self.sz_maps.items():
            has_y = entry["compton_y"] is not None
            print(f"    [{lbl}]  instrument={entry['instrument'].name}"
                  f"  rms={entry['rms']:.4f} mJy/beam"
                  f"  compton_y={'✓' if has_y else '–'}")
        print("  X-ray maps:")
        for lbl, entry in self.xray_maps.items():
            bkg = f"{entry['background']:.3e}" if entry['background'] is not None else "–"
            print(f"    [{lbl}]  instrument={entry['instrument'].name}"
                  f"  mask={'✓' if entry['mask'] is not None else '–'}"
                  f"  background={bkg}")
        print("  Thermo maps :", list(self.thermo) or "–")
        print("  l_eff       :", self.l_eff)
        print(_sep)
