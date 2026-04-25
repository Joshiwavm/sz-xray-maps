"""X-ray instrument classes."""
from __future__ import annotations

import numpy as np
from astropy import units as u


class XRayInstrument:
    """Abstract base for X-ray imaging instruments."""

    name: str
    ecf: u.Quantity  # erg cm⁻² ct⁻¹

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply instrument-specific preprocessing."""
        return data


class Chandra(XRayInstrument):
    """Chandra ACIS-I 0.5-2.0 keV maps in ct/s/pixel."""

    name: str = "Chandra"
    ecf: u.Quantity = 1.027e-10 * u.erg / u.cm**2 / u.s / (u.ct / u.s)  # PIMMS absorbed, ACIS-I 0.5-2 keV, APEC 7 keV, z=1.71, NH=1.23e20

class XMM(XRayInstrument):
    """XMM-Newton EPIC MOS thin 0.4-4.0 keV maps in ct/s/pixel."""

    name: str = "XMM"
    ecf: u.Quantity = 5.572e-12 * u.erg / u.cm**2 / u.s / (u.ct / u.s)  # PIMMS absorbed, MOS thin 0.4-4 keV, APEC 7 keV, z=1.71, NH=1.23e20

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        out[out == 0.0] = np.nan
        return out
