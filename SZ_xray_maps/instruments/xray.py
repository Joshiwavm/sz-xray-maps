"""X-ray instrument classes."""
from __future__ import annotations

import numpy as np
from astropy import units as u

from .ecf import compute_ecf, CHANDRA_FALLBACK_ECF, XMM_FALLBACK_ECF


class XRayInstrument:
    """Abstract base for X-ray imaging instruments."""

    name: str
    ecf: u.Quantity  # erg cm⁻² ct⁻¹

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply instrument-specific preprocessing."""
        return data


class Chandra(XRayInstrument):
    """Chandra ACIS-I 0.5–2.0 keV maps in ct/s/pixel.

    Parameters
    ----------
    rmf, arf        Paths to the Chandra RMF and ARF files.  If provided
                    together with *z* and *T_keV*, the ECF is computed via
                    XSPEC; otherwise the hardcoded fallback is used.
    NH_1022pcm2     Galactic N_H [10²² cm⁻²] (default 0.0123).
    z               Cluster redshift.
    T_keV           Cluster temperature [keV].
    metallicity     Metal abundance [solar] (default 0.37).
    """

    name: str = "Chandra"

    def __init__(
        self,
        rmf: str | None = None,
        arf: str | None = None,
        NH_1022pcm2: float = 0.0123,
        z: float | None = None,
        T_keV: float | None = None,
        metallicity: float = 0.37,
    ) -> None:
        self.ecf = compute_ecf(
            rmf=rmf, arf=arf,
            emin_keV=0.5, emax_keV=2.0,
            NH_1022pcm2=NH_1022pcm2,
            redshift=z, T_keV=T_keV,
            metallicity=metallicity,
            fallback=CHANDRA_FALLBACK_ECF,
        )


class XMM(XRayInstrument):
    """XMM-Newton EPIC MOS thin 0.4–4.0 keV maps in ct/s/pixel.

    Parameters
    ----------
    rmf, arf        Paths to the XMM RMF and ARF files.  If provided
                    together with *z* and *T_keV*, the ECF is computed via
                    XSPEC; otherwise the hardcoded fallback is used.
    NH_1022pcm2     Galactic N_H [10²² cm⁻²] (default 0.0123).
    z               Cluster redshift.
    T_keV           Cluster temperature [keV].
    metallicity     Metal abundance [solar] (default 0.37).
    """

    name: str = "XMM"

    def __init__(
        self,
        rmf: str | None = None,
        arf: str | None = None,
        NH_1022pcm2: float = 0.0123,
        z: float | None = None,
        T_keV: float | None = None,
        metallicity: float = 0.37,
    ) -> None:
        self.ecf = compute_ecf(
            rmf=rmf, arf=arf,
            emin_keV=0.4, emax_keV=4.0,
            NH_1022pcm2=NH_1022pcm2,
            redshift=z, T_keV=T_keV,
            metallicity=metallicity,
            fallback=XMM_FALLBACK_ECF,
        )

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        out[out == 0.0] = np.nan
        return out
