"""SZ instrument definitions and unit conversions."""
from __future__ import annotations

import numpy as np
from astropy.io.fits import Header

from ..utils.sz_units import (
    compton_to_jy_per_pix,
    jy_beam_to_jy_pix,
    k_bright_to_jy_per_beam,
)

_JY_BEAM_SCALES: dict[str, float] = {"Jy/beam": 1.0, "mJy/beam": 1e-3}


class SZInstrument:
    """Abstract base. Subclasses implement ``to_jy_beam``."""

    name: str
    frequency: float  # Hz

    def to_jy_beam(self, data: np.ndarray, header: Header) -> np.ndarray:
        raise NotImplementedError

    def to_compton_y(self, data_jy_beam: np.ndarray, header: Header) -> np.ndarray:
        """Jy/beam → Compton-y (shared by all instruments)."""
        beam_to_pix = jy_beam_to_jy_pix(
            header["CDELT1"], header["CDELT2"], header["BMAJ"], header["BMIN"]
        )
        jy_pix_to_y = compton_to_jy_per_pix(
            self.frequency, header["CDELT1"], header["CDELT2"]
        )
        return data_jy_beam * beam_to_pix / jy_pix_to_y


class ALMA(SZInstrument):
    """ALMA SZ map. ``frequency`` in Hz (e.g. ``92e9`` for Band 3)."""

    def __init__(self, frequency: float, units: str = "Jy/beam") -> None:
        self.frequency = frequency
        self.units = units
        self.name = f"ALMA_{frequency / 1e9:.0f}GHz"

    def to_jy_beam(self, data: np.ndarray, header: Header) -> np.ndarray:
        return data * _JY_BEAM_SCALES[self.units]


class Mustang2(SZInstrument):
    """Mustang-2 90 GHz maps in Jy/beam, mJy/beam, or K_RJ."""

    frequency: float = 90e9
    name: str = "Mustang2"

    def __init__(self, units: str = "Jy/beam") -> None:
        self.units = units

    def to_jy_beam(self, data: np.ndarray, header: Header) -> np.ndarray:
        if self.units == "K_RJ":
            return data * k_bright_to_jy_per_beam(
                self.frequency,
                header["CDELT1"], header["CDELT2"],
                header["BMAJ"],   header["BMIN"],
            )
        return data * _JY_BEAM_SCALES[self.units]
