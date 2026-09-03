"""Cosmology shared across the SZ_xray_maps package."""
from __future__ import annotations

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
