"""Generate per-bin cooling function .npz files using PyAtomDB."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 101 edges → 100 log-spaced observed-frame bins covering 0.4–10 keV
DEFAULT_EBINS: np.ndarray = np.array([
    0.400, 0.413, 0.427, 0.441, 0.455, 0.470, 0.485, 0.501, 0.517, 0.534,
    0.552, 0.570, 0.589, 0.608, 0.628, 0.648, 0.669, 0.691, 0.714, 0.737,
    0.761, 0.786, 0.812, 0.839, 0.866, 0.894, 0.924, 0.954, 0.985, 1.017,
    1.051, 1.085, 1.120, 1.157, 1.195, 1.234, 1.274, 1.316, 1.359, 1.404,
    1.450, 1.497, 1.546, 1.597, 1.649, 1.703, 1.758, 1.816, 1.875, 1.937,
    2.000, 2.065, 2.133, 2.203, 2.275, 2.349, 2.426, 2.505, 2.587, 2.672,
    2.759, 2.850, 2.943, 3.039, 3.139, 3.241, 3.347, 3.457, 3.570, 3.687,
    3.807, 3.932, 4.060, 4.193, 4.330, 4.472, 4.618, 4.770, 4.926, 5.087,
    5.253, 5.425, 5.602, 5.786, 5.975, 6.170, 6.372, 6.581, 6.796, 7.018,
    7.248, 7.485, 7.730, 7.983, 8.244, 8.513, 8.792, 9.079, 9.377, 9.683,
    10.000,
])

DEFAULT_TLIST: np.ndarray = np.logspace(5, 9, 1001)


def generate_bins(
    cluster_tag: str,
    z: float,
    metallicity: float,
    output_dir: str | Path,
    ebins: np.ndarray | None = None,
    tlist: np.ndarray | None = None,
) -> None:
    """Generate per-bin cooling function .npz files for a cluster.

    Files are named:
        cooling_function_{cluster_tag}_{metallicity:.2f}solar_{emin:.3f}-{emax:.3f}keV.npz

    Each file stores:
        temperature  [K]         — log-spaced values from 1e5 to 1e9 K
        power_tot    [erg cm³/s] — cooling function integrated over that observed-frame bin

    Energy bins are in observed-frame keV; rest-frame energies (E_obs * (1+z)) are
    passed to PyAtomDB which computes spectra in the plasma rest frame.
    """
    import pyatomdb
    import pyatomdb.const as atomdb_const

    if ebins is None:
        ebins = DEFAULT_EBINS
    if tlist is None:
        tlist = DEFAULT_TLIST

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cie = pyatomdb.spectrum.CIESession()
    cie.dolines  = True
    cie.docont   = True

    cie.set_abund(np.arange(3, 31), metallicity)  # metals only; H (Z=1) and He (Z=2) stay at solar

    n_bins  = len(ebins) - 1
    n_exist = 0
    for i in range(n_bins):
        emin, emax = ebins[i], ebins[i + 1]
        fname = output_dir / (
            f"cooling_function_{cluster_tag}_{metallicity:.2f}solar_"
            f"{emin:.3f}-{emax:.3f}keV.npz"
        )
        if fname.exists():
            n_exist += 1
            continue

        emin_rest     = emin * (1 + z)
        emax_rest     = emax * (1 + z)
        e_centre_rest = (emin_rest + emax_rest) / 2.0
        cie.set_response(np.array([emin_rest, emax_rest]), raw=True)

        power = np.zeros(len(tlist))
        for j, T_K in enumerate(tlist):
            kT      = T_K * atomdb_const.KBOLTZ
            spec    = cie.return_spectrum(kT)
            power[j] = float(np.sum(spec)) * e_centre_rest * atomdb_const.ERG_KEV

        np.savez(fname, temperature=tlist, power_tot=power)
        print(f"  [{i+1:3d}/{n_bins}] {emin:.3f}–{emax:.3f} keV  saved  "
              f"(peak Λ = {power.max():.3e})")

    if n_exist == n_bins:
        print(f"Cooling function: {n_bins} bins already on disk for "
              f"{cluster_tag}  z={z}  Z={metallicity:.2f} solar  — skipping generation.")
    elif n_exist > 0:
        print(f"Cooling function: {n_bins - n_exist} new bins written, "
              f"{n_exist} already existed.")
    else:
        print(f"Cooling function: {n_bins} bins written for "
              f"{cluster_tag}  z={z}  Z={metallicity:.2f} solar.")
