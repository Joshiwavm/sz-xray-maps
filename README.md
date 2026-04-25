# SZ-Xray-Maps

Python package for computing projected thermodynamic maps of galaxy clusters by combining Sunyaev-Zel'dovich (SZ) and X-ray observations.

Given a Compton-y map and an X-ray surface brightness map, the package derives per-pixel projected maps of:
- electron pressure P_e
- electron density n_e
- projected temperature T_e
- entropy K_e

along with per-pixel SNR maps for each quantity.

## Method

The thermodynamic quantities follow from the SZ pressure and X-ray surface brightness via an effective line-of-sight depth l_eff:

```
P_e = (m_e c² / σ_T) · y / l_eff
n_e = sqrt(1.2 · 4π(1+z)⁴ · S_X / (Λ(T) · l_eff))
T_e = P_e / n_e
K_e = T_e / n_e^(2/3)
```

where `1.2 = n_e/n_H` (fully ionised plasma), `Λ(T)` is the X-ray cooling function evaluated at the spectroscopic temperature, and `S_X = rate · ECF / Ω_pix` converts per-pixel count rates to surface brightness.

### Effective line-of-sight depth

Two methods are provided:

**Uniform (Monte Carlo)** — `compute_leff_uniform()`: solves for the single depth that reproduces a spectroscopic temperature prior by root-finding over Monte Carlo temperature samples. Equivalent to the shape factor `C = sqrt(r500 / l_eff)`, which must satisfy `C ≥ 1` (Cauchy–Schwarz constraint). Useful as a sanity check but assumes uniform density along the line of sight.

**Beta-model map** — `compute_leff_map()`: computes a pixel-by-pixel l_eff from a projected spherical beta model using the analytical formula for `l_eff = (∫n_e dl)² / ∫n_e² dl`. Recommended for science maps.

### Cross-calibration

`calibrate_xray()` cross-calibrates a second X-ray instrument against a reference by matching the median projected temperature within a calibration aperture:

```
factor = median(T_ref) / median(T_target)
n_target → n_target / factor   ⟹   T_target → T_target × factor
```

### Noise and SNR

Poisson variance is propagated analytically through the Gaussian smoothing kernel applied to the X-ray data (`Var(Σ gᵢXᵢ) = Σ gᵢ² Var(Xᵢ)`), so noise reflects the smoothed resolution rather than the native pixel scale. Relative uncertainties on P_e, n_e, T_e, K_e are propagated from SZ RMS, X-ray Poisson noise, and l_eff uncertainty (from MC samples).

## Supported instruments

**SZ:** ALMA (any band), Mustang-2

**X-ray:** Chandra ACIS-I (0.5–2 keV), XMM-Newton EPIC MOS thin (0.4–4 keV)

Adding a new instrument requires subclassing `SZInstrument` or `XRayInstrument` and providing an ECF. ECF values should be derived from PIMMS for the appropriate band, spectral model, redshift, and Galactic NH.

## Installation

```bash
cd scripts/Thermodynamical_maps
pip install -e .
```

## Quick start

```python
import numpy as np
from astropy import units as u
from SZ_xray_maps import Manager
from SZ_xray_maps.instruments import ALMA, Chandra, XMM
from SZ_xray_maps.utils import circle_mask

mgr = Manager(
    size_arcmin=2.5,
    center=(74.923, -49.782),   # RA, Dec [deg]
    z=1.71,
    M500=4e14 * u.Msun,
    kT=7.2 * u.keV,
    kT_std=0.5 * u.keV,
    metallicity=0.37,
)

# Load maps — first SZ call sets the reference WCS
mgr.add_sz("band3.fits", ALMA(92e9), label="band3")
mgr.add_xray("chandra.fits", Chandra(), label="chandra",
             smooth_fwhm_arcsec=4.5, expmap_path="expmap.fits",
             energy_range=(0.5, 2.0))
mgr.add_xray("xmm.fits", XMM(), label="xmm",
             smooth_fwhm_arcsec=6.0, energy_range=(0.4, 4.0))

# Compton-y map
mgr.compute_compton_y("band3")

# Background subtraction
mask = circle_mask(mgr.xray_maps["chandra"]["data"], ...)
mgr.compute_background("chandra", mask)
mgr.compute_background("xmm", mask)

# Effective line-of-sight depth (choose one)
mgr.compute_leff_map(rc_arcsec=19.5, beta=0.82)           # beta-model (recommended)
mgr.compute_leff_uniform("band3", "xmm", radius_arcsec=20) # MC uniform (sanity check)

# Cross-calibration (Chandra relative to XMM)
mgr.calibrate_xray(ref_label="xmm", target_label="chandra",
                   sz_label="band3", radius_arcsec=20)

# Thermodynamic maps
mgr.compute_thermo("band3", ["chandra", "xmm"], use_leff_map=True)

# Noise and SNR
mgr.compute_noise()
mgr.compute_snr("band3", ["chandra", "xmm"])

# Diagnostics and plotting
mgr.diagnostics()
fig, axs = mgr.plot_thermo_grid("band3", ["chandra", "xmm"])
mgr.save_thermo_fits("output/")
```

See `Notebooks/Thermo_maps.ipynb` for a full worked example on ACT-CL J0459.6−4946 (z = 1.71).

## Cooling function data

`data/cooling_function/` contains pre-computed per-energy-bin cooling functions generated with PyAtomDB for a specific cluster (J0459, z = 1.71, 0.37 solar metallicity). The bins are in **observed-frame** keV and already account for the cluster redshift — pass `energy_range` in observed-frame keV directly.

The glob pattern at runtime (`cooling_function_*{metallicity:.2f}solar*.npz`) is general: tables for other metallicities or clusters can be added alongside the existing files.

## Dependencies

- numpy, scipy, astropy, matplotlib
- [reproject](https://reproject.readthedocs.io) — reprojecting maps between WCS grids

## Citation

If you use this code, please cite the associated paper (in prep.).
