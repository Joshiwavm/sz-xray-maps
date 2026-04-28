# SZ-Xray-Maps

Python package for computing projected thermodynamic maps of galaxy clusters from combined SZ and X-ray observations. Derives per-pixel maps of electron pressure, density, temperature, and entropy with propagated uncertainties.

## Installation

```bash
cd scripts/Thermodynamical_maps
pip install -e .
```

[pyatomdb](https://atomdb.readthedocs.io/en/master/installation.html) requires a separate AtomDB data download on first use — follow the instructions at that link before running.

## Usage

See [docs/tutorial.md](docs/tutorial.md) for a full walkthrough, or `Notebooks/Thermo_maps.ipynb` for a worked example on ACT-CL J0459.6−4946 (z = 1.71).

```python
from SZ_xray_maps import Manager
from SZ_xray_maps.instruments import ALMA, Chandra, XMM

mgr = Manager(cluster_tag="J0459", size_arcmin=2.5, center=(74.923, -49.782),
              z=1.71, M500=4e14 * u.Msun, kT=7.2 * u.keV, kT_std=0.5 * u.keV,
              metallicity=0.37)
```

Cooling function tables are generated automatically on first instantiation and cached in `SZ_xray_maps/cooling_function/`.

## Citation

If you use this code, please cite the associated paper (in prep.).
