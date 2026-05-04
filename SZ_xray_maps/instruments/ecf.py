"""Compute X-ray Energy Conversion Factors (ECF) from XSPEC ARF/RMF.

ECF converts observed count rate [ct/s] to physical flux [erg/cm²/s]:
    S_X = count_rate * ECF / pixel_area_sr

Derived as:
    ECF = erg_flux_per_norm / count_rate_per_norm

where both quantities come from an APEC model evaluated at the cluster
temperature, folded through the instrument ARF/RMF.

If XSPEC is unavailable or ARF/RMF are not provided, falls back to a
hardcoded PIMMS-derived value.
"""
from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import atexit
import signal
from pathlib import Path

from astropy import units as u


# ---------------------------------------------------------------------------
# Hardcoded fallback ECFs
# PIMMS: absorbed APEC 7 keV, z=1.71, NH=1.23×10²⁰ cm⁻²
# ---------------------------------------------------------------------------
CHANDRA_FALLBACK_ECF = 1.027e-10 * u.erg / u.cm**2 / u.s / (u.ct / u.s)
XMM_FALLBACK_ECF     = 5.572e-12 * u.erg / u.cm**2 / u.s / (u.ct / u.s)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_ecf(
    rmf: str | None,
    arf: str | None,
    emin_keV: float,
    emax_keV: float,
    NH_1022pcm2: float,
    redshift: float | None,
    T_keV: float | None,
    metallicity: float,
    fallback: u.Quantity,
    abund: str = "lodd",
) -> u.Quantity:
    """Return ECF [erg cm⁻² ct⁻¹], computed from XSPEC or from *fallback*.

    Parameters
    ----------
    rmf, arf        Paths to the Chandra/XMM RMF and ARF response files.
                    Pass None to skip XSPEC and use *fallback* directly.
    emin_keV, emax_keV  Observed energy band [keV].
    NH_1022pcm2     Galactic hydrogen column [10²² cm⁻²].
    redshift        Cluster redshift.  Required for XSPEC path.
    T_keV           Cluster temperature [keV].  Required for XSPEC path.
    metallicity     Metal abundance [solar].
    fallback        ECF to use when XSPEC is unavailable.
    abund           XSPEC abundance table (default ``'lodd'``).
    """
    # --- guard: must have all inputs ---
    if None in (rmf, arf, redshift, T_keV):
        _warn_fallback("rmf, arf, redshift, or T_keV not provided", fallback)
        return fallback

    if not Path(rmf).exists():
        _warn_fallback(f"RMF not found: {rmf}", fallback)
        return fallback
    if not Path(arf).exists():
        _warn_fallback(f"ARF not found: {arf}", fallback)
        return fallback

    # --- try XSPEC (subprocess) ---
    if shutil.which("xspec") is not None:
        try:
            val = _ecf_subprocess(
                rmf, arf, emin_keV, emax_keV,
                NH_1022pcm2, redshift, T_keV, metallicity, abund,
            )
            print(
                f"ECF (XSPEC, {emin_keV:.1f}–{emax_keV:.1f} keV): "
                f"{val:.4e} erg cm⁻² ct⁻¹"
            )
            return val * u.erg / u.cm**2 / u.s / (u.ct / u.s)
        except Exception as exc:
            _warn_fallback(f"XSPEC subprocess failed ({exc})", fallback)

    # --- try PyXspec ---
    try:
        import xspec as _xspec  # noqa: PLC0415
        val = _ecf_pyxspec(
            _xspec, rmf, arf, emin_keV, emax_keV,
            NH_1022pcm2, redshift, T_keV, metallicity, abund,
        )
        print(
            f"ECF (PyXspec, {emin_keV:.1f}–{emax_keV:.1f} keV): "
            f"{val:.4e} erg cm⁻² ct⁻¹"
        )
        return val * u.erg / u.cm**2 / u.s / (u.ct / u.s)
    except ImportError:
        _warn_fallback("xspec binary not in PATH and PyXspec unavailable", fallback)
    except Exception as exc:
        _warn_fallback(f"PyXspec computation failed ({exc})", fallback)

    return fallback


def _warn_fallback(reason: str, fallback: u.Quantity) -> None:
    print(f"ECF: {reason} — using fallback {fallback.value:.4e} erg cm⁻² ct⁻¹")


# ---------------------------------------------------------------------------
# XSPEC subprocess implementation  (proven to work via xspechelper testing)
# ---------------------------------------------------------------------------

_SENTINEL = "@S@T@V"
_SENTINEL_RE = re.compile(rf"{_SENTINEL} (.*) {_SENTINEL}")
_ERG_RE = re.compile(r"\(([\d.e+\-]+)\s*ergs?/cm\^2/s\)")

_active_procs: list[subprocess.Popen] = []


def _start_xspec() -> subprocess.Popen:
    loop = (
        "autosave off\n"
        "while { 1 } {\n"
        " set s [gets stdin]\n"
        " if { [eof stdin] } { tclexit }\n"
        " eval $s\n"
        "}\n"
    )
    proc = subprocess.Popen(
        ["xspec"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )
    _active_procs.append(proc)
    _drain(proc)
    proc.stdin.write(f"set SCODE {_SENTINEL}\n")
    proc.stdin.write(loop)
    return proc


def _drain(proc: subprocess.Popen) -> None:
    while True:
        r, _, _ = select.select([proc.stdout.fileno()], [], [], 0)
        if r:
            chunk = os.read(r[0], 4096)
            if not chunk:
                break
        else:
            break


def _read_result(proc: subprocess.Popen) -> str:
    while True:
        line = proc.stdout.readline()
        m = _SENTINEL_RE.search(line)
        if m:
            return m.group(1)


def _send(proc: subprocess.Popen, cmd: str) -> None:
    proc.stdin.write(cmd)


def _stop_xspec(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.write("tclexit\n")
        proc.stdout.close()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    if proc in _active_procs:
        _active_procs.remove(proc)


def _ecf_subprocess(
    rmf: str, arf: str,
    emin: float, emax: float,
    NH: float, z: float, T: float, Z_met: float,
    abund: str,
) -> float:
    """Compute ECF via a subprocess XSPEC session."""
    tmp = f"/tmp/mbp_ecf_{os.getpid()}.fak"
    proc = _start_xspec()
    try:
        # --- load response (fakeit with a dummy powerlaw first) ---
        _send(proc, f"model TBabs(powerlaw) & 0.1 & 1.7 & 1\n")
        _drain(proc)
        _send(proc, f"fakeit none & {rmf} & {arf} & y & foo & {tmp} & 1.0\n")
        _drain(proc)
        _send(proc, f"ignore **:**-{emin:.4f} {emax:.4f}-**\n")
        _drain(proc)

        # --- set APEC model ---
        _send(proc, f"abund {abund}\n")
        _drain(proc)
        _send(proc, f"model TBabs(apec) & {NH} & {T} & {Z_met} & {z} & 1\n")
        _drain(proc)

        # --- count rate per unit norm ---
        _send(proc, f'puts "$SCODE [tcloutr rate 1] $SCODE"\n')
        rate_str = _read_result(proc)
        rate = float(rate_str.split()[2])   # predicted model rate [ct/s]

        # --- erg flux per unit norm (from text output of flux command) ---
        _send(proc, f"flux {emin:.4e} {emax:.4e}\n")
        _send(proc, f'puts "$SCODE done $SCODE"\n')
        lines = []
        while True:
            line = proc.stdout.readline()
            lines.append(line)
            if _SENTINEL_RE.search(line):
                break
        erg_flux = None
        for line in lines:
            m = _ERG_RE.search(line)
            if m:
                erg_flux = float(m.group(1))
                break
        if erg_flux is None:
            raise RuntimeError(f"Could not parse erg flux from XSPEC output: {lines}")

    finally:
        _stop_xspec(proc)
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if rate <= 0:
        raise ValueError(f"Non-positive model rate: {rate}")

    return erg_flux / rate


# ---------------------------------------------------------------------------
# PyXspec implementation
# ---------------------------------------------------------------------------

def _ecf_pyxspec(
    xspec,
    rmf: str, arf: str,
    emin: float, emax: float,
    NH: float, z: float, T: float, Z_met: float,
    abund: str,
) -> float:
    """Compute ECF via the PyXspec Python interface."""
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Xset.abund = abund

    xspec.AllData.clear()
    xspec.AllModels.clear()

    # Model must exist before fakeit
    m = xspec.Model("TBabs*apec")
    m(1).values = NH
    m(2).values = T
    m(3).values = Z_met
    m(4).values = z
    m(5).values = 1.0

    fs = xspec.FakeitSettings(response=rmf, arf=arf, exposure=1.0)
    xspec.AllData.fakeit(1, fs, noWrite=True)
    xspec.AllData.ignore(f"**-{emin:.4f} {emax:.4f}-**")

    # Re-apply model parameters after fakeit
    m = xspec.AllModels(1)
    m(1).values = NH
    m(2).values = T
    m(3).values = Z_met
    m(4).values = z
    m(5).values = 1.0

    rate = xspec.AllData(1).rate[2]          # predicted model rate [ct/s]

    xspec.AllModels.calcFlux(f"{emin:.4f} {emax:.4f}")
    erg_flux = xspec.AllModels(1).flux[0]    # erg/cm²/s (tcloutr flux index 0)

    xspec.AllData.clear()
    xspec.AllModels.clear()

    if rate <= 0:
        raise ValueError(f"Non-positive model rate: {rate}")

    return erg_flux / rate


# ---------------------------------------------------------------------------
# Clean up any lingering XSPEC processes on exit
# ---------------------------------------------------------------------------

def _cleanup() -> None:
    for proc in list(_active_procs):
        _stop_xspec(proc)


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), exit(0)))
