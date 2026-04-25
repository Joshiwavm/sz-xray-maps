"""Error propagation: per-pixel noise estimation and SNR maps."""
from __future__ import annotations

import numpy as np


class ErrorPropagator:
    """Mixin that adds noise estimation and SNR map computation to the Manager."""

    def compute_noise(self) -> "Manager":
        """Estimate per-pixel noise for the loaded SZ and X-ray maps."""
        from ..utils.sz_units import jy_beam_to_jy_pix, compton_to_jy_per_pix

        for lbl, entry in self.sz_maps.items():
            if entry["compton_y"] is None:
                continue
            instr   = entry["instrument"]
            hdr     = entry["header"]
            rms_jy  = entry["rms"] * 1e-3  # mJy/beam → Jy/beam
            beam_to_pix     = jy_beam_to_jy_pix(hdr["CDELT1"], hdr["CDELT2"],
                                                  hdr["BMAJ"], hdr["BMIN"])
            compton_per_pix = compton_to_jy_per_pix(instr.frequency,
                                                      hdr["CDELT1"], hdr["CDELT2"])
            entry["noise_y"] = float(rms_jy * beam_to_pix / compton_per_pix)

        for lbl, entry in self.xray_maps.items():
            # Noise map was propagated through the smoothing kernel in add_xray().
            # Here we only add background uncertainty in quadrature now that
            # compute_background() has been called.
            noise   = entry.get("noise")
            bkg_std = entry.get("background_std")
            with np.errstate(invalid="ignore"):
                noise = np.sqrt(noise ** 2 + bkg_std ** 2)
            entry["noise"] = noise

        return self

    def compute_snr(self, sz_label: str, xray_labels: list[str]) -> "Manager":
        """Propagate thermo uncertainties into per-pixel SNR maps."""
        sz_entry  = self.sz_maps[sz_label]
        compton_y = sz_entry["compton_y"]
        noise_y   = sz_entry["noise_y"]

        σ_l_rel = (np.std(self.l_eff_samples) / np.mean(self.l_eff_samples)
                   if self.l_eff_samples is not None else 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            σ_y_rel = noise_y / np.abs(compton_y)

        for lbl in xray_labels:
            xr    = self.xray_maps[lbl]
            noise = xr.get("noise")

            with np.errstate(invalid="ignore", divide="ignore"):
                # Propagate from the three independent observables (y, S_X, l_eff).
                # P_e ∝ y/l        → σ_P² = σ_y² + σ_l²
                # n_e ∝ √(S_X/l)   → σ_n² = ¼(σ_S² + σ_l²)
                # T_e ∝ y S_X^-½ l^-½  → σ_T² = σ_y² + ¼σ_S² + ¼σ_l²
                # K_e ∝ y S_X^-⁵/⁶ l^-⅙ → σ_K² = σ_y² + (25/36)σ_S² + (1/36)σ_l²
                # Treating P_e and n_e as independent (adding σ_P² + σ_n² for T_e)
                # would double-count σ_l, so each quantity is derived from scratch.
                σ_P_rel = np.sqrt(σ_y_rel ** 2 + σ_l_rel ** 2)

                data_net = xr["data"] - (xr["background"] or 0.0)
                σ_S_rel  = noise / np.abs(data_net)
                σ_n_rel  = 0.5 * np.sqrt(σ_S_rel ** 2 + σ_l_rel ** 2)
                σ_T_rel  = np.sqrt(σ_y_rel ** 2
                                    + 0.25 * σ_S_rel ** 2
                                    + 0.25 * σ_l_rel ** 2)
                σ_K_rel  = np.sqrt(σ_y_rel ** 2
                                    + (25 / 36) * σ_S_rel ** 2
                                    + (1  / 36) * σ_l_rel ** 2)
                
            # SNR = 1 / relative_error
            self.snr[lbl] = {
                "P_e": 1.0 / σ_P_rel,
                "n_e": 1.0 / σ_n_rel,
                "T_e": 1.0 / σ_T_rel,
                "K_e": 1.0 / σ_K_rel,
            }

        return self
