"""Plotter mixin: high-level plot methods added to the Manager."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import Planck18 as cosmo

from .maps import (
    add_colorbar, add_label_box, add_r500_circle,
    add_beam_patch, configure_wcs_ticks, hide_spines,
    make_map_grid, make_grid_figure,
)
from .style import setup_style, get_planck_cmap, get_inferno_cmap


class Plotter:
    """Mixin that adds plot_* methods to the Manager."""

    # ------------------------------------------------------------------
    # Panel configuration — edit vmin/vmax here
    # ------------------------------------------------------------------
    # Each entry: (cmap_key, scale, cbar_label, vmin, vmax)
    # cmap_key: "planck", "inferno", or any matplotlib name
    # scale:    data multiplier before plotting
    # vmin/vmax: None → auto from data
    _PANEL_CONFIGS: dict[str, tuple] = {
        "compton_y":   ("planck",  1e4,  r"Compton-$y$ [×10$^{-4}$]",                  -0.2,  None),
        "P_e":         ("planck",  1e3,  r"$\bar{P}_e$ [keV cm$^{-3}$ ×10$^{-3}$]",   -20.0,  None),
        "n_e":         ("inferno", 1e3,  r"$\bar{n}_e$ [cm$^{-3}$ ×10$^{-3}$]",         0.0,  10),
        "T_e":         ("magma",   1.0,  r"$k_{\rm b}\bar{T}_e$ [keV]",                          0.0,  13),
        "K_e":         ("cividis", 1e-3, r"$\bar{K}_e$ [keV cm$^2$ ×10$^{-3}$]",        0.0,  0.5),
        # SNR maps (quantity / σ_quantity)
        "snr_compton_y": ("viridis", 1.0,  r"SNR Compton-$y$",   0.0, None),
        "snr_P_e":       ("viridis", 1.0,  r"SNR $\bar{P}_e$",   0.0, None),
        "snr_n_e":       ("viridis", 1.0,  r"SNR $\bar{n}_e$",   0.0, 5),
        "snr_T_e":       ("viridis", 1.0,  r"SNR $k_{\rm b}\bar{T}_e$",   0.0, 5),
        "snr_K_e":       ("viridis", 1.0,  r"SNR $\bar{K}_e$",   0.0, 5),
    }

    def __init__(self) -> None:
        setup_style()
        self.planck_cmap  = get_planck_cmap()
        self.inferno_cmap = get_inferno_cmap()

    def _resolve_cmap(self, key: str):
        if key == "planck":
            return self.planck_cmap
        if key == "inferno":
            return self.inferno_cmap
        return key

    # ------------------------------------------------------------------
    # General map plotting
    # ------------------------------------------------------------------

    def plot_maps(
        self,
        plotmaps,
        panel_labels=None,
        contour_sz: bool = True,
        contour_levels: list | None = None,
        snr_contours: bool | list = False,
        zoom_arcsec: float | None = None,
        show_mask: bool = False,
        show_mask2: bool = False,
        figsize: tuple | None = None,
        save_path: str | None = None,
    ) -> tuple:
        """Plot SZ, X-ray, and derived maps in a flexible grid.

        snr_contours: False to skip, True for default levels, or a list of levels.
        zoom_arcsec:  half-width of the zoom region in arcsec, centred on the map centre.
        """
        if isinstance(plotmaps, str):
            grid = np.array([[plotmaps]])
        else:
            grid = np.atleast_2d(np.asarray(plotmaps))
        nrows, ncols = grid.shape

        plabels = None if panel_labels is None else np.atleast_2d(np.asarray(panel_labels))

        if contour_levels is None:
            contour_levels = list(np.append(np.arange(-52, 0, 4), [-2, 2]))
        snr_contour_levels = snr_contours if isinstance(snr_contours, list) else [1.5, 2.5, 3.5, 4.5, 5.5]

        if zoom_arcsec is not None:
            arcsec_per_pix = abs(self._handler.wcs_sub.wcs.cdelt[1]) * 3600
            half_px = zoom_arcsec / arcsec_per_pix
            ny, nx = next(iter(self.sz_maps.values()))["data"].shape
            cx, cy = nx / 2, ny / 2
            xlim = (cx - half_px, cx + half_px)
            ylim = (cy - half_px, cy + half_px)
        else:
            xlim = ylim = None

        fig, panel_axes = make_map_grid(nrows, ncols, self._handler.wcs_sub,
                                        figsize or (5.5 * ncols, 4.5 * nrows))

        for r in range(nrows):
            for c in range(ncols):
                lbl  = grid[r, c]
                ax, cax = panel_axes[r, c]

                if lbl in self.sz_maps:
                    img, cbar_label = self._render_sz_panel(ax, lbl, contour_sz, contour_levels)

                elif lbl in self.xray_maps:
                    img, cbar_label = self._render_xray_panel(ax, lbl, show_mask, show_mask2)

                elif ":" in lbl:
                    quantity, src = lbl.split(":", 1)
                    img, cbar_label = self._render_derived_panel(
                        ax, quantity, src, snr_contours, snr_contour_levels,
                    )  # snr_contour_levels already resolved from snr_contours above

                else:
                    raise KeyError(f"'{lbl}' not found in sz_maps, xray_maps, or derived quantities.")

                if xlim is not None:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                fig.colorbar(img, cax=cax).set_label(cbar_label, fontsize=10)
                if plabels is not None:
                    add_label_box(ax, plabels[r, c])
                configure_wcs_ticks(ax, show_ra=(r == nrows - 1), show_dec=(c == 0))
                hide_spines(ax)
                if c == 0:
                    ax.coords[1].set_axislabel("Dec [J2000]", size=10)
                if r == nrows - 1:
                    ax.coords[0].set_axislabel("RA [J2000]", size=10)

        if save_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.savefig(save_path, dpi=fig.dpi, transparent=True, bbox_inches="tight")
        return fig, panel_axes

    def _render_sz_panel(self, ax, lbl, contour_sz, contour_levels):
        entry = self.sz_maps[lbl]
        data  = entry["data"]
        vmax  = np.nanmax(np.abs(data))
        img   = ax.imshow(data, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        if contour_sz:
            ax.contour(entry["snr"], levels=contour_levels,
                       colors="white", linewidths=0.5, alpha=0.5)
        return img, "mJy/beam"

    def _render_xray_panel(self, ax, lbl, show_mask, show_mask2):
        entry   = self.xray_maps[lbl]
        data    = entry["data"]
        mask    = entry["mask"]
        display = data.copy().astype(float)
        if show_mask and mask is not None:
            display[mask] = np.nan
        cmap = copy(self.inferno_cmap)
        cmap.set_bad("black")
        img = ax.imshow(display, cmap=cmap, origin="lower", interpolation="none")
        if show_mask2 and self.calib_mask is not None:
            ax.contour(self.calib_mask.astype(float), levels=[0.5],
                       colors="yellow", linewidths=0.8)
        return img, r"S$_X$ [counts s$^{-1}$]"

    def _render_derived_panel(
        self, ax, quantity: str, src: str,
        snr_contours: bool | list = False,
        snr_contour_levels: list | None = None,
    ):
        """Render a compton_y or thermo quantity panel; return (img, cbar_label)."""
        if quantity not in self._PANEL_CONFIGS:
            raise KeyError(
                f"Unknown quantity '{quantity}'. "
                f"Choose from: {', '.join(self._PANEL_CONFIGS)}."
            )
        cmap_key, scale, cbar_label, vmin, vmax = self._PANEL_CONFIGS[quantity]
        cmap = self._resolve_cmap(cmap_key)

        if quantity.startswith("snr_"):
            q = quantity[4:]  # strip "snr_"
            if src not in self.snr:
                raise KeyError(f"SNR not computed for '{src}'. Call compute_snr() first.")
            data = self.snr[src][q] * scale
            img  = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            return img, cbar_label

        if quantity == "compton_y":
            data = self.sz_maps[src]["compton_y"] * scale
            v    = np.nanmax(np.abs(data))
            img  = ax.imshow(data, cmap=cmap, origin="lower",
                             vmin=vmin if vmin is not None else -v,
                             vmax=vmax if vmax is not None else  v)
        else:
            if src not in self.thermo:
                raise KeyError(f"Thermo label '{src}' not found. Call compute_thermo() first.")
            data = self.thermo[src][quantity].value * scale
            img  = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

        if snr_contours and self.snr and src in self.snr and quantity in self.snr[src]:
            levels = (snr_contours if isinstance(snr_contours, list)
                      else (snr_contour_levels or [1.5, 2.5, 3.5, 4.5, 5.5]))
            snr_data = self.snr[src][quantity]
            ax.contour(convolve(snr_data, Gaussian2DKernel(1)), levels=levels,
                       colors="white", linewidths=0.5, linestyles="-", alpha=0.5)

        return img, cbar_label

    # ------------------------------------------------------------------
    # Thermodynamical map grid
    # ------------------------------------------------------------------

    def plot_thermo_grid(
        self,
        sz_label: str,
        xray_labels: list[str] | None = None,
        sz_contours: bool = True,
        snr_contours: bool = False,
        contour_levels: list | None = None,
        snr_contour_levels: list | None = None,
        snr_mask_threshold: float = -4.0,
        figsize: tuple | None = None,
        save_path: str | None = None,
    ) -> tuple:
        """Plot thermodynamical maps for one or more X-ray instruments."""
        xray_labels    = xray_labels or list(self.thermo.keys())
        contour_levels = contour_levels or list(np.arange(-52, 0, 8))
        snr_contour_levels = snr_contour_levels or [1.5, 2.5, 3.5, 4.5, 5.5]
        # snr_contour_levels = snr_contour_levels or [1.0, 2.0, 3.0, 4.0, 5.0]

        sz_entry     = self.sz_maps[sz_label]
        compton_y    = sz_entry["compton_y"]
        snr_sz       = sz_entry["snr"]
        cluster_mask = snr_sz < snr_mask_threshold
        theta500     = self._theta500()
        cy_peak, cx_peak = np.unravel_index(np.nanargmax(compton_y), compton_y.shape)

        panels, ncols = self._build_thermo_panels(xray_labels, compton_y, cluster_mask)
        nrows   = len(panels) // ncols
        figsize = figsize or (4.5 * ncols, 2.8 * nrows)

        fig, axs = make_grid_figure(nrows, ncols, self._handler.wcs_sub, figsize)

        for i, panel in enumerate(panels):
            ax  = axs[i]
            row, col = i // ncols, i % ncols

            img = ax.imshow(panel["data"], cmap=panel["cmap"], origin="lower",
                            vmin=panel["vmin"], vmax=panel["vmax"])

            if sz_contours:
                ax.contour(snr_sz, levels=contour_levels, colors="k", linewidths=0.5, alpha=0.5)

            if (snr_contours and self.snr
                    and panel.get("xray_label") and panel.get("quantity")):
                snr_data = self.snr[panel["xray_label"]][panel["quantity"]]
                ax.contour(convolve(snr_data, Gaussian2DKernel(1)), levels=snr_contour_levels,
                           colors="white", linewidths=0.5, linestyles="-", alpha = 0.5)

            add_r500_circle(ax, cx_peak, cy_peak, theta500)
            add_colorbar(fig, img, ax, panel["cbar_label"])
            configure_wcs_ticks(ax, size=9, show_ra=(row == nrows - 1), show_dec=(col == 0))
            if col == 0:
                ax.coords[1].set_axislabel("Dec [J2000]", size=10)
            if row == nrows - 1:
                ax.coords[0].set_axislabel("RA [J2000]", size=10)
            if panel.get("label"):
                add_label_box(ax, panel["label"])

        plt.tight_layout()
        if save_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.savefig(save_path, dpi=fig.dpi, transparent=True, bbox_inches="tight")
        return fig, axs

    def plot_snr_grid(
        self,
        sz_label: str,
        xray_labels: list[str] | None = None,
        contour_levels: list | None = None,
        snr_mask_threshold: float = -4.0,
        figsize: tuple | None = None,
        save_path: str | None = None,
    ) -> tuple:
        """Plot SNR maps in the same layout as ``plot_thermo_grid``."""
        xray_labels    = xray_labels or list(self.snr.keys())
        contour_levels = contour_levels or list(np.arange(-52, 0, 8))

        sz_entry     = self.sz_maps[sz_label]
        compton_y    = sz_entry["compton_y"]
        snr_sz       = sz_entry["snr"]
        cluster_mask = snr_sz < snr_mask_threshold
        theta500     = self._theta500()
        cy_peak, cx_peak = np.unravel_index(np.nanargmax(compton_y), compton_y.shape)

        panels, ncols = self._build_snr_panels(xray_labels, sz_label, compton_y, cluster_mask)
        nrows   = len(panels) // ncols
        figsize = figsize or (4.5 * ncols, 2.8 * nrows)

        fig, axs = make_grid_figure(nrows, ncols, self._handler.wcs_sub, figsize)

        for i, panel in enumerate(panels):
            ax  = axs[i]
            row, col = i // ncols, i % ncols

            img = ax.imshow(panel["data"], cmap=panel["cmap"], origin="lower",
                            vmin=panel["vmin"], vmax=panel["vmax"])
            ax.contour(snr_sz, levels=contour_levels, colors="k", linewidths=0.5, alpha=0.5)
            add_r500_circle(ax, cx_peak, cy_peak, theta500)
            add_colorbar(fig, img, ax, panel["cbar_label"])
            configure_wcs_ticks(ax, size=9, show_ra=(row == nrows - 1), show_dec=(col == 0))
            if col == 0:
                ax.coords[1].set_axislabel("Dec [J2000]", size=10)
            if row == nrows - 1:
                ax.coords[0].set_axislabel("RA [J2000]", size=10)
            if panel.get("label"):
                add_label_box(ax, panel["label"])

        plt.tight_layout()
        if save_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.savefig(save_path, dpi=fig.dpi, transparent=True, bbox_inches="tight")
        return fig, axs

    def _build_thermo_panels(self, xray_labels, compton_y, cluster_mask):
        def _masked(data):
            out = data * cluster_mask.astype(float)
            out[out == 0.0] = np.nan
            return out

        def _panel(quantity, data, label=None):
            _, scale, cbar_label, vmin, vmax = self._PANEL_CONFIGS[quantity]
            return dict(
                data=data, cmap=self._resolve_cmap(self._PANEL_CONFIGS[quantity][0]),
                cbar_label=cbar_label, vmin=vmin, vmax=vmax, label=label,
            )

        th0 = self.thermo[xray_labels[0]]
        P_panel = _panel("P_e", th0["P_e"].value * 1e3)
        P_panel["vmax"] = P_panel["vmax"] or np.nanmax(th0["P_e"].value * 1e3)

        _display_names = {"chandra": "Chandra", "xmm": "XMM-Newton"}

        def _instrument_panels(lbl, show_label):
            th   = self.thermo[lbl]
            name = _display_names.get(lbl, lbl) if show_label else None
            return [
                {**_panel("n_e", _masked(th["n_e"].value * 1e3), label=name),
                 "xray_label": lbl, "quantity": "n_e"},
                {**_panel("T_e", _masked(th["T_e"].value)),
                 "xray_label": lbl, "quantity": "T_e"},
                {**_panel("K_e", _masked(th["K_e"].value / 1e3)),
                 "xray_label": lbl, "quantity": "K_e"},
            ]

        if len(xray_labels) == 1:
            return [P_panel] + _instrument_panels(xray_labels[0], show_label=False), 2

        cy_panel = _panel("compton_y", compton_y * 1e4)
        if cy_panel["vmax"] is None:
            cy_panel["vmax"] = np.nanmax(compton_y * 1e4)

        left  = _instrument_panels(xray_labels[0], show_label=True)
        right = _instrument_panels(xray_labels[1], show_label=True)
        return [cy_panel, P_panel] + [p for pair in zip(left, right) for p in pair], 2

    def _build_snr_panels(self, xray_labels, sz_label, compton_y, cluster_mask):
        def _masked(data):
            out = data * cluster_mask.astype(float)
            out[out == 0.0] = np.nan
            return out

        def _panel(quantity, data, label=None):
            _, scale, cbar_label, vmin, vmax = self._PANEL_CONFIGS[f"snr_{quantity}"]
            if vmax is None:
                vmax = float(np.nanpercentile(data, 99.5))
            return dict(
                data=data, cmap=self._resolve_cmap(self._PANEL_CONFIGS[f"snr_{quantity}"][0]),
                cbar_label=cbar_label, vmin=vmin, vmax=vmax, label=label,
            )

        # P_e SNR is SZ-only — same for all instruments; use first
        snr0    = self.snr[xray_labels[0]]
        P_panel = _panel("P_e", _masked(snr0["P_e"]))

        _display_names = {"chandra": "Chandra", "xmm": "XMM-Newton"}

        def _instrument_panels(lbl, show_label):
            s    = self.snr[lbl]
            name = _display_names.get(lbl, lbl) if show_label else None
            return [
                _panel("n_e", _masked(s["n_e"]), label=name),
                _panel("T_e", _masked(s["T_e"])),
                _panel("K_e", _masked(s["K_e"])),
            ]

        if len(xray_labels) == 1:
            return [P_panel] + _instrument_panels(xray_labels[0], show_label=False), 2

        # SNR of compton_y: use the pre-computed |data/rms| map (same ratio as |y/σ_y|)
        snr_y    = _masked(np.abs(self.sz_maps[sz_label]["snr"]))
        cy_panel = _panel("compton_y", snr_y)

        left  = _instrument_panels(xray_labels[0], show_label=True)
        right = _instrument_panels(xray_labels[1], show_label=True)
        return [cy_panel, P_panel] + [p for pair in zip(left, right) for p in pair], 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _theta500(self) -> u.Quantity:
        r500 = self._r500()
        return ((r500 / cosmo.angular_diameter_distance(self.z)) * u.rad).to(u.arcsec)
