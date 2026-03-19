"""
plots.py
--------
Publication-quality CMB figures: maps, power spectra, residuals,
parameter sensitivity, and peak analysis.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import healpy as hp
    _HP_OK = True
except ImportError:
    _HP_OK = False


class CMBPlots:
    """
    Collection of plotting functions for CMB analysis results.

    All methods return the matplotlib Figure/Axes for further customisation.
    Pass savepath to save automatically.
    """

    def __init__(self, figsize_default: Tuple[int, int] = (12, 6)) -> None:
        self.figsize_default = figsize_default
        plt.rcParams.update({
            "font.family":    "serif",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        })

    # ------------------------------------------------------------------
    # CMB temperature map (Mollweide)
    # ------------------------------------------------------------------

    def plot_map(
        self,
        cmb_map:   np.ndarray,
        title:     str           = "CMB Temperature Anisotropy",
        unit:      str           = r"$\mu$K",
        vmin_uk:   float         = -300.0,
        vmax_uk:   float         =  300.0,
        savepath:  Optional[str] = None,
    ) -> plt.Figure:
        """
        Mollweide projection of a CMB temperature map.

        Parameters
        ----------
        cmb_map  : array   HEALPix map in K (will be converted to uK for display).
        unit     : str     Label for the colour bar.
        vmin_uk  : float   Colour scale minimum in uK.
        vmax_uk  : float   Colour scale maximum in uK.
        """
        if not _HP_OK:
            raise ImportError("healpy required for map plotting.")

        map_uk = cmb_map * 1e6  # K -> uK

        fig = plt.figure(figsize=(14, 7))
        hp.mollview(
            map_uk,
            fig    = fig.number,
            title  = title,
            unit   = unit,
            cmap   = "RdBu_r",
            min    = vmin_uk,
            max    = vmax_uk,
            hold   = True,
        )
        hp.graticule(dpar=30, dmer=60, alpha=0.3)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # TT power spectrum
    # ------------------------------------------------------------------

    def plot_tt_spectrum(
        self,
        ell:         np.ndarray,
        dl_tt:       np.ndarray,
        dl_err:      Optional[np.ndarray] = None,
        label:       str                  = "Data",
        color:       str                  = "steelblue",
        ell_theory:  Optional[np.ndarray] = None,
        dl_theory:   Optional[np.ndarray] = None,
        theory_label: str                 = r"$\Lambda$CDM (Planck 2018)",
        lmax:        int                  = 2000,
        savepath:    Optional[str]        = None,
    ) -> plt.Figure:
        """
        Plot the CMB TT angular power spectrum D_ell^TT.

        Optionally overlays a theoretical ΛCDM prediction.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        mask = ell <= lmax

        if dl_err is not None:
            ax.errorbar(ell[mask], dl_tt[mask], yerr=dl_err[mask],
                        fmt=".", ms=2, lw=0.8, alpha=0.6,
                        color=color, label=label)
        else:
            ax.plot(ell[mask], dl_tt[mask], ".", ms=1.5,
                    alpha=0.5, color=color, label=label)

        if ell_theory is not None and dl_theory is not None:
            mask_th = ell_theory <= lmax
            ax.plot(ell_theory[mask_th], dl_theory[mask_th],
                    "k-", lw=1.5, alpha=0.85, label=theory_label)

        # Mark acoustic peak positions (approximate)
        for ell_peak, label_peak in [(220, r"$\ell_1$"), (540, r"$\ell_2$"),
                                      (810, r"$\ell_3$"), (1100, r"$\ell_4$")]:
            if ell_peak <= lmax:
                ax.axvline(ell_peak, color="gray", ls=":", lw=0.8, alpha=0.6)
                ax.text(ell_peak + 10, ax.get_ylim()[1] * 0.92, label_peak,
                        fontsize=9, color="gray")

        ax.set_xlabel(r"Multipole moment $\ell$")
        ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
        ax.set_title("CMB Temperature Angular Power Spectrum")
        ax.set_xlim(2, lmax)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.2)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def plot_residuals(
        self,
        ell:       np.ndarray,
        residuals: np.ndarray,
        title:     str           = r"Residuals: (Data $-$ Theory) / $\sigma$",
        savepath:  Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot normalised residuals (data - theory) / sigma vs. ell.
        """
        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(ell, residuals, ".", ms=2, alpha=0.5, color="steelblue")
        ax.axhline(0,    color="k",    ls="-",  lw=0.8)
        ax.axhline(+2,   color="red",  ls="--", lw=0.8, alpha=0.6, label=r"$\pm 2\sigma$")
        ax.axhline(-2,   color="red",  ls="--", lw=0.8, alpha=0.6)
        ax.fill_between(ell, -1, 1, alpha=0.1, color="green", label=r"$\pm 1\sigma$")

        ax.set_xlabel(r"Multipole moment $\ell$")
        ax.set_ylabel(r"$(D_\ell^\mathrm{data} - D_\ell^\mathrm{th}) / \sigma$")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-5, 5)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Parameter sensitivity
    # ------------------------------------------------------------------

    def plot_parameter_variation(
        self,
        param_name:    str,
        results:       list,
        fiducial_ell:  Optional[np.ndarray] = None,
        fiducial_dl:   Optional[np.ndarray] = None,
        lmax:          int                  = 2000,
        savepath:      Optional[str]        = None,
    ) -> plt.Figure:
        """
        Plot TT spectra for a range of values of one cosmological parameter.

        Parameters
        ----------
        param_name : str    Parameter name (e.g. 'ns', 'H0').
        results    : list   Output of TheorySpectrum.vary_parameter().
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        cmap    = plt.cm.coolwarm

        n = len(results)
        for i, (val, ell, dl) in enumerate(results):
            colour = cmap(i / max(n - 1, 1))
            mask   = ell <= lmax
            ax.plot(ell[mask], dl[mask], "-", lw=1.5,
                    color=colour, alpha=0.85, label=f"{param_name} = {val:.4g}")

        if fiducial_ell is not None and fiducial_dl is not None:
            mask = fiducial_ell <= lmax
            ax.plot(fiducial_ell[mask], fiducial_dl[mask],
                    "k--", lw=2, alpha=0.5, label="Planck 2018 best-fit")

        ax.set_xlabel(r"Multipole moment $\ell$")
        ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
        ax.set_title(f"CMB TT Spectrum — Varying {param_name}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(2, lmax)
        ax.set_ylim(bottom=0)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Full summary figure
    # ------------------------------------------------------------------

    def plot_full_summary(
        self,
        ell_data:    np.ndarray,
        dl_data:     np.ndarray,
        dl_err:      np.ndarray,
        ell_theory:  np.ndarray,
        dl_theory:   np.ndarray,
        residuals:   np.ndarray,
        cmb_map:     Optional[np.ndarray] = None,
        savepath:    Optional[str]        = None,
    ) -> plt.Figure:
        """
        Four-panel summary figure:
          (1) Mollweide CMB map  (if cmb_map provided)
          (2) TT power spectrum with theory
          (3) Residuals
          (4) Zoom on first acoustic peak
        """
        n_rows = 3 if cmb_map is None else 4
        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45)

        row = 0
        if cmb_map is not None and _HP_OK:
            ax_map = fig.add_subplot(gs[row])
            hp.mollview(cmb_map * 1e6, fig=fig.number,
                        title="Planck SMICA CMB Temperature (uK)",
                        cmap="RdBu_r", min=-300, max=300, hold=True)
            row += 1

        # TT spectrum
        ax_tt = fig.add_subplot(gs[row]); row += 1
        lmax = min(int(ell_data[-1]), 2000)
        mask_d = ell_data <= lmax
        mask_t = ell_theory <= lmax
        ax_tt.errorbar(ell_data[mask_d], dl_data[mask_d],
                       yerr=dl_err[mask_d],
                       fmt=".", ms=2, lw=0.6, alpha=0.5,
                       color="steelblue", label="Planck SMICA")
        ax_tt.plot(ell_theory[mask_t], dl_theory[mask_t],
                   "k-", lw=1.5, label=r"$\Lambda$CDM (CAMB)")
        ax_tt.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
        ax_tt.set_title("CMB TT Power Spectrum")
        ax_tt.legend(); ax_tt.grid(True, alpha=0.2)

        # Residuals
        ax_res = fig.add_subplot(gs[row]); row += 1
        finite = np.isfinite(residuals) & (ell_data <= lmax)
        ax_res.plot(ell_data[finite], residuals[finite],
                    ".", ms=1.5, alpha=0.4, color="steelblue")
        ax_res.axhline(0, color="k", lw=0.8)
        ax_res.fill_between(ell_data[finite], -1, 1,
                             alpha=0.1, color="green")
        ax_res.set_ylim(-4, 4)
        ax_res.set_xlabel(r"Multipole $\ell$")
        ax_res.set_ylabel(r"Residual / $\sigma$")
        ax_res.set_title("Normalised Residuals")
        ax_res.grid(True, alpha=0.2)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"[CMBPlots] Summary saved to {savepath}")
        return fig
