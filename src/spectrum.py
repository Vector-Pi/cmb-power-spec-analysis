"""
spectrum.py
-----------
Extract CMB angular power spectra from masked HEALPix maps using
healpy.anafast, apply f_sky correction, and compute D_ell.

Supports TT, EE, BB, TE extraction from IQU maps.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import healpy as hp

from .maps  import CMBMap
from .utils import cl_to_dl, bin_spectrum, nside_to_lmax, fsky_from_mask


# ---------------------------------------------------------------------------
# PowerSpectrum
# ---------------------------------------------------------------------------

class PowerSpectrum:
    """
    Extract angular power spectra from a CMBMap.

    Parameters
    ----------
    cmap : CMBMap
        Loaded and masked CMB map.
    lmax : int, optional
        Maximum multipole.  Defaults to 3*NSIDE - 1.
    iter : int
        Number of iterations for healpy.anafast (0=fast, 3=accurate).

    Examples
    --------
    >>> ps = PowerSpectrum(cmap, lmax=1500)
    >>> cl_tt = ps.compute_tt()
    >>> dl_tt = ps.to_dl(cl_tt)
    >>> ell_bin, dl_bin, _ = ps.binned_tt(bin_width=30)
    """

    def __init__(
        self,
        cmap:  CMBMap,
        lmax:  Optional[int] = None,
        iter:  int           = 0,
    ) -> None:
        self.cmap  = cmap
        self.lmax  = lmax if lmax is not None else nside_to_lmax(cmap.nside)
        self.iter  = iter

        self._ell:    Optional[np.ndarray] = None
        self._cl_tt:  Optional[np.ndarray] = None
        self._cl_ee:  Optional[np.ndarray] = None
        self._cl_bb:  Optional[np.ndarray] = None
        self._cl_te:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # TT spectrum
    # ------------------------------------------------------------------

    def compute_tt(
        self,
        lmax:       Optional[int] = None,
        use_masked: bool          = True,
    ) -> np.ndarray:
        """
        Compute the TT angular power spectrum from the temperature map.

        Applies the pseudo-C_ell f_sky correction:
            C_ell^true ≈ C_ell^pseudo / f_sky

        Parameters
        ----------
        lmax       : int, optional   Override the lmax set at init.
        use_masked : bool            Use masked map (recommended).

        Returns
        -------
        cl_tt : array, shape (lmax+1,)
            TT power spectrum in K^2 (not yet converted to D_ell).
        """
        lmax = lmax or self.lmax
        m    = self.cmap.masked_map if use_masked else self.cmap.map

        print(f"[PowerSpectrum] Computing TT spectrum (lmax={lmax}, iter={self.iter}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl_tt = hp.anafast(m, lmax=lmax, iter=self.iter)

        # f_sky correction
        fsky      = self.cmap.fsky if use_masked else 1.0
        cl_tt    /= fsky

        self._ell   = np.arange(len(cl_tt))
        self._cl_tt = cl_tt

        print(f"  f_sky correction: {fsky:.4f}")
        print(f"  ell range: 0 – {len(cl_tt)-1}")
        return cl_tt

    # ------------------------------------------------------------------
    # EE and BB spectra (from IQU map)
    # ------------------------------------------------------------------

    def compute_polarisation(
        self,
        Q_map:     np.ndarray,
        U_map:     np.ndarray,
        mask:      Optional[np.ndarray] = None,
        lmax:      Optional[int]        = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute EE, BB, and TE power spectra from Q and U maps.

        Parameters
        ----------
        Q_map : array   Q Stokes parameter map (full sky).
        U_map : array   U Stokes parameter map (full sky).
        mask  : array   Binary or apodised mask.  If None, uses cmap.mask.
        lmax  : int     Maximum multipole.

        Returns
        -------
        cl_ee : array   EE power spectrum.
        cl_bb : array   BB power spectrum.
        cl_te : array   TE cross spectrum.
        """
        lmax = lmax or self.lmax
        if mask is None:
            mask = self.cmap.mask
        fsky = fsky_from_mask(mask)

        T_masked = self.cmap.masked_map
        Q_masked = Q_map * mask
        U_masked = U_map * mask

        print(f"[PowerSpectrum] Computing TT+EE+BB+TE spectra (lmax={lmax}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls = hp.anafast(
                [T_masked, Q_masked, U_masked],
                lmax = lmax,
                iter = self.iter,
            )
        # cls = [TT, EE, BB, TE, EB, TB]
        cl_tt, cl_ee, cl_bb, cl_te = cls[0], cls[1], cls[2], cls[3]

        # f_sky correction
        cl_tt /= fsky
        cl_ee /= fsky
        cl_bb /= fsky
        cl_te /= fsky

        self._cl_tt = cl_tt
        self._cl_ee = cl_ee
        self._cl_bb = cl_bb
        self._cl_te = cl_te
        self._ell   = np.arange(len(cl_tt))

        print(f"  f_sky correction: {fsky:.4f}")
        return cl_ee, cl_bb, cl_te

    # ------------------------------------------------------------------
    # Load official Planck spectrum (for comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def load_planck_official(
        txt_path: str,
        lmax:     int = 2500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the official Planck TT power spectrum from the text file.

        The Planck file format is:
            ell   D_ell   sigma_low   sigma_high
        in units of uK^2.

        Parameters
        ----------
        txt_path : str   Path to COM_PowerSpect_CMB-TT-full_R3.01.txt.
        lmax     : int   Maximum multipole to return.

        Returns
        -------
        ell    : array   Multipole moments.
        dl_tt  : array   D_ell^TT in uK^2.
        dl_err : array   Average 1-sigma uncertainty.
        """
        import os
        if not os.path.exists(txt_path):
            raise FileNotFoundError(
                f"Planck spectrum file not found: {txt_path}\n"
                "Run PlanckDownloader.download_tt_spectrum() first."
            )

        data = np.loadtxt(txt_path, comments="#")
        ell    = data[:, 0].astype(int)
        dl_tt  = data[:, 1]
        # Average lower and upper uncertainties
        dl_err = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))

        mask = ell <= lmax
        return ell[mask], dl_tt[mask], dl_err[mask]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dl(
        self,
        cl:   np.ndarray,
        unit: str = "uK2",
    ) -> np.ndarray:
        """
        Convert C_ell to D_ell = ell(ell+1) C_ell / 2pi.

        Parameters
        ----------
        cl   : array   C_ell array (starts at ell=0).
        unit : {'uK2', 'K2'}
            If 'uK2', multiply by (10^6)^2 to convert K^2 to uK^2.

        Returns
        -------
        dl : array   D_ell in the specified units.
        """
        ell = np.arange(len(cl))
        dl  = cl_to_dl(ell, cl)
        if unit == "uK2":
            dl *= 1e12   # K^2 → uK^2
        return dl

    @property
    def ell(self) -> np.ndarray:
        if self._ell is None:
            raise RuntimeError("Compute a spectrum first.")
        return self._ell

    @property
    def cl_tt(self) -> np.ndarray:
        if self._cl_tt is None:
            raise RuntimeError("Call compute_tt() first.")
        return self._cl_tt

    @property
    def dl_tt(self) -> np.ndarray:
        return self.to_dl(self.cl_tt)

    # ------------------------------------------------------------------
    # Binned spectra
    # ------------------------------------------------------------------

    def binned_tt(
        self,
        bin_width:  int = 20,
        unit:       str = "uK2",
        ell_min:    int = 2,
        ell_max:    Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """
        Return binned D_ell^TT.

        Parameters
        ----------
        bin_width : int   Multipole band width.
        unit      : str   'uK2' or 'K2'.
        ell_min   : int   Minimum multipole (default 2, exclude monopole/dipole).
        ell_max   : int   Maximum multipole.

        Returns
        -------
        ell_bin : array   Bin centres.
        dl_bin  : array   Binned D_ell.
        None              (no error estimate from anafast alone).
        """
        dl  = self.dl_tt
        ell = self.ell

        lmax_use = ell_max or len(ell) - 1
        mask = (ell >= ell_min) & (ell <= lmax_use)

        return bin_spectrum(ell[mask], dl[mask], bin_width=bin_width)
