"""
theory.py
---------
Compute theoretical CMB power spectra using CAMB for the Planck 2018
best-fit ΛCDM parameters, and optionally vary individual parameters
to study their effect on the spectrum.

CAMB is optional — if not installed, this module provides a
fallback that loads the official Planck power spectrum instead.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np

# Planck 2018 best-fit ΛCDM parameters (Table 2, Planck 2018 Results VI)
PLANCK_2018_BESTFIT = {
    "H0":          67.36,      # km/s/Mpc
    "ombh2":       0.02237,    # Omega_b h^2
    "omch2":       0.1200,     # Omega_c h^2
    "tau":         0.0544,     # optical depth to reionisation
    "As":          2.101e-9,   # scalar amplitude (at k_pivot = 0.05 Mpc^-1)
    "ns":          0.9649,     # scalar spectral index
    "mnu":         0.06,       # sum of neutrino masses (eV)
    "nnu":         3.046,      # effective number of neutrino species
    "Alens":       1.0,        # CMB lensing amplitude
}


def _camb_available() -> bool:
    try:
        import camb
        return True
    except ImportError:
        return False


class TheorySpectrum:
    """
    Compute theoretical CMB power spectra with CAMB.

    If CAMB is not installed, falls back to loading the official
    Planck power spectrum from the downloaded text file.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters.  Defaults to Planck 2018 best-fit.
    lmax   : int
        Maximum multipole.  Default 2500.

    Examples
    --------
    >>> th = TheorySpectrum()
    >>> ell, dl_tt = th.compute_tt()

    >>> # Vary n_s to see the effect on the spectral tilt
    >>> th_ns = TheorySpectrum(params={"ns": 0.95})
    >>> ell, dl_ns = th_ns.compute_tt()
    """

    def __init__(
        self,
        params:            Optional[Dict] = None,
        lmax:              int            = 2500,
        planck_txt_path:   Optional[str]  = None,
    ) -> None:
        self.params          = {**PLANCK_2018_BESTFIT, **(params or {})}
        self.lmax            = lmax
        self.planck_txt_path = planck_txt_path
        self._camb_ok        = _camb_available()

        if not self._camb_ok:
            print("[TheorySpectrum] CAMB not installed. "
                  "Will use official Planck spectrum as theory reference.")

    # ------------------------------------------------------------------
    # CAMB computation
    # ------------------------------------------------------------------

    def _compute_with_camb(self) -> Tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
        """Run CAMB and return ell, Dl_TT, Dl_EE, Dl_TE in uK^2."""
        import camb

        p = camb.CAMBparams()
        p.set_cosmology(
            H0    = self.params["H0"],
            ombh2 = self.params["ombh2"],
            omch2 = self.params["omch2"],
            mnu   = self.params["mnu"],
            omk   = 0.0,
            tau   = self.params["tau"],
            nnu   = self.params["nnu"],
            Alens = self.params["Alens"],
        )
        p.InitPower.set_params(
            As = self.params["As"],
            ns = self.params["ns"],
            r  = 0,
        )
        p.set_for_lmax(self.lmax, lens_potential_accuracy=1)
        p.Want_CMB = True

        results = camb.get_results(p)
        powers  = results.get_cmb_power_spectra(p, CMB_unit="muK")

        # 'total' includes lensing; shape (lmax+1, 4) = [TT, EE, BB, TE]
        dl = powers["total"]
        ell = np.arange(dl.shape[0])

        return ell, dl[:, 0], dl[:, 1], dl[:, 3]  # TT, EE, TE

    # ------------------------------------------------------------------
    # Fallback: official Planck spectrum
    # ------------------------------------------------------------------

    def _load_planck_txt(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load official Planck TT spectrum as theory fallback."""
        from .spectrum import PowerSpectrum
        ell, dl_tt, dl_err = PowerSpectrum.load_planck_official(
            self.planck_txt_path, lmax=self.lmax
        )
        return ell, dl_tt, dl_err

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_tt(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical D_ell^TT.

        Returns
        -------
        ell   : array   Multipole moments (2 to lmax).
        dl_tt : array   D_ell^TT in uK^2.
        """
        if self._camb_ok:
            ell, dl_tt, _, _ = self._compute_with_camb()
            # Return from ell=2 (skip monopole and dipole)
            return ell[2:], dl_tt[2:]
        elif self.planck_txt_path:
            ell, dl_tt, _ = self._load_planck_txt()
            return ell, dl_tt
        else:
            raise RuntimeError(
                "CAMB is not installed and no planck_txt_path provided.\n"
                "Install CAMB: pip install camb\n"
                "Or provide planck_txt_path to use the official Planck spectrum."
            )

    def compute_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute TT, EE, and TE theoretical spectra.

        Returns
        -------
        spectra : dict
            Keys: 'TT', 'EE', 'TE'.
            Values: (ell, dl) tuples.
        """
        if not self._camb_ok:
            ell, dl_tt = self.compute_tt()
            return {"TT": (ell, dl_tt)}

        ell, dl_tt, dl_ee, dl_te = self._compute_with_camb()
        return {
            "TT": (ell[2:], dl_tt[2:]),
            "EE": (ell[2:], dl_ee[2:]),
            "TE": (ell[2:], dl_te[2:]),
        }

    def vary_parameter(
        self,
        param_name:  str,
        values:      np.ndarray,
    ) -> list[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Compute TT spectra for a range of values of a single parameter.

        Useful for visualising how the power spectrum changes with e.g.
        n_s, H_0, or Omega_b h^2.

        Parameters
        ----------
        param_name : str       Parameter name (must be a key in PLANCK_2018_BESTFIT).
        values     : array     Values to try.

        Returns
        -------
        results : list of (value, ell, dl_tt)
        """
        if param_name not in PLANCK_2018_BESTFIT:
            raise ValueError(
                f"Unknown parameter '{param_name}'. "
                f"Valid: {list(PLANCK_2018_BESTFIT.keys())}"
            )
        if not self._camb_ok:
            raise RuntimeError("Parameter variation requires CAMB.")

        results = []
        for v in values:
            params_v = {**self.params, param_name: v}
            th_v     = TheorySpectrum(params=params_v, lmax=self.lmax)
            ell, dl  = th_v.compute_tt()
            results.append((v, ell, dl))
            print(f"  {param_name} = {v:.4g}  →  first peak ≈ "
                  f"{dl[ell == 220][0]:.0f} uK^2" if 220 in ell else "")

        return results

    def peak_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the theoretical acoustic peak positions and heights.

        Returns
        -------
        peak_ells    : array   Multipoles of acoustic peaks.
        peak_heights : array   D_ell at each peak in uK^2.
        """
        from .utils import find_acoustic_peaks
        ell, dl_tt = self.compute_tt()
        return find_acoustic_peaks(ell, dl_tt)
