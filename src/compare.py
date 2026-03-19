"""
compare.py
----------
Compare extracted CMB power spectra to theoretical predictions:
  - Residuals (data - theory)
  - Chi-squared goodness of fit
  - Acoustic peak position comparison
  - Cosmological parameter sensitivity
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
from scipy.interpolate import interp1d

from .utils import find_acoustic_peaks, chi_squared, bin_spectrum


class SpectrumComparison:
    """
    Compare a measured CMB power spectrum to a theoretical prediction.

    Parameters
    ----------
    ell_data   : array   Multipoles of the measured spectrum.
    dl_data    : array   Measured D_ell^TT in uK^2.
    dl_err     : array   1-sigma uncertainties on D_ell (uK^2).
    ell_theory : array   Multipoles of the theoretical spectrum.
    dl_theory  : array   Theoretical D_ell^TT in uK^2.

    Examples
    --------
    >>> comp = SpectrumComparison(ell_data, dl_data, dl_err,
    ...                           ell_theory, dl_theory)
    >>> comp.print_chisq()
    >>> comp.print_peak_comparison()
    """

    def __init__(
        self,
        ell_data:   np.ndarray,
        dl_data:    np.ndarray,
        dl_err:     np.ndarray,
        ell_theory: np.ndarray,
        dl_theory:  np.ndarray,
    ) -> None:
        self.ell_data   = np.asarray(ell_data,   dtype=float)
        self.dl_data    = np.asarray(dl_data,    dtype=float)
        self.dl_err     = np.asarray(dl_err,     dtype=float)
        self.ell_theory = np.asarray(ell_theory, dtype=float)
        self.dl_theory  = np.asarray(dl_theory,  dtype=float)

        # Interpolate theory onto the data ell grid
        self._interp = interp1d(
            ell_theory, dl_theory,
            kind        = "linear",
            bounds_error = False,
            fill_value  = np.nan,
        )
        self._dl_theory_on_data = self._interp(self.ell_data)

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (data - theory) / sigma residuals.

        Returns
        -------
        ell       : array   Multipoles.
        residuals : array   Normalised residuals (data - theory) / sigma.
        """
        diff = self.dl_data - self._dl_theory_on_data
        res  = diff / self.dl_err
        return self.ell_data, res

    def absolute_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Residuals in uK^2 (not normalised by sigma)."""
        diff = self.dl_data - self._dl_theory_on_data
        return self.ell_data, diff

    # ------------------------------------------------------------------
    # Chi-squared
    # ------------------------------------------------------------------

    def chisq(
        self,
        ell_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, int, float]:
        """
        Compute chi-squared between data and theory.

        Parameters
        ----------
        ell_range : (ell_min, ell_max), optional
            Restrict to this multipole range.

        Returns
        -------
        chi2 : float   Chi-squared value.
        ndof : int     Number of data points.
        pte  : float   Probability-to-exceed (p-value).
        """
        mask = np.isfinite(self._dl_theory_on_data)
        if ell_range:
            mask &= (self.ell_data >= ell_range[0]) & (self.ell_data <= ell_range[1])

        return chi_squared(
            dl_data   = self.dl_data[mask],
            dl_theory = self._dl_theory_on_data[mask],
            dl_err    = self.dl_err[mask],
        )

    def print_chisq(self) -> None:
        """Print chi-squared summary over standard multipole ranges."""
        ranges = [
            ("Full range",  (2,    2500)),
            ("Sachs-Wolfe", (2,    100)),
            ("1st peak",    (100,  350)),
            ("Damping tail",(800,  2000)),
        ]
        print("\n" + "=" * 55)
        print("Chi-squared Goodness of Fit")
        print("=" * 55)
        print(f"  {'Range':20s}  {'chi2':>10s}  {'ndof':>6s}  {'PTE':>8s}")
        print("-" * 55)
        for label, r in ranges:
            try:
                c2, nd, pte = self.chisq(ell_range=r)
                print(f"  {label:20s}  {c2:10.2f}  {nd:6d}  {pte:8.4f}")
            except Exception as e:
                print(f"  {label:20s}  (error: {e})")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Peak comparison
    # ------------------------------------------------------------------

    def peak_comparison(self) -> Dict[str, np.ndarray]:
        """
        Find acoustic peaks in data and theory and compare positions.

        Returns
        -------
        result : dict
            Keys: 'data_peaks', 'theory_peaks',
                  'data_heights', 'theory_heights',
                  'ell_offsets' (data - theory peak positions).
        """
        data_ells,   data_heights   = find_acoustic_peaks(
            self.ell_data, self.dl_data
        )
        theory_ells, theory_heights = find_acoustic_peaks(
            self.ell_theory, self.dl_theory
        )

        # Match peaks by proximity
        n_match = min(len(data_ells), len(theory_ells))
        offsets = data_ells[:n_match] - theory_ells[:n_match]

        return {
            "data_peaks":    data_ells,
            "theory_peaks":  theory_ells,
            "data_heights":  data_heights,
            "theory_heights": theory_heights,
            "ell_offsets":   offsets,
        }

    def print_peak_comparison(self) -> None:
        """Print acoustic peak position and height comparison."""
        result = self.peak_comparison()
        n = min(len(result["data_peaks"]), len(result["theory_peaks"]))

        print("\n" + "=" * 65)
        print("Acoustic Peak Comparison")
        print("=" * 65)
        print(f"  {'Peak':5s}  {'ell_data':>10s}  {'ell_theory':>12s}  "
              f"{'Δell':>6s}  {'D_data (uK²)':>14s}  {'D_theory (uK²)':>16s}")
        print("-" * 65)
        for i in range(n):
            print(
                f"  {i+1:5d}  "
                f"{result['data_peaks'][i]:10.1f}  "
                f"{result['theory_peaks'][i]:12.1f}  "
                f"{result['ell_offsets'][i]:+6.1f}  "
                f"{result['data_heights'][i]:14.1f}  "
                f"{result['theory_heights'][i]:16.1f}"
            )
        print("=" * 65)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def full_summary(self) -> None:
        """Print complete comparison summary."""
        self.print_chisq()
        self.print_peak_comparison()
        _, res = self.residuals()
        finite = res[np.isfinite(res)]
        print(f"\nResidual statistics:")
        print(f"  Mean normalised residual:  {np.mean(finite):.4f}")
        print(f"  Std normalised residual:   {np.std(finite):.4f}  (expect ~1.0)")
        print(f"  Max |residual|:            {np.max(np.abs(finite)):.2f} sigma")
