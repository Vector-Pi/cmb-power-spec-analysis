"""
utils.py
--------
Unit conversions, HEALPix utilities, and spectrum manipulation
for the CMB power spectrum pipeline.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# CMB temperature in Kelvin
T_CMB_K  = 2.7255      # K
T_CMB_uK = 2.7255e6    # microkelvin


# ---------------------------------------------------------------------------
# C_ell <-> D_ell conversion
# ---------------------------------------------------------------------------

def cl_to_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """
    Convert C_ell to D_ell = ell(ell+1) C_ell / 2pi.

    D_ell is the conventional CMB power spectrum unit — it flattens
    the scale-invariant primordial spectrum and makes acoustic peaks
    clearly visible.

    Parameters
    ----------
    ell : array   Multipole moments.
    cl  : array   Power spectrum C_ell (K^2 or uK^2).

    Returns
    -------
    dl  : array   D_ell in the same units as C_ell.
    """
    ell = np.asarray(ell, dtype=float)
    cl  = np.asarray(cl,  dtype=float)
    return ell * (ell + 1) * cl / (2 * np.pi)


def dl_to_cl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """
    Convert D_ell back to C_ell.

    Parameters
    ----------
    ell : array   Multipole moments.
    dl  : array   D_ell power spectrum.

    Returns
    -------
    cl  : array   C_ell.
    """
    ell = np.asarray(ell, dtype=float)
    dl  = np.asarray(dl,  dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(ell > 0, 2 * np.pi * dl / (ell * (ell + 1)), 0.0)
    return cl


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_spectrum(
    ell:       np.ndarray,
    dl:        np.ndarray,
    bin_width: int = 20,
    dl_err:    Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Bin the power spectrum into bands of width bin_width.

    Uses weighted mean if dl_err is provided, otherwise simple mean.

    Parameters
    ----------
    ell       : array     Multipole array (integers).
    dl        : array     D_ell power spectrum.
    bin_width : int       Width of each multipole band.
    dl_err    : array     Optional uncertainty on D_ell.

    Returns
    -------
    ell_bin : array   Bin centres.
    dl_bin  : array   Binned D_ell.
    err_bin : array or None   Binned uncertainty (None if dl_err not given).
    """
    ell_min = int(ell[0])
    ell_max = int(ell[-1])
    bins    = np.arange(ell_min, ell_max + bin_width, bin_width)

    ell_bin, dl_bin, err_bin = [], [], []

    for i in range(len(bins) - 1):
        mask = (ell >= bins[i]) & (ell < bins[i + 1])
        if mask.sum() == 0:
            continue
        ell_c = np.mean(ell[mask])
        if dl_err is not None:
            w     = 1.0 / dl_err[mask]**2
            dl_c  = np.sum(w * dl[mask]) / np.sum(w)
            err_c = 1.0 / np.sqrt(np.sum(w))
            err_bin.append(err_c)
        else:
            dl_c = np.mean(dl[mask])
        ell_bin.append(ell_c)
        dl_bin.append(dl_c)

    ell_bin = np.array(ell_bin)
    dl_bin  = np.array(dl_bin)
    err_out = np.array(err_bin) if dl_err is not None else None
    return ell_bin, dl_bin, err_out


# ---------------------------------------------------------------------------
# HEALPix utilities
# ---------------------------------------------------------------------------

def nside_to_lmax(nside: int) -> int:
    """
    Recommended maximum multipole for a given HEALPix resolution.
        lmax = 3 * nside - 1
    """
    return 3 * nside - 1


def fsky_from_mask(mask: np.ndarray) -> float:
    """
    Compute the sky fraction f_sky from a binary or apodised mask.

    For an apodised mask w(n), the effective sky fraction is:
        f_sky = <w^2>^2 / <w^4>   (for power spectrum estimation)
    For a binary mask:
        f_sky = sum(mask) / len(mask)

    This function uses the simple mean for compatibility with anafast.

    Parameters
    ----------
    mask : array   HEALPix mask array (0 = masked, 1 = unmasked).

    Returns
    -------
    fsky : float   Sky fraction in [0, 1].
    """
    return float(np.mean(mask))


def tcmb_uk_to_k(map_uk: np.ndarray) -> np.ndarray:
    """Convert a CMB temperature map from microkelvin to kelvin."""
    return map_uk * 1e-6


def tcmb_k_to_uk(map_k: np.ndarray) -> np.ndarray:
    """Convert a CMB temperature map from kelvin to microkelvin."""
    return map_k * 1e6


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------

def find_acoustic_peaks(
    ell: np.ndarray,
    dl:  np.ndarray,
    ell_min: int = 100,
    ell_max: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find acoustic peak positions and heights in the TT power spectrum.

    Parameters
    ----------
    ell     : array   Multipole moments.
    dl      : array   D_ell^TT in uK^2.
    ell_min : int     Lower multipole limit for peak search.
    ell_max : int     Upper multipole limit for peak search.

    Returns
    -------
    peak_ells    : array   Multipole positions of peaks.
    peak_heights : array   D_ell values at peaks in uK^2.
    """
    from scipy.signal import find_peaks

    mask   = (ell >= ell_min) & (ell <= ell_max)
    ell_s  = ell[mask]
    dl_s   = dl[mask]

    # Smooth slightly before peak finding
    from scipy.ndimage import uniform_filter1d
    dl_smooth = uniform_filter1d(dl_s, size=5)

    # Peak prominence relative to neighbours
    peaks, props = find_peaks(
        dl_smooth,
        distance   = 50,    # minimum ell separation between peaks
        prominence = 500,   # minimum height above surrounding troughs (uK^2)
    )

    return ell_s[peaks], dl_s[peaks]


# ---------------------------------------------------------------------------
# Chi-squared statistic
# ---------------------------------------------------------------------------

def chi_squared(
    dl_data:   np.ndarray,
    dl_theory: np.ndarray,
    dl_err:    np.ndarray,
    ell_range: Optional[Tuple[int, int]] = None,
    ell:       Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
    """
    Compute chi-squared between data and theory power spectra.

    Parameters
    ----------
    dl_data   : array   Measured D_ell.
    dl_theory : array   Theoretical D_ell (interpolated to same ell grid).
    dl_err    : array   1-sigma uncertainty on D_ell.
    ell_range : tuple   (ell_min, ell_max) — restrict to this range.
    ell       : array   Multipole array (required if ell_range is given).

    Returns
    -------
    chi2     : float   Chi-squared value.
    ndof     : int     Number of degrees of freedom.
    pte      : float   Probability-to-exceed (p-value).
    """
    from scipy.stats import chi2 as chi2_dist

    if ell_range is not None and ell is not None:
        mask      = (ell >= ell_range[0]) & (ell <= ell_range[1])
        dl_data   = dl_data[mask]
        dl_theory = dl_theory[mask]
        dl_err    = dl_err[mask]

    residuals = dl_data - dl_theory
    chi2      = float(np.sum((residuals / dl_err) ** 2))
    ndof      = len(dl_data)
    pte       = float(1.0 - chi2_dist.cdf(chi2, ndof))
    return chi2, ndof, pte
