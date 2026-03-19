"""
test_cmb_pipeline.py
--------------------
Offline test suite for the CMB power spectrum pipeline.

All tests use synthetic HEALPix maps generated from known power spectra
— no Planck data download required.

Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import healpy as hp

from src.utils import (
    cl_to_dl,
    dl_to_cl,
    bin_spectrum,
    nside_to_lmax,
    fsky_from_mask,
    find_acoustic_peaks,
    chi_squared,
    tcmb_uk_to_k,
    tcmb_k_to_uk,
)
from src.maps    import CMBMap
from src.compare import SpectrumComparison


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

NSIDE_TEST = 64      # small NSIDE for fast tests
LMAX_TEST  = 3 * NSIDE_TEST - 1   # = 191


@pytest.fixture(scope="module")
def synthetic_cl():
    """
    Approximate scale-invariant CMB-like C_ell spectrum.
    Returns (ell, cl) arrays with shape (lmax+1,).
    """
    ell = np.arange(LMAX_TEST + 1, dtype=float)
    # Approximate TT spectrum: power law with acoustic peaks
    cl = np.zeros(LMAX_TEST + 1)
    cl[2:] = 1e-12 / (ell[2:] * (ell[2:] + 1))  # scale-invariant baseline
    # Add three schematic acoustic peaks
    for ell_peak, amplitude in [(50, 5e-10), (100, 3e-10), (150, 2e-10)]:
        if ell_peak <= LMAX_TEST:
            sigma = 15
            cl[2:] += amplitude * np.exp(
                -0.5 * ((ell[2:] - ell_peak) / sigma) ** 2
            ) / (ell[2:] * (ell[2:] + 1))
    return ell, cl


@pytest.fixture(scope="module")
def synthetic_map(synthetic_cl):
    """Generate a synthetic CMB map from a known C_ell."""
    ell, cl = synthetic_cl
    np.random.seed(42)
    sky_map = hp.synfast(cl, nside=NSIDE_TEST, lmax=LMAX_TEST)
    return sky_map


@pytest.fixture(scope="module")
def binary_mask():
    """Binary mask: unmask 80% of the sky (exclude Galactic belt)."""
    npix   = hp.nside2npix(NSIDE_TEST)
    mask   = np.ones(npix)
    # Mask |b| < 20 deg Galactic band
    theta, phi = hp.pix2ang(NSIDE_TEST, np.arange(npix))
    b_deg = 90.0 - np.degrees(theta)
    mask[np.abs(b_deg) < 20] = 0.0
    return mask


@pytest.fixture(scope="module")
def masked_map(synthetic_map, binary_mask):
    return synthetic_map * binary_mask


# ---------------------------------------------------------------------------
# Unit conversion tests
# ---------------------------------------------------------------------------

class TestConversions:

    def test_cl_to_dl_shape(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell, cl)
        assert dl.shape == cl.shape

    def test_dl_at_ell0_is_zero(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell, cl)
        assert dl[0] == 0.0   # ell*(ell+1) = 0 at ell=0

    def test_dl_at_ell1_is_zero(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell, cl)
        assert dl[1] == pytest.approx(2 * cl[1] / (2 * np.pi))

    def test_roundtrip_cl_dl(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl       = cl_to_dl(ell, cl)
        cl_back  = dl_to_cl(ell, dl)
        assert np.allclose(cl[2:], cl_back[2:], rtol=1e-10)

    def test_dl_positive_for_positive_cl(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell, cl)
        assert np.all(dl[2:] >= 0)

    def test_tcmb_uk_k_roundtrip(self):
        arr = np.array([1.0, 2.0, -1.0])
        assert np.allclose(tcmb_k_to_uk(tcmb_uk_to_k(arr)), arr)

    def test_nside_to_lmax(self):
        assert nside_to_lmax(64)   == 191
        assert nside_to_lmax(2048) == 6143

    def test_fsky_binary_mask_approx(self, binary_mask):
        fsky = fsky_from_mask(binary_mask)
        # We masked |b| < 20 deg → f_sky should be ~0.65-0.80
        assert 0.60 < fsky < 0.85

    def test_fsky_full_sky_is_one(self):
        mask = np.ones(hp.nside2npix(NSIDE_TEST))
        assert fsky_from_mask(mask) == pytest.approx(1.0)

    def test_fsky_empty_mask_is_zero(self):
        mask = np.zeros(hp.nside2npix(NSIDE_TEST))
        assert fsky_from_mask(mask) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Binning tests
# ---------------------------------------------------------------------------

class TestBinning:

    def test_bin_reduces_length(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell[2:], cl[2:])
        ell_bin, dl_bin, _ = bin_spectrum(ell[2:], dl, bin_width=10)
        assert len(ell_bin) < len(ell[2:])

    def test_bin_centres_in_range(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell[2:], cl[2:])
        ell_bin, _, _ = bin_spectrum(ell[2:], dl, bin_width=10)
        assert ell_bin[0] >= ell[2]
        assert ell_bin[-1] <= ell[-1]

    def test_bin_preserves_approximate_mean(self, synthetic_cl):
        """Binned mean should be close to unbinned mean over same range."""
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell[2:], cl[2:])
        _, dl_bin, _ = bin_spectrum(ell[2:], dl, bin_width=5)
        assert np.abs(np.mean(dl_bin) - np.mean(dl)) / np.mean(dl) < 0.5

    def test_bin_with_errors(self, synthetic_cl):
        ell, cl = synthetic_cl
        dl     = cl_to_dl(ell[2:], cl[2:])
        dl_err = dl * 0.1
        ell_b, dl_b, err_b = bin_spectrum(ell[2:], dl, bin_width=10,
                                           dl_err=dl_err)
        assert err_b is not None
        assert len(err_b) == len(ell_b)
        assert np.all(err_b > 0)

    def test_bin_bin_width_1_no_change(self, synthetic_cl):
        """bin_width=1 should return approximately the same array."""
        ell, cl = synthetic_cl
        dl = cl_to_dl(ell[2:], cl[2:])
        ell_b, dl_b, _ = bin_spectrum(ell[2:], dl, bin_width=1)
        # Half-open binning may drop last point; allow -1 tolerance
        assert len(ell_b) >= len(ell[2:]) - 1
        assert len(ell_b) <= len(ell[2:])


# ---------------------------------------------------------------------------
# HEALPix / map tests
# ---------------------------------------------------------------------------

class TestHEALPix:

    def test_synfast_anafast_roundtrip(self, synthetic_cl):
        """synfast then anafast should recover approximately the same C_ell."""
        ell, cl_in = synthetic_cl
        np.random.seed(0)
        sky = hp.synfast(cl_in, nside=NSIDE_TEST, lmax=LMAX_TEST)
        cl_out = hp.anafast(sky, lmax=LMAX_TEST)

        # For a single realisation, recovered C_ell ≈ input within cosmic variance
        # Only check ell >= 10 where cosmic variance is smaller
        ratio = cl_out[10:80] / (cl_in[10:80] + 1e-30)
        assert np.median(ratio) > 0.1   # at least order-of-magnitude correct

    def test_mask_reduces_power(self, synthetic_map, binary_mask):
        """Masking reduces apparent power (before f_sky correction)."""
        cl_full   = hp.anafast(synthetic_map, lmax=LMAX_TEST)
        cl_masked = hp.anafast(synthetic_map * binary_mask, lmax=LMAX_TEST)
        fsky      = fsky_from_mask(binary_mask)
        # After f_sky correction, should be closer to full-sky
        cl_corrected = cl_masked / fsky
        # Check they agree to within 50% over mid-ell range
        ratio = cl_corrected[10:80] / (cl_full[10:80] + 1e-30)
        assert np.median(np.abs(ratio - 1.0)) < 0.5

    def test_fsky_correction_improves_estimate(self, synthetic_map, binary_mask):
        """f_sky correction should reduce the mean bias."""
        cl_full   = hp.anafast(synthetic_map, lmax=LMAX_TEST)
        cl_masked = hp.anafast(synthetic_map * binary_mask, lmax=LMAX_TEST)
        fsky      = fsky_from_mask(binary_mask)

        bias_before = np.mean(np.abs(cl_masked[5:50] - cl_full[5:50]))
        bias_after  = np.mean(np.abs(cl_masked[5:50] / fsky - cl_full[5:50]))
        assert bias_after < bias_before

    def test_masked_map_has_zeros_in_masked_region(self, synthetic_map, binary_mask):
        masked = synthetic_map * binary_mask
        assert np.all(masked[binary_mask == 0] == 0.0)

    def test_nside2npix_inverse(self):
        for nside in [32, 64, 128, 256]:
            npix = hp.nside2npix(nside)
            assert hp.npix2nside(npix) == nside


# ---------------------------------------------------------------------------
# Power spectrum symmetry and physical constraints
# ---------------------------------------------------------------------------

class TestSpectrumProperties:

    def test_cl_tt_positive(self, synthetic_cl):
        ell, cl = synthetic_cl
        # C_ell should be non-negative (variance)
        assert np.all(cl[2:] >= 0)

    def test_dl_scales_as_ell_squared_for_flat_cl(self):
        """For C_ell = const, D_ell = ell(ell+1)*C/(2pi) scales as ell^2."""
        ell    = np.arange(2, 100, dtype=float)
        cl_flat = np.ones_like(ell) * 1e-10
        dl     = cl_to_dl(ell, cl_flat)
        # dl / ell^2 should be approximately constant
        ratio  = dl / ell**2
        assert np.std(ratio) / np.mean(ratio) < 0.1

    def test_power_spectrum_recovery_nside64(self, synthetic_cl):
        """Full-sky anafast should approximately recover the input spectrum."""
        ell, cl_in = synthetic_cl
        np.random.seed(1)
        sky    = hp.synfast(cl_in, nside=NSIDE_TEST, lmax=LMAX_TEST)
        cl_out = hp.anafast(sky, lmax=LMAX_TEST)

        # Check at ell >= 10 where the realisation variance is smaller
        rel_diff = np.abs(cl_out[10:60] - cl_in[10:60]) / (cl_in[10:60] + 1e-30)
        # Should agree to within a factor of a few (single realisation)
        assert np.median(rel_diff) < 5.0


# ---------------------------------------------------------------------------
# Peak finding tests
# ---------------------------------------------------------------------------

class TestPeakFinding:

    def test_finds_peaks_in_smooth_spectrum(self):
        """Simple multi-peak spectrum — find_acoustic_peaks should return peaks."""
        ell  = np.arange(2, 500, dtype=float)
        dl   = np.zeros_like(ell)
        # Inject three clear peaks
        for ell_peak in [110, 270, 420]:
            dl += 3000 * np.exp(-0.5 * ((ell - ell_peak) / 20)**2)

        peaks, heights = find_acoustic_peaks(ell, dl, ell_min=50, ell_max=500)
        assert len(peaks) >= 2, f"Expected ≥2 peaks, got {len(peaks)}"

    def test_peak_positions_near_injected(self):
        """Recovered peak positions should be close to injected ones."""
        ell = np.arange(2, 500, dtype=float)
        dl  = np.zeros_like(ell)
        injected = [120, 300]
        for ep in injected:
            dl += 5000 * np.exp(-0.5 * ((ell - ep) / 15)**2)

        peaks, _ = find_acoustic_peaks(ell, dl, ell_min=50, ell_max=500)
        for ep in injected:
            assert any(np.abs(peaks - ep) < 30), (
                f"No recovered peak near injected peak at ell={ep}. "
                f"Recovered peaks: {peaks}"
            )

    def test_no_peaks_in_flat_spectrum(self):
        """A flat (featureless) spectrum should have no significant peaks."""
        ell = np.arange(100, 500, dtype=float)
        dl  = np.ones_like(ell) * 1000.0 + np.random.default_rng(0).normal(0, 5, len(ell))
        peaks, _ = find_acoustic_peaks(ell, dl, ell_min=100, ell_max=500)
        # With prominence=500 threshold, a flat spectrum should return few/no peaks
        assert len(peaks) <= 2


# ---------------------------------------------------------------------------
# Chi-squared tests
# ---------------------------------------------------------------------------

class TestChiSquared:

    def test_chisq_perfect_fit(self):
        """Perfect fit: chi2 / ndof should be near zero."""
        dl  = np.ones(100) * 1000.0
        err = np.ones(100) * 10.0
        chi2, ndof, pte = chi_squared(dl, dl, err)
        assert chi2 == pytest.approx(0.0)
        assert ndof == 100
        assert pte  == pytest.approx(1.0)

    def test_chisq_large_residuals(self):
        """Large residuals should give large chi2 and small PTE."""
        dl_data   = np.ones(50) * 1000.0
        dl_theory = np.ones(50) * 1100.0   # 10-sigma offset
        dl_err    = np.ones(50) * 10.0
        chi2, ndof, pte = chi_squared(dl_data, dl_theory, dl_err)
        assert chi2 > ndof         # chi2 >> ndof means bad fit
        assert pte  < 0.01

    def test_chisq_unit_normal(self):
        """Random unit-normal residuals → chi2/ndof ≈ 1."""
        rng = np.random.default_rng(42)
        n   = 500
        dl_data   = rng.standard_normal(n)
        dl_theory = np.zeros(n)
        dl_err    = np.ones(n)
        chi2, ndof, pte = chi_squared(dl_data, dl_theory, dl_err)
        assert 0.7 < chi2 / ndof < 1.3, (
            f"chi2/ndof = {chi2/ndof:.3f}, expected near 1.0"
        )

    def test_chisq_ell_range(self):
        """Restricting to an ell range should reduce ndof."""
        ell  = np.arange(2, 502, dtype=float)
        dl   = np.ones(500) * 1000.0
        err  = np.ones(500) * 10.0
        _, ndof_full, _ = chi_squared(dl, dl, err)
        _, ndof_sub,  _ = chi_squared(dl, dl, err,
                                       ell_range=(100, 300), ell=ell)
        assert ndof_sub < ndof_full

    def test_chisq_positive(self):
        rng = np.random.default_rng(7)
        dl_data   = rng.uniform(900, 1100, 100)
        dl_theory = np.ones(100) * 1000.0
        dl_err    = np.ones(100) * 50.0
        chi2, _, _ = chi_squared(dl_data, dl_theory, dl_err)
        assert chi2 >= 0


# ---------------------------------------------------------------------------
# SpectrumComparison tests
# ---------------------------------------------------------------------------

class TestSpectrumComparison:

    @pytest.fixture
    def mock_comparison(self):
        """Create a SpectrumComparison with known properties."""
        rng       = np.random.default_rng(0)
        ell       = np.arange(2, 200, dtype=float)
        dl_theory = 5000 * np.exp(-((ell - 100) / 40)**2) + 2000
        dl_err    = np.ones_like(ell) * 50.0
        dl_data   = dl_theory + rng.normal(0, 50, len(ell))
        return SpectrumComparison(ell, dl_data, dl_err, ell, dl_theory)

    def test_residuals_shape(self, mock_comparison):
        ell_r, res = mock_comparison.residuals()
        assert len(ell_r) == len(res)

    def test_residuals_mean_near_zero(self, mock_comparison):
        _, res = mock_comparison.residuals()
        # With unit-normal errors, mean residual should be near zero
        assert np.abs(np.mean(res)) < 1.0

    def test_residuals_std_near_one(self, mock_comparison):
        _, res = mock_comparison.residuals()
        assert 0.5 < np.std(res) < 2.0

    def test_chisq_returns_three_values(self, mock_comparison):
        result = mock_comparison.chisq()
        assert len(result) == 3

    def test_chisq_positive(self, mock_comparison):
        chi2, _, _ = mock_comparison.chisq()
        assert chi2 >= 0

    def test_pte_in_range(self, mock_comparison):
        _, _, pte = mock_comparison.chisq()
        assert 0 <= pte <= 1

    def test_perfect_fit_gives_zero_residuals(self):
        ell = np.arange(2, 100, dtype=float)
        dl  = np.ones_like(ell) * 1000.0
        err = np.ones_like(ell) * 10.0
        comp = SpectrumComparison(ell, dl, err, ell, dl)
        _, res = comp.residuals()
        assert np.allclose(res, 0.0)

    def test_absolute_residuals_shape(self, mock_comparison):
        ell_r, diff = mock_comparison.absolute_residuals()
        assert len(ell_r) == len(diff)
