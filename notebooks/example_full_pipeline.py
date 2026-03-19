"""
example_full_pipeline.py
------------------------
Full CMB power spectrum pipeline on Planck 2018 SMICA data.

Steps:
  1. Download data from the Planck Legacy Archive (~250 MB total)
  2. Load the SMICA CMB map (NSIDE=2048)
  3. Apply the 80% sky Galactic mask
  4. Extract the TT power spectrum with f_sky correction
  5. Load the official Planck TT spectrum for comparison
  6. Compare to CAMB theoretical prediction (if CAMB installed)
  7. Compute residuals and chi-squared
  8. Find acoustic peak positions
  9. Generate publication-quality figures

Runtime: ~5 min (download) + ~10 min (anafast on NSIDE=2048)
         Reduce to NSIDE=512 for a 10x speedup in anafast.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.downloader import PlanckDownloader
from src.maps       import CMBMap
from src.spectrum   import PowerSpectrum
from src.theory     import TheorySpectrum, PLANCK_2018_BESTFIT
from src.compare    import SpectrumComparison
from src.plots      import CMBPlots
from src.utils      import find_acoustic_peaks, bin_spectrum

print("=" * 60)
print("CMB Power Spectrum — Planck 2018 SMICA")
print("=" * 60)

DATA_DIR = "data/planck"
LMAX     = 1500    # set lower for speed; 2000 for full analysis

# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------
print("\n[1] Checking Planck data files ...")
dl = PlanckDownloader(data_dir=DATA_DIR)
status = dl.check_files()
if not all(status.values()):
    print("  Downloading missing files ...")
    dl.download_all()

# ---------------------------------------------------------------------------
# 2. Load CMB map
# ---------------------------------------------------------------------------
print("\n[2] Loading SMICA CMB map ...")
cmap = CMBMap(dl.smica_path, field=0, unit="K")
cmap.load()

# ---------------------------------------------------------------------------
# 3. Apply mask (80% sky fraction, field=2)
# ---------------------------------------------------------------------------
print("\n[3] Applying Galactic mask (80% sky) ...")
cmap.apply_mask(dl.mask_path, mask_field=2)
cmap.summary()

# ---------------------------------------------------------------------------
# 4. Extract TT power spectrum
# ---------------------------------------------------------------------------
print(f"\n[4] Extracting TT power spectrum (lmax={LMAX}) ...")
ps    = PowerSpectrum(cmap, lmax=LMAX, iter=0)
cl_tt = ps.compute_tt()
dl_tt = ps.dl_tt   # D_ell in uK^2

ell   = ps.ell
print(f"  D_ell at ell=220 (first peak): {dl_tt[220]:.1f} uK^2  (expect ~5700)")

# Binned spectrum for cleaner plotting
ell_bin, dl_bin, _ = ps.binned_tt(bin_width=20, ell_min=2)

# ---------------------------------------------------------------------------
# 5. Load official Planck spectrum
# ---------------------------------------------------------------------------
print("\n[5] Loading official Planck TT spectrum ...")
ell_planck, dl_planck, dl_planck_err = PowerSpectrum.load_planck_official(
    str(dl.tt_spectrum_path), lmax=LMAX
)
print(f"  Loaded {len(ell_planck)} multipole bins from official spectrum.")
print(f"  First peak amplitude: {dl_planck[np.argmin(np.abs(ell_planck - 220))]:.1f} uK^2")

# ---------------------------------------------------------------------------
# 6. Theoretical spectrum
# ---------------------------------------------------------------------------
print("\n[6] Computing theoretical spectrum ...")
th = TheorySpectrum(lmax=LMAX,
                    planck_txt_path=str(dl.tt_spectrum_path))
ell_th, dl_th = th.compute_tt()

# Acoustic peaks in theory
peak_ells_th, peak_heights_th = th.peak_positions()
print(f"  Theoretical acoustic peaks:")
for i, (e, h) in enumerate(zip(peak_ells_th[:4], peak_heights_th[:4])):
    print(f"    Peak {i+1}: ell={e:.0f}, D_ell={h:.0f} uK^2")

# ---------------------------------------------------------------------------
# 7. Residuals and chi-squared
# ---------------------------------------------------------------------------
print("\n[7] Computing residuals ...")
# Use the official Planck spectrum as both data and reference
# (anafast + fsky gives slightly different values; use official for chi^2)
comp = SpectrumComparison(
    ell_data   = ell_planck,
    dl_data    = dl_planck,
    dl_err     = dl_planck_err,
    ell_theory = ell_th,
    dl_theory  = dl_th,
)
comp.full_summary()

ell_res, residuals = comp.residuals()

# ---------------------------------------------------------------------------
# 8. Peak analysis from extracted spectrum
# ---------------------------------------------------------------------------
print("\n[8] Acoustic peak analysis (extracted spectrum) ...")
# Use binned spectrum for peak finding (less noisy)
peak_ells_data, peak_heights_data = find_acoustic_peaks(
    ell_bin, dl_bin, ell_min=100, ell_max=LMAX
)
print(f"  Extracted peaks:")
for i, (e, h) in enumerate(zip(peak_ells_data[:4], peak_heights_data[:4])):
    print(f"    Peak {i+1}: ell={e:.0f}, D_ell={h:.0f} uK^2")

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
print("\n[9] Generating figures ...")
plts = CMBPlots()

# CMB map
plts.plot_map(cmap.map,
              title="Planck 2018 SMICA CMB Temperature",
              savepath="results/cmb_map.png")
plts.plot_map(cmap.masked_map,
              title="Planck SMICA — 80% Sky Mask Applied",
              savepath="results/cmb_map_masked.png")

# Extracted TT spectrum
fig_tt = plts.plot_tt_spectrum(
    ell_bin, dl_bin,
    label       = "Planck SMICA (extracted, binned Δℓ=20)",
    ell_theory  = ell_th,
    dl_theory   = dl_th,
    lmax        = LMAX,
    savepath    = "results/tt_spectrum.png",
)

# Official Planck spectrum vs theory
fig_official = plts.plot_tt_spectrum(
    ell_planck, dl_planck, dl_err=dl_planck_err,
    label       = "Official Planck TT spectrum",
    ell_theory  = ell_th,
    dl_theory   = dl_th,
    lmax        = LMAX,
    savepath    = "results/tt_spectrum_official.png",
)

# Residuals
plts.plot_residuals(ell_res, residuals,
                    savepath="results/tt_residuals.png")

# Summary figure
plts.plot_full_summary(
    ell_data   = ell_planck,
    dl_data    = dl_planck,
    dl_err     = dl_planck_err,
    ell_theory = ell_th,
    dl_theory  = dl_th,
    residuals  = residuals,
    cmb_map    = cmap.map,
    savepath   = "results/full_summary.png",
)

print("\nAll outputs saved to results/")
print("Done.")
