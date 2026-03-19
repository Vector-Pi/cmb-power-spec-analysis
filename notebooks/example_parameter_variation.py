"""
example_parameter_variation.py
-------------------------------
Visualise how the CMB TT power spectrum changes as individual
cosmological parameters are varied around the Planck 2018 best-fit.

Requires CAMB: pip install camb

This notebook is relevant to the QGET research programme — varying n_s
shows how a modified primordial spectral index (which QGET might predict)
would appear in the CMB power spectrum.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.theory import TheorySpectrum, PLANCK_2018_BESTFIT
from src.plots  import CMBPlots
from src.utils  import find_acoustic_peaks

try:
    import camb
    CAMB_OK = True
except ImportError:
    CAMB_OK = False
    print("CAMB not installed. Install with: pip install camb")
    print("Exiting.")
    sys.exit(0)

print("=" * 60)
print("CMB Parameter Sensitivity Study")
print("=" * 60)
print(f"Planck 2018 best-fit parameters:")
for k, v in PLANCK_2018_BESTFIT.items():
    print(f"  {k:8s} = {v}")

LMAX = 2000
plts = CMBPlots()

# Fiducial spectrum
print("\n[1] Computing fiducial ΛCDM spectrum ...")
th_fid   = TheorySpectrum(lmax=LMAX)
ell_fid, dl_fid = th_fid.compute_tt()

# ---------------------------------------------------------------------------
# Vary spectral index n_s
# ---------------------------------------------------------------------------
print("\n[2] Varying spectral index n_s ...")
th_ns     = TheorySpectrum(lmax=LMAX)
ns_values = np.array([0.93, 0.95, 0.9649, 0.98, 1.00])
results_ns = th_ns.vary_parameter("ns", ns_values)

plts.plot_parameter_variation(
    "ns", results_ns,
    fiducial_ell = ell_fid,
    fiducial_dl  = dl_fid,
    lmax         = LMAX,
    savepath     = "results/vary_ns.png",
)
print(f"  Saved: results/vary_ns.png")
print(f"  Note: n_s < 1 (red tilt) suppresses small-scale power.")
print(f"  QGET would predict a modified n_s if entanglement structure")
print(f"  leaves a signature in the primordial power spectrum.")

# ---------------------------------------------------------------------------
# Vary Hubble constant H0
# ---------------------------------------------------------------------------
print("\n[3] Varying Hubble constant H0 ...")
H0_values = np.array([65.0, 67.36, 70.0, 73.0])
results_H0 = th_ns.vary_parameter("H0", H0_values)

plts.plot_parameter_variation(
    "H0", results_H0,
    fiducial_ell = ell_fid,
    fiducial_dl  = dl_fid,
    lmax         = LMAX,
    savepath     = "results/vary_H0.png",
)
print(f"  Saved: results/vary_H0.png")
print(f"  H0 shifts the acoustic peak positions — higher H0 moves peaks to lower ell.")
print(f"  The Hubble tension: local measurements give H0~73, CMB gives H0~67.")

# ---------------------------------------------------------------------------
# Vary baryon density Omega_b h^2
# ---------------------------------------------------------------------------
print("\n[4] Varying baryon density Omega_b h^2 ...")
ombh2_values = np.array([0.020, 0.022, 0.02237, 0.024, 0.026])
results_omb  = th_ns.vary_parameter("ombh2", ombh2_values)

plts.plot_parameter_variation(
    "ombh2", results_omb,
    fiducial_ell = ell_fid,
    fiducial_dl  = dl_fid,
    lmax         = LMAX,
    savepath     = "results/vary_ombh2.png",
)
print(f"  Saved: results/vary_ombh2.png")
print(f"  Higher baryon density → higher odd/even peak ratio")
print(f"  (baryons enhance compression phases, suppress rarefaction phases).")

# ---------------------------------------------------------------------------
# Peak shift with n_s — relevant to QGET
# ---------------------------------------------------------------------------
print("\n[5] Acoustic peak positions vs n_s ...")
ns_fine = np.linspace(0.92, 1.02, 11)
peak1_ells, peak1_heights = [], []

for v in ns_fine:
    th_v = TheorySpectrum(params={"ns": v}, lmax=LMAX)
    ell_v, dl_v = th_v.compute_tt()
    peak_ells, peak_heights = find_acoustic_peaks(ell_v, dl_v)
    if len(peak_ells) > 0:
        peak1_ells.append(peak_ells[0])
        peak1_heights.append(peak_heights[0])

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ns_fine[:len(peak1_ells)], peak1_ells, "bo-", lw=2)
axes[0].axvline(0.9649, color="red", ls="--", lw=1.5,
                label="Planck best-fit $n_s=0.9649$")
axes[0].set_xlabel(r"Spectral index $n_s$")
axes[0].set_ylabel(r"First peak position $\ell_1$")
axes[0].set_title(r"First Acoustic Peak Position vs. $n_s$")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ns_fine[:len(peak1_heights)], peak1_heights, "ro-", lw=2)
axes[1].axvline(0.9649, color="red", ls="--", lw=1.5,
                label="Planck best-fit")
axes[1].set_xlabel(r"Spectral index $n_s$")
axes[1].set_ylabel(r"First peak height $D_{\ell_1}^{TT}$ [$\mu$K$^2$]")
axes[1].set_title(r"First Acoustic Peak Height vs. $n_s$")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.suptitle(r"CMB sensitivity to $n_s$ — relevant to QGET predictions",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("results/ns_peak_sensitivity.png", dpi=150, bbox_inches="tight")
print(f"  Saved: results/ns_peak_sensitivity.png")

print("\nDone. All figures in results/")
