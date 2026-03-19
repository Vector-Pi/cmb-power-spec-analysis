# CMB Power Spectrum Analysis with Planck Public Data

A Python pipeline for downloading, processing, and analysing Cosmic Microwave Background (CMB) temperature and polarisation maps from the Planck 2018 public data release, extracting angular power spectra, and comparing to theoretical $\Lambda$CDM predictions.



---

## Scientific Background

The Cosmic Microwave Background is the thermal radiation left from recombination at $z \approx 1100$, observed today at $T = 2.725$ K. Its temperature anisotropies $\delta T / T \sim 10^{-5}$ are decomposed in spherical harmonics:

$$\frac{\delta T}{T}(\hat{n}) = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat{n})$$

The angular power spectrum $C_\ell = \langle |a_{\ell m}|^2 \rangle$ encodes all cosmological information. The characteristic acoustic peaks — at $\ell \approx 220$, $540$, $800$ — constrain the geometry, baryon density, dark matter density, and primordial power spectrum of the universe.

This pipeline:
1. Downloads the Planck 2018 SMICA CMB map and Galactic mask from the Planck Legacy Archive
2. Applies the mask to remove Galactic foreground contamination
3. Extracts the $TT$, $EE$, and $BB$ power spectra using `healpy.anafast`
4. Corrects for the sky fraction ($f_\text{sky}$ correction)
5. Compares to CAMB theoretical predictions for the best-fit $\Lambda$CDM parameters
6. Computes residuals and $\chi^2$ goodness of fit

### Key references
- Planck Collaboration 2018, Results V — CMB power spectra: [arXiv:1907.12875](https://arxiv.org/abs/1907.12875)
- Planck Collaboration 2018, Results I — Overview: [arXiv:1807.06205](https://arxiv.org/abs/1807.06205)
- Lewis & Bridle (2002) — CAMB: [arXiv:astro-ph/0205436](https://arxiv.org/abs/astro-ph/0205436)

---

## What This Does

| Module | Purpose |
|--------|---------|
| `downloader.py` | Download Planck maps and masks from the Planck Legacy Archive |
| `maps.py` | Load, inspect, and preprocess HEALPix CMB maps |
| `spectrum.py` | Extract $C_\ell$ from masked maps; $f_\text{sky}$ correction; binning |
| `theory.py` | Compute theoretical $C_\ell$ with CAMB; parameter grid |
| `compare.py` | Residuals, $\chi^2$, peak finding, cosmological parameter constraints |
| `plots.py` | Publication-quality figures: maps, power spectra, residuals |
| `utils.py` | Unit conversions, HEALPix utilities, $D_\ell$ conversion |

---

## Installation

```bash
git clone https://github.com/Vector-Pi/cmb-planck.git
cd cmb-planck
pip install -e .
```

Requires Python ≥ 3.9.

**Note on CAMB:** CAMB requires a Fortran compiler. On macOS:
```bash
brew install gcc
pip install camb
```
On Linux: `apt install gfortran && pip install camb`

If CAMB is unavailable, the pipeline runs in data-only mode (no theoretical comparison).

---

## Quick Start

```python
from src.downloader import PlanckDownloader
from src.maps       import CMBMap
from src.spectrum   import PowerSpectrum
from src.plots      import CMBPlots

# 1. Download data (one-time, ~500 MB)
dl = PlanckDownloader(data_dir="data/planck")
dl.download_smica_map()
dl.download_mask()

# 2. Load and mask the CMB map
cmap = CMBMap("data/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits")
cmap.load()
cmap.apply_mask("data/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
                field=4)   # 80% sky fraction

# 3. Extract power spectrum
ps = PowerSpectrum(cmap)
cl_tt = ps.compute_tt(lmax=2000)
dl_tt = ps.to_dl(cl_tt)

# 4. Plot
plts = CMBPlots()
plts.plot_tt_spectrum(dl_tt, label="Planck SMICA")
```

---

## Data

All Planck data is public and downloaded from the Planck Legacy Archive ([pla.esac.esa.int](https://pla.esac.esa.int)).

| File | Size | Description |
|------|------|-------------|
| `COM_CMB_IQU-smica_2048_R3.00_full.fits` | ~200 MB | SMICA CMB temperature + polarisation map, NSIDE=2048 |
| `HFI_Mask_GalPlane-apo0_2048_R2.00.fits` | ~50 MB | Galactic plane masks (60%, 70%, 80%, 90% sky fractions) |
| `COM_PowerSpect_CMB-TT-full_R3.01.txt` | ~1 MB | Official Planck TT power spectrum (for comparison) |

Files are downloaded automatically by `PlanckDownloader`. If the automatic download fails (the PLA occasionally times out), direct download links are included in `data/planck_urls.txt`.

---

## Output

Running the full pipeline produces:

1. **Mollweide CMB map** — temperature anisotropy map in Galactic coordinates
2. **Masked CMB map** — with Galactic plane and point sources removed
3. **TT power spectrum** — $D_\ell^{TT}$ extracted from Planck SMICA data
4. **EE power spectrum** — $D_\ell^{EE}$ (polarisation E-modes)
5. **Theory comparison** — data vs. CAMB $\Lambda$CDM best-fit
6. **Residuals plot** — $(D_\ell^\text{data} - D_\ell^\text{theory}) / \sigma_\ell$
7. **Peak analysis** — acoustic peak positions and heights
8. **$\chi^2$ summary** — goodness of fit per multipole range

---

## Tests

```bash
pytest tests/ -v
```

All tests run offline on synthetic HEALPix maps — no Planck data download required. Covers HEALPix operations, $f_\text{sky}$ correction, $D_\ell$ conversion, power spectrum symmetry, binning, peak finding, and the $\chi^2$ statistic.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `healpy ≥1.16` | HEALPix map operations and power spectrum extraction |
| `numpy ≥1.24` | Array operations |
| `scipy ≥1.11` | Signal processing, peak finding, statistics |
| `matplotlib ≥3.7` | Plotting |
| `astropy ≥5.3` | FITS I/O, units |
| `requests ≥2.28` | HTTP download from Planck Legacy Archive |
| `tqdm ≥4.65` | Download progress bars |
| `camb ≥1.5` | Theoretical $C_\ell$ (optional but recommended) |

---

## Connection to QGET

This pipeline was developed partly to build intuition for the CMB data in support of the [QGET framework](https://omarora.netlify.app/theoretical-physics/my-research-publications/) for emergent spacetime. QGET predicts that entanglement structure at the origin of spacetime may leave signatures in the primordial power spectrum — specifically a modified spectral tilt $n_s$ or non-Gaussianity in the temperature anisotropies. The `theory.py` module supports varying $n_s$ and $A_s$ to compare to the data, which is the first step toward constraining QGET predictions from CMB observations.

---

## References

- Planck 2018 Results V (power spectra): [arXiv:1907.12875](https://arxiv.org/abs/1907.12875)
- Planck 2018 Results I (overview): [arXiv:1807.06205](https://arxiv.org/abs/1807.06205)
- CAMB: [camb.readthedocs.io](https://camb.readthedocs.io)
- Wayne Hu's CMB tutorial: [background.uchicago.edu](http://background.uchicago.edu)

---

## License

MIT. See `LICENSE`.
