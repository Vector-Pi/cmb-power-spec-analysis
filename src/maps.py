"""
maps.py
-------
Load, inspect, and preprocess HEALPix CMB maps from Planck FITS files.
Handles temperature-only and IQU (temperature + polarisation) maps.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import healpy as hp

from .utils import fsky_from_mask, tcmb_uk_to_k


class CMBMap:
    """
    Load and preprocess a Planck HEALPix CMB map.

    Parameters
    ----------
    fits_path : str or Path
        Path to the Planck FITS map file.
    field : int
        FITS field to read:
          0 = Temperature (T)
          1 = Q polarisation
          2 = U polarisation
        Default: 0 (temperature only).
    unit : {'K', 'uK'}
        Map units.  Planck SMICA maps are in K_CMB.  Default 'K'.

    Examples
    --------
    >>> cmap = CMBMap("data/planck/COM_CMB_IQU-smica_2048_R3.00_full.fits")
    >>> cmap.load()
    >>> print(cmap.nside, cmap.npix)
    >>> cmap.apply_mask("data/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
    ...                 mask_field=4)
    >>> masked = cmap.masked_map
    """

    def __init__(
        self,
        fits_path: str | Path,
        field:     int = 0,
        unit:      str = "K",
    ) -> None:
        self.fits_path = Path(fits_path)
        self.field     = field
        self.unit      = unit

        self._map:        Optional[np.ndarray] = None
        self._mask:       Optional[np.ndarray] = None
        self._masked_map: Optional[np.ndarray] = None
        self._nside:      Optional[int]        = None
        self._fsky:       Optional[float]      = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, verbose: bool = True) -> "CMBMap":
        """
        Load the CMB map from the FITS file.

        Returns self for method chaining.
        """
        if not self.fits_path.exists():
            raise FileNotFoundError(
                f"Map file not found: {self.fits_path}\n"
                "Run PlanckDownloader.download_smica_map() first."
            )

        print(f"[CMBMap] Loading {self.fits_path.name} (field={self.field}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = hp.read_map(
                str(self.fits_path),
                field   = self.field,

            )

        # healpy uses UNSEEN = -1.6375e+30 for masked pixels
        raw[raw == hp.UNSEEN] = 0.0
        # Remove any remaining NaN/inf
        raw = np.where(np.isfinite(raw), raw, 0.0)

        self._map   = raw
        self._nside = hp.npix2nside(len(raw))

        if verbose:
            print(f"  NSIDE    = {self._nside}")
            print(f"  Npix     = {len(raw):,}")
            print(f"  T_mean   = {np.mean(raw)*1e6:.2f} uK")
            print(f"  T_rms    = {np.std(raw)*1e6:.2f} uK")
            print(f"  T_min    = {np.min(raw)*1e6:.2f} uK")
            print(f"  T_max    = {np.max(raw)*1e6:.2f} uK")

        return self

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------

    def apply_mask(
        self,
        mask_path:  str | Path,
        mask_field: int = 4,
        inplace:    bool = True,
    ) -> "CMBMap":
        """
        Apply a Galactic plane mask to the CMB map.

        Parameters
        ----------
        mask_path  : str or Path   Path to the mask FITS file.
        mask_field : int
            Field in the mask FITS file:
              0 = 60% sky,  1 = 70%,  2 = 80%,
              3 = 90%,      4 = 97%,  5 = 99%
            Default: 4 (97% sky fraction — fairly aggressive).
        inplace : bool
            If True, store the masked map in self._masked_map.

        Returns self.
        """
        if self._map is None:
            raise RuntimeError("Load the CMB map first with .load()")

        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}\n"
                "Run PlanckDownloader.download_mask() first."
            )

        print(f"[CMBMap] Applying mask (field={mask_field}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = hp.read_map(str(mask_path), field=mask_field)

        # Ensure mask is in [0, 1]
        mask = np.clip(mask, 0.0, 1.0)
        self._mask       = mask
        self._fsky       = fsky_from_mask(mask)
        self._masked_map = self._map * mask

        # Count fraction of completely masked pixels
        n_zero = np.sum(mask == 0)
        print(f"  Sky fraction (f_sky)     = {self._fsky:.4f}")
        print(f"  Fully masked pixels      = {n_zero:,} / {len(mask):,}")

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def map(self) -> np.ndarray:
        if self._map is None:
            raise RuntimeError("Map not loaded. Call .load() first.")
        return self._map

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("No mask applied. Call .apply_mask() first.")
        return self._mask

    @property
    def masked_map(self) -> np.ndarray:
        if self._masked_map is None:
            raise RuntimeError("No masked map. Call .apply_mask() first.")
        return self._masked_map

    @property
    def nside(self) -> int:
        if self._nside is None:
            raise RuntimeError("Map not loaded.")
        return self._nside

    @property
    def npix(self) -> int:
        return hp.nside2npix(self.nside)

    @property
    def lmax_recommended(self) -> int:
        return 3 * self.nside - 1

    @property
    def fsky(self) -> float:
        if self._fsky is None:
            raise RuntimeError("No mask applied. Call .apply_mask() first.")
        return self._fsky

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def map_in_uk(self, use_masked: bool = True) -> np.ndarray:
        """Return the map in microkelvin."""
        m = self.masked_map if use_masked else self.map
        if self.unit == "K":
            return m * 1e6
        return m

    def downgrade(self, nside_out: int, use_masked: bool = True) -> np.ndarray:
        """
        Downgrade the map to a lower resolution for faster processing.

        Parameters
        ----------
        nside_out  : int    Target resolution (must be < current nside).
        use_masked : bool   If True, downgrade the masked map.

        Returns
        -------
        map_low : array   Downgraded map.
        """
        m = self.masked_map if use_masked else self.map
        if nside_out >= self.nside:
            raise ValueError(f"nside_out={nside_out} must be < nside={self.nside}")
        return hp.ud_grade(m, nside_out=nside_out)

    def summary(self) -> None:
        """Print a summary of the loaded map."""
        print("\n" + "=" * 50)
        print("CMBMap Summary")
        print("=" * 50)
        print(f"  File:      {self.fits_path.name}")
        print(f"  NSIDE:     {self.nside}")
        print(f"  Npix:      {self.npix:,}")
        print(f"  lmax_rec:  {self.lmax_recommended}")
        if self._fsky is not None:
            print(f"  f_sky:     {self.fsky:.4f}")
        print("=" * 50)


def load_iqumap(
    fits_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a full IQU Planck map (temperature + polarisation).

    Parameters
    ----------
    fits_path : str or Path   Path to Planck IQU FITS file.

    Returns
    -------
    T_map : array   Temperature map (K).
    Q_map : array   Q Stokes parameter (K).
    U_map : array   U Stokes parameter (K).
    """
    fits_path = Path(fits_path)
    print(f"[load_iqumap] Loading IQU map from {fits_path.name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maps = hp.read_map(str(fits_path), field=(0, 1, 2))

    T_map, Q_map, U_map = maps
    for m in (T_map, Q_map, U_map):
        m[m == hp.UNSEEN] = 0.0
        m[:] = np.where(np.isfinite(m), m, 0.0)

    nside = hp.npix2nside(len(T_map))
    print(f"  NSIDE = {nside},  Npix = {len(T_map):,}")
    return T_map, Q_map, U_map
