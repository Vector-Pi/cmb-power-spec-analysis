"""
downloader.py
-------------
Download Planck 2018 CMB maps and masks from the Planck Legacy Archive.
Handles resumable downloads with a progress bar and file verification.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Planck Legacy Archive URLs (Planck 2018 / Release 3)
# ---------------------------------------------------------------------------

_PLA_BASE = "https://pla.esac.esa.int/pla/aio/product-action"

PLANCK_FILES = {
    "smica_map": {
        "url":      f"{_PLA_BASE}?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "filename": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "desc":     "Planck 2018 SMICA CMB map (IQU, NSIDE=2048)",
        "size_mb":  210,
    },
    "galactic_mask": {
        "url":      f"{_PLA_BASE}?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
        "filename": "HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
        "desc":     "Planck Galactic plane masks (NSIDE=2048)",
        "size_mb":  50,
    },
    "tt_spectrum": {
        "url":      f"{_PLA_BASE}?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt",
        "filename": "COM_PowerSpect_CMB-TT-full_R3.01.txt",
        "desc":     "Official Planck TT power spectrum",
        "size_mb":  1,
    },
}


class PlanckDownloader:
    """
    Download Planck 2018 CMB data from the Planck Legacy Archive.

    Parameters
    ----------
    data_dir : str or Path
        Directory to save downloaded files.  Default: 'data/planck'.
    timeout : int
        HTTP request timeout in seconds.  Default: 120.

    Examples
    --------
    >>> dl = PlanckDownloader(data_dir="data/planck")
    >>> dl.download_smica_map()
    >>> dl.download_mask()
    >>> dl.download_tt_spectrum()
    >>> # or download everything at once:
    >>> dl.download_all()
    """

    def __init__(
        self,
        data_dir: str | Path = "data/planck",
        timeout:  int        = 120,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.timeout  = timeout
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(
        self,
        url:      str,
        filename: str,
        desc:     str,
        size_mb:  int,
    ) -> Path:
        """
        Download a single file with resumable support and progress bar.

        Returns
        -------
        path : Path   Path to the downloaded file.
        """
        out_path = self.data_dir / filename

        # Check if already fully downloaded
        if out_path.exists():
            print(f"[PlanckDownloader] Already exists: {filename}  (skipping)")
            return out_path

        print(f"[PlanckDownloader] Downloading: {desc}")
        print(f"  URL:  {url}")
        print(f"  Dest: {out_path}")
        print(f"  Size: ~{size_mb} MB")

        # Resumable: check existing partial download
        resume_header = {}
        tmp_path = out_path.with_suffix(".part")
        if tmp_path.exists():
            existing = tmp_path.stat().st_size
            resume_header = {"Range": f"bytes={existing}-"}
            print(f"  Resuming from {existing / 1e6:.1f} MB")

        try:
            response = requests.get(
                url,
                headers = resume_header,
                stream  = True,
                timeout = self.timeout,
            )
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            mode  = "ab" if resume_header else "wb"

            with open(tmp_path, mode) as f, tqdm(
                total     = total,
                initial   = tmp_path.stat().st_size if tmp_path.exists() else 0,
                unit      = "B",
                unit_scale = True,
                desc      = filename[:30],
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

            tmp_path.rename(out_path)
            print(f"  [OK] Saved to {out_path}")

        except requests.RequestException as e:
            print(f"  [ERROR] Download failed: {e}")
            print(f"  Manual download URL: {url}")
            print(f"  Save to: {out_path}")
            raise

        return out_path

    def download_smica_map(self) -> Path:
        """Download the Planck 2018 SMICA CMB temperature+polarisation map."""
        f = PLANCK_FILES["smica_map"]
        return self._download_file(f["url"], f["filename"], f["desc"], f["size_mb"])

    def download_mask(self) -> Path:
        """Download the Planck Galactic plane mask."""
        f = PLANCK_FILES["galactic_mask"]
        return self._download_file(f["url"], f["filename"], f["desc"], f["size_mb"])

    def download_tt_spectrum(self) -> Path:
        """Download the official Planck TT power spectrum."""
        f = PLANCK_FILES["tt_spectrum"]
        return self._download_file(f["url"], f["filename"], f["desc"], f["size_mb"])

    def download_all(self) -> dict[str, Path]:
        """Download all required Planck data files."""
        print("[PlanckDownloader] Downloading all Planck data files ...")
        paths = {}
        for key, meta in PLANCK_FILES.items():
            paths[key] = self._download_file(
                meta["url"], meta["filename"], meta["desc"], meta["size_mb"]
            )
        print("[PlanckDownloader] All files downloaded.")
        return paths

    def check_files(self) -> dict[str, bool]:
        """
        Check which files are already downloaded.

        Returns
        -------
        status : dict   {file_key: True if present else False}
        """
        status = {}
        for key, meta in PLANCK_FILES.items():
            path = self.data_dir / meta["filename"]
            status[key] = path.exists()
            flag = "✓" if path.exists() else "✗"
            size = f"{path.stat().st_size / 1e6:.0f} MB" if path.exists() else "missing"
            print(f"  [{flag}] {meta['filename']}  ({size})")
        return status

    @property
    def smica_path(self) -> Path:
        return self.data_dir / PLANCK_FILES["smica_map"]["filename"]

    @property
    def mask_path(self) -> Path:
        return self.data_dir / PLANCK_FILES["galactic_mask"]["filename"]

    @property
    def tt_spectrum_path(self) -> Path:
        return self.data_dir / PLANCK_FILES["tt_spectrum"]["filename"]
