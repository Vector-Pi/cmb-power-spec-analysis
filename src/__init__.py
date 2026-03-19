"""
cmb-planck
==========
CMB power spectrum analysis with Planck 2018 public data.

Author: Om Arora  (https://vector-pi.github.io/omarora/)
"""

from .downloader import PlanckDownloader
from .maps       import CMBMap
from .spectrum   import PowerSpectrum
from .compare    import SpectrumComparison
from .plots      import CMBPlots
from .utils      import (
    cl_to_dl,
    dl_to_cl,
    bin_spectrum,
    nside_to_lmax,
    fsky_from_mask,
    tcmb_uk_to_k,
)

__version__ = "1.0.0"
__all__ = [
    "PlanckDownloader",
    "CMBMap",
    "PowerSpectrum",
    "SpectrumComparison",
    "CMBPlots",
    "cl_to_dl",
    "dl_to_cl",
    "bin_spectrum",
    "nside_to_lmax",
    "fsky_from_mask",
    "tcmb_uk_to_k",
]
