"""Single place where every third-party/standard-library import lives.

Every module under ``analysis/src`` and every notebook under ``analysis/``
pulls its packages from here with::

    from utils.constants_and_packages import *

so that no notebook or analysis module carries its own import block. Add a
new package here (and to ``__all__``) rather than importing it downstream.

Notebooks bootstrap themselves with three lines -- locate the repo root,
put ``analysis/src`` on ``sys.path``, import this module -- and then call
:func:`add_src_to_path` so sibling analysis modules import by bare name::

    import sys
    from pathlib import Path

    _REPO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
    sys.path.insert(0, str(_REPO_ROOT / "analysis" / "src"))

    from utils.constants_and_packages import *
    add_src_to_path()

Credits: Aleksandar Marinkovic.
"""

# Standard library
import os
import re
import sys
import glob
import json
import math
import pickle
import shutil
import tempfile
import warnings
import itertools
from itertools import combinations, product
from datetime import datetime
from pathlib import Path

# Scientific stack
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import scipy.ndimage
from scipy import ndimage
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.stats import gaussian_kde

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath

# Imaging / video
import cv2
import caiman as cm
import tifffile as _tifffile
from caiman.motion_correction import MotionCorrect

# Notebook display
import ipywidgets as widgets
from IPython.display import display, Markdown
from tqdm import tqdm

# Directory holding the analysis modules (analysis/src)
SRC_DIR = Path(__file__).resolve().parent.parent


def add_src_to_path():
    """Put ``analysis/src`` and each of its area directories on ``sys.path``.

    Analysis modules import their siblings by bare name (e.g.
    ``import place_cell_analysis as pca``), so every area directory --
    ``behavior_preprocessing``, ``neural_analysis``, ``utils``, ... -- has to
    be importable. Notebooks call this once in their setup cell.
    """
    directories = [SRC_DIR] + sorted(
        p for p in SRC_DIR.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )
    for directory in directories:
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)


__all__ = [
    # Constants and helpers defined here
    'CONSTANTS', 'SRC_DIR', 'add_src_to_path',
    # Standard library
    'os', 're', 'sys', 'glob', 'json', 'math', 'pickle', 'shutil', 'tempfile',
    'warnings', 'itertools', 'combinations', 'product', 'datetime', 'Path',
    # Scientific stack
    'h5py', 'np', 'pd', 'scipy', 'ndimage', 'butter', 'lfilter', 'freqz',
    'filtfilt', 'gaussian_kde',
    # Plotting
    'plt', 'mcolors', 'LineCollection', 'MplPolygon', 'Rectangle', 'MplPath',
    # Imaging / video
    'cv2', 'cm', '_tifffile', 'MotionCorrect',
    # Notebook display
    'widgets', 'display', 'Markdown', 'tqdm',
]


class CONSTANTS:
    placeholder = None
