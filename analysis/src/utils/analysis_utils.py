"""
Path resolution utilities for MiceHPCDrift data.

Session layout
--------------
With round:    data/Mouse{ID}/Round{N}/{track}/{session_ts}/
Without round: data/Mouse{ID}/{track}/{session_ts}/

Each session directory contains:
  caiman_final/caiman_results.hdf5   -- CaImAn CNMF output
  dlc/*_filtered.h5                  -- DLC position data
  {track}_{camera}{timestamp}.avi    -- behavior video (not miniscope)

Credits: Aleksandar Marinkovic.
"""

from utils.constants_and_packages import *

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def setup_style():
    """Apply a clean white-background plot style used throughout the project."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "axes.labelcolor":  "black",
        "text.color":       "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "grid.color":       "#cccccc",
        "grid.alpha":       0.5,
        "font.family":      "sans-serif",
        "font.size":        10,
        "figure.dpi":       120,
    })


# ---------------------------------------------------------------------------
# CaImAn HDF5 loading
# ---------------------------------------------------------------------------

def _read_dims(f, est):
    """Return the (height, width) FOV shape, tolerating a corrupted ``estimates/dims``.

    Some sessions (e.g. Mouse1637) were saved with ``estimates/dims`` holding
    a placeholder object (``b'NoneType'``) instead of the actual shape. Fall
    back to the top-level ``/dims`` dataset, then to the correlation image's
    shape, either of which reflects the same FOV.
    """
    for candidate in (est.get("dims"), f.get("dims")):
        if candidate is None:
            continue
        arr = np.array(candidate)
        if arr.dtype.kind in "iuf" and arr.size >= 2:
            return tuple(int(x) for x in arr.flat[:2])
    for key in ("Cn", "correlation_image"):
        if key in est:
            return tuple(est[key].shape)
    raise ValueError("Could not determine FOV dims from 'estimates/dims', '/dims', or 'Cn'")


def load_caiman_hdf5(path):
    """Load relevant arrays from a CaImAn HDF5 file.

    Parameters
    ----------
    path : str
        Absolute path to a ``caiman_results.hdf5`` file.

    Returns
    -------
    dict with keys:
        ``A``            -- sparse spatial footprints (scipy.sparse.csc_matrix)
        ``C``            -- fluorescence traces, shape (n_components, n_frames)
        ``S``            -- deconvolved spikes, same shape as C
        ``dims``         -- (height, width) of the imaging plane
        ``snr``          -- SNR per component (if present)
        ``cnn_preds``    -- CNN classifier scores (if present)
        ``idx_accepted`` -- list of accepted component indices
        ``idx_rejected`` -- list of rejected component indices
        ``corr_img``     -- correlation image (if present)
    """
    data = {}
    with h5py.File(path, "r") as f:
        est = f["estimates"]

        if "A/data" in f["estimates"]:
            data["A"] = scipy.sparse.csc_matrix(
                (
                    np.array(est["A/data"]),
                    np.array(est["A/indices"]),
                    np.array(est["A/indptr"]),
                ),
                shape=tuple(np.array(est["A/shape"])),
            )
        else:
            data["A"] = scipy.sparse.csc_matrix(np.array(est["A"]))

        data["C"]    = np.array(est["C"])
        data["S"]    = np.array(est["S"])
        data["dims"] = _read_dims(f, est)

        for key in ("SNR_comp", "snr", "SNR"):
            if key in est:
                data["snr"] = np.array(est[key])
                break

        for key in ("cnn_preds", "cnn"):
            if key in est:
                arr = np.array(est[key])
                if arr.size > 0:
                    data["cnn_preds"] = arr
                break

        if "idx_components" in est:
            data["idx_accepted"] = list(np.array(est["idx_components"]))
        if "idx_components_bad" in est:
            data["idx_rejected"] = list(np.array(est["idx_components_bad"]))

        for ci_key in ("Cn", "correlation_image"):
            if ci_key in f:
                data["corr_img"] = np.array(f[ci_key])
                break
            elif ci_key in est:
                data["corr_img"] = np.array(est[ci_key])
                break

    n_comp = data["C"].shape[0]
    if "idx_accepted" in data and "idx_rejected" not in data:
        data["idx_rejected"] = [i for i in range(n_comp) if i not in data["idx_accepted"]]
    if "idx_accepted" not in data:
        data["idx_accepted"] = list(range(n_comp))
        data["idx_rejected"] = []

    return data

_SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}$")

BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data"
)


def _mouse_str(mouse):
    s = str(mouse)
    return s if s.startswith("Mouse") else f"Mouse{s}"


def _round_str(round_):
    s = str(round_)
    return s if s.startswith("Round") else f"Round{s}"


def get_session_dirs(mouse, track, round=None):
    """Return sorted session directories for a mouse/track combination.

    Parameters
    ----------
    mouse : str or int
        Mouse ID — accepts 1636, "1636", or "Mouse1636".
    track : str
        Track folder name, e.g. "Linear" or "TMaze".
    round : int or str, optional
        Round number, e.g. 1 or "Round1". Omit if the mouse has no Round layer.

    Returns
    -------
    list[str]
        Sorted absolute paths to each session directory.

    Raises
    ------
    FileNotFoundError
        If the resolved track directory does not exist.
    """
    if round is not None:
        base = os.path.join(BASE_DATA_DIR, _mouse_str(mouse), _round_str(round), track)
    else:
        base = os.path.join(BASE_DATA_DIR, _mouse_str(mouse), track)

    if not os.path.isdir(base):
        raise FileNotFoundError(f"Track directory not found: {base}")

    return sorted(
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and _SESSION_RE.match(d)
    )


def get_caiman_file(mouse, track, round=None, day=-1):
    """Return CaImAn HDF5 path(s) for a mouse/track.

    Parameters
    ----------
    day : int
        0-based session index.  Pass ``-1`` (default) to return paths for
        all sessions as a list.

    Returns
    -------
    str
        Absolute path when *day* >= 0.
    list[str]
        Sorted absolute paths for all sessions when *day* == -1.

    Raises
    ------
    IndexError
        If *day* is out of range for the available sessions.
    """
    files = get_caiman_files(mouse, track, round=round)
    if day == -1:
        return files
    return files[day]


def get_caiman_files(mouse, track, round=None):
    """Return sorted CaImAn HDF5 paths, one per session.

    Looks inside each session's ``caiman_final/`` subdirectory for a
    ``.hdf5`` or ``.h5`` file.  When multiple files match, ``caiman_results.hdf5``
    is preferred.

    Returns
    -------
    list[str]
        Absolute paths to CaImAn result files.

    Raises
    ------
    FileNotFoundError
        If a session has no matching file.
    """
    files = []
    for sd in get_session_dirs(mouse, track, round=round):
        matches = []
        for subdir in ("caiman_final", "caiman"):
            caiman_dir = os.path.join(sd, subdir)
            if os.path.isdir(caiman_dir):
                matches = (
                    glob.glob(os.path.join(caiman_dir, "*.hdf5"))
                    + glob.glob(os.path.join(caiman_dir, "*.h5"))
                )
                if matches:
                    break
        if not matches:
            raise FileNotFoundError(
                f"No CaImAn HDF5 found in caiman_final/ or caiman/ under {sd}"
            )
        if len(matches) > 1:
            preferred = [m for m in matches if os.path.basename(m) == "caiman_results.hdf5"]
            matches = preferred or matches[:1]
        files.append(matches[0])
    return files


def get_registration_dir(mouse, track, round=None):
    """Return the path to the multi-session registration folder, or None if absent.

    The registration folder lives alongside session directories inside the
    track directory and matches the pattern ``caiman_final_Registration*``.

    Returns
    -------
    str or None
    """
    if round is not None:
        base = os.path.join(BASE_DATA_DIR, _mouse_str(mouse), _round_str(round), track)
    else:
        base = os.path.join(BASE_DATA_DIR, _mouse_str(mouse), track)

    if not os.path.isdir(base):
        return None

    matches = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
        and d.lower().startswith("caiman_final_registration")
    ]
    return matches[0] if matches else None


def _require_reg_dir(mouse, track, round):
    """Return registration dir or raise FileNotFoundError."""
    reg_dir = get_registration_dir(mouse, track, round=round)
    if reg_dir is None:
        raise FileNotFoundError(
            f"No registration directory found for {_mouse_str(mouse)}/{track}"
        )
    return reg_dir


def get_assignments_file(mouse, track, round=None):
    """Return the path to the neuron assignments ``.npy`` file.

    File lives in the ``caiman_final_Registration*`` folder and matches
    ``assignments*.npy``.

    Returns
    -------
    str
        Absolute path to the assignments file.

    Raises
    ------
    FileNotFoundError
    """
    reg_dir = _require_reg_dir(mouse, track, round)
    matches = glob.glob(os.path.join(reg_dir, "assignments*.npy"))
    if not matches:
        raise FileNotFoundError(f"No assignments .npy found in {reg_dir}")
    return matches[0]


def get_spatial_union_file(mouse, track, round=None):
    """Return the path to the spatial-union ``.npy`` file.

    File lives in the ``caiman_final_Registration*`` folder and matches
    ``spatial_union*.npy``.

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
    """
    reg_dir = _require_reg_dir(mouse, track, round)
    matches = glob.glob(os.path.join(reg_dir, "spatial_union*.npy"))
    if not matches:
        raise FileNotFoundError(f"No spatial_union .npy found in {reg_dir}")
    return matches[0]


def get_matching_file(mouse, track, round=None):
    """Return the path to the neuron-matching ``.pkl`` file.

    File lives in the ``caiman_final_Registration*`` folder and matches
    ``matching*.pkl``.

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
    """
    reg_dir = _require_reg_dir(mouse, track, round)
    matches = glob.glob(os.path.join(reg_dir, "matching*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No matching .pkl found in {reg_dir}")
    return matches[0]


def get_binned_activity_file(mouse, track, round=None):
    """Return the path to the multisession binned-activity ``.parquet`` file.

    File lives in the ``caiman_final_Registration*`` folder and matches
    ``*binned_activity*.parquet``.

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
    """
    reg_dir = _require_reg_dir(mouse, track, round)
    matches = glob.glob(os.path.join(reg_dir, "*binned_activity*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No binned activity .parquet found in {reg_dir}")
    return matches[0]


def load_assignments(mouse, track, round=None):
    """Load the multi-session neuron assignments matrix.

    Returns
    -------
    numpy.ndarray, shape (n_global_neurons, n_sessions)
        Each row is a tracked neuron.  Column ``t`` holds its local index in
        session ``t``, or ``NaN`` if it was not detected that session.

    Raises
    ------
    FileNotFoundError
        If no registration directory or assignments file is found.
    """
    return np.load(get_assignments_file(mouse, track, round=round), allow_pickle=True)


def get_all_mice():
    """Return sorted list of mouse IDs (int) found under BASE_DATA_DIR.

    Returns
    -------
    list[int]
    """
    if not os.path.isdir(BASE_DATA_DIR):
        return []
    return sorted(
        int(d[5:])
        for d in os.listdir(BASE_DATA_DIR)
        if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
        and re.match(r"^Mouse\d+$", d)
    )


def get_all_rounds(mouse, track):
    """Return sorted round numbers for a mouse/track combination.

    Returns a list of ints when Round subdirectories exist, or ``[None]``
    when the track sits directly under the mouse directory (round-less layout).
    Returns an empty list if the mouse directory or track is absent.

    Returns
    -------
    list[int or None]
    """
    mouse_dir = os.path.join(BASE_DATA_DIR, _mouse_str(mouse))
    if not os.path.isdir(mouse_dir):
        return []
    round_dirs = [
        d for d in os.listdir(mouse_dir)
        if os.path.isdir(os.path.join(mouse_dir, d, track))
        and re.match(r"^Round\d+$", d)
    ]
    if round_dirs:
        return sorted(int(d[5:]) for d in round_dirs)
    if os.path.isdir(os.path.join(mouse_dir, track)):
        return [None]
    return []


def get_miniscope_videos(mouse, track, round=None):
    """Return sorted miniscope AVI paths, one per session.

    Miniscope videos are identified by 'miniscope' in the filename
    (case-insensitive).  Motion-correction output files are excluded.

    Returns
    -------
    list[str]

    Raises
    ------
    FileNotFoundError
        If a session directory has no qualifying miniscope video.
    """
    videos = []
    for sd in get_session_dirs(mouse, track, round=round):
        candidates = [
            f for f in glob.glob(os.path.join(sd, "*.avi"))
            if re.search(r"miniscope", os.path.basename(f), re.IGNORECASE)
            and "motion_correction" not in os.path.basename(f).lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"No miniscope AVI found in {sd}")
        videos.append(sorted(candidates)[0])
    return videos


def get_miniscope_video(mouse, track, round=None, day=0):
    """Return a single miniscope AVI path.

    Parameters
    ----------
    day : int
        0-based session index (default 0 = first session).

    Returns
    -------
    str
    """
    return get_miniscope_videos(mouse, track, round=round)[day]


def get_video_details(video_path):
    """Return the frame count and frame rate of a video file, as reported by its container.

    Parameters
    ----------
    video_path : str

    Returns
    -------
    dict
        With keys ``frame_count`` (int) and ``fps`` (float).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"frame_count": frame_count, "fps": fps}


def get_behavior_videos(mouse, track, round=None):
    """Return sorted behavior video paths, one per session.

    The behavior video is the ``.avi`` in the session directory whose name does
    NOT contain ``miniscope`` or ``motion_correction`` (case-insensitive).

    Returns
    -------
    list[str]
        Absolute paths to behavior ``.avi`` files.

    Raises
    ------
    FileNotFoundError
        If a session directory has no qualifying video file.
    """
    videos = []
    for sd in get_session_dirs(mouse, track, round=round):
        candidates = [
            f for f in glob.glob(os.path.join(sd, "*.avi"))
            if "miniscope" not in os.path.basename(f).lower()
            and "motion_correction" not in os.path.basename(f).lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"No behavior video found in {sd}")
        videos.append(sorted(candidates)[0])
    return videos


def get_dlc_csv_files(mouse, track, round=None):
    """Return sorted DLC filtered CSV paths, one per session.

    Looks inside each session's ``dlc/`` subdirectory for a file matching
    ``*_filtered.csv``.

    Parameters
    ----------
    mouse : str or int
    track : str
    round : int or str, optional

    Returns
    -------
    list[str]
        Absolute paths to DLC filtered CSV files.

    Raises
    ------
    FileNotFoundError
        If a session's ``dlc/`` directory is missing or contains no filtered CSV.
    """
    files = []
    for sd in get_session_dirs(mouse, track, round=round):
        dlc_dir = os.path.join(sd, "dlc")
        if not os.path.isdir(dlc_dir):
            raise FileNotFoundError(f"No dlc/ subdirectory in {sd}")
        matches = glob.glob(os.path.join(dlc_dir, "*_filtered.csv"))
        if not matches:
            raise FileNotFoundError(f"No *_filtered.csv DLC file found in {dlc_dir}")
        files.append(sorted(matches)[0])
    return files


def get_dlc_files(mouse, track, round=None):
    """Return sorted DLC filtered H5 paths, one per session.

    Looks inside each session's ``dlc/`` subdirectory for a file matching
    ``*_filtered.h5``.

    Parameters
    ----------
    mouse : str or int
    track : str
    round : int or str, optional

    Returns
    -------
    list[str]
        Absolute paths to DLC filtered H5 files.

    Raises
    ------
    FileNotFoundError
        If a session's ``dlc/`` directory is missing or contains no filtered H5.
    """
    files = []
    for sd in get_session_dirs(mouse, track, round=round):
        dlc_dir = os.path.join(sd, "dlc")
        if not os.path.isdir(dlc_dir):
            raise FileNotFoundError(f"No dlc/ subdirectory in {sd}")
        matches = glob.glob(os.path.join(dlc_dir, "*_filtered.h5"))
        if not matches:
            raise FileNotFoundError(f"No *_filtered.h5 DLC file found in {dlc_dir}")
        files.append(sorted(matches)[0])
    return files


def get_dlc_160000_files(mouse, track, round=None):
    """Return sorted DLC filtered H5 paths from the 160000-iteration model.

    Same as :func:`get_dlc_files` but restricted to files matching
    ``*_160000_filtered.h5`` (the higher-iteration DLC model used for some
    mice), rather than any ``*_filtered.h5``.

    Parameters
    ----------
    mouse : str or int
    track : str
    round : int or str, optional

    Returns
    -------
    list[str]
        Absolute paths to ``*_160000_filtered.h5`` DLC files.

    Raises
    ------
    FileNotFoundError
        If a session's ``dlc/`` directory is missing or contains no matching H5.
    """
    files = []
    for sd in get_session_dirs(mouse, track, round=round):
        dlc_dir = os.path.join(sd, "dlc")
        if not os.path.isdir(dlc_dir):
            raise FileNotFoundError(f"No dlc/ subdirectory in {sd}")
        matches = glob.glob(os.path.join(dlc_dir, "*_160000_filtered.h5"))
        if not matches:
            raise FileNotFoundError(f"No *_160000_filtered.h5 DLC file found in {dlc_dir}")
        files.append(sorted(matches)[0])
    return files


def get_events_csv_files(mouse, track, round=None):
    """Return sorted trial-events CSV paths, one per session.

    Derives each path from the corresponding ``*_160000_filtered.h5`` DLC
    file the same way ``behavior_event_creation.create_events_csv`` names
    its output (``.h5`` suffix replaced with ``_events.csv``). Does not run
    event detection itself -- raises if the CSV hasn't been generated yet
    via ``behavior_event_creation.create_events_csv``.

    Parameters
    ----------
    mouse : str or int
    track : str
    round : int or str, optional

    Returns
    -------
    list[str]
        Absolute paths to ``*_160000_filtered_events.csv`` files.

    Raises
    ------
    FileNotFoundError
        If a session's ``*_160000_filtered.h5`` has no matching events CSV.
    """
    files = []
    for dlc_path in get_dlc_160000_files(mouse, track, round=round):
        out_path = dlc_path[:-len(".h5")] + "_events.csv"
        if not os.path.exists(out_path):
            raise FileNotFoundError(
                f"No events CSV found for {os.path.basename(dlc_path)}. "
                f"Run behavior_event_creation.create_events_csv() first."
            )
        files.append(out_path)
    return files


_DLC_TRACK_ALIASES = {
    "linear":      "Linear",
    "lineartrack": "Linear",
    "tmaze":       "TMaze",
}


def _dlc_track_str(track):
    return _DLC_TRACK_ALIASES.get(track.lower(), track)


def get_dlc_trials_csv_files(mouse, track, round=None):
    """Return sorted DLC trial-summary CSV paths, one per session.

    Looks inside each session's ``dlc/`` subdirectory for a file matching
    ``*_trials.csv`` (per-trial start/end frame, duration, and direction,
    derived from DLC tracking rather than Bpod).

    Parameters
    ----------
    mouse : str or int
    track : str
        Track name or shorthand (e.g. ``"linear"`` / ``"LinearTrack"`` / ``"Linear"``).
    round : int or str, optional

    Returns
    -------
    list[str]
        Absolute paths to DLC trials CSV files.

    Raises
    ------
    FileNotFoundError
        If a session's ``dlc/`` directory is missing or contains no trials CSV.
    """
    files = []
    for sd in get_session_dirs(mouse, _dlc_track_str(track), round=round):
        dlc_dir = os.path.join(sd, "dlc")
        if not os.path.isdir(dlc_dir):
            raise FileNotFoundError(f"No dlc/ subdirectory in {sd}")
        matches = glob.glob(os.path.join(dlc_dir, "*_trials.csv"))
        if not matches:
            raise FileNotFoundError(f"No *_trials.csv DLC file found in {dlc_dir}")
        files.append(sorted(matches)[0])
    return files


def load_dlc_csv(csv_path):
    """Load a DLC filtered CSV and return x, y, and likelihood per bodypart.

    Parameters
    ----------
    csv_path : str
        Absolute path to a ``*_filtered.csv`` DLC output file.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Maps bodypart name -> dict with keys ``"x"``, ``"y"``, ``"likelihood"``,
        each a 1-D float array of per-frame values.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df.columns.get_level_values(1).unique()
    return {
        bp: {
            coord: df[scorer, bp, coord].to_numpy(dtype=float)
            for coord in ("x", "y", "likelihood")
        }
        for bp in bodyparts
    }

# ---------------------------------------------------------------------------
# Bpod behavior .mat files
# ---------------------------------------------------------------------------

BPOD_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "Bpod Local", "Data"
)

_BPOD_TRACK_ALIASES = {
    "linear":      "LinearTrack",
    "lineartrack": "LinearTrack",
    "tmaze":       "TMaze",
}


def _bpod_track_str(track):
    return _BPOD_TRACK_ALIASES.get(track.lower(), track)


def _resolve_bpod_subject_dir(mouse):
    """Return the Bpod subject directory, trying Mouse{ID} then bare {ID}."""
    s = str(mouse)
    candidates = []
    if not s.startswith("Mouse"):
        candidates.append(os.path.join(BPOD_DATA_DIR, f"Mouse{s}"))
    candidates.append(os.path.join(BPOD_DATA_DIR, s))
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"No Bpod subject directory found for mouse '{mouse}' under {BPOD_DATA_DIR}"
    )


def get_bpod_mat_files(mouse, track, day=-1):
    """Return sorted Bpod behavior .mat paths for a mouse/track combination.

    Parameters
    ----------
    mouse : str or int
        Mouse ID — accepts 1637, "1637", or "Mouse1637".
    track : str
        Track name or shorthand.  Common values:

        * ``"linear"`` / ``"LinearTrack"``
        * ``"tmaze"``  / ``"TMaze"`` / ``"TMaze2"`` / ``"TMaze4"``
    day : int
        0-based session index.  Pass ``-1`` (default) to return paths for
        **all** sessions as a list.

    Returns
    -------
    str
        Absolute path when *day* >= 0.
    list[str]
        Sorted absolute paths for all sessions when *day* == -1.

    Raises
    ------
    IndexError
        If *day* is out of range for the available sessions.
    FileNotFoundError
        If the subject or Session Data directory is absent.
    """
    subject_dir = _resolve_bpod_subject_dir(mouse)
    track_name = _bpod_track_str(track)
    session_data_dir = os.path.join(subject_dir, track_name, "Session Data")
    if not os.path.isdir(session_data_dir):
        raise FileNotFoundError(
            f"No Bpod Session Data directory found: {session_data_dir}"
        )
    files = sorted(glob.glob(os.path.join(session_data_dir, "*.mat")))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {session_data_dir}")
    
    if day == -1:
        return files
    return files[day]


def _load_mat_v5(path):
    """Load a MATLAB v5 .mat file and return the SessionData struct."""
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return data["SessionData"]


def _h5_to_python(item, f):
    """Recursively convert an h5py Group/Dataset to Python dicts and NumPy arrays."""
    if isinstance(item, h5py.Dataset):
        data = item[()]
        # Object arrays hold HDF5 references (MATLAB cell arrays)
        if data.dtype.kind == 'O':
            resolved = [_h5_to_python(f[ref], f) for ref in data.flat]
            return resolved[0] if data.size == 1 else resolved
        # uint16 datasets with MATLAB_class='char' are strings
        cls = item.attrs.get("MATLAB_class", b"")
        if isinstance(cls, bytes):
            cls = cls.decode()
        if cls == "char":
            return "".join(chr(c) for c in data.flat)
        return data.squeeze() if data.ndim > 0 else data[()]
    if isinstance(item, h5py.Group):
        return {key: _h5_to_python(item[key], f) for key in item.keys()}
    return item


def load_bpod_mat(path):
    """Load a Bpod ``.mat`` (MATLAB v7.3 / HDF5) file via h5py.

    Parameters
    ----------
    path : str
        Absolute path to a Bpod session ``.mat`` file.

    Returns
    -------
    dict
        The ``SessionData`` mapping when present; otherwise the full
        top-level dict of the file.
    """
    with h5py.File(path, "r") as f:
        if "SessionData" in f:
            return _h5_to_python(f["SessionData"], f)
        return _h5_to_python(f, f)


def load_bpod_session(mouse, track, day=-1):
    """Load Bpod behavior data for a mouse/track session.

    Convenience wrapper that calls :func:`get_bpod_mat_file` then
    :func:`load_bpod_mat`.

    Parameters
    ----------
    mouse : str or int
        Mouse ID — accepts 1637, "1637", or "Mouse1637".
    track : str
        Track name or shorthand (see :func:`get_bpod_mat_files`).
    day : int
        0-based session index, or ``-1`` (default) to load **all** sessions.

    Returns
    -------
    dict
        ``SessionData`` for the requested session when *day* >= 0.
    list[dict]
        One ``SessionData`` dict per session when *day* == -1.
    """
    if day == -1:
        return [load_bpod_mat(p) for p in get_bpod_mat_files(mouse, track)]
    return load_bpod_mat(get_bpod_mat_files(mouse, track, day=day))


def get_all_limbs(mouse, track, round=None, day=0):
    """Return the bodypart names tracked by DLC for a given animal.

    Reads the DLC filtered H5 for one session and extracts bodypart names
    from the MultiIndex column headers (scorer, bodyparts, coords).

    Parameters
    ----------
    mouse : str or int
    track : str
    round : int or str, optional
    day : int
        0-based session index to read bodypart names from (default 0).
        All sessions share the same bodypart set, so the default is fine.

    Returns
    -------
    list[str]
        Bodypart names as defined in the DLC project (e.g. ["bodypart1",
        "bodypart2", "bodypart3", "objectA"]).

    Raises
    ------
    FileNotFoundError
        If no DLC file is found for the specified session.
    """
    files = get_dlc_files(mouse, track, round=round)
    df = pd.read_hdf(files[day])
    return list(df.columns.get_level_values(1).unique())