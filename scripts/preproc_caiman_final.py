import cv2
import logging
import numpy as np
import os
import psutil
import re
import datetime
from pathlib import Path
from argparse import ArgumentParser

try:
    cv2.setNumThreads(0)
except ():
    pass

# Store memmaps on Seagate2 — root disk is nearly full (~70 GB free); Seagate2 has 1.1 TB free
os.environ['CAIMAN_DATA'] = "/media/toor/Seagate2/caiman_data"

# --- GPU setup and fixed random seed ---
import cupy as cp
np.random.seed(42)
cp.random.seed(42)
print("CuPy device:", cp.zeros((1,)).device)   # should print <CUDA Device 0>

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params
from pynwb import NWBHDF5IO
from pynwb.ophys import DfOverF

# Aging cohort — larger somas than the young cohort (different lens), so CNMF-E needs a
# bigger gSig/gSig_filt. Confirmed via parameter sweep on all 4 mice (2026-08-23):
# gSig=7/gSig_filt=10 consistently beat gSig=5/8 (young default) and gSig=9+ (overshoots,
# noisier background, more false-positive components) on visual contour-to-soma fit and
# rejected-component rate. nb and min_pnr showed no benefit from changing and are shared
# across cohorts.
OLD_MICE_IDS = {"163", "1636", "1637", "1639"}
OLD_MICE_GSIG = [7, 7]
OLD_MICE_GSIG_FILT = [10, 10]
YOUNG_MICE_GSIG = [5, 5]
YOUNG_MICE_GSIG_FILT = [8, 8]


def _match_mouse_id(video_path, ids):
    """Match "Mouse<id>" as a whole ID (no trailing digit) in a path string — a plain
    substring check would let "163" wrongly match Mouse1636/1637/1639 paths, which all
    contain "163" as a prefix."""
    return next((k for k in ids if re.search(rf"Mouse{re.escape(k)}(?!\d)", str(video_path))), None)


def parse_args():
    parser = ArgumentParser(description="Parse arguments for motion correction and source extraction")

    # general dataset-dependent parameters
    parser.add_argument("--fr", type=int, default=25, help="Imaging rate in frames per second")
    parser.add_argument("--decay_time", type=float, default=0.56, help="Length of a typical transient in seconds")
    parser.add_argument("--dxy", type=float, nargs=2, default=[0.83, 0.83], help="Spatial resolution in x and y in (um per pixel)")

    # motion correction parameters
    parser.add_argument("--strides", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--overlaps", type=int, nargs=2, default=[32, 32])
    parser.add_argument("--max_shifts", type=int, nargs=2, default=[25, 25])
    parser.add_argument("--max_deviation_rigid", type=int, default=8)
    parser.add_argument("--gSig_filt", type=float, nargs=2, default=None,
                         help=f"Gaussian filter size for motion correction. Defaults to "
                              f"{OLD_MICE_GSIG_FILT} for the aging cohort (Mouse{{{','.join(sorted(OLD_MICE_IDS))}}}), "
                              f"{YOUNG_MICE_GSIG_FILT} otherwise.")
    parser.add_argument("--pw_rigid", type=bool, default=True)

    # CNMF parameters
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--gSig", type=float, nargs=2, default=None,
                         help=f"CNMF-E spatial footprint size. Defaults to {OLD_MICE_GSIG} for "
                              f"the aging cohort (Mouse{{{','.join(sorted(OLD_MICE_IDS))}}}), "
                              f"{YOUNG_MICE_GSIG} otherwise.")
    parser.add_argument("--merge_thr", type=float, default=0.65)
    parser.add_argument("--rf", type=int, default=32)
    parser.add_argument("--stride_cnmf", type=int, default=16)
    parser.add_argument("--ssub", type=int, default=1)
    parser.add_argument("--tsub", type=int, default=1)
    parser.add_argument("--gnb", type=int, default=0)
    parser.add_argument("--min_corr", type=float, default=0.8)
    parser.add_argument("--min_pnr", type=float, default=6.5)
    parser.add_argument("--ssub_B", type=int, default=2)

    # component evaluation parameters
    parser.add_argument("--min_SNR", type=float, default=2.5)
    parser.add_argument("--rval_thr", type=float, default=0.75)

    # script-specific parameters
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="caiman_final",
                         help="Output subdirectory name under the session folder (for parameter sweeps)")
    parser.add_argument("--log_severity", type=str, default="WARNING")
    parser.add_argument("--use_log_file", action="store_true")
    parser.add_argument("--delete_logs", action="store_false")
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--save_nwb", action="store_true")
    parser.add_argument("--delete_memmaps", action="store_false")
    parser.add_argument("--save_comparison", action="store_true")

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


def package_arguments_to_dict(args, video_path: Path):
    is_old_mouse = _match_mouse_id(video_path, OLD_MICE_IDS) is not None
    gSig = args.gSig if args.gSig is not None else (OLD_MICE_GSIG if is_old_mouse else YOUNG_MICE_GSIG)
    gSig_filt = args.gSig_filt if args.gSig_filt is not None else (OLD_MICE_GSIG_FILT if is_old_mouse else YOUNG_MICE_GSIG_FILT)
    print(f"Resolved gSig={gSig} gSig_filt={gSig_filt} (aging cohort: {is_old_mouse})")

    parameter_dict = {
        "fnames": str(video_path),
        "fr": args.fr,
        "dxy": args.dxy,
        "decay_time": args.decay_time,
        "strides": args.strides,
        "overlaps": args.overlaps,
        "max_shifts": args.max_shifts,
        "max_deviation_rigid": args.max_deviation_rigid,
        "gSig_filt": [int(x) for x in gSig_filt],  # cast to int: newer cv2/numpy reject float ksize/indices
        "pw_rigid": args.pw_rigid,
        "nonneg_movie": False,
        "p": args.p,
        "nb": args.gnb,
        "min_corr": args.min_corr,
        "min_pnr": args.min_pnr,
        "ssub_B": args.ssub_B,
        "rf": args.rf,
        "gSig": np.array(gSig, dtype=int),  # cast to int: newer cv2/numpy reject float ksize/indices
        "gSiz": 2 * np.array(gSig, dtype=int) + 1,
        "stride": args.stride_cnmf,
        "ssub": args.ssub,
        "tsub": 1,  
        "merge_thr": args.merge_thr,
        "min_SNR": 2.5,
        "rval_thr": 0.75,

        # CNMF-E setup
        "nb_patch": 0,
        "K": None,
        "method_init": "corr_pnr",
        "center_psf": True,
        "only_init": True,
        "ring_size_factor": 1.5,
        "use_cnn": True,  # enable CNN classifier for refinement
        "border_pix": 8, 

        # GPU usage
        "use_cuda": False,
    }

    return params.CNMFParams(params_dict=parameter_dict)


def get_params():
    args = parse_args()
    input_path = Path(args.input_path)
    cnmf_params = package_arguments_to_dict(args, input_path)

    cnmf_params.change_params({
        'quality': {
            'min_SNR': 2.5,
            'SNR_lowest': 2.0,       
            'rval_thr': 0.8,
            'rval_lowest': 0.65,      
            'use_cnn': True,         
            'min_cnn_thr': 0.9,      
            'cnn_lowest': 0.4
        }
    })
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_severity = log_levels.get(args.log_severity.upper(), logging.WARNING)
    return (
        cnmf_params,
        input_path,
        log_severity,
        args.use_log_file,
        args.delete_logs,
        args.synchronous,
        args.delete_memmaps,
        args.save_comparison,
        args.run_name,
    )




def setup(use_log_file: bool, log_severity: Path, synchronous: bool):
    if use_log_file:
        current_datetime = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        log_filename = 'caiman' + current_datetime + '.log'
        log_path = Path(cm.paths.get_tempdir()) / log_filename
        print(f"Will save logging data to {log_path}")
    else:
        log_path = None
    # set up logging
    logging.basicConfig(
        format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
        filename=log_path,
        level=log_severity,
        style="{",
    )

    if synchronous:
        print("Running on one core.")
        num_processors_to_use = 1
    else:
        # set env variables to avoid multithreading in dependencies !DO NOT CHANGE!
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        print(
            f"You have {psutil.cpu_count()} CPUs available in your current environment, using {psutil.cpu_count() - 1 if  psutil.cpu_count() <= 32 else 32} for parallel processing."
        )
        num_processors_to_use = None if psutil.cpu_count() <= 20 else 20

    if "cluster" in locals():  # 'locals' contains list of current local variables
        print("Closing previous cluster")
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, num_processes = cm.cluster.setup_cluster(
        backend="multiprocessing",
        n_processes=num_processors_to_use,
        ignore_preexisting=False,
    )
    print(
        f"Successfully initilialized multicore processing with a pool of {num_processes} CPU cores"
    )

    return cluster, num_processes


def cleanup(cluster, delete_logs: bool, delete_memmaps: bool, memmap_paths=None):
    cm.stop_server(dview=cluster)
    logging.shutdown()

    if delete_logs:
        logging_dir = Path(cm.paths.get_tempdir())
        log_files = logging_dir.glob("caiman*.log")
        for log_file in log_files:
            print(f"Deleting {log_file}")
            os.remove(log_file)
    
    if delete_memmaps and memmap_paths:
        # Ensure arrays that may hold the mmap are gone before deletion
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        for k in ["normcorre_mmap", "cnmf_mmap"]:
            f = memmap_paths.get(k)
            if not f:
                continue
            # mot_correct.mmap_file is sometimes returned as a list
            paths = f if isinstance(f, list) else [f]
            for p in paths:
                try:
                    print(f"Deleting memmap file: {p}")
                    Path(p).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Failed to delete {p}: {e}")
    

def save_motion_correction_comparison(
    input_path: Path,
    output_path: Path,
    mot_correct: MotionCorrect,
    crop: tuple = None,  # (row_start, row_stop, col_start, col_stop)
):
    movie_orig = cm.load(str(input_path), subindices=slice(2000))
    movie_corrected = cm.load(mot_correct.mmap_file, subindices=slice(2000))
    if crop is not None:
        rs, row_end, cs, ce = crop
        movie_orig = movie_orig[:, rs:row_end, cs:ce]
        movie_corrected = movie_corrected[:, rs:row_end, cs:ce]
    ds_ratio = 0.2
    cm.concatenate(
        [
            movie_orig.resize(1, 1, ds_ratio)
            - mot_correct.min_mov * mot_correct.nonneg_movie,
            movie_corrected.resize(1, 1, ds_ratio),
        ],
        axis=2,
    ).save(str(output_path / "motion_correction_comparison_final2.avi"))


def preproc(parameters: params.CNMFParams, video_path: Path, cluster, num_processes: int, save_nwb=False, save_comparison=True, run_name="caiman_final"):
    print(parameters)
    mot_correct = MotionCorrect(str(video_path), dview=cluster, **parameters.motion)
    mot_correct.motion_correct(save_movie=True)

    print(f"Motion correction results saved to {mot_correct.mmap_file}")

    CROPS = {
        "946": (70, -200, 70, -70),
        "847": (90, -150, 150, -20),
        "945": (20, -180, 150, -1),
        "943": (120, -250, 150, -1),
        "944": (200, 550, 100, 500),
        "163": (0, -50, 70, -50),
        "1636": (100, -1, 80, -20),
        "1637": (30, -1, 130, -10),
        "1639": (10, -10, 70, -20),
    }  # (row_start, row_stop, col_start, col_stop)
    mouse_id = _match_mouse_id(video_path, CROPS)
    if mouse_id is None:
        raise ValueError(f"No CROP defined for {video_path.name} — add an entry to CROPS")
    CROP = CROPS[mouse_id]

    caiman_output_dir = video_path.parent / run_name
    caiman_output_dir.mkdir(parents=True, exist_ok=True)

    if save_comparison:
        save_motion_correction_comparison(video_path, caiman_output_dir, mot_correct, crop=CROP)
        print("Saved motion correction comparison to disk")

    # Load F-order mmap directly — crop removes border artifacts (CROP 70px > max_shifts 25px)
    movie_mc = cm.load(mot_correct.mmap_file)
    rs, row_end, cs, ce = CROP
    images = np.array(movie_mc[:, rs:row_end, cs:ce])  # copy only cropped region to RAM
    del movie_mc

    images -= images.mean(axis=0)  # remove static vignette/illumination gradient
    img_min = images.min()
    if img_min < 0:
        images -= img_min  # shift to non-negative so CaImAn doesn't have to
    dims = images.shape[1:]

    cropped_mmap_path = cm.save_memmap(
        [images],
        base_name="memmap_cropped_",
        order='C',
        dview=cluster,
    )

    print(f"Cropped memory-mapped file saved to {cropped_mmap_path}")

    Yr, dims, num_frames = cm.load_memmap(cropped_mmap_path)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")
    parameters.change_params({'dims': dims})

    cnmf_model = cnmf.CNMF(num_processes, params=parameters, dview=cluster)
    cnmf_fit = cnmf_model.fit(images)
    print("CNMF-E model fit to data")

    correlation_image, _ = cm.summary_images.correlation_pnr(
        images[::max(num_frames//1000, 1)], # subsample if needed
        gSig=parameters.init["gSig"][0],
        swap_dim=False,
    ) # change swap dim if output looks weird, it is a problem with tiffile

    print("Computed correlation image")

    cnmf_fit.estimates.evaluate_components(images, cnmf_fit.params, dview=cluster)

    print(
        f"Num accepted/rejected: {len(cnmf_fit.estimates.idx_components)}, {len(cnmf_fit.estimates.idx_components_bad)}"
    )


    cnmf_fit.estimates.detrend_df_f(
        quantileMin=8, frames_window=250, flag_auto=False, use_residuals=False, detrend_only=True
    )

    cnmf_fit.estimates.Cn = (
        correlation_image  # squirrel away correlation image with cnmf object
    )

    # save caiman format
    caiman_results_path = video_path.parent / run_name / "caiman_results.hdf5"
    caiman_results_path.parent.mkdir(exist_ok=True, parents=True)
    cnmf_fit.save(str(caiman_results_path))
    print(f"Results saved to {str(caiman_results_path)}!")

    if save_nwb:

        raise NotImplementedError(
            "Unfortunately, saving caiman results to an NWB file isn't fully implemented yet :/"
        )

        # Hacky change to cnmf object to allow builtin save
        cnmf_model.estimates.cnn_preds = None
        cnmf_model.estimates.b = np.zeros((cnmf_model.estimates.A.shape[0], 1))
        cnmf_model.estimates.f = np.zeros((1, cnmf_model.estimates.C.shape[1]))

        # save nwb
        cnmf_model.estimates.save_NWB(
            str(nwb_path), 
            imaging_series_name="gcamp",
            imaging_rate=parameters.data["fr"],
            imaging_plane_name="ImagingPlane",
        )

        # Add F_dff later since builtin save doesn't
        with NWBHDF5IO(str(nwb_path), 'r+') as io:
            nwb = io.read()
            f_dff = DfOverF()
            f_dff.add_roi_response_series(
                "f_dff",
                data=cnmf_model.estimates.F_dff,
                rois=nwb.processing["ophys"].get("Fluorescence").roi_response_series["RoiResponseSeries"].rois,
                unit="N/A",
                timestamps=nwb.acquisition["gcamp"].timestamps,
            )
            nwb.processing["ophys"].add_data_interface(f_dff)
            io.write(nwb)
            io.close()
        
        print(f"Results saved!")
    return {
        "normcorre_mmap": mot_correct.mmap_file,
        "cnmf_mmap": cropped_mmap_path,
    }


def main():
    cnmf_params, input_path, log_severity, use_log_file, delete_logs, synchronous, delete_memmaps, save_comparison, run_name = get_params()
    cluster, n_processes = setup(use_log_file, log_severity, synchronous)
    memmap_paths = preproc(cnmf_params, input_path, cluster, n_processes, save_comparison=save_comparison, run_name=run_name)
    cleanup(cluster, delete_logs, delete_memmaps=delete_memmaps, memmap_paths=memmap_paths)
    print("Done!")


if __name__ == "__main__":
    main()