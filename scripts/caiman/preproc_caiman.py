import cv2
import logging
import numpy as np
import os
import psutil
import datetime
from pathlib import Path
from argparse import ArgumentParser

try:
    cv2.setNumThreads(0)
except ():
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params


def parse_args():
    parser = ArgumentParser(
        description="Parse arguments for motion correction and source extraction"
    )

    # general dataset-dependent parameters
    parser.add_argument(
        "--fr", type=int, default=25, help="Imaging rate in frames per second"
    )
    parser.add_argument(
        "--decay_time",
        type=float,
        default=0.56,
        help="Length of a typical transient in seconds",
    )
    parser.add_argument(
        "--dxy",
        type=float,
        nargs=2,
        default=[0.83, 0.83],
        help="Spatial resolution in x and y in (um per pixel)",
    )

    # motion correction parameters
    parser.add_argument(
        "--strides",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Start a new patch for pw-rigid motion correction every x pixels",
    )
    parser.add_argument(
        "--overlaps",
        type=int,
        nargs=2,
        default=[32, 32],
        help="Overlap between patches (width of patch = strides+overlaps)",
    )
    parser.add_argument(
        "--max_shifts",
        type=int,
        nargs=2,
        default=[25, 25],
        help="Maximum allowed rigid shifts (in pixels)",
    )
    parser.add_argument(
        "--max_deviation_rigid",
        type=int,
        default=8,
        help="Maximum shifts deviation allowed for patch with respect to rigid shifts",
    )
    parser.add_argument(
        "--gSig_filt",
        type=int,
        nargs=2,
        default=[8, 8],
        help="Size of high pass spatial filtering, used in 1p data",
    )
    parser.add_argument(
        "--pw_rigid",
        type=bool,
        default=True,
        help="Flag for performing non-rigid motion correction",
    )

    # CNMF parameters for source extraction and deconvolution
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="Order of the autoregressive system (set p=2 if there is visible rise time in data)",
    )
    parser.add_argument(
        "--gSig",
        type=int,
        nargs=2,
        default=[8, 8],
        help="Expected half-width of neurons in pixels (Gaussian kernel standard deviation)",
    )
    parser.add_argument(
        "--merge_thr",
        type=float,
        default=0.8,
        help="Merging threshold, max correlation allowed",
    )
    parser.add_argument(
        "--rf",
        type=int,
        default=32,
        help="Half-size of the patches in pixels (patch width is rf*2 + 1)",
    )
    parser.add_argument(
        "--stride_cnmf",
        type=int,
        default=16,
        help="Amount of overlap between the patches in pixels (overlap is stride_cnmf+1)",
    )
    parser.add_argument(
        "--ssub", type=int, default=1, help="Spatial subsampling during initialization"
    )
    parser.add_argument(
        "--tsub", type=int, help="Temporal subsampling during initialization"
    )
    parser.add_argument(
        "--gnb",
        type=int,
        default=-1,
        help="Number of global background components (set to 0 for lower ram, -1 for faster runtime)",
    )
    parser.add_argument(
        "--min_corr",
        type=float,
        default=0.8,
        help="Min peak value from correlation image",
    )
    parser.add_argument(
        "--min_pnr", type=float, default=4.4, help="Min peak to noise ratio"
    )
    parser.add_argument(
        "--ssub_B",
        type=int,
        default=2,
        help="Spatial subsampling factor for background",
    )

    # component evaluation parameters
    parser.add_argument(
        "--min_SNR",
        type=float,
        default=2.0,
        help="Signal to noise ratio for accepting a component",
    )
    parser.add_argument(
        "--rval_thr",
        type=float,
        default=0.75,
        help="Space correlation threshold for accepting a component",
    )

    # script-specific parameters
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input session",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output session",
    )
    parser.add_argument(
        "--log_severity",
        type=str,
        default="WARNING",
        help="Logging severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--use_log_file",
        action="store_true",
        help="Path to log file",
    )
    parser.add_argument(
        "--delete_logs",
        action="store_true",
        help="Flag for deleting logs after script completion",
    )
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Flag for synchronous processing (useful for debugging)",
    )

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


def package_arguments_to_dict(args):
    parameter_dict = {
        "fnames": args.input_path,
        "fr": args.fr,
        "dxy": args.dxy,
        "decay_time": args.decay_time,
        "strides": args.strides,
        "overlaps": args.overlaps,
        "max_shifts": args.max_shifts,
        "max_deviation_rigid": args.max_deviation_rigid,
        "gSig_filt": args.gSig_filt,
        "pw_rigid": args.pw_rigid,
        "p": args.p,
        "nb": args.gnb,
        "min_corr": args.min_corr,
        "min_pnr": args.min_pnr,
        "ssub_B": args.ssub_B,
        "rf": args.rf,
        "gSig": np.array(args.gSig),
        "gSiz": 2 * np.array(args.gSig) + 1,
        "stride": args.stride_cnmf,
        "ssub": args.ssub,
        "tsub": args.tsub if args.tsub is not None else args.fr // 10 + 1,
        "merge_thr": args.merge_thr,
        "min_SNR": args.min_SNR,
        "rval_thr": args.rval_thr,

        # required for CNMF-E, will not change
        "nb_patch": 0,
        "K": None,
        "method_init": "corr_pnr",
        "center_psf": True,
        "only_init": True,
        "use_cnn": False,
    }

    return params.CNMFParams(
        params_dict=parameter_dict
    )


def get_params():
    args = parse_args()

    cnmf_params = package_arguments_to_dict(args)

    if args.log_severity == "DEBUG":
        log_severity = logging.DEBUG
    elif args.log_severity == "INFO":
        log_severity = logging.INFO
    elif args.log_severity == "WARNING":
        log_severity = logging.WARNING
    elif args.log_severity == "ERROR":
        log_severity = logging.ERROR
    elif args.log_severity == "CRITICAL":
        log_severity = logging.CRITICAL
    else:
        raise ValueError(
            "Invalid log severity level. Please choose from DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    return (
        cnmf_params,
        Path(args.input_path),
        output_path,
        log_severity,
        args.use_log_file,
        args.delete_logs,
        args.synchronous,
    )


def setup(use_log_file: bool, log_severity: Path, delete_logs: bool, synchronous: bool):
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

        print(
            f"You have {psutil.cpu_count()} CPUs available in your current environment, using {psutil.cpu_count() - 1 if  psutil.cpu_count() <= 32 else 31} for parallel processing."
        )
        num_processors_to_use = None if psutil.cpu_count() <= 32 else 31

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


def cleanup(cluster, delete_logs: bool):
    cm.stop_server(dview=cluster)
    logging.shutdown()

    if delete_logs:
        logging_dir = cm.paths.get_tempdir()
        log_files = logging_dir.glob("demo_pipeline*.log")
        for log_file in log_files:
            print(f"Deleting {log_file}")
            os.remove(log_file)


def save_motion_correction_comparison(input_path: Path, output_path: Path, mot_correct: MotionCorrect):
    movie_orig = cm.load(str(input_path), subindices=slice(2000))  # in case it was not loaded earlier
    movie_corrected = cm.load(mot_correct.mmap_file, subindices=slice(2000))  # load motion corrected movie
    ds_ratio = 0.2
    cm.concatenate(
        [
            movie_orig.resize(1, 1, ds_ratio)
            - mot_correct.min_mov * mot_correct.nonneg_movie,
            movie_corrected.resize(1, 1, ds_ratio),
        ],
        axis=2,
    ).save(str(output_path / "motion_correction_comparison.avi"))


def preproc(parameters: params.CNMFParams, input_path: Path, output_path: Path, cluster, num_processes: int):
    mot_correct = MotionCorrect(str(input_path), dview=cluster, **parameters.motion)
    mot_correct.motion_correct(save_movie=True)

    print(f"Motion correction results saved to {mot_correct.mmap_file}")

    save_motion_correction_comparison(input_path, output_path, mot_correct)

    print("Saved motion correction comparison to disk")

    border_to_0 = (
        0 if mot_correct.border_nan == "copy" else mot_correct.border_to_0
    )  # trim border against NaNs
    mc_memmapped_fname = cm.save_memmap(
        mot_correct.mmap_file,
        base_name="memmap_",
        order="C",
        border_to_0=border_to_0,  # exclude borders, if that was done
        dview=cluster,
    )

    print(f"Memory-mapped file saved to {mc_memmapped_fname}")

    Yr, dims, num_frames = cm.load_memmap(mc_memmapped_fname)
    images = np.reshape(
        Yr.T, [num_frames] + list(dims), order="F"
    )  # reshape frames in standard 3d format (T x X x Y)

    print("Loaded memory-mapped file into memory for CNMF processing")

    cnmf_model = cnmf.CNMF(num_processes, params=parameters, dview=cluster)
    cnmf_fit = cnmf_model.fit(images)

    print("CNMF-E model fit to data")

    correlation_image, _ = cm.summary_images.correlation_pnr(
        images[::max(num_frames//1000, 1)], # subsample if needed
        gSig=parameters.init["gSig"][0],
        swap_dim=False,
    ) # change swap dim if output looks weird, it is a problem with tiffile

    cnmf_fit.estimates.evaluate_components(images, cnmf_fit.params, dview=cluster)

    print(
        f"Num accepted/rejected: {len(cnmf_fit.estimates.idx_components)}, {len(cnmf_fit.estimates.idx_components_bad)}"
    )

    # cnmf_fit.estimates.detrend_df_f(
    #     quantileMin=8, frames_window=250, flag_auto=False, use_residuals=False, detrend_only=True
    # )

    save_path = output_path / "cnmfe_results.hdf5"
    cnmf_fit.estimates.Cn = (
        correlation_image  # squirrel away correlation image with cnmf object
    )
    cnmf_fit.save(str(save_path))
    print(f"Results saved to {save_path}")


def main():
    cnmf_params, input_path, output_path, log_severity, use_log_file, delete_logs, synchronous = get_params()
    cluster, n_processes = setup(use_log_file, log_severity, delete_logs, synchronous)
    preproc(cnmf_params, input_path, output_path, cluster, n_processes)
    cleanup(cluster, delete_logs)
    print("Done!")


if __name__ == "__main__":
    main()
