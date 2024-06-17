import argparse as ap
import deeplabcut

from pathlib import Path

from pynwb import NWBHDF5IO

def handle_args():
    parser = ap.ArgumentParser(description='Estimate pose using DeepLabCut')
    parser.add_argument('config', type=str, help='Path to the config.yaml file')
    parser.add_argument('nwb_file', type=str, help='Path to the NWB file')
    parser.add_argument('acquisition', type=str, help='Name of the acquisition')
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID to use")
    parser.add_argument("--save_summary", action='store_true', help="Save a summary of the results")
    return parser.parse_args()

def main(config_path: Path, nwb_path: Path, video_key: str, gpu_id: int = None, save_summary: bool = False):
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        video_path = nwb_path.parent / nwb.acquisition[video_key].external_file[0]

    data_dir = video_path.parent / 'dlc'
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)

    deeplabcut.analyze_videos(
        str(config_path),
        [str(video_path)],
        videotype='avi',
        destfolder=str(data_dir),
        gputouse=gpu_id
    )
    deeplabcut.filterpredictions(
        str(config_path),
        [str(video_path)],
        videotype='avi',
        destfolder=str(data_dir),
    )
    if save_summary:
        deeplabcut.create_labeled_video(str(config_path), [str(video_path)], videotype='avi', destfolder=str(data_dir))
        deeplabcut.plot_trajectories(str(config_path), [str(video_path)], videotype='avi', destfolder=str(data_dir))


if __name__ == "__main__":
    args = handle_args()
    config_path = Path(args.config)
    nwb_path = Path(args.nwb_file)

    main(config_path=config_path, nwb_path=nwb_path, video_key=args.acquisition, gpu_id=args.gpu_id, save_summary=args.save_summary)