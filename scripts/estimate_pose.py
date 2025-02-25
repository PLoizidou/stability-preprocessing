import argparse as ap
import deeplabcut

from pathlib import Path

from pynwb import NWBHDF5IO

def handle_args():
    parser = ap.ArgumentParser(description='Estimate pose using DeepLabCut')
    parser.add_argument('config', type=str, help='Path to the config.yaml file')
    parser.add_argument('video_path', type=str, help='Path to the NWB file')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--save_summary", action='store_true', help="Save a summary of the results")
    return parser.parse_args()

def main(config_path: Path, video_path: Path, gpu_id: int = None, save_summary: bool = False):
    
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
    video_path = Path(args.video_path)
    config_path = Path(args.config)

    main(config_path=config_path, video_path=video_path, gpu_id=args.gpu_id, save_summary=args.save_summary)