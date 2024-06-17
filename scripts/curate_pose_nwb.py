import argparse as ap
import h5py

from pathlib import Path

from pynwb import NWBHDF5IO
from neuroconv.datainterfaces import DeepLabCutInterface

def handle_args():
    parser = ap.ArgumentParser(description='Estimate pose using DeepLabCut')
    parser.add_argument('config', type=str, help='Path to the config.yaml file')
    parser.add_argument('nwb_file', type=str, help='Path to the NWB file')
    parser.add_argument('acquisition', type=str, help='Name of the acquisition')
    parser.add_argument("storage_name", type=str, help="Name of the storage group in the NWB file")
    parser.add_argument("--match_regex", type=str, default=None, help="Regex to filter pose files") 
    return parser.parse_args()

def main(config_path: Path, nwb_path: Path, video_key: str, storage_key: str, match_regex: str = None):
    with NWBHDF5IO(nwb_path, 'r+') as io:
        nwb = io.read()
        video_path = nwb_path.parent / nwb.acquisition[video_key].external_file[0]
        subject = nwb.subject.subject_id

        data_dir = video_path.parent / 'dlc'
        if not data_dir.exists():
            raise FileNotFoundError(f'DLC directory not found: {data_dir}')

        pose_regex = match_regex if not match_regex is None else f'*filtered.h5'
        filtered_pose = str(list(data_dir.glob(pose_regex))[0])
        print(f'Filtered pose: {filtered_pose}')

        dlc_interface = DeepLabCutInterface(
            file_path=filtered_pose,
            config_file_path=config_path,
            subject_name=subject,
            verbose=False,
        )
        metadata = dlc_interface.get_metadata()
        metadata
        dlc_interface.add_to_nwbfile(nwbfile=nwb, metadata=metadata)
        io.write(nwb)

    with h5py.File(nwb_path, "r+") as nwb_file:
        nwb_file["processing/behavior"].move("PoseEstimation", storage_key)
    print(f'Added pose data to NWB file: {nwb_path}')
    print(f'Pose data stored in: processing/behavior/{storage_key}')

    


if __name__ == "__main__":
    args = handle_args()
    config_path = Path(args.config)
    nwb_path = Path(args.nwb_file)

    main(
        config_path=config_path,
        nwb_path=nwb_path,
        video_key=args.acquisition,
        storage_key=args.storage_name,
        match_regex=args.match_regex
    )