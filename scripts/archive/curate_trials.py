import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil
import argparse
import pandas as pd
from pynwb import NWBHDF5IO, NWBFile
from pynwb.image import ImageSeries
from pynwb.ophys import OnePhotonSeries, OpticalChannel
from pynwb.file import Subject
import uuid

def process_animal_files(animal_id, files, start_date=None):
    """
    Process the files for a single animal and organize them by date and time.

    Args:
        animal_id (str): The animal ID.
        files (list): List of files belonging to the animal.
        start_date (datetime, optional): The start date to filter sessions.

    Returns:
        dict: A dictionary with the structure {date: {time: [file_paths]}} for the given animal.
    """
    # Regular expression to extract date and time
    file_pattern = re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}_\d{2}_\d{2})')
    
    sessions = defaultdict(lambda: defaultdict(list))
    
    for file in files:
        match = file_pattern.search(file.name)
        if match:
            date_str = match.group('date')
            time_str = match.group('time')
            session_datetime = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Include the file only if it is after the start_date (if start_date is provided)
            if not start_date or session_datetime > start_date:
                sessions[date_str][time_str].append(file)
    
    return sessions

def organize_sessions(base_dir, start_date=None, animals=None):
    """
    Organizes files by sessions based on animal ID, date, and time.

    Args:
        base_dir (str): The base directory containing the files.
        start_date (str, optional): The start date in 'YYYY-MM-DD' format. Only sessions after this date will be included.
        animals (list, optional): List of animal IDs to include. Only files for these animals will be processed.

    Returns:
        dict: A dictionary with the structure {animal_id: {date: {time: [file_paths]}}}.
    """
    # Convert start_date to datetime object if provided
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    # Dictionary to hold sessions
    sessions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Dictionary to hold files for each animal
    animal_files = defaultdict(list)

    base_path = Path(base_dir)

    for child in base_path.glob('*'):
        if child.is_dir() and "Mouse" in child.stem and (not animals or child.stem in animals):
            animal_dir = child
            animal_id = child.stem
            animal_files[animal_id] = list(animal_dir.glob("**/*"))

    # Process files for each animal
    for animal_id, files in animal_files.items():
        animal_sessions = process_animal_files(animal_id, files, start_date)
        sessions[animal_id] = animal_sessions

    return sessions


def curate_sessions(sessions, output_dir):
    """
    Save each session to an NWB file.

    Args:
        sessions (dict): The organized sessions dictionary.
        output_dir (str): The directory to save the NWB files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for animal_id, dates in sessions.items():
        sub_str = f"sub-{animal_id}"
        for date, times in dates.items():
            print(f"Saving session on {date} for animal {animal_id}")
            for time, files in times.items():
                session_id = f"ses-{date.replace('-', '')}T{time.replace('_', '')}"
                csv_files = [f for f in files if f.name.endswith(".csv")]
                if csv_files:
                    timestamps_df = pd.read_csv(str(csv_files[0]), header=None)
                else:
                    raise ValueError(f"No CSV file found for session {date} .")
                timestamps_df["timestamps"] = pd.to_datetime(timestamps_df[0])
                timestamps = timestamps_df["timestamps"]
                start_time = timestamps[0]
                timestamps = (timestamps - start_time).dt.total_seconds().values

                for file in files:
                    new_file = output_path / sub_str / session_id / file.name
                    new_file.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Copying {file} to {new_file}")
                    shutil.copy(file, new_file)
                    
                    



def save_sessions_to_nwb(sessions, output_dir):
    """
    Save each session to an NWB file.

    Args:
        sessions (dict): The organized sessions dictionary.
        output_dir (str): The directory to save the NWB files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for animal_id, dates in sessions.items():
        sub_str = f"sub-{animal_id}"
        for date, times in dates.items():
            for time, files in times.items():
                session_id = f"ses-{date.replace('-', '')}T{time.replace('_', '')}"
                timestamps_df = pd.read_csv([f for f in files if f.name.endswith(".csv")][0], header=None)
                timestamps_df["timestamps"] = pd.to_datetime(timestamps_df[0])
                timestamps = timestamps_df["timestamps"]
                start_time = timestamps[0]
                timestamps = (timestamps - start_time).dt.total_seconds().values

                new_files = []

                for file in files:
                    new_file = output_path / sub_str / session_id / file.name
                    new_files.append(new_file)
                    new_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, new_file)
                    
                
                nwbfile = NWBFile(
                    session_description=f"Mouse exploring T-maze and linear track on {animal_id} on {date} at {time}",
                    identifier=str(uuid.uuid4()),
                    session_start_time=start_time,
                    session_id=session_id,
                    experimenter="Panagiota Loizidou",
                    lab="Lois Laboritory",
                    institution="California Institute of Technology",
                )
                subject = Subject(
                    subject_id=animal_id,
                    age="PLACEHOLDER",
                    description=animal_id,
                    species="Mus musulus",
                    sex="M",
                )
                nwbfile.subject = subject

                device = nwbfile.create_device(
                    name="miniscope",
                    description="Custom, chronically implanted miniscope",
                    manufacturer="?", # TODO: fill in
                )
                optical_channel = OpticalChannel(
                    name="OpticalChannel",
                    description="One photon channel",
                    emission_lambda=520.,
                )

                imaging_plane = nwbfile.create_imaging_plane(
                    name="ImagingPlane",
                    optical_channel=optical_channel,
                    imaging_rate=25.,
                    description="Hippocampus",
                    device=device,
                    excitation_lambda=500.,
                    indicator="GFP",
                    location="CA1",
                    grid_spacing=[0.83, 0.83],
                    grid_spacing_unit="micrometers",
                    origin_coords=[0., 0.],
                    origin_coords_unit="micrometers",
                )

                imaging_data = OnePhotonSeries(
                    name="gcamp",
                    dimension=[640, 480],
                    format="external",
                    external_file=[f.name for f in new_files if "miniscope" in f.name],
                    imaging_plane=imaging_plane,
                    starting_frame=[0],
                    timestamps=timestamps,
                )
                nwbfile.add_acquisition(imaging_data)


                device_linear = nwbfile.create_device(
                    name="linear_track",
                    description="Camera mounted above linear track",
                    manufacturer="?", # TODO: fill in
                )
                device_tmaze = nwbfile.create_device(
                    name="t_maze",
                    description="Camera mounted above T-maze",
                    manufacturer="?", # TODO: fill in
                )
                devive_home = nwbfile.create_device(
                    name="home",
                    description="Camera mounted above home cage",
                    manufacturer="?", # TODO: fill in
                )

                behavior_linear = ImageSeries(
                    name="behavior_linear",
                    external_file=[f.name for f in new_files if "behaviorLinear" in f.name],
                    starting_frame=[0],
                    format="external",
                    timestamps=timestamps,
                    device=device_linear,
                )
                behavior_home = ImageSeries(
                    name="behavior_home",
                    external_file=[f.name for f in new_files if "behavior" in f.name and not "Linear" in f.name],
                    starting_frame=[0],
                    format="external",
                    timestamps=timestamps,
                    device=devive_home,
                )
                nwbfile.add_acquisition(behavior_linear)
                nwbfile.add_acquisition(behavior_home)



                output_file = output_path / sub_str / session_id / f"{sub_str}-{session_id}.nwb"
                print(f"Saving session {session_id} for animal {animal_id} to {output_file}")
                with NWBHDF5IO(output_file, 'w') as io:
                    io.write(nwbfile)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Organize animal session files by date and time.")
    parser.add_argument('base_dir', type=str, help="The base directory containing the files.")
    parser.add_argument("output_dir", type=str, help="The directory to save the NWB files.")
    parser.add_argument('--start_date', type=str, help="The start date in 'YYYY-MM-DD' format. Only sessions after this date will be included.", default=None)
    parser.add_argument('--animals', type=str, nargs='*', help="List of animal IDs to include. Only files for these animals will be processed.", default=None)
    parser.add_argument("--save_nwb", action='store_true', help="Save the organized sessions to NWB files.")
    
    args = parser.parse_args()
    
    # Call the organize_sessions function with parsed arguments
    organized_sessions = organize_sessions(args.base_dir, args.start_date, args.animals)

    if not args.save_nwb:
        curate_sessions(organized_sessions, args.output_dir)
    else:
        # Save the organized sessions to NWB files, same directory structure
        save_sessions_to_nwb(organized_sessions, args.output_dir)

if __name__ == "__main__":
    main()
