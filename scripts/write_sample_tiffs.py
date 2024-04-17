import cv2
import random
import argparse
import os


def extract_random_frames(input_file, output_folder, num_frames):
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate random frame numbers
    random_frame_numbers = random.sample(range(total_frames), num_frames)

    # Iterate over the random frame numbers and extract frames
    for i, frame_number in enumerate(random_frame_numbers):
        # Set the frame number to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Couldn't read frame", frame_number)
            continue

        # Save the frame as TIFF
        output_file = os.path.join(output_folder, f"frame_{i}.tiff")
        cv2.imwrite(output_file, frame)
        print("Frame", frame_number, "extracted and saved successfully as", output_file)

    # Release the video file
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract random frames from an AVI file and save them as TIFF images."
    )
    parser.add_argument("--input_file", help="Path to the input AVI file")
    parser.add_argument(
        "--output_folder", help="Path to the output folder to save the frames"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of random frames to extract (default: 10)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    extract_random_frames(args.input_file, args.output_folder, args.num_frames)
