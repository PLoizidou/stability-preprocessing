import cv2
import argparse


def save_frames_range_as_avi(input_file, output_file, start_frame, end_frame):
    # Open the AVI file
    cap = cv2.VideoCapture(input_file)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if start_frame and end_frame are valid
    if (
        start_frame < 0
        or start_frame >= total_frames
        or end_frame < start_frame
        or end_frame >= total_frames
    ):
        print("Error: Invalid start_frame or end_frame.")
        return

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Set the frame position to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Write frames in the specified range to output file
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame {i}.")
            break
        out.write(frame)
        print(f"Frame {i} written successfully.")

    # Release the video file and VideoWriter object
    cap.release()
    out.release()

    print(f"Frames {start_frame} to {end_frame} saved as {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a range of frames from an AVI file as a new AVI file."
    )
    parser.add_argument("--input_file", help="Path to the input AVI file")
    parser.add_argument("--output_file", help="Path to the output AVI file")
    parser.add_argument("--start_frame", type=int, help="Start frame index")
    parser.add_argument("--end_frame", type=int, help="End frame index")
    args = parser.parse_args()

    save_frames_range_as_avi(
        args.input_file, args.output_file, args.start_frame, args.end_frame
    )
