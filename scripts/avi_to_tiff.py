import cv2
import os
from tqdm import tqdm
import tifffile
from pathlib import Path
import argparse

def avi_to_tif_stack(input_file: Path, output_file: Path):
    print(f"Converting {input_file} to {output_file}")

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    # Open the AVI file
    cap = cv2.VideoCapture(str(input_file))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Iterate through frames
    with tifffile.TiffWriter(output_file, bigtiff=True) as stack:
        for i in tqdm(range(frame_count), desc="Converting Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            stack.write(frame, contiguous=True)

    # Release video capture object
    cap.release()

    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert AVI file to TIFF stack')
    parser.add_argument('--input_file', help='Input AVI file path')
    parser.add_argument('--output_file', help='Output file to save TIFF stack')
    args = parser.parse_args()

    # Convert AVI to TIFF stack
    avi_to_tif_stack(Path(args.input_file), Path(args.output_file))
