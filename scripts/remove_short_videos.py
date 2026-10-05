import subprocess
import glob
import os
import argparse

MIN_DURATION = 10.0  # seconds

def get_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def main(root_dir):
    pattern = os.path.join(root_dir, "*.avi")

    for video in glob.glob(pattern):
        duration = get_duration(video)

        if duration is None:
            print(f"[SKIP] Could not read duration: {video}")
            continue

        if duration < MIN_DURATION:
            print(f"[DELETE] {video} ({duration:.2f}s)")
            os.remove(video)

            ts = (
                video
                .replace(".avi", "")
                .replace("Miniscope", "timestamps")
                + ".csv"
            )
            if os.path.exists(ts):
                os.remove(ts)
                print(f"         deleted {ts}")
        else:
            print(f"[KEEP]   {video} ({duration:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete .avi videos shorter than 10 seconds (and matching timestamps CSV)."
    )
    parser.add_argument(
        "directory",
        help="Directory containing the video/CSV files"
    )

    args = parser.parse_args()
    main(args.directory)
