import os
import argparse
import subprocess
import pandas as pd
import glob
import shutil
import tempfile

def trim_video(in_video, out_video, trim_sec):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", in_video,
            "-t", str(trim_sec),
            "-c", "copy",
            out_video
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def trim_timestamps(in_csv, out_csv, trim_sec, fps=25):
    max_rows = int(trim_sec * fps)

    df = pd.read_csv(in_csv, header=None)

    if len(df) == 0:
        raise ValueError(f"Empty CSV: {in_csv}")

    df_trimmed = df.iloc[:max_rows]

    if df_trimmed.empty:
        raise ValueError(f"Trimming produced empty CSV: {in_csv}")

    df_trimmed.to_csv(out_csv, index=False, header=False)


def main(work_dir, archive_dir, trim_sec):
    work_dir = os.path.abspath(work_dir)
    archive_dir = os.path.abspath(archive_dir)

    os.makedirs(archive_dir, exist_ok=True)

    for csv_path in glob.glob(os.path.join(work_dir, "timestamps*.csv")):
        fname = os.path.basename(csv_path)
        token = fname.replace("timestamps", "").replace(".csv", "")

        print(f"[PROCESS] {token}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # temp outputs
            tmp_csv = os.path.join(tmpdir, fname)
            trim_timestamps(csv_path, tmp_csv, trim_sec, fps = 25)

            tmp_videos = []
            for prefix in ["Miniscope", "Home", "Linear"]:
                vname = f"{prefix}{token}.avi"
                vpath = os.path.join(work_dir, vname)
                if os.path.exists(vpath):
                    tmp_out = os.path.join(tmpdir, vname)
                    trim_video(vpath, tmp_out, trim_sec)
                    tmp_videos.append(vname)

            # move originals to archive
            shutil.move(csv_path, os.path.join(archive_dir, fname))
            for vname in tmp_videos:
                orig = os.path.join(work_dir, vname)
                shutil.move(orig, os.path.join(archive_dir, vname))

            # move trimmed files into place
            shutil.move(tmp_csv, os.path.join(work_dir, fname))
            for vname in tmp_videos:
                shutil.move(
                    os.path.join(tmpdir, vname),
                    os.path.join(work_dir, vname)
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim videos and timestamps, archive originals, keep filenames unchanged."
    )
    parser.add_argument("work_dir", help="Directory with original videos and CSV")
    parser.add_argument("archive_dir", help="Directory to move original files into")
    parser.add_argument("seconds", type=float, help="Trim duration in seconds")

    args = parser.parse_args()
    main(args.work_dir, args.archive_dir, args.seconds)
