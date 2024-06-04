import subprocess
import argparse
from pathlib import Path

def run_caiman_preprocessing(input_dir, output_base_dir, min_corr, min_pnr, min_SNR, rval_thr, gnb):
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)

    # Ensure the output base directory exists
    output_base_path.mkdir(parents=True, exist_ok=True)

    # List all .avi files in the input directory
    for file_path in input_path.glob('miniscope*.avi'):
        # Generate the output path
        output_path = output_base_path / file_path.stem
        # Construct the command
        command = [
            'python', 'scripts/caiman/preproc_caiman.py',
            '--input_path', str(file_path),
            '--output_path', str(output_path),
            '--min_corr', str(min_corr),
            '--min_pnr', str(min_pnr),
            '--min_SNR', str(min_SNR),
            '--rval_thr', str(rval_thr),
            '--gnb', str(gnb)
        ]
        # Run the command
        print(f'Running command: {" ".join(command)}')
        subprocess.run(command)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run caiman preprocessing on all miniscope files in a directory.')
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input .avi files.')
    parser.add_argument('output_base_dir', type=str, help='Path to the base directory for output files.')
    parser.add_argument('--min_corr', type=float, default=0.85, help='Minimum correlation threshold.')
    parser.add_argument('--min_pnr', type=float, default=6.5, help='Minimum peak-to-noise ratio threshold.')
    parser.add_argument('--min_SNR', type=float, default=3, help='Minimum signal-to-noise ratio.')
    parser.add_argument('--rval_thr', type=float, default=0.8, help='R-value threshold.')
    parser.add_argument('--gnb', type=int, default=0, help='Number of gnb.')

    # Parse arguments
    args = parser.parse_args()

    # Run the preprocessing for all miniscope files
    run_caiman_preprocessing(
        args.input_dir,
        args.output_base_dir,
        args.min_corr,
        args.min_pnr,
        args.min_SNR,
        args.rval_thr,
        args.gnb
    )
