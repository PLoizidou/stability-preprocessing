#!/usr/bin/env python3
"""
Simple Miniscope denoiser for a single .avi file.
- No dependency on framesPerFile or startingFileNum
- Reads until end-of-file
- Applies frequency-domain vertical-notch (removes full-width horizontal bands)
- Optionally applies temporal deflicker (low-pass on per-frame mean)
- Preserves overall brightness (zero-mean correction and DC keep)

Usage:
  python miniscope_denoise_simple.py \
    --input /path/to/input.avi \
    --output /path/to/output.avi \
    --codec FFV1 --keep-dc --deflicker --cutoff 3.0 --order 6 --channel g --verbose

Install:
  pip install opencv-python scipy

Notes:
- Output is 8-bit grayscale. Internal processing uses float32.
- For scientific analysis, prefer --codec FFV1 (lossless).
"""
import argparse
import os
import sys
import cv2
import numpy as np
from math import cos, pi
from scipy.signal import butter, filtfilt

def parse_args():
    p = argparse.ArgumentParser(description="Single-file Miniscope banding denoise")
    p.add_argument("--input", "-i", required=True, help="Input .avi path")
    p.add_argument("--output", "-o", required=True, help="Output .avi path")
    p.add_argument("--codec", default="FFV1", help="FourCC, e.g., FFV1 (lossless), MJPG, XVID")
    p.add_argument("--channel", choices=["r","g","b","gray"], default="g", help="Color channel to use")
    p.add_argument("--notch", type=int, default=2, help="Vertical k half-width to suppress (bins)")
    p.add_argument("--soften", type=int, default=6, help="Feather width around notch (bins)")
    p.add_argument("--keep-dc", action="store_true", help="Keep DC exactly to preserve mean brightness")
    p.add_argument("--deflicker", action="store_true", help="Apply temporal deflicker via low-pass")
    p.add_argument("--cutoff", type=float, default=3.0, help="Deflicker cutoff Hz (if --deflicker)")
    p.add_argument("--order", type=int, default=6, help="Butterworth order (if --deflicker)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

def fourcc_from_string(s: str) -> int:
    s = (s or "FFV1")
    if len(s) == 4:
        return cv2.VideoWriter_fourcc(*s)
    return cv2.VideoWriter_fourcc(*"FFV1")

def make_vertical_notch_mask(rows:int, cols:int, notch:int=2, soften:int=6, keep_dc:bool=True) -> np.ndarray:
    """
    Build multiplicative mask (after fftshift) suppressing ky≈0 to remove horizontal bands.
    Returns shape (rows, cols, 2) for OpenCV complex DFT multiplication.
    """
    cy, cx = rows // 2, cols // 2
    wv = np.ones(rows, dtype=np.float32)
    for y in range(rows):
        dv = abs(y - cy)
        if dv <= notch:
            w = 0.0
        elif dv <= notch + soften and soften > 0:
            t = (dv - notch) / float(soften)
            w = 0.5 * (1 - cos(pi * t))
        else:
            w = 1.0
        wv[y] = w
    mask2d = np.repeat(wv[:, None], cols, axis=1)
    if keep_dc:
        mask2d[cy, cx] = 1.0
    return np.dstack([mask2d, mask2d]).astype(np.float32)

def get_channel(frame: np.ndarray, which: str) -> np.ndarray:
    if frame.ndim == 2 or which == "gray":
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # BGR indexing in OpenCV
    if which == "b":
        return frame[:,:,0]
    if which == "g":
        return frame[:,:,1]
    if which == "r":
        return frame[:,:,2]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def collect_mean_trace(path_in, channel, maskFFT):
    """Read video once, compute mean after notch filtering for each frame."""
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path_in}")
    means = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = get_channel(frame, channel).astype(np.float32)
        dft = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * maskFFT
        img_back = cv2.idft(np.fft.ifftshift(fshift))
        mag = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        means.append(float(mag.mean()))
    cap.release()
    return np.array(means, dtype=np.float32)

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Probe video geometry & FPS
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open input video", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if args.verbose:
        print(f"Input: {cols}x{rows} @ {fps:.3f} Hz")

    # Build notch mask once
    maskFFT = make_vertical_notch_mask(rows, cols, notch=args.notch, soften=args.soften, keep_dc=args.keep_dc)

    # Optional: compute temporal mean trace (after notch) and low-pass it
    meanFiltered = None
    if args.deflicker:
        if args.verbose:
            print("Collecting mean trace for deflicker...")
        meanFrame = collect_mean_trace(args.input, args.channel, maskFFT)
        if len(meanFrame) >= 4:
            b, a = butter(args.order, args.cutoff / (0.5 * fps), btype="low", analog=False)
            meanFiltered = filtfilt(b, a, meanFrame)
        else:
            print("Warning: not enough frames for deflicker, skipping.", file=sys.stderr)
            meanFiltered = None

    # Prepare writer (grayscale output)
    fourcc = fourcc_from_string(args.codec)
    out = cv2.VideoWriter(args.output, fourcc, fps, (cols, rows), isColor=False)
    if not out.isOpened():
        print("Could not open output writer", file=sys.stderr)
        sys.exit(1)

    # Second pass: process and write frames, reading until EOF
    cap = cv2.VideoCapture(args.input)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = get_channel(frame, args.channel).astype(np.float32)

        # Spatial notch
        dft = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * maskFFT
        img_back = cv2.idft(np.fft.ifftshift(fshift))
        mag = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # Optional temporal deflicker: scale to match filtered mean
        if meanFiltered is not None and idx < len(meanFiltered):
            mf = float(mag.mean())
            target = float(meanFiltered[idx])
            if mf > 1e-6:
                mag = mag * (1.0 + (target - mf)/mf)

        # Clip to 8-bit
        mag = np.clip(mag, 0.0, 255.0).astype(np.uint8)
        out.write(mag)
        idx += 1

    cap.release()
    out.release()
    if args.verbose:
        print(f"Done. Wrote: {args.output}")

if __name__ == "__main__":
    main()
