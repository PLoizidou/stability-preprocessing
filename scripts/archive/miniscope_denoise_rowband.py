#!/usr/bin/env python3
"""
Miniscope V4 denoise: remove horizontal rolling bands (row-wise flicker)
and global fast brightness flicker (~3 Hz), with brightness preserved.

Two steps per the Aharoni Lab notebook idea:
1) Spatial frequency-domain notch to suppress vertical k≈0 content (horizontal stripes).
2) Temporal low-pass on per-frame mean, then scale frames to remove fast flicker.

Usage:
  python miniscope_denoise_rowband.py --input INPUT.avi --output OUTPUT.avi

Optional flags:
  --cutoff 3.0            Low-pass cutoff in Hz for deflicker
  --order 6               Butterworth filter order
  --fps auto              Use "auto" to read from file, or provide a number
  --notch 2               Half-width (in vertical frequency bins) to fully suppress
  --soften 6              Feather width (bins) to smoothly restore pass-band
  --keep-dc               Preserve DC exactly (recommended to avoid brightness shift)
  --channel g             Channel: g (default), r, b, or gray
  --codec FFV1            FourCC, e.g., FFV1 (lossless), MJPG, XVID
  --display               Show live side-by-side preview (slower)
  --every 1               Save or preview every Nth frame (default 1)
  --verbose               Print extra logs

Notes:
- Output is grayscale. Processing uses float32 internally.
- For science analysis, prefer lossless codec (FFV1).
"""
import argparse
import os
import sys
import cv2
import numpy as np
from math import cos, pi
from scipy.signal import butter, filtfilt

def parse_args():
    p = argparse.ArgumentParser(description="Miniscope banding + deflicker denoise")
    p.add_argument("--input", "-i", required=True, help="Input video path")
    p.add_argument("--output", "-o", required=True, help="Output video path")
    p.add_argument("--cutoff", type=float, default=3.0, help="Deflicker low-pass cutoff Hz")
    p.add_argument("--order", type=int, default=6, help="Butterworth filter order")
    p.add_argument("--fps", default="auto", help='FPS, "auto" or a number')
    p.add_argument("--notch", type=int, default=2, help="Vertical k half-width to suppress (bins)")
    p.add_argument("--soften", type=int, default=6, help="Feather width around notch (bins)")
    p.add_argument("--keep-dc", action="store_true", help="Preserve DC coefficient exactly")
    p.add_argument("--channel", choices=["r","g","b","gray"], default="g", help="Which channel to use")
    p.add_argument("--codec", default="FFV1", help="FourCC codec (e.g., FFV1, MJPG, XVID)")
    p.add_argument("--display", action="store_true", help="Display preview window")
    p.add_argument("--every", type=int, default=1, help="Process/save every Nth frame")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p.parse_args()

def fourcc_from_string(s: str) -> int:
    s = (s or "FFV1")
    if len(s) == 4:
        return cv2.VideoWriter_fourcc(*s)
    # try named constants
    return cv2.VideoWriter_fourcc(*"FFV1")

def make_vertical_notch_mask(rows:int, cols:int, notch:int=2, soften:int=6, keep_dc:bool=True) -> np.ndarray:
    """
    Build a multiplicative real-valued mask in frequency domain (after fftshift),
    that zeros a horizontal strip around ky=0 (i.e., removes horizontal bands).
    - notch: |ky| <= notch is set to 0 (full suppression)
    - soften: smooth ramp from 0 to 1 for notch < |ky| <= notch+soften
    - keep_dc: set the single DC bin back to 1 to preserve mean brightness
    Returns shape (rows, cols, 2) to multiply an OpenCV complex DFT.
    """
    cy, cx = rows // 2, cols // 2
    # vertical 1D weights
    wv = np.ones(rows, dtype=np.float32)
    for y in range(rows):
        dv = abs(y - cy)
        if dv <= notch:
            w = 0.0
        elif dv <= notch + soften and soften > 0:
            # smooth cosine ramp 0 -> 1
            t = (dv - notch) / float(soften)
            w = 0.5 * (1 - cos(pi * t))
        else:
            w = 1.0
        wv[y] = w
    mask2d = np.repeat(wv[:, None], cols, axis=1)
    if keep_dc:
        mask2d[cy, cx] = 1.0  # preserve exact DC
    # expand to 2 channels
    mask2 = np.dstack([mask2d, mask2d]).astype(np.float32)
    return mask2

def get_channel(frame: np.ndarray, which: str) -> np.ndarray:
    if frame.ndim == 2 or which == "gray":
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # BGR to channel
    if which == "b":
        return frame[:,:,0]
    if which == "g":
        return frame[:,:,1]
    if which == "r":
        return frame[:,:,2]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def collect_mean_trace(path_in, channel, maskFFT, step, verbose=False):
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path_in}")
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    means = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        g = get_channel(frame, channel).astype(np.float32)
        # DFT (scaled) -> notch -> inverse
        dft = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * maskFFT
        img_back = cv2.idft(np.fft.ifftshift(fshift))
        mag = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        means.append(float(mag.mean()))
        idx += 1
    cap.release()
    return np.array(means, dtype=np.float32)

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open input video", file=sys.stderr)
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if args.fps == "auto":
        fps = fps_in if fps_in and fps_in > 0 else 30.0
    else:
        try:
            fps = float(args.fps)
        except:
            fps = 30.0
    if args.verbose:
        print(f"Video: {cols}x{rows} @ {fps:.3f} Hz, frames≈{total}")

    # Build notch mask (once)
    maskFFT = make_vertical_notch_mask(rows, cols, notch=args.notch, soften=args.soften, keep_dc=args.keep_dc)

    # Pass 1: mean trace after spatial FFT cleaning
    if args.verbose:
        print("Collecting mean intensity trace...")
    meanFrame = collect_mean_trace(args.input, args.channel, maskFFT, step=1, verbose=args.verbose)
    if len(meanFrame) < 4:
        print("Too few frames to process.", file=sys.stderr)
        sys.exit(1)

    # Temporal low-pass filter for fast deflicker
    b, a = butter(args.order, args.cutoff / (0.5 * fps), btype="low", analog=False)
    meanFiltered = filtfilt(b, a, meanFrame)

    # Pass 2: apply FFT notch + temporal scaling, write out
    if args.verbose:
        print("Writing output...")
    fourcc = fourcc_from_string(args.codec)
    # grayscale (False) writer
    out = cv2.VideoWriter(args.output, fourcc, fps, (cols, rows), isColor=False)
    if not out.isOpened():
        print("Could not open output writer", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.every != 0:
            idx += 1
            continue
        g = get_channel(frame, args.channel).astype(np.float32)

        # Spatial notch
        dft = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * maskFFT
        img_back = cv2.idft(np.fft.ifftshift(fshift))
        mag = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # Temporal deflicker: scale to match filtered mean
        mf = float(mag.mean())
        target = float(meanFiltered[idx])
        # scale factor preserves brightness
        if mf > 1e-6:
            mag = mag * (1.0 + (target - mf)/mf)

        # Clip to 8-bit
        mag[mag > 255.0] = 255.0
        mag[mag < 0.0] = 0.0
        out_frame = mag.astype(np.uint8)

        out.write(out_frame)

        if args.display:
            # Build quick visualization: raw gray, filtered, difference
            raw = get_channel(frame, args.channel)
            raw8 = raw.astype(np.uint8) if raw.dtype != np.uint8 else raw
            diff = 128.0 + (raw8.astype(np.float32) - out_frame.astype(np.float32)) * 2.0
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            vis = cv2.hconcat([raw8, out_frame, diff])
            cv2.imshow("Raw | Filtered | Difference", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        idx += 1

    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()

    if args.verbose:
        print(f"Done. Wrote: {args.output}")

if __name__ == "__main__":
    main()
