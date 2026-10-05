import cv2
import numpy as np

inp = "/media/toor/Seagate2/Mouse847/20251110T172458/trimmed_video.avi"          # your original
outp = "/media/toor/Seagate2/Mouse847/20251110T172458/trimmed_video_denoised4.avi"



cap = cv2.VideoCapture(inp)
if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Pass 1: collect row means over time (no frames stored)
row_means_all = []  # list of (h,) arrays

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame

    g = g.astype(np.float32)
    row_means_all.append(g.mean(axis=1))  # (h,)

cap.release()

row_means_all = np.stack(row_means_all, axis=0)  # shape: (T, h)
T = row_means_all.shape[0]
print(f"Frames counted: {T}")

# Robust baseline per row (median across time)
baseline_row = np.median(row_means_all, axis=0)  # (h,)

# Optional: smooth baseline a bit along vertical axis, to avoid weird kinks
def smooth_1d(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid")

nine = 9  # small, gentle smoothing window
baseline_row = smooth_1d(baseline_row, k=nine)


# Reopen video for Pass 2
cap = cv2.VideoCapture(inp)
if not cap.isOpened():
    raise RuntimeError("Could not reopen input video for pass 2")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # robust codec
out = cv2.VideoWriter(outp, fourcc, fps, (w, h), False)
if not out.isOpened():
    raise RuntimeError("Could not open output writer")

strength = 0.8        # reduce banding strongly but not fully
smooth_k = 7          # smooth per-frame row deviation to avoid jitter (odd number)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame

    g = g.astype(np.float32)

    # Per-frame row mean
    row_mean = g.mean(axis=1)  # (h,)

    # Deviation from stable baseline
    dev = row_mean - baseline_row  # banding estimate for this frame

    # Smooth deviation along rows, so we only capture broad bands
    dev_s = smooth_1d(dev, smooth_k)

    # Enforce zero-mean correction so we do not change global brightness
    dev_s = dev_s - dev_s.mean()

    # Scale
    row_bias = dev_s * strength  # (h,)

    # Broadcast to full frame
    bias_img = row_bias[:, None]

    # Subtract
    corrected = g - bias_img
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Safety
    assert corrected.shape == (h, w)
    out.write(corrected)
    frame_idx += 1

cap.release()
out.release()
print(f"Done. Wrote {frame_idx} frames to {outp}")