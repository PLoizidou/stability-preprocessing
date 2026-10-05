#!/bin/bash
# Run preproc_caiman_final.py once with an explicit --run_name and arbitrary
# extra CLI args, skipping if that run already has results. Used for one-off
# parameter variants (e.g. nb or min_pnr) that don't fit the gSig/gSig_filt
# pair-sweep shape of run_caiman_param_sweep.sh.
#
# Usage:
#   ./run_caiman_param_variant.sh /path/to/clip.avi caiman_final_g7_f10_nb1 --gSig 7 7 --gSig_filt 10 10 --gnb 1

INPUT_PATH="$1"
RUN_NAME="$2"
shift 2

if [ -z "$INPUT_PATH" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 /path/to/clip.avi run_name [extra preproc_caiman_final.py args...]"
    exit 1
fi

SCRIPT_DIR="/home/toor/Desktop/stability-preprocessing/scripts"
ENV_NAME="stability-preprocessing"
OUT_DIR="$(dirname "$INPUT_PATH")"
LOG_DIR="$OUT_DIR/sweep_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

if [ -f "$OUT_DIR/$RUN_NAME/caiman_results.hdf5" ]; then
    echo "=== $RUN_NAME already exists, skipping ==="
    exit 0
fi

echo "=== Running $RUN_NAME (args: $*) -> $LOG_FILE ==="
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/preproc_caiman_final.py" \
    --input_path "$INPUT_PATH" \
    --run_name "$RUN_NAME" \
    "$@" \
    > "$LOG_FILE" 2>&1
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "=== $RUN_NAME done ==="
else
    echo "=== $RUN_NAME FAILED (exit $STATUS) — see $LOG_FILE ==="
fi
exit $STATUS
