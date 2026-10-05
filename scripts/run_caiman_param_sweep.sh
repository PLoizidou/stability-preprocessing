#!/bin/bash
# Run preproc_caiman_final.py once per gSig/gSig_filt pair on a single test clip,
# writing each run to its own caiman_final_g<gSig>_f<gSig_filt>/ subdirectory
# alongside the clip (matches the young-mice test_cropping sweep layout).
#
# Usage:
#   ./run_caiman_param_sweep.sh /path/to/clip.avi "5,8 5,10 7,10 7,12" [extra preproc_caiman_final.py args...]
#
# Each pair is "gSig,gSig_filt". Any trailing arguments are passed through
# unchanged to every run (e.g. --gnb 1, --min_pnr 4.5, --rf 48).

INPUT_PATH="$1"
PAIRS="$2"
shift 2 || true
EXTRA_ARGS=("$@")

if [ -z "$INPUT_PATH" ] || [ -z "$PAIRS" ]; then
    echo "Usage: $0 /path/to/clip.avi \"gSig,gSigFilt gSig,gSigFilt ...\" [extra args...]"
    exit 1
fi

SCRIPT_DIR="/home/toor/Desktop/stability-preprocessing/scripts"
ENV_NAME="stability-preprocessing"
OUT_DIR="$(dirname "$INPUT_PATH")"
LOG_DIR="$OUT_DIR/sweep_logs"
mkdir -p "$LOG_DIR"

echo "Input clip:  $INPUT_PATH"
echo "Output root: $OUT_DIR"
echo "Pairs:       $PAIRS"
echo "Extra args:  ${EXTRA_ARGS[*]}"
echo

for PAIR in $PAIRS; do
    GSIG="${PAIR%,*}"
    GFILT="${PAIR#*,}"
    RUN_NAME="caiman_final_g${GSIG}_f${GFILT}"
    LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

    if [ -f "$OUT_DIR/$RUN_NAME/caiman_results.hdf5" ]; then
        echo "=== $RUN_NAME already exists, skipping ==="
        continue
    fi

    echo "=== Running $RUN_NAME (gSig=$GSIG gSig_filt=$GFILT) -> $LOG_FILE ==="
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/preproc_caiman_final.py" \
        --input_path "$INPUT_PATH" \
        --gSig "$GSIG" "$GSIG" \
        --gSig_filt "$GFILT" "$GFILT" \
        --run_name "$RUN_NAME" \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "=== $RUN_NAME done ==="
    else
        echo "=== $RUN_NAME FAILED (exit $STATUS) — see $LOG_FILE ==="
    fi
done

echo
echo "Sweep complete."
