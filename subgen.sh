#!/bin/bash
#
# Title: WSL Batch Video Transcriber (Whisper.cpp Pipeline)
# Description: Automates the batch transcription of videos using ffmpeg and whisper.cpp.
#              Optimized for performance within the Windows Subsystem for Linux (WSL)
#              environment, handling cross-OS paths and file cleanup.
#
# PROJECT STRUCTURE:
#   subgen/
#   ├── subgen.sh              <- This script
#   └── whisper.cpp/           <- Git submodule (https://github.com/ggml-org/whisper.cpp)
#       ├── build/bin/         <- Compiled whisper-cli executable
#       └── models/            <- Model files (.bin) and download scripts
#
# USAGE:
# 1. Init the submodule:  git submodule update --init --recursive
# 2. Build whisper-cli:   cd whisper.cpp && cmake -B build -DGGML_CUDA=1 && cmake --build build -j$(nproc)
# 3. Install ffmpeg:      sudo apt install ffmpeg
# 4. Download the model:  cd whisper.cpp/models && bash download-ggml-model.sh large-v3-q5_0
# 5. Download VAD model:  cd whisper.cpp/models && bash download-vad-model.sh silero-v5.1.2
# 6. Run:                 ./subgen.sh "/mnt/c/Users/YourName/Videos/ProjectFolder"

set -euo pipefail

# ---------------------------------------------------------------------------
# PRETTY CLI COLORS
# ---------------------------------------------------------------------------
BOLD=$(tput bold)
DIM=$(tput dim 2>/dev/null || echo "")
BLUE=$(tput setaf 4)
CYAN=$(tput setaf 6)
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
MAGENTA=$(tput setaf 5)
RESET=$(tput sgr0)

# ---------------------------------------------------------------------------
# ROBUST PATHING
# ---------------------------------------------------------------------------
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WHISPER_SUBMODULE_DIR="$SCRIPT_DIR/whisper.cpp"
WHISPER_EXECUTABLE="$WHISPER_SUBMODULE_DIR/build/bin/whisper-cli"

# ---------------------------------------------------------------------------
# MODEL, LANGUAGE, AND TASK CONFIGURATION
# ---------------------------------------------------------------------------
WHISPER_MODEL_NAME="large-v3-q5_0"
LANGUAGE="en"
TASK="transcribe"
WHISPER_MODEL="$WHISPER_SUBMODULE_DIR/models/ggml-$WHISPER_MODEL_NAME.bin"

# ---------------------------------------------------------------------------
# GPU CONFIGURATION
# ---------------------------------------------------------------------------
USE_CUDA=true
NVIDIA_GPU_INDEX=0
GPU_FEED_THREADS=8

# ---------------------------------------------------------------------------
# VAD (VOICE ACTIVITY DETECTION) CONFIGURATION
# ---------------------------------------------------------------------------
USE_VAD=true
VAD_MODEL_NAME="ggml-silero-v5.1.2.bin"
VAD_MODEL_PATH="$WHISPER_SUBMODULE_DIR/models/$VAD_MODEL_NAME"

# ---------------------------------------------------------------------------
# STATE & ARGUMENT VALIDATION
# ---------------------------------------------------------------------------
NUM_THREADS=$(nproc)

if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 \"/path/to/windows/video/folder\"${RESET}" >&2
    echo -e "${YELLOW}Example: $0 \"/mnt/c/Users/User/Videos/Client Project\"${RESET}" >&2
    exit 1
fi

INPUT_DIR="$1"
ERROR_LOG_FILE="$INPUT_DIR/transcription_errors.log"
TEMP_DIR="/tmp/whisper_pipeline_$(date +%s)"
mkdir -p "$TEMP_DIR"

# Batch-level counters and timer
BATCH_START_TIME=$(date +%s)
FILES_OK=0
FILES_FAILED=0
FILES_SKIPPED=0

# GPU verification: set after first file confirms CUDA is active
GPU_VERIFIED=false
VRAM_BASELINE_MB=0

# ---------------------------------------------------------------------------
# HELPERS: UI primitives
# ---------------------------------------------------------------------------

# Print a styled section header
section() {
    local title="$1"
    echo -e "\n${BLUE}${BOLD}══ ${title} ${RESET}"
}

# Print a key/value info line
info() {
    local key="$1"
    local val="$2"
    printf "   ${CYAN}%-22s${RESET} %s\n" "$key" "$val"
}

# Status badges
ok()   { echo -e "   ${GREEN}✔ ${1}${RESET}"; }
warn() { echo -e "   ${YELLOW}⚠ ${1}${RESET}"; }
err()  { echo -e "   ${RED}✘ ${1}${RESET}" >&2; }
hint() { echo -e "   ${DIM}→ ${1}${RESET}"; }

# ---------------------------------------------------------------------------
# GRACEFUL EXIT TRAP
# ---------------------------------------------------------------------------
function cleanup() {
    local exit_code=$?

    echo ""
    if [ "$exit_code" -eq 130 ]; then
        warn "Interrupted by user (Ctrl+C). Cleaning up..."
    elif [ "$exit_code" -ne 0 ]; then
        err "Script exited with an error (code: $exit_code)."
    fi

    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        hint "Temporary directory cleaned up."
    fi

    echo -n "${RESET}"
    exit $exit_code
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# ERROR LOG FUNCTION
# ---------------------------------------------------------------------------
function log_error() {
    local file_path="$1"
    local error_msg="$2"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %T")
    echo "[$timestamp] FAILED: $file_path" >> "$ERROR_LOG_FILE"
    echo "       REASON: $error_msg"       >> "$ERROR_LOG_FILE"
    echo "       ---------------------------------" >> "$ERROR_LOG_FILE"
}

# ---------------------------------------------------------------------------
# PROGRESS BAR: parse "progress = N%" from whisper log + live VRAM from nvidia-smi
# ---------------------------------------------------------------------------
# Usage: show_progress_bar <whisper_log_file> <pid_to_wait_for>
# Runs in the FOREGROUND, tailing the log until the PID exits.
function show_progress_bar() {
    local log_file="$1"
    local pid="$2"
    local bar_width=30
    local pct=0
    local last_draw=""
    local vram_used="--"
    local vram_poll_counter=0

    # Wait for log file to appear
    local waited=0
    while [ ! -f "$log_file" ] && kill -0 "$pid" 2>/dev/null; do
        sleep 0.2
        waited=$((waited + 1))
        [ $waited -gt 25 ] && break
    done

    printf "\n   "

    while kill -0 "$pid" 2>/dev/null; do
        if [ -f "$log_file" ]; then
            local line
            line=$(grep -oP 'progress\s*=\s*\K[0-9]+' "$log_file" 2>/dev/null | tail -1 || true)
            if [[ -n "$line" ]]; then
                pct=$line
            fi
        fi

        # Poll VRAM every ~1.5s (every 5 iterations of 0.3s sleep)
        vram_poll_counter=$(( vram_poll_counter + 1 ))
        if [ "$USE_CUDA" = true ] && command -v nvidia-smi &>/dev/null && [ $(( vram_poll_counter % 5 )) -eq 0 ]; then
            local v
            v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$NVIDIA_GPU_INDEX 2>/dev/null | xargs || true)
            if [[ -n "$v" ]]; then
                vram_used="${v}"
            fi
        fi

        local filled=$(( pct * bar_width / 100 ))
        local empty=$(( bar_width - filled ))
        local bar_filled bar_empty
        bar_filled=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) 2>/dev/null || true)
        bar_empty=$(printf '░%.0s' $(seq 1 $empty 2>/dev/null) 2>/dev/null || true)
        local draw="${pct}|${vram_used}"

        if [ "$draw" != "$last_draw" ]; then
            if [ "$USE_CUDA" = true ]; then
                printf "\r   [${GREEN}%s${DIM}%s${RESET}] ${BOLD}%3d%%${RESET}  ${MAGENTA}VRAM: %s/${GPU_VRAM} MiB${RESET}   " \
                    "$bar_filled" "$bar_empty" "$pct" "$vram_used"
            else
                printf "\r   [${GREEN}%s${DIM}%s${RESET}] ${BOLD}%3d%%${RESET}  " \
                    "$bar_filled" "$bar_empty" "$pct"
            fi
            last_draw="$draw"
        fi
        sleep 0.3
    done

    # Final: ensure 100% is shown on success
    wait "$pid" 2>/dev/null && true
    local final_exit=$?

    if [ "$final_exit" -eq 0 ]; then
        local bar
        bar=$(printf '█%.0s' $(seq 1 $bar_width))
        if [ "$USE_CUDA" = true ]; then
            printf "\r   [${GREEN}%s${RESET}] ${BOLD}%3d%%${RESET}  ${MAGENTA}VRAM: %s/${GPU_VRAM} MiB${RESET}   \n" \
                "$bar" "100" "$vram_used"
        else
            printf "\r   [${GREEN}%s${RESET}] ${BOLD}%3d%%${RESET}  \n" "$bar" "100"
        fi
    else
        printf "\n"
    fi

    return $final_exit
}

# ---------------------------------------------------------------------------
# GPU DIAGNOSTICS: one-time proof after first successful transcription
# ---------------------------------------------------------------------------
# Parses the whisper log to extract: backend, VRAM loaded, encode speed,
# and computes a real-time speed ratio to prove GPU is active.
function gpu_diagnostics() {
    local log_file="$1"
    local audio_seconds="$2"   # duration of the WAV in seconds

    echo ""
    echo -e "   ${MAGENTA}${BOLD}┌── GPU PROOF (first-file diagnostic) ──────────────────────┐${RESET}"

    # 1. Backend confirmation
    local backend
    backend=$(grep -oP 'whisper_backend_init_gpu: using \K.*' "$log_file" 2>/dev/null | head -1 || true)
    if [[ -n "$backend" ]]; then
        echo -e "   ${MAGENTA}│${RESET}  ${GREEN}✔${RESET} Backend:       ${BOLD}$backend${RESET}"
    else
        echo -e "   ${MAGENTA}│${RESET}  ${RED}✘${RESET} Backend:       ${RED}No CUDA backend found in log!${RESET}"
    fi

    # 2. Model VRAM loaded
    local model_vram
    model_vram=$(grep -oP 'CUDA0 total size\s*=\s*\K[0-9.]+' "$log_file" 2>/dev/null | head -1 || true)
    if [[ -n "$model_vram" ]]; then
        echo -e "   ${MAGENTA}│${RESET}  ${GREEN}✔${RESET} Model in VRAM: ${BOLD}${model_vram} MB${RESET} loaded to GPU"
    fi

    # 3. Encode speed (the killer metric)
    local encode_per_run
    encode_per_run=$(grep -oP 'encode time\s*=.*?\(\s*\K[0-9.]+(?=\s*ms per run)' "$log_file" 2>/dev/null | head -1 || true)
    if [[ -n "$encode_per_run" ]]; then
        echo -e "   ${MAGENTA}│${RESET}  ${GREEN}✔${RESET} Encode speed:  ${BOLD}${encode_per_run} ms/pass${RESET}  (CPU would be ~3000-5000 ms)"
    fi

    # 4. Real-time speed ratio
    local total_ms
    total_ms=$(grep -oP 'total time\s*=\s*\K[0-9.]+' "$log_file" 2>/dev/null | head -1 || true)
    if [[ -n "$total_ms" ]] && [[ -n "$audio_seconds" ]]; then
        local total_sec
        total_sec=$(awk "BEGIN {printf \"%.1f\", $total_ms / 1000}")
        local ratio
        ratio=$(awk "BEGIN {printf \"%.1f\", $audio_seconds / ($total_ms / 1000)}")
        echo -e "   ${MAGENTA}│${RESET}  ${GREEN}✔${RESET} Speed ratio:   ${BOLD}${ratio}× real-time${RESET}  (${audio_seconds}s audio → ${total_sec}s wall)"
        # Verdict
        local ratio_int
        ratio_int=$(awk "BEGIN {printf \"%d\", $audio_seconds / ($total_ms / 1000)}")
        if [ "$ratio_int" -ge 5 ]; then
            echo -e "   ${MAGENTA}│${RESET}"
            echo -e "   ${MAGENTA}│${RESET}  ${GREEN}${BOLD}   ✓ VERDICT: GPU is confirmed active.${RESET}"
            echo -e "   ${MAGENTA}│${RESET}  ${DIM}   (>5× real-time on large-v3-q5_0 is impossible on CPU)${RESET}"
        else
            echo -e "   ${MAGENTA}│${RESET}"
            echo -e "   ${MAGENTA}│${RESET}  ${YELLOW}${BOLD}   ⚠ VERDICT: Speed is suspiciously low.${RESET}"
            echo -e "   ${MAGENTA}│${RESET}  ${YELLOW}   This may indicate GPU is NOT being used.${RESET}"
            echo -e "   ${MAGENTA}│${RESET}  ${YELLOW}   Expected >5× for GPU; got ${ratio}×.${RESET}"
        fi
    fi

    # 5. Live VRAM delta
    if [ "$USE_CUDA" = true ] && command -v nvidia-smi &>/dev/null; then
        local current_vram
        current_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$NVIDIA_GPU_INDEX 2>/dev/null | xargs || true)
        if [[ -n "$current_vram" ]] && [ "$VRAM_BASELINE_MB" -gt 0 ]; then
            local delta=$(( current_vram - VRAM_BASELINE_MB ))
            echo -e "   ${MAGENTA}│${RESET}  ${GREEN}✔${RESET} VRAM delta:    ${BOLD}+${delta} MiB${RESET} above idle baseline (${VRAM_BASELINE_MB} → ${current_vram} MiB)"
        fi
    fi

    echo -e "   ${MAGENTA}${BOLD}└───────────────────────────────────────────────────────────┘${RESET}"
    echo ""
}

# ---------------------------------------------------------------------------
# DEPENDENCY CHECKS
# ---------------------------------------------------------------------------
section "DEPENDENCY CHECKS"

if [ ! -f "$WHISPER_EXECUTABLE" ]; then
    err "whisper-cli not found at: $WHISPER_EXECUTABLE"
    hint "Build it: cd whisper.cpp && cmake -B build -DGGML_CUDA=1 && cmake --build build -j\$(nproc)"
    exit 1
fi
ok "whisper-cli   $(basename "$WHISPER_EXECUTABLE")"

if ! command -v ffmpeg &> /dev/null; then
    err "ffmpeg is not installed."
    hint "Fix: sudo apt install ffmpeg"
    exit 1
fi
ok "ffmpeg        $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

if [ ! -f "$WHISPER_MODEL" ]; then
    err "Model '$WHISPER_MODEL_NAME' not found at: $WHISPER_MODEL"
    hint "Download: cd whisper.cpp/models && bash download-ggml-model.sh $WHISPER_MODEL_NAME"
    exit 1
fi
ok "model         $WHISPER_MODEL_NAME"

if [ "$USE_VAD" = true ]; then
    if [ ! -f "$VAD_MODEL_PATH" ]; then
        err "VAD model not found at: $VAD_MODEL_PATH"
        hint "Download: cd whisper.cpp/models && bash download-vad-model.sh silero-v5.1.2"
        exit 1
    fi
    ok "VAD model     $VAD_MODEL_NAME"
fi

# ---------------------------------------------------------------------------
# GPU VALIDATION
# ---------------------------------------------------------------------------
GPU_NAME=""
if [ "$USE_CUDA" = true ]; then
    section "GPU VALIDATION"
    if command -v nvidia-smi &> /dev/null; then
        set +e
        GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total,driver_version \
                   --format=csv,noheader,nounits --id=$NVIDIA_GPU_INDEX 2>/dev/null)
        NVIDIA_SMI_EXIT_CODE=$?
        set -e

        GPU_MODEL=$(echo "$GPU_NAME" | cut -d',' -f1 | xargs)
        GPU_VRAM=$(echo  "$GPU_NAME" | cut -d',' -f2 | xargs)
        GPU_DRIVER=$(echo "$GPU_NAME" | cut -d',' -f3 | xargs)

        if [ $NVIDIA_SMI_EXIT_CODE -eq 0 ] && [[ "$GPU_MODEL" == *"NVIDIA"* ]]; then
            # Capture idle VRAM baseline for later delta comparison
            VRAM_BASELINE_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$NVIDIA_GPU_INDEX 2>/dev/null | xargs || echo "0")
            ok "GPU detected"
            info "Device [${NVIDIA_GPU_INDEX}]:" "$GPU_MODEL"
            info "VRAM (total):" "${GPU_VRAM} MiB"
            info "VRAM (idle):" "${VRAM_BASELINE_MB} MiB"
            info "Driver:" "$GPU_DRIVER"
            hint "CUDA_VISIBLE_DEVICES=${NVIDIA_GPU_INDEX} for all transcriptions."
        elif [ $NVIDIA_SMI_EXIT_CODE -eq 0 ]; then
            err "GPU at index $NVIDIA_GPU_INDEX is not an NVIDIA card: '$GPU_MODEL'"
            hint "Check your NVIDIA_GPU_INDEX in this script or your WSL/CUDA setup."
            exit 1
        else
            err "nvidia-smi failed to query GPU at index $NVIDIA_GPU_INDEX."
            hint "Run 'nvidia-smi' manually to diagnose."
            exit 1
        fi
    else
        warn "nvidia-smi not found. Cannot pre-validate GPU."
        hint "CUDA may still work, but we cannot confirm the correct device."
    fi
fi

# ---------------------------------------------------------------------------
# PIPELINE CONFIGURATION SUMMARY (printed once)
# ---------------------------------------------------------------------------
section "PIPELINE CONFIGURATION"
info "Input directory:"   "$INPUT_DIR"
info "Model:"             "$WHISPER_MODEL_NAME"
info "Language:"          "$LANGUAGE"
info "Task:"              "$TASK"
if [ "$USE_CUDA" = true ]; then
    info "Acceleration:"  "CUDA (GPU ${NVIDIA_GPU_INDEX}) — ${GPU_FEED_THREADS} feeder threads"
else
    info "Acceleration:"  "CPU only — ${NUM_THREADS} threads"
fi
info "VAD:"               "$([ "$USE_VAD" = true ] && echo "enabled (${VAD_MODEL_NAME})" || echo "disabled")"
info "Temp directory:"    "$TEMP_DIR"
info "Error log:"         "$ERROR_LOG_FILE"

# ---------------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------------
mapfile -t VIDEO_FILES < <(
    find "$INPUT_DIR" -type f \( \
        -iname "*.mp4" -o \
        -iname "*.mkv" -o \
        -iname "*.avi" -o \
        -iname "*.webm" -o \
        -iname "*.ts"  -o \
        -iname "*.mov" \
    \)
)

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    err "No video files found in: $INPUT_DIR"
    hint "Supported extensions (case-insensitive): mp4, mkv, avi, webm, ts, mov"
    exit 1
fi

info "Files found:"       "${#VIDEO_FILES[@]}"

# ---------------------------------------------------------------------------
# MAIN BATCH PROCESSING LOOP
# ---------------------------------------------------------------------------
TOTAL_FILES=${#VIDEO_FILES[@]}
FILE_COUNT=0

for VIDEO_PATH in "${VIDEO_FILES[@]}"; do
    FILE_COUNT=$((FILE_COUNT + 1))

    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    VIDEO_BASENAME="${VIDEO_FILENAME%.*}"
    VIDEO_DIR=$(dirname "$VIDEO_PATH")

    TEMP_WAV_PATH="$TEMP_DIR/$VIDEO_BASENAME.wav"
    FINAL_SRT_PATH="$VIDEO_DIR/$VIDEO_BASENAME.srt"
    TEMP_SRT_BASE_PATH="$TEMP_DIR/$VIDEO_BASENAME"
    EXPECTED_SRT_PATH="$TEMP_SRT_BASE_PATH.srt"

    FFMPEG_LOG="$TEMP_DIR/$VIDEO_BASENAME.ffmpeg.log"
    WHISPER_LOG="$TEMP_DIR/$VIDEO_BASENAME.whisper.log"

    FILE_START_TIME=$(date +%s)

    echo ""
    echo -e "${BOLD}[${FILE_COUNT}/${TOTAL_FILES}]${RESET} ${CYAN}${VIDEO_FILENAME}${RESET}"
    echo -e "   ${DIM}──────────────────────────────────────────────────${RESET}"

    # ------------------------------------------------------------------
    # RESUME: skip if SRT already exists
    # ------------------------------------------------------------------
    if [ -f "$FINAL_SRT_PATH" ]; then
        warn "Already done — SRT exists, skipping."
        FILES_SKIPPED=$((FILES_SKIPPED + 1))
        continue
    fi

    # ------------------------------------------------------------------
    # PHASE 1: Audio extraction
    # ------------------------------------------------------------------
    printf "   ${BLUE}Phase 1${RESET}  Extracting audio...    "

    if ! ffmpeg -i "$VIDEO_PATH" -vn \
        -acodec pcm_s16le \
        -ar 16000 \
        -ac 1 \
        -y "$TEMP_WAV_PATH" -loglevel error 2> "$FFMPEG_LOG"; then

        exit_code=$?
        printf "\n"
        if [ "$exit_code" -eq 130 ]; then
            warn "Interrupted during audio extraction."
            rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG"
            exit 130
        fi
        err "ffmpeg failed (code: $exit_code). See log for details."
        if [ -s "$FFMPEG_LOG" ]; then
            sed 's/^/             /' "$FFMPEG_LOG" >&2
        fi
        log_error "$VIDEO_PATH" "FFmpeg failed (code: $exit_code)."
        rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG"
        FILES_FAILED=$((FILES_FAILED + 1))
        continue
    fi

    if [ ! -s "$TEMP_WAV_PATH" ]; then
        printf "\n"
        err "FFmpeg succeeded but produced an empty WAV file."
        hint "The video likely has no audio track."
        log_error "$VIDEO_PATH" "FFmpeg produced a zero-byte WAV file."
        rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG"
        FILES_FAILED=$((FILES_FAILED + 1))
        continue
    fi

    # Show a short duration summary inline and stash raw seconds for GPU diagnostics
    WAV_DURATION_RAW=$(ffprobe -v error -show_entries format=duration \
                       -of default=noprint_wrappers=1:nokey=1 "$TEMP_WAV_PATH" 2>/dev/null || echo "0")
    WAV_DURATION=$(echo "$WAV_DURATION_RAW" | awk '{printf "%dm%02ds", int($1/60), int($1)%60}' || echo "?")
    WAV_DURATION_INT=$(echo "$WAV_DURATION_RAW" | awk '{printf "%d", int($1)}' || echo "0")
    rm -f "$FFMPEG_LOG"
    echo -e "${GREEN}✔${RESET}  (${WAV_DURATION})"

    # ------------------------------------------------------------------
    # PHASE 2: Transcription
    # ------------------------------------------------------------------
    printf "   ${BLUE}Phase 2${RESET}  Transcribing...        "

    # Build whisper args
    WHISPER_CMD_ARGS=(
        -m "$WHISPER_MODEL"
        -f "$TEMP_WAV_PATH"
        -l "$LANGUAGE"
    )

    if [ "$USE_CUDA" = true ]; then
        export CUDA_VISIBLE_DEVICES=$NVIDIA_GPU_INDEX
        WHISPER_CMD_ARGS+=( -t "$GPU_FEED_THREADS" )
    else
        WHISPER_CMD_ARGS+=( -t "$NUM_THREADS" -ng )
    fi

    if [ "$TASK" = "translate" ]; then
        WHISPER_CMD_ARGS+=( -tr )
    fi

    if [ "$USE_VAD" = true ]; then
        WHISPER_CMD_ARGS+=( --vad -vm "$VAD_MODEL_PATH" )
    fi

    WHISPER_CMD_ARGS+=( -osrt -of "$TEMP_SRT_BASE_PATH" -pp )

    # Run whisper in background, redirect ALL output to log, show progress bar
    rm -f "$WHISPER_LOG"
    "$WHISPER_EXECUTABLE" "${WHISPER_CMD_ARGS[@]}" >> "$WHISPER_LOG" 2>&1 &
    WHISPER_PID=$!

    # show_progress_bar tails the log file for "progress = N%" lines.
    # It returns the exit code of the background process.
    set +e
    show_progress_bar "$WHISPER_LOG" "$WHISPER_PID"
    WHISPER_EXIT=$?
    set -e

    if [ "$WHISPER_EXIT" -eq 130 ]; then
        printf "\n"
        warn "Interrupted during transcription."
        rm -f "$WHISPER_LOG" "$TEMP_WAV_PATH"
        exit 130
    fi

    if [ "$WHISPER_EXIT" -ne 0 ]; then
        # --- AUTO-RETRY WITHOUT VAD ---
        if [ "$USE_VAD" = true ]; then
            printf "   ${YELLOW}⚠ Phase 2${RESET}  VAD crash — retrying without VAD...  "

            WHISPER_CMD_ARGS_RETRY=(
                -m "$WHISPER_MODEL"
                -f "$TEMP_WAV_PATH"
                -l "$LANGUAGE"
            )
            if [ "$USE_CUDA" = true ]; then
                export CUDA_VISIBLE_DEVICES=$NVIDIA_GPU_INDEX
                WHISPER_CMD_ARGS_RETRY+=( -t "$GPU_FEED_THREADS" )
            else
                WHISPER_CMD_ARGS_RETRY+=( -t "$NUM_THREADS" -ng )
            fi
            if [ "$TASK" = "translate" ]; then
                WHISPER_CMD_ARGS_RETRY+=( -tr )
            fi
            WHISPER_CMD_ARGS_RETRY+=( -osrt -of "$TEMP_SRT_BASE_PATH" -pp )

            rm -f "$WHISPER_LOG"
            "$WHISPER_EXECUTABLE" "${WHISPER_CMD_ARGS_RETRY[@]}" >> "$WHISPER_LOG" 2>&1 &
            RETRY_PID=$!

            set +e
            show_progress_bar "$WHISPER_LOG" "$RETRY_PID"
            RETRY_EXIT=$?
            set -e

            if [ "$RETRY_EXIT" -eq 130 ]; then
                printf "\n"
                warn "Interrupted during retry."
                rm -f "$WHISPER_LOG" "$TEMP_WAV_PATH"
                exit 130
            fi

            if [ "$RETRY_EXIT" -ne 0 ]; then
                err "Transcription failed (VAD off retry also failed, code: $RETRY_EXIT)."
                log_error "$VIDEO_PATH" "Whisper failed on both VAD and non-VAD attempts (code: $RETRY_EXIT)."
                hint "Temp WAV kept for debugging: $TEMP_WAV_PATH"
                rm -f "$WHISPER_LOG"
                FILES_FAILED=$((FILES_FAILED + 1))
                continue
            fi
            hint "Retry without VAD succeeded."
        else
            err "Transcription failed (code: $WHISPER_EXIT)."
            log_error "$VIDEO_PATH" "Whisper failed with VAD disabled (code: $WHISPER_EXIT)."
            hint "Temp WAV kept for debugging: $TEMP_WAV_PATH"
            rm -f "$WHISPER_LOG"
            FILES_FAILED=$((FILES_FAILED + 1))
            continue
        fi
    fi

    # ------------------------------------------------------------------
    # CUDA VALIDATION (from log — not shown to user unless there's a problem)
    # ------------------------------------------------------------------
    if [ "$USE_CUDA" = true ]; then
        if grep -q "failed to initialize CUDA" "$WHISPER_LOG" 2>/dev/null; then
            err "CUDA runtime failure — driver version insufficient."
            hint "Update your NVIDIA drivers on Windows, reboot, and retry."
            rm -f "$WHISPER_LOG"
            exit 1
        elif ! grep -q "ggml_cuda_init" "$WHISPER_LOG" 2>/dev/null; then
            err "CUDA silent failure — whisper fell back to CPU."
            hint "Possible VRAM overflow. Try a smaller model (e.g. medium.en or medium.en-q5_0)."
            log_error "$VIDEO_PATH" "CUDA silent failure — whisper fell back to CPU."
            hint "Temp WAV kept for debugging: $TEMP_WAV_PATH"
            rm -f "$WHISPER_LOG"
            FILES_FAILED=$((FILES_FAILED + 1))
            continue
        fi

        # One-time GPU diagnostics after the FIRST successful transcription
        if [ "$GPU_VERIFIED" = false ]; then
            gpu_diagnostics "$WHISPER_LOG" "$WAV_DURATION_INT"
            GPU_VERIFIED=true
        fi
    fi
    rm -f "$WHISPER_LOG"

    # ------------------------------------------------------------------
    # SRT existence check
    # ------------------------------------------------------------------
    if [ ! -f "$EXPECTED_SRT_PATH" ]; then
        err "Whisper exited cleanly but produced no SRT file (silent failure)."
        hint "Expected: $EXPECTED_SRT_PATH"
        hint "Temp WAV kept for debugging: $TEMP_WAV_PATH"
        log_error "$VIDEO_PATH" "Whisper ran but produced no SRT (silent failure)."
        FILES_FAILED=$((FILES_FAILED + 1))
        continue
    fi

    # ------------------------------------------------------------------
    # PHASE 3: Delivery
    # ------------------------------------------------------------------
    printf "   ${BLUE}Phase 3${RESET}  Delivering SRT...      "

    if cp "$EXPECTED_SRT_PATH" "$FINAL_SRT_PATH"; then
        rm -f "$TEMP_WAV_PATH" "$EXPECTED_SRT_PATH"
        FILE_ELAPSED=$(( $(date +%s) - FILE_START_TIME ))
        ELAPSED_FMT=$(printf "%dm%02ds" $((FILE_ELAPSED/60)) $((FILE_ELAPSED%60)))
        echo -e "${GREEN}✔${RESET}  (took ${ELAPSED_FMT})"
        FILES_OK=$((FILES_OK + 1))
    else
        printf "\n"
        err "Failed to copy SRT to destination."
        hint "Dest: $FINAL_SRT_PATH"
        log_error "$VIDEO_PATH" "Failed to copy SRT from temp to destination."
        FILES_FAILED=$((FILES_FAILED + 1))
    fi

done

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
BATCH_ELAPSED=$(( $(date +%s) - BATCH_START_TIME ))
BATCH_FMT=$(printf "%dh %dm %02ds" $((BATCH_ELAPSED/3600)) $(( (BATCH_ELAPSED%3600)/60 )) $((BATCH_ELAPSED%60)))

section "BATCH COMPLETE"
echo ""
info "Total time:"    "$BATCH_FMT"
info "Processed:"     "$TOTAL_FILES file(s)"
echo -e "   ${GREEN}✔ Success:${RESET}  ${FILES_OK}"
[ "$FILES_SKIPPED" -gt 0 ] && echo -e "   ${YELLOW}⟳ Skipped:${RESET}  ${FILES_SKIPPED}  (SRT already existed)"
[ "$FILES_FAILED"  -gt 0 ] && echo -e "   ${RED}✘ Failed:${RESET}   ${FILES_FAILED}  (see: $ERROR_LOG_FILE)"
echo ""