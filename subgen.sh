#!/bin/bash
#
# Title: WSL Batch Video Transcriber (Whisper.cpp Pipeline)
# Description: Automates the batch transcription of videos using ffmpeg and whisper.cpp
#           Optimized for performance within the Windows Subsystem for Linux (WSL)
#           environment, handling cross-OS paths and file cleanup.
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
# 4. Download the model:  cd whisper.cpp/models && bash download-ggml-model.sh medium.en
# 5. Download VAD model:  cd whisper.cpp/models && bash download-vad-model.sh silero-v5.1.2
# 6. Run:                 ./subgen.sh "/mnt/c/Users/YourName/Videos/ProjectFolder"

# --- GLOBAL CONFIGURATION (Phase 2 Optimization) ---
# Use 'set -e' for immediate exit on error, 'set -u' for unset variables, 'set -o pipefail' for pipeline safety.
set -euo pipefail

# --- PRETTY CLI COLORS ---
BOLD=$(tput bold)
BLUE=$(tput setaf 4)
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# --- ROBUST PATHING ---
# Get the absolute directory where this script is located.
# This makes all paths absolute and removes ambiguity, fixing the 'cd' bug.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Root of the whisper.cpp git submodule
WHISPER_SUBMODULE_DIR="$SCRIPT_DIR/whisper.cpp"

# Path to the compiled whisper.cpp executable (inside the submodule)
WHISPER_EXECUTABLE="$WHISPER_SUBMODULE_DIR/build/bin/whisper-cli"

# --- MODEL, LANGUAGE, AND TASK CONFIGURATION ---
# Using the standard (unquantized) English-only medium model.
# Model file lives inside the whisper.cpp submodule's models/ directory.
WHISPER_MODEL_NAME="medium.en"
# Set the spoken language (e.g., 'en' for English, 'auto' for auto-detect).
LANGUAGE="en"
# Set the task: "transcribe" (speech-to-text) or "translate" (speech-to-English)
TASK="transcribe"

# Path to the model file (inside the submodule's models/ directory)
WHISPER_MODEL="$WHISPER_SUBMODULE_DIR/models/ggml-$WHISPER_MODEL_NAME.bin"

# --- GPU CONFIGURATION ---
# IMPORTANT: Assumes you compiled with -DGGML_CUDA=1.
# GPU is used BY DEFAULT. Setting this to 'false' will add the '-ng'
# (no-gpu) flag to force CPU-only processing.
USE_CUDA=true
# Set the *specific* GPU index for CUDA to use. This fixes issues on
# multi-GPU systems (like Intel+NVIDIA) where CUDA defaults to the
# wrong device (e.g., the iGPU). '0' refers to the first GPU
# listed in 'nvidia-smi', which should be your NVIDIA card.
NVIDIA_GPU_INDEX=0
# Number of CPU threads to use. $(nproc) is for CPU-only.
# A smaller number (like 8) is *much* more efficient for feeding a GPU.
GPU_FEED_THREADS=8

# --- VAD (VOICE ACTIVITY DETECTION) CONFIGURATION ---
# Use VAD to skip silence and significantly speed up transcription.
USE_VAD=true
# The VAD model to use.
VAD_MODEL_NAME="ggml-silero-v5.1.2.bin"
VAD_MODEL_PATH="$WHISPER_SUBMODULE_DIR/models/$VAD_MODEL_NAME"

# --- STATE & ARGUMENT VALIDATION ---
NUM_THREADS=$(nproc)

if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 \"/path/to/windows/video/folder\"${RESET}" >&2
    echo -e "${YELLOW}Example: $0 \"/mnt/c/Users/User/Videos/Client Project\"${RESET}" >&2
    exit 1
fi

INPUT_DIR="$1"
ERROR_LOG_FILE="$INPUT_DIR/transcription_errors.log"
# Create a unique temporary directory in the fast WSL environment
TEMP_DIR="/tmp/whisper_pipeline_$(date +%s)"
mkdir -p "$TEMP_DIR"

# --- GRACEFUL EXIT TRAP (CTRL+C) ---
# This function will run on ANY script exit (normal, error, or interrupt)
function cleanup() {
    # '$?' holds the exit code of the last command. 0 = success.
    local exit_code=$? 
    
    if [ "$exit_code" -eq 130 ]; then
        # Special code for SIGINT (CTRL+C)
        echo -e "\n\n${YELLOW}[INTERRUPT]${RESET} User interruption (CTRL+C) detected. Cleaning up..."
    elif [ "$exit_code" -ne 0 ]; then
        echo -e "\n\n${RED}[ERROR]${RESET} Script exited abnormally (Code: $exit_code). Cleaning up..."
    else
        echo -e "\n${GREEN}[FINISH]${RESET} Batch process complete."
    fi

    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        echo -e "   -> Temporary directory ${YELLOW}$TEMP_DIR${RESET} has been cleaned up."
    fi
    
    # Reset terminal colors just in case
    echo -n "${RESET}"
    exit $exit_code
}
# 'trap' calls the 'cleanup' function on any EXIT signal
trap cleanup EXIT
# --- END NEW TRAP ---

# --- NEW ERROR LOGGING FUNCTION ---
function log_error() {
    local file_path="$1"
    local error_msg="$2"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %T")
    echo "[$timestamp] FAILED: $file_path" >> "$ERROR_LOG_FILE"
    echo "       REASON: $error_msg" >> "$ERROR_LOG_FILE"
    echo "       ---------------------------------" >> "$ERROR_LOG_FILE"
}
# --- END NEW FUNCTION ---


# --- DEPENDENCY CHECK & INITIALIZATION ---
echo -e "${BLUE}--- DEPENDENCY CHECKS ---${RESET}"
# Check for whisper.cpp executable
if [ ! -f "$WHISPER_EXECUTABLE" ]; then
    echo -e "${RED}ERROR: Whisper executable not found at '$WHISPER_EXECUTABLE'.${RESET}" >&2
    echo "         (Checked absolute path: whisper.cpp/build/bin/whisper-cli)." >&2
    echo -e "         ${YELLOW}Please build the submodule:${RESET}" >&2
    echo -e "         ${YELLOW}  cd whisper.cpp && cmake -B build -DGGML_CUDA=1 && cmake --build build -j\$(nproc)${RESET}" >&2
    exit 1
fi
echo -e "   ${GREEN}[SUCCESS]${RESET} Found whisper-cli: $WHISPER_EXECUTABLE"

# Check for FFmpeg dependency
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}ERROR: FFmpeg is not installed. Please run: sudo apt install ffmpeg${RESET}" >&2
    exit 1
fi
echo -e "   ${GREEN}[SUCCESS]${RESET} Found ffmpeg."

# Check for main model file
if [ ! -f "$WHISPER_MODEL" ]; then
    echo -e "${RED}ERROR: Whisper model '$WHISPER_MODEL_NAME' not found.${RESET}" >&2
    echo "         Expected file at: '$WHISPER_MODEL'." >&2
    echo "         (Inside the whisper.cpp submodule's models/ directory)." >&2
    echo -e "         ${YELLOW}Please run: cd whisper.cpp/models && bash download-ggml-model.sh $WHISPER_MODEL_NAME${RESET}" >&2
    exit 1
fi
echo -e "   ${GREEN}[SUCCESS]${RESET} Found main model: $WHISPER_MODEL_NAME"

# Check for VAD model file (if enabled)
if [ "$USE_VAD" = true ]; then
    if [ ! -f "$VAD_MODEL_PATH" ]; then
        echo -e "${RED}ERROR: VAD model not found at '$VAD_MODEL_PATH'.${RESET}" >&2
        echo "         (Inside the whisper.cpp submodule's models/ directory)." >&2
        echo -e "         ${YELLOW}VAD is enabled, but the model is missing.${RESET}" >&2
        echo -e "         ${YELLOW}Please run: cd whisper.cpp/models && bash download-vad-model.sh silero-v5.1.2${RESET}" >&2
        exit 1
    fi
    echo -e "   ${GREEN}[SUCCESS]${RESET} Found VAD model: $VAD_MODEL_NAME"
fi

# --- GPU VALIDATION ---
# If CUDA is enabled, try to validate that the target GPU is actually an NVIDIA card.
if [ "$USE_CUDA" = true ]; then
    echo -e "${BLUE}--- GPU VALIDATION ---${RESET}"
    if command -v nvidia-smi &> /dev/null; then
        echo -e "   -> 'nvidia-smi' found. Querying for GPU at index ${YELLOW}$NVIDIA_GPU_INDEX${RESET}..."
        # Query for the name of the GPU at the specified index.
        # Use 'set +e' to temporarily allow this command to fail without exiting the script
        set +e
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits --id=$NVIDIA_GPU_INDEX 2> /dev/null)
        NVIDIA_SMI_EXIT_CODE=$?
        set -e # Re-enable immediate exit on error

        if [ $NVIDIA_SMI_EXIT_CODE -eq 0 ] && [[ "$GPU_NAME" == *"NVIDIA"* ]]; then
            # Success!
            echo -e "   ${GREEN}[SUCCESS]${RESET} Found '${BOLD}$GPU_NAME${RESET}' at index $NVIDIA_GPU_INDEX."
            echo "   -> Pipeline will proceed with NVIDIA hardware acceleration."
        elif [ $NVIDIA_SMI_EXIT_CODE -eq 0 ]; then
            # Found a GPU, but it's not NVIDIA (e.g., Intel)
            echo -e "${RED}!!! FATAL GPU ERROR: Found a GPU at index $NVIDIA_GPU_INDEX, but it's not an NVIDIA card:${RESET}" >&2
            echo -e "       -> Found: '$GPU_NAME'" >&2
            echo "       -> This pipeline requires an NVIDIA GPU for CUDA acceleration." >&2
            echo -e "       -> ${YELLOW}Please check your NVIDIA_GPU_INDEX setting in this script or your WSL/CUDA setup.${RESET}" >&2
            exit 1
        else
            # nvidia-smi failed (e.g., index out of bounds, driver issue)
            echo -e "${RED}!!! FATAL GPU ERROR: 'nvidia-smi' failed to query GPU at index $NVIDIA_GPU_INDEX.${RESET}" >&2
            echo "       -> This could mean the index is wrong, or there's an NVIDIA driver issue inside WSL." >&2
            echo -e "       -> ${YELLOW}Run 'nvidia-smi' manually to check.${RESET}" >&2
            exit 1
        fi
    else
        # nvidia-smi command not found
        echo -e "${YELLOW}WARNING: 'nvidia-smi' command not found.${RESET}" >&2
        echo "   Cannot validate GPU. Will attempt to run with CUDA, but this may fail or use the wrong device." >&2
        echo "   Please ensure NVIDIA drivers are correctly installed and exposed to WSL." >&2
    fi
fi
# --- END GPU VALIDATION ---

echo -e "${BLUE}--- STARTING BATCH TRANSCRIPTION PIPELINE ---${RESET}"
echo -e "   ${BOLD}Input Directory:${RESET} $INPUT_DIR"
echo -e "   ${BOLD}Temporary Staging:${RESET} $TEMP_DIR"
echo -e "   ${BOLD}Failure Log:${RESET} $ERROR_LOG_FILE"

# --- ROBUST FILE DISCOVERY (Using safer -iname operator) ---
mapfile -t VIDEO_FILES < <(
    find "$INPUT_DIR" -type f \( \
        -iname "*.mp4" -o \
        -iname "*.mkv" -o \
        -iname "*.avi" -o \
        -iname "*.webm" -o \
        -iname "*.ts" -o \
        -iname "*.mov" \
    \)
)

# Check if the array is empty (i.e., no files were found)
if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    echo ""
    echo -e "${RED}!!! FATAL ERROR: No video files found in the specified directory.${RESET}" >&2
    echo -e "       Path checked: '$INPUT_DIR'" >&2
    echo -e "       Extensions checked (case-insensitive): mp4, mkv, avi, webm, ts, mov" >&2
    echo -e "       ${YELLOW}Ensure the folder exists and contains files with one of the supported extensions.${RESET}" >&2
    echo ""
    exit 1
fi

echo -e "   ${BOLD}Found ${#VIDEO_FILES[@]} video file(s)${RESET} for processing."
echo -e "${BLUE}-------------------------------------------------${RESET}"

# --- MAIN BATCH PROCESSING LOOP ---
FILE_COUNT=0
TOTAL_FILES=${#VIDEO_FILES[@]}

for VIDEO_PATH in "${VIDEO_FILES[@]}"; do
    FILE_COUNT=$((FILE_COUNT + 1))
    
    # 1. Path Safety and Naming
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    VIDEO_BASENAME="${VIDEO_FILENAME%.*}"
    VIDEO_DIR=$(dirname "$VIDEO_PATH")

    TEMP_WAV_PATH="$TEMP_DIR/$VIDEO_BASENAME.wav"
    FINAL_SRT_PATH="$VIDEO_DIR/$VIDEO_BASENAME.srt"
    
    # Define the output *base* path. whisper-cli will add '.srt' to this.
    TEMP_SRT_BASE_PATH="$TEMP_DIR/$VIDEO_BASENAME"
    # Define the *actual* output file path we expect whisper-cli to create
    EXPECTED_SRT_PATH="$TEMP_SRT_BASE_PATH.srt"

    # Log file for FFmpeg errors
    FFMPEG_LOG="$TEMP_DIR/$VIDEO_BASENAME.ffmpeg.log" 
    # Log file for Whisper stderr (to check for CUDA init)
    WHISPER_LOG="$TEMP_DIR/$VIDEO_BASENAME.whisper.log"
    
    echo -e "\n${BOLD}Processing ($FILE_COUNT/$TOTAL_FILES): $VIDEO_FILENAME${RESET}"

    # --- PHASE 1: PRE-PROCESSING (FFMPEG AUDIO EXTRACTION) ---
    
    # *** THIS IS THE RESUME LOGIC ***
    # Check if the final SRT already exists and skip if it does.
    if [ -f "$FINAL_SRT_PATH" ]; then
        echo -e "   ${YELLOW}[SKIP]${RESET} Final SRT already exists at $FINAL_SRT_PATH"
        continue
    fi

    echo -e "   ${BLUE}[Phase 1]${RESET} Extracting audio (this may take a moment)..."
    
    # 1. Run FFmpeg
    # Use 'tee' to show live progress while also logging.
    # '-loglevel warning' hides verbose info, '-stats' shows the progress bar.
    if ! ffmpeg -i "$VIDEO_PATH" -vn \
        -acodec pcm_s16le \
        -ar 16000 \
        -ac 1 \
        -y "$TEMP_WAV_PATH" -loglevel warning -stats 2> >(tee "$FFMPEG_LOG" >&2); then
        
        # --- CTRL+C (130) CHECK ---
        exit_code=$?
        if [ "$exit_code" -eq 130 ]; then
            # 130 is the exit code for SIGINT (CTRL+C)
            echo -e "\n   ${YELLOW}[INTERRUPT]${RESET} CTRL+C detected during FFmpeg. Exiting..."
            rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG"
            exit 130 # This will trigger the main EXIT trap
        fi
        # --- END NEW ---

        echo -e "   ${RED}[ERROR]${RESET} FFmpeg failed to extract audio from $VIDEO_FILENAME (Code: $exit_code)." >&2
        log_error "$VIDEO_PATH" "FFmpeg failed to extract audio (Code: $exit_code)."
        echo -e "   --- FFMPEG STDERR/STDOUT (see details above) ---" >&2
        rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG" 
        continue
    fi
    
    # 2. Audio Integrity Check
    if [ ! -s "$TEMP_WAV_PATH" ]; then
        echo -e "   ${RED}[ERROR]${RESET} FFmpeg exited successfully, but generated a zero-byte WAV file." >&2
        echo -e "   ${YELLOW}This usually means the video has no detectable audio track. Skipping.${RESET}" >&2
        log_error "$VIDEO_PATH" "FFmpeg generated a zero-byte (empty) audio file."
        rm -f "$TEMP_WAV_PATH" "$FFMPEG_LOG"
        continue
    fi

    rm -f "$FFMPEG_LOG" # Cleanup successful log
    echo -e "   ${GREEN}[SUCCESS]${RESET} Phase 1 Complete: Audio extracted and integrity validated."

    # --- PHASE 2: TRANSCRIPTION (WHISPER.CPP EXECUTION) ---

    echo -e "   ${BLUE}[Phase 2]${RESET} Transcribing audio (this may take several minutes)..."

    # Use a Bash array for clean, safe argument handling
    WHISPER_CMD_ARGS=(
        -m "$WHISPER_MODEL"
        -f "$TEMP_WAV_PATH"
        -l "$LANGUAGE"
    )
    
    # *** THREADING OPTIMIZATION ***
    if [ "$USE_CUDA" = true ]; then
        echo "   -> Using GPU (default, no flag added)."
        echo "   -> Forcing CUDA to use NVIDIA GPU at index $NVIDIA_GPU_INDEX."
        export CUDA_VISIBLE_DEVICES=$NVIDIA_GPU_INDEX
        
        # *** CPU OPTIMIZATION FOR GPU ***
        echo "   -> Setting CPU threads to $GPU_FEED_THREADS (optimal for feeding GPU)."
        WHISPER_CMD_ARGS+=( -t "$GPU_FEED_THREADS" )
    else
        echo "   -> Forcing CPU (-ng flag)."
        echo "   -> Setting CPU threads to $NUM_THREADS (max)."
        WHISPER_CMD_ARGS+=( -t "$NUM_THREADS" )
        WHISPER_CMD_ARGS+=( -ng )
    fi

    # *** FLEXIBLE TASK ***
    if [ "$TASK" = "translate" ]; then
        echo "   -> Task set to: translate (adding -tr flag)."
        WHISPER_CMD_ARGS+=( -tr )
    else
        echo "   -> Task set to: transcribe."
    fi

    # --- VAD OPTIMIZATION ---
    if [ "$USE_VAD" = true ]; then
        echo "   -> VAD enabled (skipping silence)."
        WHISPER_CMD_ARGS+=( --vad )
        WHISPER_CMD_ARGS+=( -vm "$VAD_MODEL_PATH" )
    fi

    # Add output flags
    WHISPER_CMD_ARGS+=( -osrt )
    WHISPER_CMD_ARGS+=( -of "$TEMP_SRT_BASE_PATH" )
    
    # --- ADD PROGRESS BAR ---
    # This flag tells whisper-cli to print progress to stderr.
    WHISPER_CMD_ARGS+=( -pp )
    
    # Execute whisper.cpp ("Try 1", with VAD if enabled)
    # We pipe stderr (2>) to 'tee'
    # 'tee' sends one copy to our log file and another copy to >&2 (stderr)
    # This lets the user see the progress bar in real-time.
    if ! "$WHISPER_EXECUTABLE" "${WHISPER_CMD_ARGS[@]}" 2> >(tee "$WHISPER_LOG" >&2); then
        
        # --- "Catch" Block ---
        exit_code=$?
        
        # --- CTRL+C (130) CHECK ---
        if [ "$exit_code" -eq 130 ]; then
            # 130 is the exit code for SIGINT (CTRL+C)
            echo -e "\n   ${YELLOW}[INTERRUPT]${RESET} CTRL+C detected during Whisper. Exiting..."
            rm -f "$WHISPER_LOG"
            exit 130 # This will trigger the main EXIT trap
        fi
        
        # --- *** NEW: AUTO-RETRY LOGIC *** ---
        # Check if VAD was enabled on this failed attempt.
        if [ "$USE_VAD" = true ]; then
            # VAD was on. This is likely the malloc() crash.
            echo -e "   ${YELLOW}[RETRY]${RESET} Whisper.cpp failed with VAD enabled (Code: $exit_code)." >&2
            echo -e "   -> Suspected VAD-related crash. Retrying *without* VAD..."
            
            # Re-build the command arguments, but *force* VAD to be off.
            # We must re-build from scratch to safely remove VAD args.
            WHISPER_CMD_ARGS_RETRY=(
                -m "$WHISPER_MODEL"
                -f "$TEMP_WAV_PATH"
                -l "$LANGUAGE"
            )

            # Copy thread/task/GPU logic
            if [ "$USE_CUDA" = true ]; then
                export CUDA_VISIBLE_DEVICES=$NVIDIA_GPU_INDEX
                WHISPER_CMD_ARGS_RETRY+=( -t "$GPU_FEED_THREADS" )
            else
                WHISPER_CMD_ARGS_RETRY+=( -t "$NUM_THREADS" )
                WHISPER_CMD_ARGS_RETRY+=( -ng )
            fi

            if [ "$TASK" = "translate" ]; then
                WHISPER_CMD_ARGS_RETRY+=( -tr )
            fi

            # Add output flags
            WHISPER_CMD_ARGS_RETRY+=( -osrt )
            WHISPER_CMD_ARGS_RETRY+=( -of "$TEMP_SRT_BASE_PATH" )
            WHISPER_CMD_ARGS_RETRY+=( -pp )
            
            # --- "Try 2 (Without VAD)" ---
            if ! "$WHISPER_EXECUTABLE" "${WHISPER_CMD_ARGS_RETRY[@]}" 2> >(tee "$WHISPER_LOG" >&2); then
                # This *second* attempt failed. This is a fatal error for this file.
                retry_exit_code=$?
                if [ "$retry_exit_code" -eq 130 ]; then
                    echo -e "\n   ${YELLOW}[INTERRUPT]${RESET} CTRL+C detected during retry. Exiting..."
                    rm -f "$WHISPER_LOG"
                    exit 130
                fi
                
                echo -e "   ${RED}[FATAL ERROR]${RESET} Whisper.cpp failed *again* even with VAD disabled (Code: $retry_exit_code)." >&2
                log_error "$VIDEO_PATH" "Whisper.cpp failed on retry (VAD off). (Code: $retry_exit_code)."
                echo -e "   -> ${YELLOW}Leaving temp WAV file for debugging: $TEMP_WAV_PATH${RESET}"
                rm -f "$WHISPER_LOG"
                continue # Give up on this file, move to the next.
            fi
            
            # If we are here, the *retry* (Try 2) was successful!
            echo -e "   ${GREEN}[SUCCESS]${RESET} Retry without VAD was successful."
            # The script will now proceed to the CUDA checks and Phase 3 as normal.

        else
            # VAD was *already* off. This is a non-VAD-related crash.
            echo -e "   ${RED}[ERROR]${RESET} Whisper.cpp failed to transcribe $VIDEO_FILENAME (Code: $exit_code)." >&2
            echo -e "   -> VAD was *disabled*, so this is a different issue." >&2
            log_error "$VIDEO_PATH" "Whisper.cpp crashed or failed with VAD disabled (Code: $exit_code)."
            echo -e "   -> The error message from whisper-cli should be visible directly above this message." >&2
            echo -e "   -> ${YELLOW}Leaving temp WAV file for debugging: $TEMP_WAV_PATH${RESET}"
            rm -f "$WHISPER_LOG"
            continue # Give up on this file, move to the next.
        fi
        # --- *** END NEW RETRY LOGIC *** ---
        
    fi
    
    # --- *** NEW, SMARTER CUDA SILENT FAILURE CHECK *** ---
    if [ "$USE_CUDA" = true ]; then
        # Check if CUDA *failed* to initialize.
        if grep -q "failed to initialize CUDA" "$WHISPER_LOG"; then
            echo -e "   ${RED}[ERROR]${RESET} ${BOLD}CUDA RUNTIME FAILURE DETECTED.${RESET}" >&2
            echo -e "   -> ${YELLOW}Log shows: 'failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version'${RESET}" >&2
            echo -e "   -> ${BOLD}ACTION REQUIRED: You must update your NVIDIA drivers on your host Windows machine.${RESET}" >&2
            echo -e "   -> After updating, reboot your computer and re-run this script." >&2
            rm -f "$WHISPER_LOG"
            # We exit the whole script here. Skipping is pointless as all files will fail.
            exit 1
        
        # Check if CUDA *succeeded*
        elif ! grep -q "ggml_cuda_init" "$WHISPER_LOG"; then
            echo -e "   ${RED}[ERROR]${RESET} ${BOLD}CUDA SILENT FAILURE DETECTED.${RESET}" >&2
            echo -e "   -> ${YELLOW}Whisper-cli ran but did NOT initialize the CUDA GPU.${RESET}" >&2
            echo -e "   -> This means it fell back to CPU, which is why it's so slow." >&2
            echo -e "   -> This is likely a VRAM error (model too large) or a build issue (cuBLAS mismatch)." >&2
            echo -e "   -> ${YELLOW}Try using a smaller model (e.g., 'small.en') or a more quantized model.${RESET}" >&2
            echo -e "   -> ${YELLOW}Leaving temp WAV file for debugging: $TEMP_WAV_PATH${RESET}"
            log_error "$VIDEO_PATH" "CUDA silent failure. Whisper-cli fell back to CPU."
            rm -f "$WHISPER_LOG"
            continue
        else
            echo -e "   ${GREEN}[SUCCESS]${RESET} CUDA runtime validated (found 'ggml_cuda_init' in log)."
        fi
    fi
    rm -f "$WHISPER_LOG" # Clean up the whisper log
    # --- END NEW CHECK ---
    
    # Enhanced check for silent failures
    # Check if the transcription run was successful (Exit Code 0) AND produced the file
    if [ ! -f "$EXPECTED_SRT_PATH" ]; then
        echo -e "   ${RED}[ERROR]${RESET} Whisper.cpp execution finished (Exit Code 0), but no SRT file was generated." >&2
        echo -e "   -> This indicates a silent failure (e.g., a CUDA issue not caught by the exit code)." >&2
        echo -e "   -> Expected file at: $EXPECTED_SRT_PATH"
        echo -e "   -> ${YELLOW}Leaving temp WAV file for debugging: $TEMP_WAV_PATH${RESET}"
        log_error "$VIDEO_PATH" "Whisper.cpp ran but produced no SRT file (Silent failure)."
        continue
    fi

    echo -e "   ${GREEN}[SUCCESS]${RESET} Phase 2 Complete: Transcription successful."

    # --- PHASE 3: DELIVERY AND CLEANUP ---
    
    echo -e "   ${BLUE}[Phase 3]${RESET} Delivering SRT file..."
    # 1. Deliver the final artifact back to the Windows mounted path
    cp "$EXPECTED_SRT_PATH" "$FINAL_SRT_PATH"

    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}[SUCCESS]${RESET} SRT file delivered to: $FINAL_SRT_PATH"
        # 2. Cleanup (The Responsible Engineer)
        rm -f "$TEMP_WAV_PATH"
        rm -f "$EXPECTED_SRT_PATH"
    else
        echo -e "   ${RED}[ERROR]${RESET} Failed to copy final SRT to Windows path. Keeping temp files for debugging." >&2
        log_error "$VIDEO_PATH" "Failed to copy final SRT from temp to destination."
    fi

    echo -e "${BLUE}--- Finished $VIDEO_FILENAME ---${RESET}"

done

# --- FINAL SUMMARY ---
# The 'trap cleanup EXIT' will run here automatically.
# No 'rmdir' command is needed.