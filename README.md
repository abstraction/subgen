# subgen

Batch subtitle generator for WSL. Point it at a folder of videos and it spits out `.srt` files next to each one. Under the hood it's just ffmpeg for audio extraction and **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** for transcription, running on your NVIDIA GPU via CUDA.

---

## Project Structure

```
subgen/
├── subgen.sh          ← entry point
├── .gitmodules        ← whisper.cpp declared as a submodule
└── whisper.cpp/       ← git submodule (ggml-org/whisper.cpp)
    ├── build/
    │   └── bin/
    │       └── whisper-cli     ← compiled in the build step
    └── models/
        ├── download-ggml-model.sh
        ├── download-vad-model.sh
        ├── ggml-large-v3-q5_0.bin      ← main transcription model
        └── ggml-silero-v5.1.2.bin      ← VAD model (optional but recommended)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| WSL 2 (Ubuntu) | Windows Subsystem for Linux |
| NVIDIA GPU + drivers | Install the NVIDIA Graphics Driver on the Windows host. It exposes the GPU to WSL automatically via GPU-PV. |
| CUDA Toolkit (WSL build) | Install inside WSL from NVIDIA's CUDA repo (see [CUDA Setup](#cuda-setup-wsl)) |
| `cmake` >= 3.14 | `sudo apt install cmake build-essential` |
| `ffmpeg` | `sudo apt install ffmpeg` |
| `git` | For cloning and submodule init |

---

## Setup

### 1. Clone & initialise the submodule

```bash
git clone --recurse-submodules https://github.com/your-username/subgen.git
cd subgen
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Build `whisper-cli` with CUDA support

```bash
cd whisper.cpp
cmake -B build -DGGML_CUDA=1
cmake --build build --config Release -j$(nproc)
cd ..
```

> **Verify:** `whisper.cpp/build/bin/whisper-cli` should now exist.

> **Build crashing or WSL resetting?** NVCC generates large memory structures for CUDA templates (`fattn`, `ggml-cuda`), peaking at 3-4 GB per compiler thread. On RAM-limited machines this kills the build. Two options:
>
> Use a memory-aware job count instead of `-j$(nproc)`:
> ```bash
> cmake --build build -j $(free -g | awk '/^Mem:/{j=int($7/3); print j<1?1:j}') --config Release
> ```
>
> Or add swap on the Windows host. Create/edit `%USERPROFILE%\.wslconfig`:
> ```ini
> [wsl2]
> swap=8GB
> ```
> Then apply with `wsl --shutdown` in PowerShell.

### 3. Install ffmpeg

```bash
sudo apt install ffmpeg
```

### 4. Download the transcription model

```bash
cd whisper.cpp/models
bash download-ggml-model.sh large-v3-q5_0
cd ../..
```

Downloads `ggml-large-v3-q5_0.bin` (~1.0 GB) into `whisper.cpp/models/`.

> **Why large-v3-q5_0?** Whisper large-v3 is OpenAI's most accurate speech recognition model. The `q5_0` variant quantizes weights to 5-bit integers, cutting VRAM from ~5.8 GB down to ~1.9 GB with barely any accuracy cost. That's what makes it a good fit for 4-6 GB cards like the RTX 3050 6 GB.

Verify the model and GPU are working with the bundled JFK sample:

```bash
cd whisper.cpp
./build/bin/whisper-cli -m models/ggml-large-v3-q5_0.bin -l en -f samples/jfk.wav
```

You should see a transcript. `ggml_cuda_init` in the output confirms the GPU was picked up.

### 5. Download the VAD model *(optional but recommended)*

**[Silero VAD](https://github.com/snakers4/silero-vad)** is a small neural Voice Activity Detection model (~864 KB). With it enabled, whisper-cli identifies speech regions first and skips silence entirely. This makes a real difference on videos with long pauses, intros, or music.

The model is distributed in GGML format by [ggml-org](https://huggingface.co/ggml-org/whisper-vad) and comes with whisper.cpp's own download helper.

```bash
# From the project root:
bash whisper.cpp/models/download-vad-model.sh silero-v5.1.2
```

Downloads `ggml-silero-v5.1.2.bin` (~864 KB) into `whisper.cpp/models/`.

> **Tip:** VAD is on by default (`USE_VAD=true`). To skip the download and run without it, set `USE_VAD=false` in `subgen.sh`.

---

## CUDA Setup (WSL)

### Driver architecture

NVIDIA GPU acceleration in WSL2 works through GPU Paravirtualization (GPU-PV):

- **Windows host:** install the NVIDIA Graphics Driver from the [official vendor portal](https://www.nvidia.com/drivers). The Windows kernel-mode driver surfaces the GPU to WSL automatically.
- **WSL2:** don't install a Linux graphics driver (`.run` or `.deb`) inside the WSL instance. It corrupts the paravirtualization layer and breaks the passthrough.

Verify the passthrough from inside WSL:

```bash
nvidia-smi
```

Expected output: GPU name, driver version, and CUDA version pulled from the Windows host.

### CUDA Toolkit and nvcc

If CMake can't find the CUDA compiler (`No CMAKE_CUDA_COMPILER could be found`), install the CUDA toolkit from NVIDIA's official WSL repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit
```

Then expose the compiler to your shell. Add these to `~/.zshrc` or `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Reload and verify:

```bash
source ~/.zshrc   # or ~/.bashrc
nvcc --version
```

---

## Usage

```bash
./subgen.sh "/mnt/c/Users/YourName/Videos/ProjectFolder"
```

The script will:
1. Discover all `.mp4 .mkv .avi .webm .ts .mov` files in the given directory (recursive).
2. Skip any video that already has a matching `.srt` beside it (resume-safe).
3. Extract a mono 16 kHz WAV to `/tmp/` via ffmpeg.
4. Transcribe with `whisper-cli` on your NVIDIA GPU.
5. Write the `.srt` next to the original video and clean up temp files.
6. Log any failures to `transcription_errors.log` inside the input directory.

### Example

```bash
./subgen.sh "/mnt/c/Users/Alice/Videos/Lectures"
# output:
# /mnt/c/Users/Alice/Videos/Lectures/lecture_01.srt
# /mnt/c/Users/Alice/Videos/Lectures/lecture_02.srt
# ...
```

---

## Configuration

All options live at the top of `subgen.sh`. No flags needed at runtime.

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_NAME` | `large-v3-q5_0` | Whisper model variant to use |
| `LANGUAGE` | `en` | Spoken language code |
| `TASK` | `transcribe` | `transcribe` or `translate` (to English) |
| `USE_CUDA` | `true` | Enable NVIDIA GPU acceleration |
| `NVIDIA_GPU_INDEX` | `0` | GPU index from `nvidia-smi` (set to `1` on dual-GPU systems) |
| `GPU_FEED_THREADS` | `8` | CPU threads used to feed the GPU |
| `USE_VAD` | `true` | Enable Voice Activity Detection (skip silence) |

### Switching models

Edit `WHISPER_MODEL_NAME` in `subgen.sh` to any variant, then download it:

```bash
cd whisper.cpp/models && bash download-ggml-model.sh <model-name>
```

See the **[Model Reference](#model-reference)** section below for a full breakdown.

---

## Model Reference

### `.en` vs multilingual models

`.en` models are English-only, which gives a marginal accuracy edge on clean English audio. They're only available for the smaller architectures:

| `.en` model | Multilingual equivalent |
|---|---|
| `tiny.en` | `tiny` |
| `base.en` | `base` |
| `small.en` | `small` |
| `medium.en` | `medium` |

`large-v3` is **multilingual only** — there's no official `large-v3.en`. It handles English fine; just pass `-l en` to lock the language and stop it from guessing wrong.

---

### Quantization tiers

Quantization swaps 16-bit float weights for smaller integers, cutting file size and VRAM at a small accuracy cost.

**Quality order:** `FP16 > Q8_0 > Q5_0 > Q4`

| Quantization | Bits | Accuracy loss | Notes |
|---|---|---|---|
| `FP16` (none) | 16 | baseline | Full precision, highest VRAM |
| `Q8_0` | 8 | Negligible | Essentially identical to FP16 in practice |
| `Q5_0` | 5 | Very small | Best balance, recommended for most GPUs |
| `Q4` | 4 | Noticeable | Only worth it when VRAM is very tight |

The real-world gap between Q8 and Q5 is small. Most files produce identical transcripts. You're more likely to notice a difference with heavy accents, noisy audio, overlapping speakers, or dense technical vocabulary.

---

### Model comparison (RTX 3050 6 GB)

Measured VRAM with `medium.en` during active transcription: **~2243 MiB / 6144 MiB**.

| Model | File size | VRAM est. | Quality | Languages | Best for |
|---|---|---|---|---|---|
| `large-v3-q5_0` | ~1.0 GB | ~1.9 GB | Excellent | 99 | **Default, best all-rounder** |
| `large-v3-q8_0` | ~1.6 GB | ~2.9 GB | Near-lossless | 99 | Maximum quantized quality |
| `large-v3` (FP16) | ~2.9 GB | ~5.8 GB | Baseline | 99 | 8 GB+ VRAM only |
| `medium.en-q5_0` | ~0.5 GB | ~1.0 GB | Good | English | Speed priority or low-VRAM fallback |
| `medium.en` | ~1.5 GB | ~2.9 GB | Good | English | Unquantized medium baseline |

> `large-v3-q5_0` fits comfortably on the RTX 3050 6 GB. `large-v3-q8_0` fits too. Full `large-v3` (FP16) is borderline, so stick with `q5_0` unless you specifically need FP16.

---

### English language forcing

Because `large-v3` is multilingual, it auto-detects the language from audio. For English-only content, always pass `-l en` to prevent mis-detection:

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-q5_0.bin \
  -l en \
  audio.wav
```

`subgen.sh` passes `LANGUAGE="en"` by default, so no manual change is needed unless you want multilingual output.

---

### Manual quantization

Pre-quantized models are available via the download script and are the easiest option:

```bash
bash ./models/download-ggml-model.sh large-v3-q5_0
```

To quantize a full model yourself (e.g. to Q8_0):

```bash
# Download the full FP16 model first
bash ./models/download-ggml-model.sh large-v3

# Quantize it
./build/bin/whisper-quantize \
    models/ggml-large-v3.bin \
    models/ggml-large-v3-q8_0.bin \
    q8_0
```

Then set `WHISPER_MODEL_NAME="large-v3-q8_0"` in `subgen.sh`.

---

### Quick decision guide

| Goal | Model to use |
|---|---|
| Best overall (default) | `large-v3-q5_0` + `LANGUAGE=en` |
| Maximum quantized quality | `large-v3-q8_0` |
| Speed over accuracy | `medium.en-q5_0` |
| Multilingual / language switching | `large-v3-q5_0` (omit `LANGUAGE=en`) |

---

## Updating whisper.cpp

The `whisper.cpp` directory is a Git submodule pinned to a specific upstream commit. To pull in upstream changes:

```bash
cd whisper.cpp
git fetch
git checkout master
git pull
cd ..
git add whisper.cpp
git commit -m "chore: bump whisper.cpp upstream commit reference"
```

Rebuild after updating:

```bash
cd whisper.cpp && cmake -B build -DGGML_CUDA=1 && cmake --build build --config Release -j$(nproc)
```

---

## Supported Video Formats

`.mp4` · `.mkv` · `.avi` · `.webm` · `.ts` · `.mov`

Detection is case-insensitive (`.MP4`, `.Mkv`, etc. all work).

---

## Robustness Features

- **Resume support** — already-transcribed videos are skipped automatically.
- **VAD auto-retry** — if whisper-cli crashes with VAD enabled (a known malloc issue), the script retries without VAD before giving up on that file.
- **CUDA silent failure detection** — inspects the whisper-cli log for `ggml_cuda_init` to catch cases where the GPU was silently skipped.
- **CUDA version mismatch detection** — detects `failed to initialize CUDA` and exits with a clear message to update host NVIDIA drivers.
- **Graceful Ctrl+C** — temp files are cleaned up even on interrupt.
- **Error log** — every failed file is recorded in `transcription_errors.log` so you can handle failures selectively.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `whisper-cli: not found` | Run the cmake build step inside `whisper.cpp/`. |
| Model not found | Run `bash whisper.cpp/models/download-ggml-model.sh large-v3-q5_0`. |
| `VAD model not found` | Run `bash whisper.cpp/models/download-vad-model.sh silero-v5.1.2`, or set `USE_VAD=false` in `subgen.sh`. |
| CUDA silent failure (fell back to CPU) | Not enough VRAM. Try `medium.en-q5_0` (~1.0 GB VRAM). |
| `failed to initialize CUDA` | Update NVIDIA drivers on the **Windows** host, then reboot. |
| `No CMAKE_CUDA_COMPILER could be found` | Install the CUDA toolkit and add `nvcc` to `PATH`. See [CUDA Setup](#cuda-setup-wsl). |
| Build crashes / WSL resets during compilation | NVCC OOM. Use the memory-aware build command or add swap. See [Build step 2](#2-build-whisper-cli-with-cuda-support). |
| Wrong GPU used (iGPU instead of dGPU) | Set `NVIDIA_GPU_INDEX=1` (or the correct index from `nvidia-smi`). |
| Zero-byte WAV file | The video has no audio track; that file is skipped automatically. |

---

## License

This script is released under the MIT License.
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) is a separate project with its own MIT License.
