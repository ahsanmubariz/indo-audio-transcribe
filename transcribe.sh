#!/usr/bin/env bash
# transcribe.sh — Indonesian speech-to-text with PixelML/sea-asr-id, CPU only.
#
#   ./transcribe.sh rekaman.mp3 hasil.txt
#   ./transcribe.sh -i rapat.m4a -o rapat.srt --mode lm
#
# Creates/activates ./.venv in the current directory, installs sea_asr (CPU
# PyTorch), then transcribes. Output format follows the extension of OUTPUT:
# .txt (default) | .srt | .vtt | .json
#
# Probing, refusal and the ffmpeg clean-up live in mytranscribe/audio.py, the
# same module the web app calls, so the two front doors cannot drift apart.
# This script owns the argument parsing, the venv and the human-facing messages
# (ADR-0001); it does not own the audio handling.
set -Eeuo pipefail

SEA_ASR_REF="v0.2.1"                 # version the model card was published with
TORCH_VERSION="2.14.1"               # version pinned by sea_asr
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
VENV_DIR="${VENV_DIR:-$PWD/.venv}"

# ---------------------------------------------------------------- helpers ---
c_info=$'\e[36m'; c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_off=$'\e[0m'
[[ -t 2 ]] || { c_info=; c_ok=; c_warn=; c_err=; c_off=; }
info() { printf '%s[*]%s %s\n' "$c_info" "$c_off" "$*" >&2; }
ok()   { printf '%s[+]%s %s\n' "$c_ok"   "$c_off" "$*" >&2; }
warn() { printf '%s[!]%s %s\n' "$c_warn" "$c_off" "$*" >&2; }
die()  { printf '%s[x]%s %s\n' "$c_err"  "$c_off" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") INPUT OUTPUT [options]
       $(basename "$0") -i INPUT -o OUTPUT [options]

Output format follows OUTPUT's extension: .txt (default), .srt, .vtt, .json

Options:
  -i, --input FILE     audio file (mp3, m4a, aac, ogg, opus, flac, wav, wma, amr, ...)
  -o, --output FILE    transcript file to write
  -m, --mode MODE      greedy  : fastest, no language model
                       lm      : + Indonesian n-gram LM, more accurate (default)
                       rescore : + Sailor2-1B rescoring, most accurate, slow on CPU
  -t, --threads N      CPU threads to use (default: half the cores = $DEFAULT_THREADS of $MAX_CORES)
      --raw            only convert to 16 kHz mono WAV, no clean-up filters
      --keep-wav       keep the optimised WAV next to the output file
      --allow-video    accept video files too (their audio track is used)
      --reinstall      reinstall the Python packages in the venv
  -h, --help           show this help

Env: VENV_DIR (default ./.venv), HF_TOKEN (if Hugging Face asks you to log in)
EOF
}

# ------------------------------------------------------------------- args ---
MAX_CORES="$(nproc --all)"
DEFAULT_THREADS=$(( MAX_CORES / 2 )); (( DEFAULT_THREADS < 1 )) && DEFAULT_THREADS=1
INPUT=""; OUTPUT=""; MODE="lm"; THREADS="$DEFAULT_THREADS"
RAW=0; KEEP_WAV=0; ALLOW_VIDEO=0; REINSTALL=0
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)   INPUT="${2:-}"; shift 2 ;;
    -o|--output)  OUTPUT="${2:-}"; shift 2 ;;
    -m|--mode)    MODE="${2:-}"; shift 2 ;;
    -t|--threads) THREADS="${2:-}"; shift 2 ;;
    --raw)        RAW=1; shift ;;
    --keep-wav)   KEEP_WAV=1; shift ;;
    --allow-video) ALLOW_VIDEO=1; shift ;;
    --reinstall)  REINSTALL=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; positional+=("$@"); break ;;
    -*)           die "Unknown option: $1 (see --help)" ;;
    *)            positional+=("$1"); shift ;;
  esac
done
[[ -z "$INPUT"  && ${#positional[@]} -ge 1 ]] && INPUT="${positional[0]}"
[[ -z "$OUTPUT" && ${#positional[@]} -ge 2 ]] && OUTPUT="${positional[1]}"
[[ -n "$INPUT" && -n "$OUTPUT" ]] || { usage; exit 1; }
[[ -f "$INPUT" ]] || die "Input not found: $INPUT"
[[ "$MODE" =~ ^(greedy|lm|rescore)$ ]] || die "--mode must be greedy, lm or rescore"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive number"
if (( THREADS > MAX_CORES )); then
  warn "--threads $THREADS is more than the $MAX_CORES cores available, using $MAX_CORES"
  THREADS=$MAX_CORES
fi
# Every library below honours these, so the whole run stays within $THREADS cores
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"

# Output format from extension; no extension -> .txt
ext="${OUTPUT##*.}"; [[ "$OUTPUT" == *.* ]] || { ext="txt"; OUTPUT="$OUTPUT.txt"; }
FMT="${ext,,}"
[[ "$FMT" =~ ^(txt|srt|vtt|json)$ ]] || die "Output extension must be .txt, .srt, .vtt or .json (got .$ext)"
mkdir -p "$(dirname -- "$OUTPUT")"

# ------------------------------------------------------- system commands ---
missing=()
for cmd in ffmpeg ffprobe git; do command -v "$cmd" >/dev/null || missing+=("$cmd"); done
if [[ ${#missing[@]} -gt 0 ]]; then
  die "Missing: ${missing[*]}. Install with:  sudo apt install ffmpeg git"
fi

# ------------------------------------- 1-2) probe and optimise the audio ---
# Both halves live in mytranscribe/audio.py, which the web app calls too, so the
# two front doors cannot drift. The module is standard-library only, so it runs
# here with the system Python, before the venv exists. It reports facts as
# key=value on stdout and exits non-zero with a reason on stderr.
find_python() {
  local c
  for c in python3.13 python3.12 python3; do
    if command -v "$c" >/dev/null &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PREP_PY="$(find_python)" || die "Python >= 3.12 is needed to prepare the audio. On older Ubuntu:
      sudo apt install python3.12 python3.12-venv    (or install uv)"

info "Checking input: $INPUT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
WAV="$TMP_DIR/input_16k_mono.wav"

# The module cleans up after itself, so it writes the WAV straight to where the
# rest of this script expects it rather than leaving it to be moved.
prep_args=("$INPUT" --threads "$THREADS" --keep-wav "$WAV")
[[ $RAW -eq 1 ]] && prep_args+=(--raw)
[[ $ALLOW_VIDEO -eq 1 ]] && prep_args+=(--allow-video)

info "Optimising audio -> 16 kHz mono 16-bit WAV$([[ $RAW -eq 1 ]] && echo ' (raw)')"
# The module's stderr is the reason a person should read; keep it, but print it
# through die() so it carries the same [x] prefix as every other refusal here.
if ! prep_err="$(MYTRANSCRIBE_ROOT="$PWD" \
  "$PREP_PY" -c 'import sys, os; sys.path.insert(0, os.environ["MYTRANSCRIBE_ROOT"]);
from mytranscribe.prepare import main; raise SystemExit(main(sys.argv[1:]))' \
  "${prep_args[@]}" 2>&1 >"$TMP_DIR/prep.out")"; then
  die "${prep_err:-Could not prepare the audio}"
fi
prep_out="$(cat "$TMP_DIR/prep.out")"

declare -A meta=()
while IFS='=' read -r k v; do [[ -n "$k" ]] && meta["$k"]="$v"; done <<<"$prep_out"

dur="${meta[duration]:-0}"
printf -v dur_h '%02d:%02d:%02d' $(( ${dur%.*}/3600 )) $(( ${dur%.*}%3600/60 )) $(( ${dur%.*}%60 ))
ok "Format: ${meta[format_name]:-?} | codec: ${meta[codec]:-?} | ${meta[sample_rate]:-?} Hz | ${meta[channels]:-?} ch | $dur_h"
if [[ "${meta[audio_tracks]:-1}" -gt 1 ]]; then
  warn "File has ${meta[audio_tracks]} audio tracks; using the first one."
fi
ok "Audio ready ($(du -h "$WAV" | cut -f1))"

if [[ $KEEP_WAV -eq 1 ]]; then
  cp -- "$WAV" "${OUTPUT%.*}.16k.wav"
  info "Kept WAV: ${OUTPUT%.*}.16k.wav"
fi

# A hook for the tests: stop after preparation, before any model or venv work,
# so the audio stage can be compared against mytranscribe/audio.py on its own.
if [[ "${MYTRANSCRIBE_STOP_AFTER_PREPARE:-0}" == "1" ]]; then
  if [[ $KEEP_WAV -eq 1 ]]; then
    exit 0
  fi
  die "MYTRANSCRIBE_STOP_AFTER_PREPARE=1 needs --keep-wav so the WAV survives"
fi

# ------------------------------------------------ 3) venv + dependencies ---
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  info "Creating venv: $VENV_DIR"
  if PY="$(find_python)"; then
    "$PY" -m venv "$VENV_DIR" 2>/dev/null || {
      ver="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      die "venv module missing. Run:  sudo apt install python${ver}-venv python${ver}-dev"
    }
  elif command -v uv >/dev/null; then
    info "No Python >= 3.12 on the system, using uv to fetch one"
    uv venv --seed --python 3.12 "$VENV_DIR"
  else
    die "sea_asr needs Python >= 3.12 (Ubuntu 24.04 has it). On older Ubuntu either:
      curl -LsSf https://astral.sh/uv/install.sh | sh     # then re-run this script
   or sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv python3.12-dev"
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || die "$VENV_DIR uses Python $(python -V 2>&1); sea_asr needs >= 3.12. Delete it or set VENV_DIR."

pip_i() { python -m pip install --disable-pip-version-check "$@"; }

if [[ $REINSTALL -eq 1 ]]; then rm -f "$VENV_DIR"/.sea_asr_*; fi

if [[ ! -f "$VENV_DIR/.sea_asr_base" ]]; then
  info "Installing CPU-only PyTorch (avoids the ~3 GB CUDA build)"
  pip_i -q --upgrade pip wheel setuptools
  pip_i --index-url "$TORCH_CPU_INDEX" "torch==$TORCH_VERSION" \
    || { warn "torch $TORCH_VERSION CPU wheel not found, installing latest CPU build";
         pip_i --index-url "$TORCH_CPU_INDEX" torch; }
  info "Installing sea_asr $SEA_ASR_REF"
  pip_i --extra-index-url "$TORCH_CPU_INDEX" \
    "sea-asr @ git+https://github.com/PixelML/sea-asr@$SEA_ASR_REF"
  touch "$VENV_DIR/.sea_asr_base"
fi

if [[ "$MODE" != "greedy" && ! -f "$VENV_DIR/.sea_asr_lm" ]]; then
  info "Installing n-gram LM support (kenlm + pyctcdecode; kenlm compiles, takes a minute)"
  pip_i --extra-index-url "$TORCH_CPU_INDEX" \
    "sea-asr[lm] @ git+https://github.com/PixelML/sea-asr@$SEA_ASR_REF" \
    || die "LM install failed (usually the kenlm C++ build). Run:
      sudo apt install build-essential cmake python3-dev zlib1g-dev libbz2-dev liblzma-dev
   (on Ubuntu 22.04 + deadsnakes use python3.12-dev), then re-run, or use --mode greedy."
  touch "$VENV_DIR/.sea_asr_lm"
fi
# rescore needs no extra packages; Sailor2-1B (~2 GB) downloads on its first run.

# ------------------------------------------------------- 4) transcribe ---
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
[[ "$MODE" == "rescore" ]] && warn "rescore mode runs a 1B LM on CPU: expect it to be several times slower."

info "Transcribing (mode=$MODE, threads=$THREADS of $MAX_CORES cores). First run downloads the model."
export MYTRANSCRIBE_ROOT="$PWD"
python - "$WAV" "$OUTPUT" "$FMT" "$MODE" "$THREADS" "$dur" <<'PY'
import sys
sys.path.insert(0, __import__("os").environ["MYTRANSCRIBE_ROOT"])
from mytranscribe.cli import main
raise SystemExit(main(sys.argv[1:]))
PY

ok "Saved: $OUTPUT"
