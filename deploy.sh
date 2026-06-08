#!/usr/bin/env bash
# =============================================================================
# Whisper ASR API — Non-invasive Deployment Script
# =============================================================================
# Designed for AutoDL and similar GPU cloud environments.
# - Creates an isolated venv (does NOT touch system Python)
# - Downloads model from bucket or converts from HuggingFace
# - Manages the server via PID file
# - Supports HTTP proxy for pip and model downloads
#
# Usage:
#   bash deploy.sh start          # Start the server
#   bash deploy.sh stop           # Stop the server
#   bash deploy.sh restart        # Restart the server
#   bash deploy.sh status         # Check server status
#   bash deploy.sh logs [N]       # Tail last N lines of logs (default 50)
#   bash deploy.sh setup          # Install deps + download model (no start)
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
# These can be overridden by environment variables or a .env file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Defaults (matching .env.example)
APP_NAME="whisper_api"
VENV_DIR="${VENV_DIR:-venv}"
PID_FILE="${PID_FILE:-${APP_NAME}.pid}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${APP_NAME}.log"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_PATH="${MODEL_PATH:-models/whisper-large-v3-turbo-ct2}"
MODEL_DOWNLOAD_URL="${MODEL_DOWNLOAD_URL:-}"
HF_MODEL_ID="${HF_MODEL_ID:-}"
MODEL_COMPUTE_TYPE="${MODEL_COMPUTE_TYPE:-float16}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
MODEL_DEVICE_INDEX="${MODEL_DEVICE_INDEX:-0}"
DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-zh}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_FORMAT="${LOG_FORMAT:-json}"

# Proxy settings
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helpers -----------------------------------------------------------------

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

die() {
    log_error "$*"
    exit 1
}

# --- Proxy Setup -------------------------------------------------------------

setup_proxy() {
    if [ -n "$HTTP_PROXY" ]; then
        log_info "Using HTTP proxy: $HTTP_PROXY"
        export http_proxy="$HTTP_PROXY"
        export HTTP_PROXY="$HTTP_PROXY"
    fi
    if [ -n "$HTTPS_PROXY" ]; then
        log_info "Using HTTPS proxy: $HTTPS_PROXY"
        export https_proxy="$HTTPS_PROXY"
        export HTTPS_PROXY="$HTTPS_PROXY"
    fi
}

# --- Prerequisite Checks -----------------------------------------------------

check_prerequisites() {
    log_step "Checking prerequisites..."

    # Python 3.10+
    if ! command -v python3 &>/dev/null; then
        die "python3 not found. Please install Python 3.10+."
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"

    # CUDA / GPU detection (must run before ffmpeg install to decide GPU vs CPU)
    HAS_NVIDIA=false
    if command -v nvidia-smi &>/dev/null; then
        HAS_NVIDIA=true
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    else
        log_warn "nvidia-smi not found. GPU may not be available."
    fi

    # ffmpeg
    install_ffmpeg

    # Verify ffmpeg works
    if ! ffmpeg -version &>/dev/null; then
        die "ffmpeg installation failed. Please install it manually."
    fi
    log_info "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

    # Report GPU acceleration status
    if $HAS_NVIDIA; then
        if ffmpeg -codecs 2>/dev/null | grep -q h264_nvenc; then
            log_info "ffmpeg NVIDIA GPU acceleration: ENABLED (h264_nvenc, hevc_nvenc)"
        else
            log_warn "ffmpeg NVIDIA GPU acceleration: NOT available (CPU decode only)"
        fi
    fi
}

# --- ffmpeg Installation -------------------------------------------------------

install_ffmpeg() {
    # Already installed — check if it meets our needs
    if command -v ffmpeg &>/dev/null; then
        if $HAS_NVIDIA && ! ffmpeg -codecs 2>/dev/null | grep -q h264_nvenc; then
            log_warn "ffmpeg found but lacks NVIDIA GPU codecs. Attempting upgrade..."
            # Fall through to install GPU-enabled version
        else
            log_info "ffmpeg already installed with required capabilities"
            return 0
        fi
    else
        log_warn "ffmpeg not found. Installing..."
    fi

    # Strategy 1: NVIDIA GPU → download BtbN static build with NVENC/NVDEC/CUDA
    if $HAS_NVIDIA && install_ffmpeg_nvidia; then
        return 0
    fi

    # Strategy 2: conda (often ships GPU-enabled ffmpeg on CUDA machines)
    if command -v conda &>/dev/null; then
        log_info "Installing ffmpeg via conda-forge..."
        if conda install -y -c conda-forge ffmpeg 2>/dev/null; then
            if command -v ffmpeg &>/dev/null; then
                log_info "ffmpeg installed via conda"
                return 0
            fi
        fi
        log_warn "conda install failed, trying next method..."
    fi

    # Strategy 3: apt-get (standard Ubuntu ffmpeg, CPU-only but reliable)
    if command -v apt-get &>/dev/null; then
        log_info "Installing ffmpeg via apt-get..."
        if sudo apt-get update -qq 2>/dev/null && sudo apt-get install -y -qq ffmpeg 2>/dev/null; then
            if command -v ffmpeg &>/dev/null; then
                log_info "ffmpeg installed via apt-get"
                return 0
            fi
        fi
        log_warn "apt-get install failed, trying next method..."
    fi

    # Strategy 4: static CPU-only build (last resort, works anywhere)
    if install_ffmpeg_static; then
        return 0
    fi

    return 1
}

install_ffmpeg_nvidia() {
    # Download pre-built ffmpeg from BtbN/FFmpeg-Builds
    # Includes: h264_nvenc, hevc_nvenc, h264_cuvid, hevc_cuvid, and CUDA filters
    local BASE_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
    local ARCHIVE="ffmpeg-master-latest-linux64-gpl-shared.tar.xz"
    local TMP_DIR="/tmp/ffmpeg_nvidia_$$"

    log_info "Downloading GPU-accelerated ffmpeg (BtbN build with NVENC/NVDEC)..."

    mkdir -p "$TMP_DIR"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${TMP_DIR}/${ARCHIVE}" "${BASE_URL}/${ARCHIVE}" || {
            log_warn "Failed to download GPU ffmpeg"
            rm -rf "$TMP_DIR"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -# -L -o "${TMP_DIR}/${ARCHIVE}" "${BASE_URL}/${ARCHIVE}" || {
            log_warn "Failed to download GPU ffmpeg"
            rm -rf "$TMP_DIR"
            return 1
        }
    else
        rm -rf "$TMP_DIR"
        return 1
    fi

    log_info "Extracting ffmpeg..."
    tar -xf "${TMP_DIR}/${ARCHIVE}" -C "$TMP_DIR" 2>/dev/null || {
        log_warn "Failed to extract GPU ffmpeg archive"
        rm -rf "$TMP_DIR"
        return 1
    }

    # Find extracted directory (name varies by build date)
    local EXTRACTED_DIR
    EXTRACTED_DIR=$(find "$TMP_DIR" -maxdepth 1 -type d -name "ffmpeg-master-*" | head -1)
    if [ -z "$EXTRACTED_DIR" ]; then
        log_warn "Could not find extracted ffmpeg directory"
        rm -rf "$TMP_DIR"
        return 1
    fi

    # Install to /usr/local (requires sudo) or ~/.local/bin (user-local)
    local BIN_DIR
    if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
        BIN_DIR="/usr/local/bin"
        sudo cp "${EXTRACTED_DIR}/bin/ffmpeg" "${EXTRACTED_DIR}/bin/ffprobe" "$BIN_DIR/" 2>/dev/null || {
            BIN_DIR="${HOME}/.local/bin"
            mkdir -p "$BIN_DIR"
            cp "${EXTRACTED_DIR}/bin/ffmpeg" "${EXTRACTED_DIR}/bin/ffprobe" "$BIN_DIR/"
        }
        # Install shared libraries so ffmpeg can find them
        sudo cp -r "${EXTRACTED_DIR}/lib/"* /usr/local/lib/ 2>/dev/null || true
        sudo ldconfig 2>/dev/null || true
    else
        BIN_DIR="${HOME}/.local/bin"
        mkdir -p "$BIN_DIR"
        cp "${EXTRACTED_DIR}/bin/ffmpeg" "${EXTRACTED_DIR}/bin/ffprobe" "$BIN_DIR/"
        # Add to PATH for this session
        export PATH="${HOME}/.local/bin:${PATH}"
        log_info "Installed ffmpeg to ${HOME}/.local/bin (add to PATH if not already)"
    fi

    # Cleanup
    rm -rf "$TMP_DIR"

    # Verify NVIDIA codecs are available
    if command -v ffmpeg &>/dev/null && ffmpeg -codecs 2>/dev/null | grep -q h264_nvenc; then
        log_info "GPU-accelerated ffmpeg installed successfully"
        return 0
    fi

    log_warn "GPU ffmpeg installed but NVIDIA codecs not detected (driver mismatch?)"
    return 1
}

install_ffmpeg_static() {
    # Fallback: johnvansickle.com static build (CPU-only, works on any amd64 Linux)
    local BASE_URL="https://johnvansickle.com/ffmpeg/releases"
    local ARCHIVE="ffmpeg-release-amd64-static.tar.xz"
    local TMP_DIR="/tmp/ffmpeg_static_$$"

    log_info "Downloading static ffmpeg build (CPU-only)..."

    mkdir -p "$TMP_DIR"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${TMP_DIR}/${ARCHIVE}" "${BASE_URL}/${ARCHIVE}" || {
            rm -rf "$TMP_DIR"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -# -L -o "${TMP_DIR}/${ARCHIVE}" "${BASE_URL}/${ARCHIVE}" || {
            rm -rf "$TMP_DIR"
            return 1
        }
    else
        rm -rf "$TMP_DIR"
        return 1
    fi

    tar -xf "${TMP_DIR}/${ARCHIVE}" -C "$TMP_DIR" 2>/dev/null || {
        rm -rf "$TMP_DIR"
        return 1
    }

    local EXTRACTED_DIR
    EXTRACTED_DIR=$(find "$TMP_DIR" -maxdepth 1 -type d -name "ffmpeg-*-static" | head -1)
    if [ -z "$EXTRACTED_DIR" ]; then
        rm -rf "$TMP_DIR"
        return 1
    fi

    local BIN_DIR
    if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
        BIN_DIR="/usr/local/bin"
        sudo cp "${EXTRACTED_DIR}/ffmpeg" "${EXTRACTED_DIR}/ffprobe" "$BIN_DIR/" 2>/dev/null || {
            BIN_DIR="${HOME}/.local/bin"
            mkdir -p "$BIN_DIR"
            cp "${EXTRACTED_DIR}/ffmpeg" "${EXTRACTED_DIR}/ffprobe" "$BIN_DIR/"
        }
    else
        BIN_DIR="${HOME}/.local/bin"
        mkdir -p "$BIN_DIR"
        cp "${EXTRACTED_DIR}/ffmpeg" "${EXTRACTED_DIR}/ffprobe" "$BIN_DIR/"
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    rm -rf "$TMP_DIR"

    if command -v ffmpeg &>/dev/null; then
        log_info "Static ffmpeg installed to ${BIN_DIR}"
        return 0
    fi
    return 1
}

# --- Virtual Environment -----------------------------------------------------

setup_venv() {
    log_step "Setting up virtual environment..."

    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_info "Created venv at $VENV_DIR"
    else
        log_info "Venv already exists at $VENV_DIR"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip -q

    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt -q

    log_info "Dependencies installed successfully"
}

# --- Model Setup -------------------------------------------------------------

setup_model() {
    log_step "Setting up model..."

    MODEL_DIR="$(dirname "$MODEL_PATH")"
    mkdir -p "$MODEL_DIR"

    # Case 1: Model already exists
    if [ -f "$MODEL_PATH/model.bin" ] || [ -f "$MODEL_PATH/config.json" ]; then
        log_info "Model already exists at $MODEL_PATH"
        return 0
    fi

    # Case 2: Download from bucket URL
    if [ -n "$MODEL_DOWNLOAD_URL" ]; then
        log_info "Downloading model from $MODEL_DOWNLOAD_URL ..."

        ARCHIVE="/tmp/whisper_model_$$.tar.gz"
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "$ARCHIVE" "$MODEL_DOWNLOAD_URL" || {
                rm -f "$ARCHIVE"
                die "Failed to download model"
            }
        elif command -v curl &>/dev/null; then
            curl -L -o "$ARCHIVE" "$MODEL_DOWNLOAD_URL" || {
                rm -f "$ARCHIVE"
                die "Failed to download model"
            }
        else
            die "Neither wget nor curl found"
        fi

        log_info "Extracting model..."
        mkdir -p "$MODEL_PATH"
        tar -xzf "$ARCHIVE" -C "$MODEL_PATH" --strip-components=1 2>/dev/null || \
            tar -xzf "$ARCHIVE" -C "$MODEL_PATH" 2>/dev/null || \
            unzip -q "$ARCHIVE" -d "$MODEL_PATH" 2>/dev/null || \
            die "Failed to extract model archive"

        rm -f "$ARCHIVE"
        log_info "Model downloaded and extracted to $MODEL_PATH"
        return 0
    fi

    # Case 3: Convert from HuggingFace model
    if [ -n "$HF_MODEL_ID" ]; then
        log_info "Converting HuggingFace model '$HF_MODEL_ID' to CTranslate2 format..."

        # Install ct2-transformers-converter if needed
        pip install ctranslate2 -q 2>/dev/null || true

        python3 -c "
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = '$HF_MODEL_ID'
target = '$MODEL_PATH'
os.makedirs(target, exist_ok=True)

print(f'Downloading {model_id}...')
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

print(f'Saving to {target}...')
model.save_pretrained(target)
processor.save_pretrained(target)
print('Done. Now converting with ct2-transformers-converter...')
" || die "Failed to download HF model"

        ct2-transformers-converter \
            --model "$MODEL_PATH" \
            --output_dir "$MODEL_PATH" \
            --copy_files tokenizer.json preprocessor_config.json \
            --quantization "$MODEL_COMPUTE_TYPE" \
            --force 2>/dev/null || {
            log_warn "ct2-transformers-converter failed. Trying alternative method..."
            # Alternative: use faster-whisper's built-in conversion
            python3 -c "
from faster_whisper.utils import download_model
download_model('$HF_MODEL_ID', output_dir='$MODEL_PATH')
" || die "Model conversion failed"
        }

        log_info "Model converted to CTranslate2 format at $MODEL_PATH"
        return 0
    fi

    # Case 4: Try local HuggingFace model directory
    LOCAL_HF_DIR="whisper-large-v3-turbo-finetuned"
    if [ -d "$LOCAL_HF_DIR" ] && [ -f "$LOCAL_HF_DIR/config.json" ]; then
        log_info "Found local HuggingFace model at $LOCAL_HF_DIR, converting..."

        pip install ctranslate2 -q 2>/dev/null || true

        ct2-transformers-converter \
            --model "$LOCAL_HF_DIR" \
            --output_dir "$MODEL_PATH" \
            --copy_files tokenizer.json preprocessor_config.json \
            --quantization "$MODEL_COMPUTE_TYPE" \
            --force 2>/dev/null || {
            log_warn "Automatic conversion failed."
            log_warn "Please convert manually or set MODEL_DOWNLOAD_URL / HF_MODEL_ID in .env"
            die "Model setup failed"
        }

        log_info "Local model converted to $MODEL_PATH"
        return 0
    fi

    die "No model found. Set MODEL_DOWNLOAD_URL or HF_MODEL_ID in .env, or place a model at $MODEL_PATH"
}

# --- Server Management -------------------------------------------------------

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_server() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        log_warn "Server is already running (PID: $pid)"
        return 1
    fi

    log_step "Starting server..."

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    # Activate venv
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

    # Export env vars for the server process
    export HOST PORT MODEL_PATH MODEL_COMPUTE_TYPE MODEL_DEVICE MODEL_DEVICE_INDEX
    export DEFAULT_LANGUAGE LOG_LEVEL LOG_FORMAT

    # Start server in background
    nohup python3 -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --workers 1 \
        >> "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Wait a moment and check if it's still running
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        log_info "Server started successfully (PID: $pid)"
        log_info "Listening on http://${HOST}:${PORT}"
        log_info "Health check: http://${HOST}:${PORT}/health/live"
        log_info "API docs:   http://${HOST}:${PORT}/docs"
        log_info "Logs:       tail -f $LOG_FILE"
    else
        log_error "Server failed to start. Check logs:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    if ! is_running; then
        log_warn "Server is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    log_step "Stopping server (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 30 seconds for graceful shutdown
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ $waited -lt 30 ]; do
        sleep 1
        waited=$((waited + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Server did not stop gracefully, force killing..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$PID_FILE"
    log_info "Server stopped"
}

show_status() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo -e "${GREEN}Server is running${NC}"
        echo "  PID:       $pid"
        echo "  Host:      $HOST"
        echo "  Port:      $PORT"
        echo "  Model:     $MODEL_PATH"
        echo "  Device:    $MODEL_DEVICE"
        echo "  Log file:  $LOG_FILE"

        # Try health check
        if command -v curl &>/dev/null; then
            echo ""
            echo "Health check:"
            curl -s "http://${HOST}:${PORT}/health/live" 2>/dev/null || echo "  (unreachable)"
        fi
    else
        echo -e "${RED}Server is not running${NC}"
        rm -f "$PID_FILE"
    fi
}

show_logs() {
    local lines="${1:-50}"
    if [ -f "$LOG_FILE" ]; then
        tail -n "$lines" "$LOG_FILE"
    else
        log_warn "No log file found at $LOG_FILE"
    fi
}

# --- Main --------------------------------------------------------------------

main() {
    local cmd="${1:-}"

    case "$cmd" in
        setup)
            setup_proxy
            check_prerequisites
            setup_venv
            setup_model
            log_info "Setup complete. Run 'bash deploy.sh start' to start the server."
            ;;

        start)
            setup_proxy
            check_prerequisites
            setup_venv
            setup_model
            start_server
            ;;

        stop)
            stop_server
            ;;

        restart)
            stop_server
            sleep 2
            start_server
            ;;

        status)
            show_status
            ;;

        logs)
            show_logs "${2:-50}"
            ;;

        *)
            echo "Whisper ASR API — Deployment Script"
            echo ""
            echo "Usage: bash deploy.sh <command> [options]"
            echo ""
            echo "Commands:"
            echo "  setup      Install dependencies and download model (don't start)"
            echo "  start      Start the server (setup + start)"
            echo "  stop       Stop the server gracefully"
            echo "  restart    Stop then start the server"
            echo "  status     Show server status and health"
            echo "  logs [N]   Tail last N lines of logs (default: 50)"
            echo ""
            echo "Environment (.env or export):"
            echo "  MODEL_DOWNLOAD_URL   URL to download model archive"
            echo "  HF_MODEL_ID          HuggingFace model ID for conversion"
            echo "  HTTP_PROXY           Proxy for pip/downloads"
            echo "  PORT                 Server port (default: 8080)"
            echo "  MODEL_PATH           CTranslate2 model directory"
            echo ""
            echo "First time: copy .env.example to .env and configure."
            exit 0
            ;;
    esac
}

main "$@"
