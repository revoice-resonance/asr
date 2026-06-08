#!/usr/bin/env bash
# =============================================================================
# Model Packaging Script
# =============================================================================
# Converts whisper-large-v3-turbo from HuggingFace format to CTranslate2,
# then creates a tar.gz archive ready for upload to a bucket.
#
# Usage:
#   bash package_model.sh                          # convert + package
#   bash package_model.sh --skip-convert            # package existing ct2 model
#   bash package_model.sh --output my-model.tar.gz  # custom output name
#
# Prerequisites:
#   pip install ctranslate2 transformers
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Config -------------------------------------------------------------------

HF_MODEL_DIR="${HF_MODEL_DIR:-whisper-large-v3-turbo-finetuned}"
CT2_MODEL_DIR="${CT2_MODEL_DIR:-models/whisper-large-v3-turbo-ct2}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"   # float16 | int8_float16 | int8
OUTPUT="${OUTPUT:-whisper-large-v3-turbo-ct2.tar.gz}"
SKIP_CONVERT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --compute-type) COMPUTE_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers ------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Step 1: Convert HF → CTranslate2 -----------------------------------------

if $SKIP_CONVERT; then
    log_info "Skipping conversion, using existing model at $CT2_MODEL_DIR"
else
    log_info "Converting $HF_MODEL_DIR → $CT2_MODEL_DIR (compute_type=$COMPUTE_TYPE)"

    if [ ! -d "$HF_MODEL_DIR" ]; then
        log_error "HuggingFace model not found at $HF_MODEL_DIR"
        log_error "Place your fine-tuned model in $HF_MODEL_DIR/ or use --skip-convert"
        exit 1
    fi

    if [ ! -f "$HF_MODEL_DIR/model.safetensors" ] && [ ! -f "$HF_MODEL_DIR/pytorch_model.bin" ]; then
        log_error "No model weights found in $HF_MODEL_DIR/"
        exit 1
    fi

    # Install converter if needed
    pip install ctranslate2 transformers -q 2>/dev/null || {
        log_error "Failed to install ctranslate2. Run: pip install ctranslate2 transformers"
        exit 1
    }

    # Convert
    log_info "Running ct2-transformers-converter..."
    ct2-transformers-converter \
        --model "$HF_MODEL_DIR" \
        --output_dir "$CT2_MODEL_DIR" \
        --copy_files tokenizer.json preprocessor_config.json \
        --quantization "$COMPUTE_TYPE" \
        --force

    log_info "Conversion complete"
fi

# --- Step 2: Verify CTranslate2 model -----------------------------------------

if [ ! -f "$CT2_MODEL_DIR/model.bin" ] && [ ! -f "$CT2_MODEL_DIR/config.json" ]; then
    log_error "CTranslate2 model not found at $CT2_MODEL_DIR"
    exit 1
fi

log_info "Model files:"
find "$CT2_MODEL_DIR" -type f -exec ls -lh {} \; | awk '{print "  " $NF " (" $5 ")"}'

# --- Step 3: Create archive ---------------------------------------------------

log_info "Creating archive: $OUTPUT"

# Get the parent dir and basename for clean extraction paths
PARENT_DIR="$(dirname "$CT2_MODEL_DIR")"
MODEL_BASENAME="$(basename "$CT2_MODEL_DIR")"

tar -czf "$OUTPUT" -C "$PARENT_DIR" "$MODEL_BASENAME"

ARCHIVE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
log_info "Archive created: $OUTPUT ($ARCHIVE_SIZE)"

# --- Step 4: Print upload instructions ----------------------------------------

echo ""
echo "=============================================="
echo "  Model package ready for upload"
echo "=============================================="
echo ""
echo "  Archive:  $(realpath "$OUTPUT")"
echo "  Size:     $ARCHIVE_SIZE"
echo ""
echo "  Upload to your bucket, then set in .env:"
echo "    MODEL_DOWNLOAD_URL=https://your-bucket.example.com/$OUTPUT"
echo ""
echo "  Example upload commands:"
echo "    # AWS S3"
echo "    aws s3 cp $OUTPUT s3://your-bucket/models/$OUTPUT"
echo ""
echo "    # Alibaba Cloud OSS (AutoDL常用)"
echo "    ossutil cp $OUTPUT oss://your-bucket/models/$OUTPUT"
echo ""
echo "    # Google Cloud Storage"
echo "    gsutil cp $OUTPUT gs://your-bucket/models/$OUTPUT"
echo ""
echo "    # rclone (any provider)"
echo "    rclone copy $OUTPUT remote:bucket/models/"
echo ""
echo "  On the server, deploy.sh will auto-download and extract:"
echo "    bash deploy.sh start"
echo "=============================================="
