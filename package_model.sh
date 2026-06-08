#!/usr/bin/env bash
# =============================================================================
# Model Packaging Script — 把模型目录打成 tar.gz 准备上传 bucket
# =============================================================================
#
# 用法:
#   bash package_model.sh                           # 打包默认目录
#   bash package_model.sh -d my-model-dir            # 指定模型目录
#   bash package_model.sh -o my-model.tar.gz         # 指定输出文件名
#
# 上传后把链接写入 MODEL_URL 文件，deploy.sh 会自动读取。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Config -------------------------------------------------------------------

MODEL_DIR="${MODEL_DIR:-whisper-large-v3-turbo-finetuned}"
OUTPUT="${OUTPUT:-whisper-large-v3-turbo-finetuned.tar.gz}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)    MODEL_DIR="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash package_model.sh [-d MODEL_DIR] [-o OUTPUT.tar.gz]"
            echo ""
            echo "  默认打包 whisper-large-v3-turbo-finetuned/ → whisper-large-v3-turbo-finetuned.tar.gz"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers ------------------------------------------------------------------

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Check --------------------------------------------------------------------

if [ ! -d "$MODEL_DIR" ]; then
    log_error "模型目录不存在: $MODEL_DIR"
    exit 1
fi

FILE_COUNT=$(find "$MODEL_DIR" -type f | wc -l)
DIR_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
log_info "模型目录: $MODEL_DIR ($FILE_COUNT 个文件, $DIR_SIZE)"

# --- Package ------------------------------------------------------------------

PARENT_DIR="$(dirname "$MODEL_DIR")"
MODEL_BASENAME="$(basename "$MODEL_DIR")"

log_info "打包中..."
tar -czf "$OUTPUT" -C "$PARENT_DIR" "$MODEL_BASENAME"

ARCHIVE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
log_info "完成: $OUTPUT ($ARCHIVE_SIZE)"

# --- Instructions -------------------------------------------------------------

echo ""
echo "=============================================="
echo "  上传到 bucket 后，把链接写入 MODEL_URL 文件"
echo "=============================================="
echo ""
echo "  当前目录: $(pwd)"
echo "  压缩包:   $OUTPUT ($ARCHIVE_SIZE)"
echo ""
echo "  上传示例:"
echo "    # 阿里云 OSS (AutoDL 常用)"
echo "    ossutil cp $OUTPUT oss://your-bucket/models/$OUTPUT"
echo ""
echo "    # AWS S3"
echo "    aws s3 cp $OUTPUT s3://your-bucket/models/$OUTPUT"
echo ""
echo "    # rclone"
echo "    rclone copy $OUTPUT remote:bucket/models/"
echo ""
echo "  上传后编辑 MODEL_URL 文件，写入下载地址:"
echo "    echo 'https://your-bucket.example.com/models/$OUTPUT' > MODEL_URL"
echo ""
echo "  部署时 deploy.sh 会自动读取 MODEL_URL 下载模型。"
echo "=============================================="
