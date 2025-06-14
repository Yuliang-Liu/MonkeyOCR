#!/bin/bash

set -e

MODEL_DIR="/app/MonkeyOCR/model_weight"
TOOLS_DIR="/app/MonkeyOCR/tools"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if models already exist
check_models_exist() {
    local model_files=(
        "models--echo840--MonkeyOCR/snapshots"
        "Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
        "Structure/layout_zh.pt"
    )
    
    for model_file in "${model_files[@]}"; do
        if [ -d "$MODEL_DIR/$model_file" ] || [ -f "$MODEL_DIR/$model_file" ]; then
            log_info "Found existing model file: $model_file"
            return 0
        fi
    done
    return 1
}

# Download models using ModelScope
download_with_modelscope() {
    log_info "Downloading models using ModelScope..."
    
    cd /app/MonkeyOCR
    
    if python "$TOOLS_DIR/download_model.py" -t modelscope; then
        log_info "ModelScope download successful!"
        return 0
    else
        log_error "ModelScope download failed"
        return 1
    fi
}

# Download models using HuggingFace
download_with_huggingface() {
    log_info "Downloading models using HuggingFace..."
    
    cd /app/MonkeyOCR
    
    if python "$TOOLS_DIR/download_model.py"; then
        log_info "HuggingFace download successful!"
        return 0
    else
        log_error "HuggingFace download failed"
        return 1
    fi
}

# Main download logic
download_models() {
    if check_models_exist; then
        log_info "Model files already exist, skipping download"
        return 0
    fi
    
    log_info "Starting MonkeyOCR model download..."
    
    # Check if download script exists
    if [ ! -f "$TOOLS_DIR/download_model.py" ]; then
        log_error "Download script not found: $TOOLS_DIR/download_model.py"
        return 1
    fi
    
    # Try ModelScope first
    if download_with_modelscope; then
        return 0
    fi
    
    log_warn "ModelScope download failed, switching to HuggingFace..."
    sleep 2
    
    # Fallback to HuggingFace
    if download_with_huggingface; then
        return 0
    fi
    
    log_error "All download methods failed!"
    log_error "Please check network connection or manually download models"
    log_error "Manual download commands:"
    log_error "  python tools/download_model.py -t modelscope  # or"
    log_error "  python tools/download_model.py              # HuggingFace"
    
    return 1
}

# Execute download
download_models
