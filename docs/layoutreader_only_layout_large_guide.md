# LayoutReader-only-layout-large Usage Guide

This guide provides comprehensive instructions for using the **LayoutReader-only-layout-large** model in MonkeyOCR for document reading order prediction.

## 🚀 Quick Start

### Prerequisites

Ensure you have MonkeyOCR installed with CUDA support. Refer to the [CUDA Installation Guide](install_cuda.md) for detailed setup instructions.

### 1. Model Configuration

Edit your `model_configs.yaml` file to use the LayoutReader-only-layout-large model:

```yaml
weights:
  layoutReader-only-layout-large: LayoutReader-only-layout-large  # Relative path to `model_weight`
layout_config: 
  model: doclayout_yolo
  reader:
    name: layoutReader-only-layout-large  # Update the layoutreader model name

```

### 2. Model Download

Download the LayoutReader-only-layout-large model files and place them in the appropriate directory:

```bash
mkdir -p model_weight/LayoutReader-only-layout-large

modelscope download --model yujunhuinlp/LayoutReader-only-layout-large --local_dir model_weight/LayoutReader-only-layout-large

```

