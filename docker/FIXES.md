# Dockerfile 修复说明

根据官方安装指南 [install_cuda_pp.md](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support)，对 Dockerfile 进行了以下修复：

## 主要修改

### 1. 更新基础镜像
- **修改前**: `FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **修改后**: `FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04`
- **原因**: 官方指南推荐使用 CUDA 12.6

### 2. 严格按照官方指南的安装顺序

#### Step 1: 安装 PaddlePaddle 和 PaddleX
```dockerfile
# Step 1: Install PaddlePaddle and PaddleX (as per official guide)
ARG CUDA_VERSION=126
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/

# Execute the following command to install the base version of PaddleX
RUN pip install "paddlex[base]"
```

#### Step 2: 安装 PyTorch
```dockerfile
# Step 2: Install PyTorch (as per official guide)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

#### Step 3: 安装项目依赖
```dockerfile
# Step 3: Install project dependencies from requirements.txt
RUN pip install -r requirements.txt
```

#### Step 4: 安装项目本身
```dockerfile
# Step 4: Install project itself
RUN pip install -e .
```

#### Step 5: 安装 LMDeploy (推荐的后端)
```dockerfile
# Step 5: Install LMDeploy (recommended backend)
RUN pip install lmdeploy==0.8.0
```

### 3. 关键改进

1. **正确的安装顺序**: 严格按照官方指南的步骤进行安装
2. **依赖管理**: 使用 `requirements.txt` 确保所有依赖都正确安装
3. **CUDA 版本一致性**: 使用 CUDA 12.6 与官方指南保持一致
4. **PaddlePaddle 安装**: 使用官方推荐的安装命令和版本
5. **构建时模型下载**: 在构建阶段下载模型，加快容器启动速度
6. **移除代理设置**: 保持干净的安装环境，不使用镜像源
7. **简化模型下载**: 只使用 huggingface_hub，移除 modelscope 依赖

### 4. 模型下载优化

**构建时下载模型**:
```dockerfile
# Install model download related dependencies (only huggingface_hub)
RUN pip install huggingface_hub

# Download models during build (as root for better permissions)
RUN python tools/download_model.py -n MonkeyOCR
```

**运行时检查模型**:
```bash
# Check if models exist (they should be downloaded during build)
log_info "Checking if models are available..."
if [ -d "/app/MonkeyOCR/model_weight/Recognition" ] && [ -d "/app/MonkeyOCR/model_weight/Structure" ]; then
    log_info "Models are ready (downloaded during build)"
else
    log_warn "Models not found, attempting to download..."
    # Fallback to runtime download
fi
```

## 与官方指南的对应关系

| 官方指南步骤 | Dockerfile 对应部分 |
|-------------|-------------------|
| Step 1: Install PaddleX | Step 1: Install PaddlePaddle and PaddleX |
| Step 2: Install Inference Backend | Step 2-5: PyTorch + LMDeploy |
| LMDeploy (Recommended) | Step 5: Install LMDeploy |
| Model Download | Build-time download + Runtime check |

## 优势

1. **更快的容器启动**: 模型在构建时下载，容器启动时无需等待下载
2. **更可靠的部署**: 模型文件包含在镜像中，减少网络依赖
3. **更干净的安装**: 移除代理设置，使用官方源
4. **更好的权限管理**: 以 root 权限下载模型，确保完整下载
5. **简化的依赖**: 只使用 huggingface_hub，减少不必要的依赖
6. **明确的模型指定**: 明确指定下载 MonkeyOCR 模型

## 测试方法

1. **构建测试**:
   ```bash
   cd docker
   docker build -t monkeyocr:test .
   ```

2. **运行测试**:
   ```bash
   docker compose build monkeyocr
   docker compose up monkeyocr-demo
   ```

## 注意事项

1. **CUDA 版本**: 确保主机系统支持 CUDA 12.6
2. **GPU 驱动**: 确保安装了兼容的 NVIDIA 驱动
3. **内存要求**: 建议至少 8GB GPU 内存用于模型推理
4. **LMDeploy 补丁**: 对于 20/30/40 系列 GPU，可能需要应用 LMDeploy 补丁
5. **构建时间**: 由于包含模型下载，构建时间会较长
6. **镜像大小**: 包含模型文件后，镜像大小会显著增加
7. **网络要求**: 需要稳定的网络连接来下载模型

## 故障排除

如果遇到构建问题：

1. **检查 CUDA 版本兼容性**
2. **验证网络连接** (用于下载依赖和模型)
3. **检查磁盘空间** (构建过程需要大量空间)
4. **查看构建日志** 以识别具体错误
5. **模型下载失败**: 检查网络连接和 HuggingFace 可用性

## 相关文件

- `docker/Dockerfile` - 修复后的 Dockerfile
- `docker/entrypoint.sh` - 更新后的容器启动脚本
- `docker/download_models.sh` - 简化的模型下载脚本
- `requirements.txt` - 项目依赖列表 