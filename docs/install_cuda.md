# Install with CUDA Support

This guide walks you through setting up the environment for **MonkeyOCR** with CUDA support. You can choose **one** of the backends — [**LMDeploy**](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#using-lmdeploy-as-the-inference-backend-optional)(recomended), [**vLLM**](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#using-vllm-as-the-inference-backend-optional), or [**transformers**](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#using-transformers-as-the-inference-backend-optional) — to install and use. It covers installation instructions for each of them.

> **Note:** Based on our internal test, inference speed ranking is: **LMDeploy ≥ vLLM >>> transformers**

## Using **LMDeploy** as the Inference Backend (Optional)
> **Supporting CUDA 12.4/12.1/11.8**

If you're using **CUDA 12.4** or **CUDA 12.1**, follow these steps:

```bash
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR

git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

export CUDA_VERSION=124 # for CUDA 12.4
# export CUDA_VERSION=121 # for CUDA 12.1

# Install PyTorch. Refer to https://pytorch.org/get-started/previous-versions/ for version compatibility
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

pip install -e .

pip install lmdeploy==0.8.0
```

If you're using **CUDA 11.8**, use the following instead:

```bash
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR

git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

# Install PyTorch. Refer to https://pytorch.org/get-started/previous-versions/ for version compatibility
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e .

pip install https://github.com/InternLM/lmdeploy/releases/download/v0.8.0/lmdeploy-0.8.0+cu118-cp310-cp310-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

> [!IMPORTANT]
> ### Fixing the **Shared Memory Error** on **20/30/40 series / V100 ...** GPUs (Optional)
> 
> Our 3B model runs smoothly on the NVIDIA RTX 30/40 series. However, when using **LMDeploy** as the inference backend, you might run into compatibility issues on these GPUs — typically this error:
> 
> ```
> triton.runtime.errors.OutOfResources: out of resource: shared memory
> ```
> 
> To resolve this issue, apply the following patch:
> 
> ```bash
> python tools/lmdeploy_patcher.py patch
> ```
> **Note:** This command modifies LMDeploy’s source code in your environment.
> To undo the changes, simply run:
> 
> ```bash
> python tools/lmdeploy_patcher.py restore
> ```
> 
> Based on our tests on the **NVIDIA RTX 3090**, inference speed was **0.338 pages/second** using **LMDeploy** (with the patch applied), compared to only **0.015 pages/second** using **transformers**.
> 
> **Special thanks to [@pineking](https://github.com/pineking) for the solution!**

---

## Using **vLLM** as the Inference Backend (Optional)
> **Supporting CUDA 12.6/12.8/11.8**
```bash
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR

git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

pip install uv --upgrade
export CUDA_VERSION=126 # for CUDA 12.6
# export CUDA_VERSION=128 # for CUDA 12.8
# export CUDA_VERSION=118 # for CUDA 11.8
uv pip install vllm==0.9.1 --torch-backend=cu${CUDA_VERSION}

pip install -e .
```

Then, update the `chat_config.backend` field in your `model_configs.yaml` config file:

```yaml
chat_config:
    backend: vllm
```

---

## Using **transformers** as the Inference Backend (Optional)
> **Supporting CUDA 12.4/12.1**
```bash
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR

git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

pip install -e .
```

Install PyTorch according to your CUDA version:

```bash
export CUDA_VERSION=124 # for CUDA 12.4
# export CUDA_VERSION=121 # for CUDA 12.1

# Install pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

Install Flash Attention 2:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
Then, update the `chat_config` in your `model_configs.yaml` config file:
```yaml
chat_config:
  backend: transformers
  batch_size: 10  # Adjust based on your available GPU memory
```
