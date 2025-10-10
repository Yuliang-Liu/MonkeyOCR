# Install with NPU Support
此向导仅用于在NPU上运行monkey-ocr，需提前下载模型权重到/root/.cache目录，已在910B4显卡上面做过验证，后端使用vllm_queue, 性能大致0.7page/s。
# step1: 下载vllm-ascend镜像并启动，需要vllm-ascend:v0.9.2rc1及更高版本
```
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.2rc1
docker run --rm \
--name vllm-ascend \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```
# step2: 进入镜像内部安装对应的包
```
git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR
pip install -e .
# 下面的命令不执行会报错缺少一些opencv图形化显示的so，我们不需要
pip uninstall opencv-python-headless opencv-python -y
pip install opencv-python-headless==4.11.0.86
```
# step3:修改model_config.yaml配置
```
device: npu
weights:
  doclayout_yolo: Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt # or Structure/layout_zh.pt
  # PP-DocLayout_plus-L: Structure/PP-DocLayout_plus-L
  layoutreader: Relation
layout_config:
  model: doclayout_yolo # PP-DocLayout_plus-L (MonkeyOCR-pro, npu不支持) / doclayout_yolo (MonkeyOCR)
  reader:
      name: layoutreader
models_dir: /root/.cache
chat_config:
  weight_path: /root/.cache/Recognition
  backend: vllm_queue
...
```
# step4: 启动服务
```
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
# 可选，如果是卡0可以不设置
export ASCEND_RT_VISIBLE_DEVICEAS=0
univcore api.main:app --host 0.0.0.0 --port 8000
```
