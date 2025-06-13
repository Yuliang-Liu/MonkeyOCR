<div align="center" xmlns="http://www.w3.org/1999/html">
<h1 align="center">
MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)
</div>


> **MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm**<br>
> Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/MonkeyOCR)
[![Model Weight](https://img.shields.io/badge/ModelScope-green)](https://modelscope.cn/models/l1731396519/MonkeyOCR)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:7685/)



## Introduction
MonkeyOCR adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.

1. Compared with the pipeline-based method MinerU, our approach achieves an average improvement of 5.1% across nine types of Chinese and English documents, including a 15.0% gain on formulas and an 8.6% gain on tables.
2. Compared to end-to-end models, our 3B-parameter model achieves the best average performance on English documents, outperforming models such as Gemini 2.5 Pro and Qwen2.5 VL-72B.
3. For multi-page document parsing, our method reaches a processing speed of 0.84 pages per second, surpassing MinerU (0.65) and Qwen2.5 VL-7B (0.12).

<img src="https://v1.ax1x.com/2025/06/05/7jQ3cm.png" alt="7jQ3cm.png" border="0" />

MonkeyOCR currently does not support photographed documents, but we will continue to improve it in future updates. Stay tuned!
Currently, our model is deployed on a single GPU, so if too many users upload files at the same time, issues like “This application is currently busy” may occur. We're actively working on supporting Ollama and other deployment solutions to ensure a smoother experience for more users. Additionally, please note that the processing time shown on the demo page does not reflect computation time alone—it also includes result uploading and other overhead. During periods of high traffic, this time may be longer. The inference speeds of MonkeyOCR, MinerU, and Qwen2.5 VL-7B were measured on an H800 GPU.




## News 
* ```2025.06.12 ``` 🚀 The model’s trending on [Hugging Face](https://huggingface.co/models?sort=trending). Thanks for the love!
* ```2025.06.05 ``` 🚀 We release MonkeyOCR, an English and Chinese documents parsing model.


## Quick Start

### 1. Install MonkeyOCR
```bash
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR

git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR

# Install pytorch, see https://pytorch.org/get-started/previous-versions/ for your cuda version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 
pip install -e .
```
### 2. Download Model Weights
Download our model from Huggingface.
```python
pip install huggingface_hub

python tools/download_model.py
```
You can also download our model from ModelScope.

```python
pip install modelscope

python tools/download_model.py -t modelscope
```
### 3. Inference
```bash
# Make sure in MonkeyOCR directory
# End to end parsing
python parse.py path/to/pdf_or_image
# Single task recognition (only output markdown)
python parse.py path/to/pdf_or_image -t text/formula/table
# Specify output path and model configs path
python parse.py path/to/your.pdf -o ./output -c config.yaml
```

#### 💡 Gentle Reminder

For Chinese scenarios or cases where text, tables, etc., are recognized as images, you can try using the following structure detection model:  
[layout_zh.pt](https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt)

To use this model, update the configuration file as follows:  
In [`model_configs.yaml`](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/model_configs.yaml#L3), replace:  

```yaml
Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt
```

with:  

```yaml
Structure/layout_zh.pt
```

#### Output Results
MonkeyOCR generates three types of output files:

1. **Processed Markdown File** (`your.md`): The final parsed document content in markdown format, containing text, formulas, tables, and other structured elements.
2. **Layout Results** (`your_layout.pdf`): The layout results drawed on origin PDF.
2. **Intermediate Block Results** (`your_middle.json`): A JSON file containing detailed information about all detected blocks, including:
   - Block coordinates and positions
   - Block content and type information
   - Relationship information between blocks

These files provide both the final formatted output and detailed intermediate results for further analysis or processing.

### 4. Gradio Demo
```bash
# Prepare your env for gradio
pip install gradio==5.23.3
pip install pdf2image==1.17.0
```
```bash
# Start demo
python demo/demo_gradio.py
```
### Fix **shared memory error** on **RTX 3090 / 4090 / ...** GPUs (Optional)

Our 3B model runs efficiently on NVIDIA RTX 3090. However, when using **LMDeploy** as the inference backend, you may encounter compatibility issues on **RTX 3090 / 4090** GPUs — particularly the following error:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory
```

To work around this issue, you can apply the patch below:

```bash
python tools/lmdeploy_patcher.py patch
```

> ⚠️ **Note:** This command will modify LMDeploy's source code in your environment.
> To revert the changes, simply run:

```bash
python tools/lmdeploy_patcher.py restore
```

**Special thanks to [@pineking](https://github.com/pineking) for the solution!**

### Switch inference backend (Optional)

You can switch inference backend to `transformers` following the steps below:

1. Install required dependency (if not already installed):
   ```bash
   # install flash attention 2, you can download the corresponding version from https://github.com/Dao-AILab/flash-attention/releases/
   pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```
2. Open the `model_configs.yaml` file
3. Set `chat_config.backend` to `transformers`
4. Adjust the `batch_size` according to your GPU's memory capacity to ensure stable performance

Example configuration:

```yaml
chat_config:
  backend: transformers
  batch_size: 10  # Adjust based on your available GPU memory
```


## Benchmark Results


Here are the evaluation results of our model on OmniDocBench. MonkeyOCR-3B uses DocLayoutYOLO as the structure detection model, while MonkeyOCR-3B* uses our trained structure detection model with improved Chinese performance.


### 1. The end-to-end evaluation results of different tasks.

<table style="width:100%; border-collapse:collapse; text-align:center;" border="0">
  <thead>
    <tr>
      <th rowspan="2">Model Type</th>
      <th rowspan="2">Methods</th>
      <th colspan="2">Overall Edit↓</th>
      <th colspan="2">Text Edit↓</th>
      <th colspan="2">Formula Edit↓</th>
      <th colspan="2">Formula CDM↑</th>
      <th colspan="2">Table TEDS↑</th>
      <th colspan="2">Table Edit↓</th>
      <th colspan="2">Read Order Edit↓</th>
    </tr>
    <tr>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Pipeline Tools</td>
      <td>MinerU</td>
      <td>0.150</td>
      <td>0.357</td>
      <td>0.061</td>
      <td>0.215</td>
      <td>0.278</td>
      <td>0.577</td>
      <td>57.3</td>
      <td>42.9</td>
      <td>78.6</td>
      <td>62.1</td>
      <td>0.180</td>
      <td>0.344</td>
      <td><strong>0.079</strong></td>
      <td>0.292</td>
    </tr>
    <tr>
      <td>Marker</td>
      <td>0.336</td>
      <td>0.556</td>
      <td>0.080</td>
      <td>0.315</td>
      <td>0.530</td>
      <td>0.883</td>
      <td>17.6</td>
      <td>11.7</td>
      <td>67.6</td>
      <td>49.2</td>
      <td>0.619</td>
      <td>0.685</td>
      <td>0.114</td>
      <td>0.340</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td>0.191</td>
      <td>0.365</td>
      <td>0.105</td>
      <td>0.384</td>
      <td>0.306</td>
      <td><strong>0.454</strong></td>
      <td>62.7</td>
      <td><strong>62.1</strong></td>
      <td>77.0</td>
      <td>67.1</td>
      <td>0.243</td>
      <td>0.320</td>
      <td>0.108</td>
      <td>0.304</td>
    </tr>
    <tr>
      <td>Docling</td>
      <td>0.589</td>
      <td>0.909</td>
      <td>0.416</td>
      <td>0.987</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>61.3</td>
      <td>25.0</td>
      <td>0.627</td>
      <td>0.810</td>
      <td>0.313</td>
      <td>0.837</td>
    </tr>
    <tr>
      <td>Pix2Text</td>
      <td>0.320</td>
      <td>0.528</td>
      <td>0.138</td>
      <td>0.356</td>
      <td>0.276</td>
      <td>0.611</td>
      <td>78.4</td>
      <td>39.6</td>
      <td>73.6</td>
      <td>66.2</td>
      <td>0.584</td>
      <td>0.645</td>
      <td>0.281</td>
      <td>0.499</td>
    </tr>
    <tr>
      <td>Unstructured</td>
      <td>0.586</td>
      <td>0.716</td>
      <td>0.198</td>
      <td>0.481</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0.06</td>
      <td>1</td>
      <td>0.998</td>
      <td>0.145</td>
      <td>0.387</td>
    </tr>
    <tr>
      <td>OpenParse</td>
      <td>0.646</td>
      <td>0.814</td>
      <td>0.681</td>
      <td>0.974</td>
      <td>0.996</td>
      <td>1</td>
      <td>0.11</td>
      <td>0</td>
      <td>64.8</td>
      <td>27.5</td>
      <td>0.284</td>
      <td>0.639</td>
      <td>0.595</td>
      <td>0.641</td>
    </tr>
    <tr>
      <td rowspan="5">Expert VLMs</td>
      <td>GOT-OCR</td>
      <td>0.287</td>
      <td>0.411</td>
      <td>0.189</td>
      <td>0.315</td>
      <td>0.360</td>
      <td>0.528</td>
      <td>74.3</td>
      <td>45.3</td>
      <td>53.2</td>
      <td>47.2</td>
      <td>0.459</td>
      <td>0.520</td>
      <td>0.141</td>
      <td>0.280</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.452</td>
      <td>0.973</td>
      <td>0.365</td>
      <td>0.998</td>
      <td>0.488</td>
      <td>0.941</td>
      <td>15.1</td>
      <td>16.8</td>
      <td>39.9</td>
      <td>0</td>
      <td>0.572</td>
      <td>1.000</td>
      <td>0.382</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>Mistral OCR</td>
      <td>0.268</td>
      <td>0.439</td>
      <td>0.072</td>
      <td>0.325</td>
      <td>0.318</td>
      <td>0.495</td>
      <td>64.6</td>
      <td>45.9</td>
      <td>75.8</td>
      <td>63.6</td>
      <td>0.600</td>
      <td>0.650</td>
      <td>0.083</td>
      <td>0.284</td>
    </tr>
    <tr>
      <td>OLMOCR-sglang</td>
      <td>0.326</td>
      <td>0.469</td>
      <td>0.097</td>
      <td>0.293</td>
      <td>0.455</td>
      <td>0.655</td>
      <td>74.3</td>
      <td>43.2</td>
      <td>68.1</td>
      <td>61.3</td>
      <td>0.608</td>
      <td>0.652</td>
      <td>0.145</td>
      <td>0.277</td>
    </tr>
    <tr>
      <td>SmolDocling-256M</td>
      <td>0.493</td>
      <td>0.816</td>
      <td>0.262</td>
      <td>0.838</td>
      <td>0.753</td>
      <td>0.997</td>
      <td>32.1</td>
      <td>0.55</td>
      <td>44.9</td>
      <td>16.5</td>
      <td>0.729</td>
      <td>0.907</td>
      <td>0.227</td>
      <td>0.522</td>
    </tr>
    <tr>
      <td rowspan="3">General VLMs</td>
      <td>GPT4o</td>
      <td>0.233</td>
      <td>0.399</td>
      <td>0.144</td>
      <td>0.409</td>
      <td>0.425</td>
      <td>0.606</td>
      <td>72.8</td>
      <td>42.8</td>
      <td>72.0</td>
      <td>62.9</td>
      <td>0.234</td>
      <td>0.329</td>
      <td>0.128</td>
      <td>0.251</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B</td>
      <td>0.312</td>
      <td>0.406</td>
      <td>0.157</td>
      <td>0.228</td>
      <td>0.351</td>
      <td>0.574</td>
      <td><strong>79.0</strong></td>
      <td>50.2</td>
      <td>76.4</td>
      <td>72.2</td>
      <td>0.588</td>
      <td>0.619</td>
      <td>0.149</td>
      <td>0.203</td>
    </tr>
    <tr>
      <td>InternVL3-8B</td>
      <td>0.314</td>
      <td>0.383</td>
      <td>0.134</td>
      <td>0.218</td>
      <td>0.417</td>
      <td>0.563</td>
      <td>78.3</td>
      <td>49.3</td>
      <td>66.1</td>
      <td>73.1</td>
      <td>0.586</td>
      <td>0.564</td>
      <td>0.118</td>
      <td>0.186</td>
    </tr>
    <tr>
      <td rowspan="2">Mix</td>
      <td>MonkeyOCR-3B <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt">[Weight]</a></td>
      <td><strong>0.140</strong></td>
      <td>0.297</td>
      <td><strong>0.058</strong></td>
      <td>0.185</td>
      <td><strong>0.238</strong></td>
      <td>0.506</td>
      <td>78.7</td>
      <td>51.4</td>
      <td><strong>80.2</strong></td>
      <td><strong>77.7</strong></td>
      <td><strong>0.170</strong></td>
      <td><strong>0.253</strong></td>
      <td>0.093</td>
      <td>0.244</td>
    </tr>
    <tr>
      <td>MonkeyOCR-3B* <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt">[Weight]</a></td>
      <td>0.154</td>
      <td><strong>0.277</strong></td>
      <td>0.073</td>
      <td><strong>0.134</strong></td>
      <td>0.255</td>
      <td>0.529</td>
      <td>78.5</td>
      <td>50.8</td>
      <td>78.2</td>
      <td>76.2</td>
      <td>0.182</td>
      <td>0.262</td>
      <td>0.105</td>
      <td><strong>0.183</strong></td>
    </tr>
  </tbody>
</table>




### 2. The end-to-end text recognition performance across 9 PDF page types.
<table style="width: 100%; border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="border-bottom: 2px solid #000;">
      <th><b>Model Type</b></th>
      <th><b>Models</b></th>
      <th><b>Book</b></th>
      <th><b>Slides</b></th>
      <th><b>Financial Report</b></th>
      <th><b>Textbook</b></th>
      <th><b>Exam Paper</b></th>
      <th><b>Magazine</b></th>
      <th><b>Academic Papers</b></th>
      <th><b>Notes</b></th>
      <th><b>Newspaper</b></th>
      <th><b>Overall</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Pipeline Tools</b></td>
      <td>MinerU</td>
      <td><u>0.055</u></td>
      <td>0.124</td>
      <td><u>0.033</u></td>
      <td><u>0.102</u></td>
      <td><u>0.159</u></td>
      <td><b>0.072</b></td>
      <td><u>0.025</u></td>
      <td>0.984</td>
      <td>0.171</td>
      <td>0.206</td>
    </tr>
    <tr>
      <td>Marker</td>
      <td>0.074</td>
      <td>0.340</td>
      <td>0.089</td>
      <td>0.319</td>
      <td>0.452</td>
      <td>0.153</td>
      <td>0.059</td>
      <td>0.651</td>
      <td>0.192</td>
      <td>0.274</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td>0.131</td>
      <td>0.220</td>
      <td>0.202</td>
      <td>0.216</td>
      <td>0.278</td>
      <td>0.147</td>
      <td>0.091</td>
      <td>0.634</td>
      <td>0.690</td>
      <td>0.300</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Expert VLMs</b></td>
      <td>GOT-OCR</td>
      <td>0.111</td>
      <td>0.222</td>
      <td>0.067</td>
      <td>0.132</td>
      <td>0.204</td>
      <td>0.198</td>
      <td>0.179</td>
      <td>0.388</td>
      <td>0.771</td>
      <td>0.267</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.734</td>
      <td>0.958</td>
      <td>1.000</td>
      <td>0.820</td>
      <td>0.930</td>
      <td>0.830</td>
      <td>0.214</td>
      <td>0.991</td>
      <td>0.871</td>
      <td>0.806</td>
    </tr>
    <tr>
      <td rowspan="3"><b>General VLMs</b></td>
      <td>GPT4o</td>
      <td>0.157</td>
      <td>0.163</td>
      <td>0.348</td>
      <td>0.187</td>
      <td>0.281</td>
      <td>0.173</td>
      <td>0.146</td>
      <td>0.607</td>
      <td>0.751</td>
      <td>0.316</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B</td>
      <td>0.148</td>
      <td><b>0.053</b></td>
      <td>0.111</td>
      <td>0.137</td>
      <td>0.189</td>
      <td>0.117</td>
      <td>0.134</td>
      <td>0.204</td>
      <td>0.706</td>
      <td>0.205</td>
    </tr>
    <tr>
      <td>InternVL3-8B</td>
      <td>0.163</td>
      <td><u>0.056</u></td>
      <td>0.107</td>
      <td>0.109</td>
      <td><b>0.129</b></td>
      <td>0.100</td>
      <td>0.159</td>
      <td><b>0.150</b></td>
      <td>0.681</td>
      <td>0.188</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Mix</b></td>
      <td>MonkeyOCR-3B <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt">[Weight]</a></td>
      <td><b>0.046</b></td>
      <td>0.120</td>
      <td><b>0.024</b></td>
      <td><b>0.100</b></td>
      <td><b>0.129</b></td>
      <td><u>0.086</u></td>
      <td><b>0.024</b></td>
      <td>0.643</td>
      <td><b>0.131</b></td>
      <td><u>0.155</u></td>
    </tr>
    <tr>
      <td>MonkeyOCR-3B* <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt">[Weight]</a></td>
      <td>0.054</td>
      <td>0.203</td>
      <td>0.038</td>
      <td>0.112</td>
      <td>0.138</td>
      <td>0.111</td>
      <td>0.032</td>
      <td><u>0.194</u></td>
      <td><u>0.136</u></td>
      <td><b>0.120</b></td>
    </tr>
  </tbody>
</table>

### 3. Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<img src="https://v1.ax1x.com/2025/06/05/7jQlj4.png" alt="7jQlj4.png" border="0" />


## Visualization Demo

Get a Quick Hands-On Experience with Our Demo:  http://vlrlabmonkey.xyz:7685

> Our demo is simple and easy to use:
>
> 1. Upload a PDF or image.
> 2. Click “Parse (解析)” to let the model perform structure detection, content recognition, and relationship prediction on the input document. The final output will be a markdown-formatted version of the document.
> 3. Select a prompt and click “Test by prompt” to let the model perform content recognition on the image based on the selected prompt.



### Support diverse Chinese and English PDF types

<p align="center">
  <img src="asserts/Visualization.GIF?raw=true" width="600"/>
</p>

### Example for formula document
<img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

### Example for table document
<img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

### Example for newspaper
<img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

### Example for financial report
<img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
<img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

## Citing MonkeyOCR

If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@misc{li2025monkeyocrdocumentparsingstructurerecognitionrelation,
      title={MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm}, 
      author={Zhang Li and Yuliang Liu and Qiang Liu and Zhiyin Ma and Ziyang Zhang and Shuo Zhang and Zidun Guo and Jiarui Zhang and Xinyu Wang and Xiang Bai},
      year={2025},
      eprint={2506.05218},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05218}, 
}
```



## Acknowledgments
We would like to thank [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), and [InternVL3](https://github.com/OpenGVLab/InternVL) for providing base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets.


## Copyright
Please don’t hesitate to share your valuable feedback — it’s a key motivation that drives us to continuously improve our framework. The current technical report only presents the results of the 3B model. Our model is intended for non-commercial use. If you are interested in larger one, please contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn.
