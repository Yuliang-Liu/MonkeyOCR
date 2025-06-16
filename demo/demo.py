# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.custom_model import MonkeyOCR

MonkeyOCR_model = MonkeyOCR('model_configs.yaml')
# args

total_time = 0
# for i in range(1,11):
pdf_file_name = f"demo/demo1.pdf"  # replace with the real pdf path
name_without_suff = '.'.join(os.path.basename(pdf_file_name).split(".")[:-1])

# prepare env
local_image_dir, local_md_dir = f"output/{name_without_suff}/images", f"output/{name_without_suff}"
image_dir = str(os.path.basename(local_image_dir))

os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
    local_md_dir
)

# read bytes
reader1 = FileBasedDataReader()
pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

# proc
## Create Dataset Instance
if pdf_file_name.split(".")[-1] == "pdf":
    ds = PymuDocDataset(pdf_bytes)
else:
    ds = ImageDataset(pdf_bytes)

t1 = time.time()
infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model)

## pipeline
pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=MonkeyOCR_model)
single_time = time.time() - t1
print(f"parsing time: {single_time:.2f}s")

infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

### get model inference result
model_inference_result = infer_result.get_infer_res()

### draw layout result on each page
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

### draw spans result on each page
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

### get markdown content
md_content = pipe_result.get_markdown(image_dir)

### dump markdown
pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

### get content list content
content_list_content = pipe_result.get_content_list(image_dir)

### dump content list
pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

### get middle json
middle_json_content = pipe_result.get_middle_json()

### dump middle json
pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

print(f"Results saved to {local_md_dir}")
