import time
from loguru import logger
from magic_pdf.model.batch_analyze_llm import BatchAnalyzeLLM
from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.operators.models_llm import InferenceResultLLM
from magic_pdf.data.dataset import ImageDataset
from io import BytesIO
from PIL import Image


def doc_analyze_llm(
    dataset: Dataset,
    MonkeyOCR_model,
    start_page_id=0,
    end_page_id=None,
    split_pages=False,
) -> InferenceResultLLM:

    end_page_id = end_page_id if end_page_id else len(dataset) - 1

    device = MonkeyOCR_model.device

    batch_model = BatchAnalyzeLLM(model=MonkeyOCR_model)

    model_json = []
    doc_analyze_start = time.time()

    images = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
    analyze_result = batch_model(images,split_pages=split_pages)

    inference_results = []
    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        page_width = img_dict['width']
        page_height = img_dict['height']
        if start_page_id <= index <= end_page_id:
            result = analyze_result.pop(0)
        else:
            result = []

        if split_pages:
            # If split_pages is True, we create a separate entry for each page
            page_info = {'page_no': 0, 'height': page_height, 'width': page_width}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            # Convert PIL image to bytes
            img_bytes = BytesIO()
            img = Image.fromarray(img_dict['img'])
            img.save(img_bytes, format='PNG')
            img_ds = ImageDataset(img_bytes.getvalue())
            inference_result = InferenceResultLLM([page_dict], img_ds)
            inference_results.append(inference_result)
        else:
            page_info = {'page_no': index, 'height': page_height, 'width': page_width}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)
    if not split_pages:
        inference_results = InferenceResultLLM(model_json, dataset)

    gc_start = time.time()
    clean_memory(device)
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f'gc time: {gc_time}')

    doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    doc_analyze_speed = round((end_page_id + 1 - start_page_id) / doc_analyze_time, 2)
    logger.info(
        f'doc analyze time: {round(time.time() - doc_analyze_start, 2)},'
        f'speed: {doc_analyze_speed} pages/second'
    )

    return inference_results
