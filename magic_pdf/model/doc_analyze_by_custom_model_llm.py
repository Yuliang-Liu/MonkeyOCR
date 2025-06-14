import os
import time

from loguru import logger

from magic_pdf.model.batch_analyze_llm import BatchAnalyzeLLM

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.operators.models_llm import InferenceResultLLM
        

def doc_analyze_llm(
    dataset: Dataset,
    MonkeyOCR_model,
    start_page_id=0,
    end_page_id=None,
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
    analyze_result = batch_model(images)

    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        page_width = img_dict['width']
        page_height = img_dict['height']
        if start_page_id <= index <= end_page_id:
            result = analyze_result.pop(0)
        else:
            result = []

        page_info = {'page_no': index, 'height': page_height, 'width': page_width}
        page_dict = {'layout_dets': result, 'page_info': page_info}
        model_json.append(page_dict)

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

    return InferenceResultLLM(model_json, dataset)
