import time
from loguru import logger
from magic_pdf.model.batch_analyze_llm import BatchAnalyzeLLM
from magic_pdf.data.dataset import Dataset, MultiFileDataset
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
    split_files=False,
    pred_abandon=False,
) -> InferenceResultLLM:

    end_page_id = end_page_id if end_page_id else len(dataset) - 1

    device = MonkeyOCR_model.device

    batch_model = BatchAnalyzeLLM(model=MonkeyOCR_model)

    model_json = []
    doc_analyze_start = time.time()

    image_dicts = []
    images = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            image_dicts.append(img_dict)
            images.append(img_dict['img'])
    
    logger.info(f'images load time: {round(time.time() - doc_analyze_start, 2)}')
    analyze_result = batch_model(images, split_pages=split_pages or split_files, pred_abandon=pred_abandon)

    # Handle MultiFileDataset with split_files
    if split_files and isinstance(dataset, MultiFileDataset):
        file_results = []
        for file_index in range(len(dataset.file_info)):
            file_info = dataset.file_info[file_index]
            file_start_page = file_info['start_page']
            file_end_page = file_info['end_page']
            file_page_count = file_info['page_count']
            
            # Create file-specific dataset
            file_dataset = dataset.export_file_as_dataset(file_index)
            
            # Collect results for this file
            file_model_json = []
            for page_idx in range(file_page_count):
                global_page_idx = file_start_page + page_idx
                if start_page_id <= global_page_idx <= end_page_id:
                    result = analyze_result.pop(0)
                else:
                    result = []
                
                img_dict = image_dicts[global_page_idx]
                page_width = img_dict['width']
                page_height = img_dict['height']
                
                if split_pages:
                    # For split_pages, create individual InferenceResultLLM for each page
                    page_info = {'page_no': 0, 'height': page_height, 'width': page_width}
                    page_dict = {'layout_dets': result, 'page_info': page_info}
                    
                    # For ImageDataset, we can reuse the file_dataset directly since it's already single-page
                    if isinstance(file_dataset, ImageDataset) and file_page_count == 1:
                        page_inference_result = InferenceResultLLM([page_dict], file_dataset)
                    else:
                        # For multi-page files (PDFs), convert page to bytes
                        img_bytes = BytesIO()
                        img = Image.fromarray(img_dict['img'])
                        img.save(img_bytes, format='PNG')
                        img_ds = ImageDataset(img_bytes.getvalue())
                        page_inference_result = InferenceResultLLM([page_dict], img_ds)
                    
                    # Initialize file_results structure if needed
                    if len(file_results) <= file_index:
                        file_results.extend([[] for _ in range(file_index + 1 - len(file_results))])
                    if not isinstance(file_results[file_index], list):
                        file_results[file_index] = []
                    file_results[file_index].append(page_inference_result)
                else:
                    # For file-level results, use relative page numbers starting from 0
                    page_info = {'page_no': page_idx, 'height': page_height, 'width': page_width}
                    page_dict = {'layout_dets': result, 'page_info': page_info}
                    file_model_json.append(page_dict)
            
            if not split_pages:
                # Create one InferenceResultLLM per file
                file_inference_result = InferenceResultLLM(file_model_json, file_dataset)
                file_results.append(file_inference_result)
        
        inference_results = file_results
    else:
        # Original logic for non-split_files cases
        inference_results = []
        for index in range(len(dataset)):
            img_dict = image_dicts[index]
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
