import base64
import copy
import time

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from magic_pdf.config.constants import MODEL_NAME
from io import BytesIO
from PIL import Image
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img)

YOLO_LAYOUT_BASE_BATCH_SIZE = 1

class BatchAnalyzeLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, images: list) -> list:
        images_layout_res = []

        layout_start_time = time.time()
        if self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_images = []
            modified_images = []
            for image_index, image in enumerate(images):
                pil_img = Image.fromarray(image)
                layout_images.append(pil_img)

            images_layout_res += self.model.layout_model.batch_predict(
                # layout_images, self.batch_ratio * YOLO_LAYOUT_BASE_BATCH_SIZE
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )

            for image_index, useful_list in modified_images:
                for res in images_layout_res[image_index]:
                    for i in range(len(res['poly'])):
                        if i % 2 == 0:
                            res['poly'][i] = (
                                res['poly'][i] - useful_list[0] + useful_list[2]
                            )
                        else:
                            res['poly'][i] = (
                                res['poly'][i] - useful_list[1] + useful_list[3]
                            )
        logger.info(
            f'layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}'
        )

        clean_vram(self.model.device, vram_threshold=8)

        llm_ocr_start = time.time()
        new_images_all = []
        cids_all = []
        page_idxs = []
        for index in range(len(images)):
            layout_res = images_layout_res[index]
            pil_img = Image.fromarray(images[index])
            new_images = []
            cids = []
            for res in layout_res:
                new_image, useful_list = crop_img(
                    res, pil_img, crop_paste_x=50, crop_paste_y=50
                )
                new_images.append(new_image)
                cids.append(res['category_id'])
            new_images_all.extend(new_images)
            cids_all.extend(cids)
            page_idxs.append(len(new_images_all) - len(new_images))
        logger.info('VLM OCR start...')
        ocr_result = self.batch_llm_ocr(new_images_all, cids_all)
        for index in range(len(images)):
            ocr_results = []
            layout_res = images_layout_res[index]
            for i in range(len(layout_res)):
                res = layout_res[i]
                ocr = ocr_result[page_idxs[index]+i]
                # ocr = self.llm_ocr(new_image, res['category_id'])
                if res['category_id'] in [8, 14]:
                    temp_res = copy.deepcopy(res)
                    temp_res['category_id'] = 14
                    temp_res['score'] = 1.0
                    temp_res['latex'] = ocr
                    ocr_results.append(temp_res)
                elif res['category_id'] in [0, 1, 2, 4, 6, 7, 101]:
                    temp_res = copy.deepcopy(res)
                    temp_res['category_id'] = 15
                    temp_res['score'] = 1.0
                    temp_res['text'] = ocr
                    ocr_results.append(temp_res)
                elif res['category_id'] == 5:
                    res['score'] = 1.0
                    res['html'] = ocr
            layout_res.extend(ocr_results)
            logger.info(f'OCR processed images / total images: {index+1} / {len(images)}')
        logger.info(
            f'llm ocr time: {round(time.time() - llm_ocr_start, 2)}, image num: {len(images)}'
        )

        return images_layout_res

    def batch_llm_ocr(self, images, cat_ids, version='lmdeploy',max_batch_size=8):
        import re
        def sanitize_md(output):
            cleaned = re.match(r'<md>.*</md>', output, flags=re.DOTALL)
            if cleaned is None:
                return output.replace('<md>', '').replace('</md>', '').replace('md\n','').strip()
            return f"{cleaned[0].replace('<md>', '').replace('</md>', '').strip()}"
        def sanitize_mf(output):
            cleaned = re.match(r'\$\$.*\$\$', output, flags=re.DOTALL)
            if cleaned is None:
                return output.replace('$$', '').strip()
            return f"{cleaned[0].replace('$$', '').strip()}"
        def sanitize_html(output):
            # cleaned = re.match(r'<html>.*</html>', output, flags=re.DOTALL)
            cleaned = re.match(r'```html.*```', output, flags=re.DOTALL)
            if cleaned is None:
                return '<html>\n'+output.replace('```html','<html>').replace('```','</html>').strip()+'\n</html>'
            return f"{cleaned[0].replace('```html','<html>').replace('```','</html>').strip()}"
        assert len(images) == len(cat_ids)
        instruction = f'''Please output the text content from the image.'''
        instruction_mf = f'''Please write out the expression of the formula in the image using LaTeX format.'''
        instruction_table = f'''This is the image of a table. Please output the table in html format.'''
        cid2instruction = {
            0: instruction,
            1: instruction,
            # 2: instruction,
            4: instruction,
            5: instruction_table,
            6: instruction,
            7: instruction,
            8: instruction_mf,
            # 9: instruction,
            14: instruction_mf,
            101: instruction,
        }
        new_images = []
        messages = []
        ignore_idx = []
        outs = []
        if version in ['vllm', 'lmdeploy']:
            for i in range(len(images)):
                if cat_ids[i] not in cid2instruction:
                    ignore_idx.append(i)
                    continue
                new_images.append(images[i])
                messages.append(cid2instruction[cat_ids[i]])
            out = self.model.chat_model.batch_inference(new_images, messages)
            outs.extend(out)
        else:
            buffer = BytesIO()
            for i in range(len(images)):
                if cat_ids[i] not in cid2instruction:
                    ignore_idx.append(i)
                    continue
                images[i].save(buffer, format='JPEG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                messages.append(
                    [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "data:image/jpeg;base64," + image_base64,
                            },
                            {"type": "text", "text": "{}".format(cid2instruction[cat_ids[i]])},
                        ],
                    },]
                )
                buffer.seek(0)
                buffer.truncate(0)
                # if len(messages) == max_batch_size or i == len(images) - 1:
            outs.extend(self.model.llm_model.batch_inference(messages))
        for j in ignore_idx:
            outs.insert(j, '')
        messages.clear()
        ignore_idx.clear()
        for j in range(len(outs)):
            if cat_ids[j] in cid2instruction:
                if cat_ids[j] == 5:
                    outs[j] = sanitize_html(outs[j])
                elif cat_ids[j] in [8, 14]:
                    outs[j] = sanitize_mf(outs[j])
                else:
                    outs[j] = sanitize_md(outs[j])
        return outs