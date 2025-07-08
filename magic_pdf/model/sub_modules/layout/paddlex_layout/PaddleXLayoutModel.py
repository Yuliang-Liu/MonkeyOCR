import numpy as np
from PIL import Image
from typing import List, Union
from loguru import logger

from magic_pdf.config.ocr_content_type import CategoryId

try:
    from paddlex import create_model
except ImportError:
    raise ImportError("Paddlex is not installed. Please install it using 'pip install paddlex'.")


class PaddleXLayoutModelWrapper:
    def __init__(self, model_name: str, device: str, model_dir: str = None):
        self.model_name = model_name
        self.device = device  # Note: Device may not be directly used by paddlex.create_model
        logger.info(f"Loading {self.model_name} model from {model_dir}...")
        if model_dir is not None:
            self.model_dir = model_dir
            self.model = create_model(model_name=self.model_name, model_dir=self.model_dir)
        else:
            self.model = create_model(model_name=self.model_name)

        self.category_mapping = {
            "paragraph_title": CategoryId.Title,
            "image": CategoryId.ImageBody,
            "text": CategoryId.Text,
            "number": CategoryId.Abandon,
            "abstract": CategoryId.Text,
            "content": CategoryId.Text,
            "figure_title": CategoryId.ImageCaption,
            "formula": CategoryId.InterlineEquation_Layout,
            "table": CategoryId.TableBody,
            "reference": CategoryId.Text,
            "doc_title": CategoryId.Title,
            "footnote": CategoryId.Abandon,
            "header": CategoryId.Abandon,
            "algorithm": CategoryId.Text,
            "footer": CategoryId.Abandon,
            "seal": CategoryId.Abandon,
            "chart": CategoryId.ImageBody,
            "formula_number": CategoryId.Abandon,
            "aside_text": CategoryId.Text,
            "reference_content": CategoryId.Text,
        }

    def _process_paddlex_result(self, paddlex_result_obj: dict) -> List[dict]:
        layout_res = []
        for det in paddlex_result_obj.get('boxes', []):
            label_name = det.get('label')
            category_id = self.category_mapping.get(label_name, -1)
            
            # Skip unknown or incomplete detections
            if category_id == -1 or not det.get('coordinate') or not det.get('score'):
                continue
            
            xmin, ymin, xmax, ymax = [int(p) for p in det['coordinate']]
            new_item = {
                "category_id": category_id,
                "original_label": label_name,
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(det['score']), 3),
            }
            layout_res.append(new_item)
        return layout_res

    def _prepare_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return np.stack([image] * 3, axis=-1)
            if image.shape[2] == 4:
                return image[:, :, :3]
            return image
        else:
            raise TypeError("Unsupported image type. Expected PIL.Image or numpy.ndarray.")

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[dict]:
        image_input = self._prepare_image(image)
        paddlex_output = list(self.model.predict(image_input, batch_size=1, layout_nms=True))
        if not paddlex_output:
            return []
        return self._process_paddlex_result(paddlex_output[0])

    def batch_predict(self, images: List[Union[np.ndarray, Image.Image]], batch_size: int) -> List[List[dict]]:
        prepared_images = [self._prepare_image(img) for img in images]
        
        # The model.predict itself handles batching, but we call it once.
        paddlex_outputs = list(self.model.predict(prepared_images, batch_size=batch_size, layout_nms=True))
        return [self._process_paddlex_result(res) for res in paddlex_outputs]
