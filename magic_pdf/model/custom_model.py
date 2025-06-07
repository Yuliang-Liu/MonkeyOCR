import os
import torch
from magic_pdf.config.constants import *
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.model_list import AtomicModel
from transformers import LayoutLMv3ForTokenClassification
from loguru import logger
import yaml


class MonkeyOCR:
    def __init__(self, chat_path, config_path):
        current_file_path = os.path.abspath(__file__)

        current_dir = os.path.dirname(current_file_path)

        root_dir = os.path.dirname(current_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        logger.info('using configs: {}'.format(self.configs))
        self.device = self.configs.get('device', 'cpu')
        logger.info('using device: {}'.format(self.device))
        models_dir = self.configs.get(
            'models_dir', os.path.join(root_dir, 'resources', 'models')
        )
        logger.info('using models_dir: {}'.format(models_dir))
        self.layout_config = self.configs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )
        atom_model_manager = AtomModelSingleton()
        if self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                device=self.device,
            )
        logger.info(f'layout model loaded: {self.layout_model_name}')


        layout_reader_config = self.layout_config.get('reader')
        self.layout_reader_name = layout_reader_config.get('name')
        if self.layout_reader_name == 'layoutreader':
            layoutreader_model_dir = os.path.join(models_dir, self.configs['weights'][self.layout_reader_name])
            if os.path.exists(layoutreader_model_dir):
                model = LayoutLMv3ForTokenClassification.from_pretrained(
                    layoutreader_model_dir
                )
            else:
                logger.warning(
                    'local layoutreader model not exists, use online model from huggingface'
                )
                model = LayoutLMv3ForTokenClassification.from_pretrained(
                    'hantian/layoutreader'
                )

            if self.device == 'cuda' and torch.cuda.is_bf16_supported():
                model.bfloat16()
            model.to(self.device).eval()
        else:
            logger.error('model name not allow')
        self.layoutreader_model = model
        logger.info(f'layoutreader model loaded: {self.layout_reader_name}')


        self.chat_model = MonkeyChat(chat_path)
        logger.info(f'VLM loaded: {self.chat_model.model_name}')

class MonkeyChat:
    def __init__(self, model_path, engine_config=None): 
        self.model_name = os.path.basename(model_path)
        self.engine_config = self.auto_config_dtype(engine_config)
        self.pipe = pipeline(model_path, backend_config=self.engine_config, chat_template_config=ChatTemplateConfig('qwen2d5-vl'))
        self.gen_config=GenerationConfig(max_new_tokens=4096,do_sample=True,temperature=0,repetition_penalty=1.05)

    def auto_config_dtype(self, engine_config=None):
        if engine_config is None:
            engine_config = PytorchEngineConfig(session_len=10240)
        dtype = "bfloat16"
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            sm_version = capability[0] * 10 + capability[1]  # e.g. sm75 = 7.5
            
            # use float16 if computing capability <= sm75 (7.5)
            if sm_version <= 75:
                dtype = "float16"
        engine_config.dtype = dtype
        return engine_config
    
    def batch_inference(self, images, questions):
        inputs = [(question, load_image(image)) for image, question in zip(images, questions)]
        outputs = self.pipe(inputs, gen_config=self.gen_config)
        return [output.text for output in outputs]
