import os
import torch
from magic_pdf.config.constants import *
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.utils.load_image import load_image, encode_image_base64
from transformers import LayoutLMv3ForTokenClassification
from loguru import logger
import yaml
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List, Union
from openai import OpenAI


class MonkeyOCR:
    def __init__(self, config_path):
        current_file_path = os.path.abspath(__file__)

        current_dir = os.path.dirname(current_file_path)

        root_dir = os.path.dirname(current_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        logger.info('using configs: {}'.format(self.configs))

        self.device = self.configs.get('device', 'cpu')
        logger.info('using device: {}'.format(self.device))

        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True
        
        models_dir = self.configs.get(
            'models_dir', os.path.join(root_dir, 'model_weight')
        )

        logger.info('using models_dir: {}'.format(models_dir))
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"Model directory '{models_dir}' not found. "
                "Please run 'python download_model.py' to download the required models."
            )
        
        self.layout_config = self.configs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        layout_model_path = os.path.join(models_dir, self.configs['weights'][self.layout_model_name])
        if not os.path.exists(layout_model_path):
            raise FileNotFoundError(
                f"Layout model file not found at '{layout_model_path}'. "
                "Please run 'python download_model.py' to download the required models."
            )


        atom_model_manager = AtomModelSingleton()
        if self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=layout_model_path,
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

            if bf16_supported:
                model.to(self.device).eval().bfloat16()
            else:
                model.to(self.device).eval()
        else:
            logger.error('model name not allow')
        self.layoutreader_model = model
        logger.info(f'layoutreader model loaded: {self.layout_reader_name}')

        self.chat_config = self.configs.get('chat_config', {})
        chat_backend = self.chat_config.get('backend', 'lmdeploy')
        chat_path = self.chat_config.get('weight_path', 'model_weight/Recognition')
        if chat_backend == 'lmdeploy':
            logger.info('Use LMDeploy as backend')
            self.chat_model = MonkeyChat_LMDeploy(chat_path)
        elif chat_backend == 'vllm':
            logger.info('Use vLLM as backend')
            self.chat_model = MonkeyChat_vLLM(chat_path)
        elif chat_backend == 'transformers':
            logger.info('Use transformers as backend')
            batch_size = self.chat_config.get('batch_size', 5)
            self.chat_model = MonkeyChat_transformers(chat_path, batch_size, device=self.device)
        elif chat_backend == 'api':
            logger.info('Use API as backend')
            api_config = self.configs.get('api_config', {})
            if not api_config:
                raise ValueError("API configuration is required for API backend.")
            self.chat_model = MonkeyChat_OpenAIAPI(
                url=api_config.get('url'),
                model_name=api_config.get('model_name'),
                api_key=api_config.get('api_key', None)
            )
        else:
            logger.warning('Use LMDeploy as default backend')
            self.chat_model = MonkeyChat_LMDeploy(chat_path)
        logger.info(f'VLM loaded: {self.chat_model.model_name}')

class MonkeyChat_LMDeploy:
    def __init__(self, model_path, engine_config=None): 
        try:
            from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig, ChatTemplateConfig
        except ImportError:
            raise ImportError("LMDeploy is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md "
                              "to use MonkeyChat_LMDeploy.")
        self.model_name = os.path.basename(model_path)
        self.engine_config = self._auto_config_dtype(engine_config, PytorchEngineConfig)
        self.pipe = pipeline(model_path, backend_config=self.engine_config, chat_template_config=ChatTemplateConfig('qwen2d5-vl'))
        self.gen_config=GenerationConfig(max_new_tokens=4096,do_sample=True,temperature=0,repetition_penalty=1.05)

    def _auto_config_dtype(self, engine_config=None, PytorchEngineConfig=None):
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
        inputs = [(question, load_image(image, max_size=1600)) for image, question in zip(images, questions)]
        outputs = self.pipe(inputs, gen_config=self.gen_config)
        return [output.text for output in outputs]
    
class MonkeyChat_vLLM:
    def __init__(self, model_path):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md "
                               "to use MonkeyChat_vLLM.")
        self.model_name = os.path.basename(model_path)
        self.pipe = LLM(model=model_path,
                        max_seq_len_to_capture=10240,
                        mm_processor_kwargs={'use_fast': True},
                        gpu_memory_utilization=self._auto_gpu_mem_ratio(0.9))
        self.gen_config = SamplingParams(max_tokens=4096,temperature=0,repetition_penalty=1.05)
    
    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio

    def batch_inference(self, images, questions):
        placeholder = "<|image_pad|>"
        prompts = [
            ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n") for question in questions
        ]
        inputs = [{
            "prompt": prompts[i],
            "multi_modal_data": {
                "image": load_image(images[i], max_size=1600),
            }
        } for i in range(len(prompts))]
        outputs = self.pipe.generate(inputs, sampling_params=self.gen_config)
        return [o.outputs[0].text for o in outputs]

class MonkeyChat_transformers:
    def __init__(self, model_path: str, max_batch_size: int = 10, max_new_tokens=4096, device: str = None):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md "
                              "to use MonkeyChat_transformers.")
        self.model_name = os.path.basename(model_path)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True
            
        logger.info(f"Loading Qwen2.5VL model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16 if bf16_supported else torch.float16,
                        attn_implementation="flash_attention_2" if self.device.startswith("cuda") else 'sdpa',
                        device_map=self.device,
                    )
                
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"
            
            self.model.eval()
            logger.info("Qwen2.5VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def prepare_messages(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[List[dict]]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        all_messages = []
        for image, question in zip(images, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": load_image(image, max_size=1600),
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            all_messages.append(messages)
        
        return all_messages
    
    def batch_inference(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[str]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        results = []
        total_items = len(images)
        
        for i in range(0, total_items, self.max_batch_size):
            batch_end = min(i + self.max_batch_size, total_items)
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]
            
            logger.info(f"Processing batch {i//self.max_batch_size + 1}/{(total_items-1)//self.max_batch_size + 1} "
                       f"(items {i+1}-{batch_end})")
            
            try:
                batch_results = self._process_batch(batch_images, batch_questions)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed for items {i+1}-{batch_end}: {e}")
                logger.info("Falling back to single processing...")
                for img, q in zip(batch_images, batch_questions):
                    try:
                        single_result = self._process_single(img, q)
                        results.append(single_result)
                    except Exception as single_e:
                        logger.error(f"Single processing also failed: {single_e}")
                        results.append(f"Error: {str(single_e)}")
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def _process_batch(self, batch_images: List[Union[str, Image.Image]], batch_questions: List[str]) -> List[str]:
        all_messages = self.prepare_messages(batch_images, batch_questions)
        
        texts = []
        image_inputs = []
        
        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs.append(process_vision_info(messages)[0])
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return [text.strip() for text in output_texts]
    
    def _process_single(self, image: Union[str, Image.Image], question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def single_inference(self, image: Union[str, Image.Image], question: str) -> str:
        return self._process_single(image, question)
    
class MonkeyChat_OpenAIAPI:
    def __init__(self, url: str, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=url
        )
        if not self.validate_connection():
            raise ValueError("Invalid API URL or API key. Please check your configuration.")

    def validate_connection(self) -> bool:
        """
        Validate the effectiveness of API URL and key
        """
        try:
            # Try to get model list to validate connection
            response = self.client.models.list()
            logger.info("API connection validation successful")
            return True
        except Exception as e:
            logger.error(f"API connection validation failed: {e}")
            return False
    
    def img2base64(self, image: Union[str, Image.Image]) -> tuple[str, str]:
        if hasattr(image, 'format') and image.format:
            img_format = image.format
        else:
            # Default to PNG if format is not specified
            img_format = "PNG"
        image = encode_image_base64(image)
        return image, img_format.lower()

    def batch_inference(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[str]:
        results = []
        for image, question in zip(images, questions):
            try:
                # Load and resize image
                image = load_image(image, max_size=1600)
                img, img_type = self.img2base64(image)

                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{img_type};base64,{img}"
                        },
                        {
                            "type": "input_text", 
                            "text": question
                        }
                    ],
                }]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                results.append(f"Error: {e}")
        return results