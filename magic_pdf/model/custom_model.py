import os
import time
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
import asyncio
import uuid


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
                "Please run 'python tools/download_model.py' to download the required models."
            )
        
        self.layout_config = self.configs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        atom_model_manager = AtomModelSingleton()
        if self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            layout_model_path = os.path.join(models_dir, self.configs['weights'][self.layout_model_name])
            if not os.path.exists(layout_model_path):
                raise FileNotFoundError(
                    f"Layout model file not found at '{layout_model_path}'. "
                    "Please run 'python tools/download_model.py' to download the required models."
                )
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=layout_model_path,
                device=self.device,
            )
        elif self.layout_model_name == MODEL_NAME.PaddleXLayoutModel:
            layout_model_path = None
            if self.layout_model_name in self.configs['weights']:
                layout_model_path = os.path.join(models_dir, self.configs['weights'][self.layout_model_name])
                if not os.path.exists(layout_model_path):
                    raise FileNotFoundError(
                        f"Layout model file not found at '{layout_model_path}'. "
                        "Please run 'python tools/download_model.py' to download the required models."
                    )
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.PaddleXLayoutModel,
                paddlexlayout_model_dir=layout_model_path,
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
                raise FileNotFoundError(
                    f"Reading Order model file not found at '{layoutreader_model_dir}'. "
                    "Please run 'python tools/download_model.py' to download the required models."
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
        if not os.path.exists(chat_path):
            chat_path = os.path.join(models_dir, chat_path)
            if not os.path.exists(chat_path):
                raise FileNotFoundError(
                    f"Chat model file not found at '{chat_path}'. "
                    "Please run 'python tools/download_model.py' to download the required models."
                )
        if chat_backend == 'lmdeploy':
            logger.info('Use LMDeploy as backend')
            dp = self.chat_config.get('data_parallelism', 1)
            tp = self.chat_config.get('model_parallelism', 1)
            self.chat_model = MonkeyChat_LMDeploy(chat_path, dp=dp, tp=tp)
        elif chat_backend == 'lmdeploy_queue':
            logger.info('Use LMDeploy Queue as backend')
            dp = self.chat_config.get('data_parallelism', 1)
            tp = self.chat_config.get('model_parallelism', 1)
            queue_config = self.chat_config.get('queue_config', {})
            self.chat_model = MonkeyChat_LMDeploy_queue(chat_path, dp=dp, tp=tp, **queue_config)
        elif chat_backend == 'vllm':
            logger.info('Use vLLM as backend')
            tp = self.chat_config.get('model_parallelism', 1)
            self.chat_model = MonkeyChat_vLLM(chat_path, tp=tp)
        elif chat_backend == 'vllm_queue':
            logger.info('Use vLLM Queue as backend')
            tp = self.chat_config.get('model_parallelism', 1)
            queue_config = self.chat_config.get('queue_config', {})
            self.chat_model = MonkeyChat_vLLM_queue(chat_path, tp=tp, **queue_config)
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
    def __init__(self, model_path, dp=1, tp=1): 
        try:
            from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig
        except ImportError:
            raise ImportError("LMDeploy is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md "
                              "to use MonkeyChat_LMDeploy.")
        self.model_name = os.path.basename(model_path)
        self.engine_config = self._auto_config_dtype(dp=dp, tp=tp)
        self.pipe = pipeline(model_path, backend_config=self.engine_config, chat_template_config=ChatTemplateConfig('qwen2d5-vl'))
        self.gen_config=GenerationConfig(max_new_tokens=4096,do_sample=True,temperature=0,repetition_penalty=1.05)

    def _auto_config_dtype(self, dp=1, tp=1):
        from lmdeploy import PytorchEngineConfig
        engine_config = PytorchEngineConfig(session_len=10240, dp=dp, tp=tp)
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
    def __init__(self, model_path, tp=1):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md "
                               "to use MonkeyChat_vLLM.")
        self.model_name = os.path.basename(model_path)
        self.pipe = LLM(model=model_path,
                        max_seq_len_to_capture=10240,
                        mm_processor_kwargs={'use_fast': True},
                        gpu_memory_utilization=self._auto_gpu_mem_ratio(0.9),
                        tensor_parallel_size=tp)
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
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md "
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

class MonkeyChat_LMDeploy_queue:
    """
    Hybrid architecture: Combines synchronous batch processing with asynchronous concurrency for LMDeploy
    Designed for multi-user large-batch concurrent inference scenarios using LMDeploy backend
    
    Features:
    1. Uses request queue to collect requests from multiple users
    2. Dynamic batch merging to maximize GPU utilization
    3. Supports multi-user concurrency, each user can submit large batch tasks
    4. Achieves inference speed close to MonkeyChat_LMDeploy
    5. Uses LMDeploy's efficient pipeline for batch processing
    """
    
    def __init__(self, model_path, dp=1, tp=1, max_batch_size=32, queue_timeout=0.1, max_queue_size=1000):
        try:
            from lmdeploy import pipeline, GenerationConfig, ChatTemplateConfig
            import asyncio
            import threading
            from collections import deque
        except ImportError:
            raise ImportError("LMDeploy is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md")
        
        self.model_name = os.path.basename(model_path)
        self.max_batch_size = max_batch_size
        self.queue_timeout = queue_timeout
        self.max_queue_size = max_queue_size
        
        # Clear GPU memory before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize LMDeploy pipeline (for efficient batch processing)
        self.engine_config = self._auto_config_dtype(dp=dp, tp=tp)
        self.pipe = pipeline(
            model_path, 
            backend_config=self.engine_config, 
            chat_template_config=ChatTemplateConfig('qwen2d5-vl')
        )
        
        self.gen_config = GenerationConfig(
            max_new_tokens=4096,
            do_sample=True,
            temperature=0,
            repetition_penalty=1.05
        )
        
        # Request queue and processing related
        self.request_queue = deque()
        self.result_futures = {}
        self.queue_lock = threading.Lock()
        self.processing = False
        self.shutdown_flag = False
        
        # Start background processing thread
        self.processor_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processor_thread.start()
        
        logger.info(f"LMDeploy MultiUser engine initialized for model: {self.model_name}")
        logger.info(f"Max batch size: {max_batch_size}, Queue timeout: {queue_timeout}s")
    
    def _auto_config_dtype(self, dp=1, tp=1):
        """Auto configure dtype based on GPU capability"""
        from lmdeploy import PytorchEngineConfig
        engine_config = PytorchEngineConfig(session_len=10240, dp=dp, tp=tp)
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
    
    def _background_processor(self):
        """Background thread: continuously process request queue"""
        import time
        
        while not self.shutdown_flag:
            try:
                # Collect a batch of requests
                batch_requests = self._collect_batch_requests()
                
                if batch_requests:
                    # Process batch requests
                    self._process_batch_requests(batch_requests)
                else:
                    # Sleep briefly when no requests
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                time.sleep(0.1)
    
    def _collect_batch_requests(self):
        """Collect a batch of requests, supports dynamic batch size"""
        import time
        
        batch_requests = []
        start_time = time.time()
        
        with self.queue_lock:
            # Collect requests until batch size reached or timeout
            while (len(batch_requests) < self.max_batch_size and 
                   time.time() - start_time < self.queue_timeout and
                   self.request_queue):
                
                request = self.request_queue.popleft()
                batch_requests.append(request)
        
        return batch_requests
    
    def _process_batch_requests(self, batch_requests):
        """Process batch requests using LMDeploy pipeline"""
        try:
            # Prepare batch inputs for LMDeploy
            inputs = []
            request_ids = []
            
            for request in batch_requests:
                request_id, image_path, question, future = request
                
                # Load image and prepare input tuple for LMDeploy
                image = load_image(image_path, max_size=1600)
                inputs.append((question, image))
                request_ids.append(request_id)
            
            # Batch inference using LMDeploy pipeline
            start_time = time.time()
            outputs = self.pipe(inputs, gen_config=self.gen_config)
            processing_time = time.time() - start_time
            
            logger.info(f"Processed batch of {len(batch_requests)} requests in {processing_time:.2f}s "
                       f"({len(batch_requests)/processing_time:.1f} req/s)")
            
            # Distribute results to corresponding futures
            for i, output in enumerate(outputs):
                request_id = request_ids[i]
                result_text = output.text
                
                # Get corresponding future from batch_requests and set result
                request = batch_requests[i]
                _, _, _, future = request
                
                try:
                    if not future.done():
                        # Need to set future result in correct event loop
                        if hasattr(future, '_loop') and future._loop is not None:
                            future._loop.call_soon_threadsafe(future.set_result, result_text)
                        else:
                            future.set_result(result_text)
                    
                    # Clean from dictionary
                    if request_id in self.result_futures:
                        del self.result_futures[request_id]
                        
                except Exception as e:
                    logger.error(f"Failed to set result for request {request_id}: {e}")
                    # Try to set error result
                    try:
                        if not future.done():
                            if hasattr(future, '_loop') and future._loop is not None:
                                future._loop.call_soon_threadsafe(future.set_result, f"Error: {str(e)}")
                            else:
                                future.set_result(f"Error: {str(e)}")
                    except Exception:
                        pass
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set all requests to error state
            for i, request in enumerate(batch_requests):
                request_id, _, _, future = request
                
                try:
                    if not future.done():
                        error_msg = f"Error: {str(e)}"
                        if hasattr(future, '_loop') and future._loop is not None:
                            future._loop.call_soon_threadsafe(future.set_result, error_msg)
                        else:
                            future.set_result(error_msg)
                    
                    # Clean from dictionary
                    if request_id in self.result_futures:
                        del self.result_futures[request_id]
                        
                except Exception as set_error:
                    logger.error(f"Failed to set error result for request {request_id}: {set_error}")
    
    async def async_single_inference(self, image: str, question: str) -> str:
        """Asynchronous single inference"""
        request_id = f"lmdeploy_multiuser_{uuid.uuid4().hex[:8]}"
        
        # Create future to receive result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Add request to queue
        with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                logger.warning(f"Request queue full, rejecting request {request_id}")
                return "Error: Request queue full"
            
            self.request_queue.append((request_id, image, question, future))
            self.result_futures[request_id] = future
        
        try:
            # Wait for result with timeout
            result = await future
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            # Clean up timed out request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            return "Error: Request timeout"
        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            # Clean up cancelled request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            raise
        except Exception as e:
            logger.error(f"Request {request_id} failed with exception: {e}")
            # Clean up failed request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            return f"Error: {str(e)}"
    
    def single_inference(self, image: str, question: str) -> str:
        """Synchronous single inference (wraps async method)"""
        try:
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, use thread executor
                import concurrent.futures
                
                def run_async_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.async_single_inference(image, question)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    return future.result()
                    
            except RuntimeError:
                # No running event loop
                return asyncio.run(self.async_single_inference(image, question))
                
        except Exception as e:
            logger.error(f"Single inference failed: {e}")
            return f"Error: {str(e)}"
    
    async def async_batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        """Asynchronous batch inference (decompose large batches into multiple concurrent requests)"""
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        # Create concurrent tasks
        tasks = []
        for image, question in zip(images, questions):
            task = self.async_single_inference(image, question)
            tasks.append(task)
        
        # Execute all tasks concurrently
        logger.info(f"Processing {len(tasks)} requests concurrently")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exception results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        """Synchronous batch inference"""
        try:
            try:
                loop = asyncio.get_running_loop()
                # Already in async context
                import concurrent.futures
                
                def run_async_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.async_batch_inference(images, questions)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    return future.result()
                    
            except RuntimeError:
                # No running event loop
                return asyncio.run(self.async_batch_inference(images, questions))
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [f"Error: {str(e)}"] * len(images)
    
    def get_queue_status(self):
        """Get queue status (for monitoring)"""
        with self.queue_lock:
            return {
                "queue_size": len(self.request_queue),
                "pending_results": len(self.result_futures),
                "max_queue_size": self.max_queue_size,
                "processing": self.processing,
                "processor_thread_alive": self.processor_thread.is_alive(),
                "shutdown_flag": self.shutdown_flag
            }
    
    def shutdown(self):
        """Shutdown service"""
        self.shutdown_flag = True
        
        # Wait for background thread to finish
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        
        # Clean up unfinished requests
        with self.queue_lock:
            for request_id, future in self.result_futures.items():
                if not future.done():
                    future.set_result("Error: Service shutdown")
            self.result_futures.clear()
            self.request_queue.clear()
        
        # Clean up pipeline and GPU memory
        try:
            if hasattr(self, 'pipe') and self.pipe is not None:
                del self.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        logger.info("LMDeploy MultiUser engine shutdown completed")
    
    def __del__(self):
        """Destructor"""
        try:
            self.shutdown()
        except Exception:
            pass

class MonkeyChat_vLLM_queue:
    """
    Hybrid architecture: Combines synchronous batch processing with asynchronous concurrency
    Designed for multi-user large-batch concurrent inference scenarios
    
    Features:
    1. Uses request queue to collect requests from multiple users
    2. Dynamic batch merging to maximize GPU utilization
    3. Supports multi-user concurrency, each user can submit large batch tasks
    4. Achieves inference speed close to MonkeyChat_vLLM
    """
    
    def __init__(self, model_path, tp=1, max_batch_size=64, queue_timeout=0.1, max_queue_size=1000):
        try:
            from vllm import LLM, SamplingParams
            import threading
            from collections import deque
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md")
        
        self.model_name = os.path.basename(model_path)
        self.max_batch_size = max_batch_size
        self.queue_timeout = queue_timeout
        self.max_queue_size = max_queue_size
        
        # Initialize synchronous vLLM engine (for efficient batch processing)
        self.engine = LLM(
            model=model_path,
            max_seq_len_to_capture=10240,
            mm_processor_kwargs={'use_fast': True},
            gpu_memory_utilization=self._auto_gpu_mem_ratio(0.9),
            max_num_seqs=max_batch_size * 2,  # Allow larger sequence numbers
            tensor_parallel_size=tp
        )
        
        self.gen_config = SamplingParams(
            max_tokens=4096, 
            temperature=0, 
            repetition_penalty=1.05
        )
        
        # Request queue and processing related
        self.request_queue = deque()
        self.result_futures = {}
        self.queue_lock = threading.Lock()
        self.processing = False
        self.shutdown_flag = False
        
        # Start background processing thread
        self.processor_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processor_thread.start()
        
        logger.info(f"vLLM MultiUser engine initialized for model: {self.model_name}")
        logger.info(f"Max batch size: {max_batch_size}, Queue timeout: {queue_timeout}s")
    
    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio
    
    def _background_processor(self):
        """Background thread: continuously process request queue"""
        import time
        
        while not self.shutdown_flag:
            try:
                # Collect a batch of requests
                batch_requests = self._collect_batch_requests()
                
                if batch_requests:
                    # Process batch requests
                    self._process_batch_requests(batch_requests)
                else:
                    # Sleep briefly when no requests
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                time.sleep(0.1)
    
    def _collect_batch_requests(self):
        """Collect a batch of requests, supports dynamic batch size"""
        import time
        
        batch_requests = []
        start_time = time.time()
        
        with self.queue_lock:
            # Collect requests until batch size reached or timeout
            while (len(batch_requests) < self.max_batch_size and 
                   time.time() - start_time < self.queue_timeout and
                   self.request_queue):
                
                request = self.request_queue.popleft()
                batch_requests.append(request)
        
        return batch_requests
    
    def _process_batch_requests(self, batch_requests):
        """Process batch requests using synchronous engine"""
        try:
            # Prepare batch inputs
            placeholder = "<|image_pad|>"
            inputs = []
            request_ids = []
            
            for request in batch_requests:
                request_id, image_path, question, future = request
                
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": load_image(image_path, max_size=1600),
                    }
                })
                request_ids.append(request_id)
            
            # Batch inference (using high-efficiency batch processing of synchronous engine)
            start_time = time.time()
            outputs = self.engine.generate(inputs, sampling_params=self.gen_config)
            processing_time = time.time() - start_time
            
            logger.info(f"Processed batch of {len(batch_requests)} requests in {processing_time:.2f}s "
                       f"({len(batch_requests)/processing_time:.1f} req/s)")
            
            # Distribute results to corresponding futures
            for i, output in enumerate(outputs):
                request_id = request_ids[i]
                result_text = output.outputs[0].text
                
                # Get corresponding future from batch_requests and set result
                request = batch_requests[i]
                _, _, _, future = request
                
                try:
                    if not future.done():
                        # Need to set future result in correct event loop
                        if hasattr(future, '_loop') and future._loop is not None:
                            future._loop.call_soon_threadsafe(future.set_result, result_text)
                        else:
                            future.set_result(result_text)
                    
                    # Clean from dictionary
                    if request_id in self.result_futures:
                        del self.result_futures[request_id]
                        
                except Exception as e:
                    logger.error(f"Failed to set result for request {request_id}: {e}")
                    # Try to set error result
                    try:
                        if not future.done():
                            if hasattr(future, '_loop') and future._loop is not None:
                                future._loop.call_soon_threadsafe(future.set_result, f"Error: {str(e)}")
                            else:
                                future.set_result(f"Error: {str(e)}")
                    except Exception:
                        pass
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set all requests to error state
            for i, request in enumerate(batch_requests):
                request_id, _, _, future = request
                
                try:
                    if not future.done():
                        error_msg = f"Error: {str(e)}"
                        if hasattr(future, '_loop') and future._loop is not None:
                            future._loop.call_soon_threadsafe(future.set_result, error_msg)
                        else:
                            future.set_result(error_msg)
                    
                    # Clean from dictionary
                    if request_id in self.result_futures:
                        del self.result_futures[request_id]
                        
                except Exception as set_error:
                    logger.error(f"Failed to set error result for request {request_id}: {set_error}")
                
    async def async_single_inference(self, image: str, question: str) -> str:
        """Asynchronous single inference"""
        request_id = f"vllm_multiuser_{uuid.uuid4().hex[:8]}"
        
        # Create future to receive result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Add request to queue
        with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                logger.warning(f"Request queue full, rejecting request {request_id}")
                return "Error: Request queue full"
            
            self.request_queue.append((request_id, image, question, future))
            self.result_futures[request_id] = future
        
        try:
            # Wait for result without timeout
            result = await future
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            # Clean up timed out request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            return "Error: Request timeout"
        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            # Clean up cancelled request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            raise
        except Exception as e:
            logger.error(f"Request {request_id} failed with exception: {e}")
            # Clean up failed request
            with self.queue_lock:
                if request_id in self.result_futures:
                    del self.result_futures[request_id]
            return f"Error: {str(e)}"
    
    def single_inference(self, image: str, question: str) -> str:
        """Synchronous single inference (wraps async method)"""
        try:
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, use thread executor
                import concurrent.futures
                
                def run_async_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.async_single_inference(image, question)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    return future.result()
                    
            except RuntimeError:
                # No running event loop
                return asyncio.run(self.async_single_inference(image, question))
                
        except Exception as e:
            logger.error(f"Single inference failed: {e}")
            return f"Error: {str(e)}"
    
    async def async_batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        """Asynchronous batch inference (decompose large batches into multiple concurrent requests)"""
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        # Create concurrent tasks
        tasks = []
        for image, question in zip(images, questions):
            task = self.async_single_inference(image, question)
            tasks.append(task)
        
        # Execute all tasks concurrently
        logger.info(f"Processing {len(tasks)} requests concurrently")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exception results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        """Synchronous batch inference"""
        try:
            try:
                loop = asyncio.get_running_loop()
                # Already in async context
                import concurrent.futures
                
                def run_async_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.async_batch_inference(images, questions)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    return future.result()
                    
            except RuntimeError:
                # No running event loop
                return asyncio.run(self.async_batch_inference(images, questions))
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [f"Error: {str(e)}"] * len(images)
    
    def get_queue_status(self):
        """Get queue status (for monitoring)"""
        with self.queue_lock:
            return {
                "queue_size": len(self.request_queue),
                "pending_results": len(self.result_futures),
                "max_queue_size": self.max_queue_size,
                "processing": self.processing,
                "processor_thread_alive": self.processor_thread.is_alive(),
                "shutdown_flag": self.shutdown_flag
            }
    
    def shutdown(self):
        """Shutdown service"""
        self.shutdown_flag = True
        
        # Wait for background thread to finish
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        
        # Clean up unfinished requests
        with self.queue_lock:
            for request_id, future in self.result_futures.items():
                if not future.done():
                    future.set_result("Error: Service shutdown")
            self.result_futures.clear()
            self.request_queue.clear()
        
        # Clean up engine and GPU memory
        try:
            if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        logger.info("vLLM MultiUser engine shutdown completed")
    
    def __del__(self):
        """Destructor"""
        try:
            self.shutdown()
        except Exception:
            pass
