#!/usr/bin/env python3
import os
import asyncio
from threading import Lock
from typing import Optional
from loguru import logger
from magic_pdf.model.custom_model import MonkeyOCR

class ModelManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.monkey_ocr_model: Optional[MonkeyOCR] = None
            self.supports_async = False
            self.model_lock = asyncio.Lock()
            self._initialized = True
    
    def initialize_model(self, config_path: str = None) -> MonkeyOCR:
        if self.monkey_ocr_model is None:
            if config_path is None:
                config_path = os.getenv("MONKEYOCR_CONFIG", "model_configs.yaml")
            
            logger.info(f"Initializing MonkeyOCR model with config: {config_path}")
            self.monkey_ocr_model = MonkeyOCR(config_path)
            self.supports_async = self._is_async_model(self.monkey_ocr_model)
            
            model_type = "async-capable" if self.supports_async else "sync-only"
            logger.info(f"MonkeyOCR model initialized successfully ({model_type})")
        
        return self.monkey_ocr_model
    
    def _is_async_model(self, model: MonkeyOCR) -> bool:
        if hasattr(model, 'chat_model'):
            chat_model = model.chat_model
            is_async = hasattr(chat_model, 'async_batch_inference')
            logger.info(f"Model {chat_model.__class__.__name__} supports async: {is_async}")
            return is_async
        return False
    
    def get_model(self) -> Optional[MonkeyOCR]:
        return self.monkey_ocr_model
    
    def is_model_loaded(self) -> bool:
        return self.monkey_ocr_model is not None
    
    def get_async_support(self) -> bool:
        return self.supports_async
    
    def get_model_lock(self):
        return self.model_lock

model_manager = ModelManager()