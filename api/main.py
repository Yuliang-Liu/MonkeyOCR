#!/usr/bin/env python3
"""
MonkeyOCR FastAPI Application
"""

import os
import io
import tempfile
from typing import Optional, List
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tempfile import gettempdir
import zipfile
from loguru import logger
import time

from magic_pdf.model.custom_model import MonkeyOCR
import uvicorn

# Response models
class TaskResponse(BaseModel):
    success: bool
    task_type: str
    content: str
    message: Optional[str] = None

class ParseResponse(BaseModel):
    success: bool
    message: str
    output_dir: Optional[str] = None
    files: Optional[List[str]] = None
    download_url: Optional[str] = None

# Global model instance and lock
monkey_ocr_model = None
supports_async = False
model_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=4)

def initialize_model():
    """Initialize MonkeyOCR model"""
    global monkey_ocr_model
    global supports_async
    if monkey_ocr_model is None:
        config_path = os.getenv("MONKEYOCR_CONFIG", "model_configs.yaml")
        monkey_ocr_model = MonkeyOCR(config_path)
        supports_async = is_async_model(monkey_ocr_model)
    return monkey_ocr_model

def is_async_model(model: MonkeyOCR) -> bool:
    """Check if the model supports async concurrent calls"""
    if hasattr(model, 'chat_model'):
        chat_model = model.chat_model
        # More specific check for async models
        is_async = hasattr(chat_model, 'async_batch_inference')
        logger.info(f"Model {chat_model.__class__.__name__} supports async: {is_async}")
        return is_async
    return False

async def smart_model_call(func, *args, **kwargs):
    """
    Smart wrapper that automatically chooses between concurrent and blocking calls
    based on the model's capabilities
    """
    global monkey_ocr_model, model_lock
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if supports_async:
        # For async models, no need for model_lock, can run concurrently
        logger.info("Using concurrent execution (async model detected)")
        # Use asyncio's thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    else:
        # For sync models, use model_lock to prevent conflicts
        logger.info("Using blocking execution with lock (sync model detected)")
        async with model_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

async def smart_batch_model_call(images_and_questions_list, batch_func):
    """
    Smart batch processing that can handle multiple requests efficiently
    """
    global monkey_ocr_model
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if supports_async and hasattr(monkey_ocr_model.chat_model, 'async_batch_inference'):
        # Use native async batch processing for maximum efficiency
        logger.info(f"Using native async batch processing for {len(images_and_questions_list)} requests")
        
        # Flatten all images and questions
        all_images = []
        all_questions = []
        request_indices = []
        
        for i, (images, questions) in enumerate(images_and_questions_list):
            for img, q in zip(images, questions):
                all_images.append(img)
                all_questions.append(q)
                request_indices.append(i)
        
        # Single batch call for all requests
        try:
            # Use the chat model's async batch inference method properly
            all_results = await monkey_ocr_model.chat_model.async_batch_inference(all_images, all_questions)
        except Exception as e:
            logger.error(f"Async batch inference failed: {e}, falling back to individual processing")
            # Fallback to individual processing using the corrected method
            results = []
            for images, questions in images_and_questions_list:
                try:
                    # Use the thread-safe smart_model_call wrapper
                    result = await smart_model_call(batch_func, images, questions)
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Individual processing also failed: {inner_e}")
                    results.append([f"Error: {str(inner_e)}"] * len(images))
            return results
        
        # Reconstruct results for each original request
        results = []
        result_idx = 0
        for images, questions in images_and_questions_list:
            request_results = []
            for _ in range(len(images)):
                request_results.append(all_results[result_idx])
                result_idx += 1
            results.append(request_results)
        
        return results
    
    elif supports_async:
        # Concurrent processing for async models
        logger.info(f"Using concurrent batch processing for {len(images_and_questions_list)} requests")
        tasks = []
        for images, questions in images_and_questions_list:
            task = smart_model_call(batch_func, images, questions)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Sequential processing for sync models
        logger.info(f"Using sequential batch processing for {len(images_and_questions_list)} requests")
        results = []
        for images, questions in images_and_questions_list:
            result = await smart_model_call(batch_func, images, questions)
            results.append(result)
        return results

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    try:
        initialize_model()
        model_type = "async-capable" if supports_async else "sync-only"
        logger.info(f"✅ MonkeyOCR model initialized successfully ({model_type})")
    except Exception as e:
        logger.info(f"❌ Failed to initialize MonkeyOCR model: {e}")
        raise
    
    yield
    
    # Shutdown
    global executor
    executor.shutdown(wait=True)
    logger.info("🔄 Application shutdown complete")

app = FastAPI(
    title="MonkeyOCR API",
    description="OCR and Document Parsing API using MonkeyOCR",
    version="1.0.0",
    lifespan=lifespan
)

temp_dir = os.getenv("TMPDIR", gettempdir())
logger.info(f"Using temporary directory: {temp_dir}")
os.makedirs(temp_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=temp_dir), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MonkeyOCR API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": monkey_ocr_model is not None}

@app.post("/ocr/text", response_model=TaskResponse)
async def extract_text(file: UploadFile = File(...)):
    """Extract text from image or PDF"""
    return await perform_ocr_task(file, "text")

@app.post("/ocr/formula", response_model=TaskResponse)
async def extract_formula(file: UploadFile = File(...)):
    """Extract formulas from image or PDF"""
    return await perform_ocr_task(file, "formula")

@app.post("/ocr/table", response_model=TaskResponse)
async def extract_table(file: UploadFile = File(...)):
    """Extract tables from image or PDF"""
    return await perform_ocr_task(file, "table")

@app.post("/parse", response_model=ParseResponse)
async def parse_document(file: UploadFile = File(...)):
    """Parse complete document (PDF or image)"""
    return await parse_document_internal(file, split_pages=False)

@app.post("/parse/split", response_model=ParseResponse)
async def parse_document_split(file: UploadFile = File(...)):
    """Parse complete document and split result by pages (PDF or image)"""
    return await parse_document_internal(file, split_pages=True)

async def async_parse_file(input_file_path: str, output_dir: str, split_pages: bool = False):
    """
    Optimized async version of parse_file that breaks down processing into async chunks
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import uuid
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Get filename with unique identifier to avoid conflicts
    name_without_suff = '.'.join(os.path.basename(input_file_path).split(".")[:-1])
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    safe_name = f"{name_without_suff}_{unique_id}"
    
    # Prepare output directory with unique name
    local_image_dir = os.path.join(output_dir, safe_name, "images")
    local_md_dir = os.path.join(output_dir, safe_name)
    image_dir = os.path.basename(local_image_dir)
    
    # Create directories asynchronously with better error handling
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_image_dir)
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_md_dir)
    
    logger.info(f"Output dir: {local_md_dir}")
    
    # Read file content in thread pool
    def read_file_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataReader
        reader = FileBasedDataReader()
        return reader.read(input_file_path)
    
    file_bytes = await asyncio.get_event_loop().run_in_executor(None, read_file_sync)
    
    # Create dataset instance in thread pool
    def create_dataset_sync():
        from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
        file_extension = input_file_path.split(".")[-1].lower()
        if file_extension == "pdf":
            return PymuDocDataset(file_bytes)
        else:
            return ImageDataset(file_bytes)
    
    ds = await asyncio.get_event_loop().run_in_executor(None, create_dataset_sync)
    
    # Run inference in thread pool
    def run_inference_sync():
        from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
        return ds.apply(doc_analyze_llm, MonkeyOCR_model=monkey_ocr_model, split_pages=split_pages)
    
    logger.info("Starting document parsing...")
    start_time = time.time()
    
    # Use smart model call for inference
    if supports_async:
        # For async models, run without lock
        infer_result = await asyncio.get_event_loop().run_in_executor(None, run_inference_sync)
    else:
        # For sync models, use lock
        async with model_lock:
            infer_result = await asyncio.get_event_loop().run_in_executor(None, run_inference_sync)
    
    parsing_time = time.time() - start_time
    logger.info(f"Parsing time: {parsing_time:.2f}s")
    
    # Process results asynchronously
    await process_inference_results_async(
        infer_result, output_dir, safe_name, 
        local_image_dir, local_md_dir, image_dir, split_pages
    )
    
    return local_md_dir

async def process_inference_results_async(infer_result, output_dir, name_without_suff, 
                                        local_image_dir, local_md_dir, image_dir, split_pages):
    """
    Process inference results asynchronously
    """
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    
    def create_writers():
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        return image_writer, md_writer
    
    # Check if infer_result is a list (split pages)
    if isinstance(infer_result, list):
        logger.info(f"Processing {len(infer_result)} pages separately...")
        
        # Process pages concurrently
        tasks = []
        for page_idx, page_infer_result in enumerate(infer_result):
            task = process_single_page_async(
                page_infer_result, page_idx, output_dir, name_without_suff
            )
            tasks.append(task)
        
        # Wait for all page processing to complete
        await asyncio.gather(*tasks)
        
        logger.info(f"All {len(infer_result)} pages processed and saved in separate subdirectories")
    else:
        # Process single result
        logger.info("Processing as single result...")
        await process_single_result_async(
            infer_result, name_without_suff, local_image_dir, local_md_dir, image_dir
        )

async def process_single_page_async(page_infer_result, page_idx, output_dir, name_without_suff):
    """
    Process a single page result asynchronously
    """
    import uuid
    
    page_dir_name = f"page_{page_idx}"
    page_local_image_dir = os.path.join(output_dir, name_without_suff, page_dir_name, "images")
    page_local_md_dir = os.path.join(output_dir, name_without_suff, page_dir_name)
    page_image_dir = os.path.basename(page_local_image_dir)
    
    # Create page-specific directories with better error handling
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create page directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, page_local_image_dir)
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, page_local_md_dir)
    
    def process_page_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        # Create page-specific writers
        page_image_writer = FileBasedDataWriter(page_local_image_dir)
        page_md_writer = FileBasedDataWriter(page_local_md_dir)
        
        logger.info(f"Processing page {page_idx} - Output dir: {page_local_md_dir}")
        
        # Pipeline processing for this page
        page_pipe_result = page_infer_result.pipe_ocr_mode(page_image_writer, MonkeyOCR_model=monkey_ocr_model)
        
        # Save page-specific results
        page_infer_result.draw_model(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_model.pdf"))
        page_pipe_result.draw_layout(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_layout.pdf"))
        page_pipe_result.draw_span(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_spans.pdf"))
        page_pipe_result.dump_md(page_md_writer, f"{name_without_suff}_page_{page_idx}.md", page_image_dir)
        page_pipe_result.dump_content_list(page_md_writer, f"{name_without_suff}_page_{page_idx}_content_list.json", page_image_dir)
        page_pipe_result.dump_middle_json(page_md_writer, f'{name_without_suff}_page_{page_idx}_middle.json')
    
    # Run page processing in thread pool
    await asyncio.get_event_loop().run_in_executor(None, process_page_sync)

async def process_single_result_async(infer_result, name_without_suff, local_image_dir, local_md_dir, image_dir):
    """
    Process single result asynchronously
    """
    def process_single_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # Pipeline processing for single result
        pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=monkey_ocr_model)
        
        # Save single result
        infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
        pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
        pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)
        pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
    
    # Run processing in thread pool
    await asyncio.get_event_loop().run_in_executor(None, process_single_sync)

async def async_single_task_recognition(input_file_path: str, output_dir: str, task: str):
    """
    Optimized async version of single_task_recognition
    """
    import uuid
    
    logger.info(f"Starting async single task recognition: {task}")
    
    # Get filename with unique identifier to avoid conflicts
    name_without_suff = '.'.join(os.path.basename(input_file_path).split(".")[:-1])
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    safe_name = f"{name_without_suff}_{unique_id}"
    
    # Prepare output directory with unique name
    local_md_dir = os.path.join(output_dir, safe_name)
    
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_md_dir)
    
    # Get task instruction
    from parse import TASK_INSTRUCTIONS
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])
    
    # Load images asynchronously
    def load_images_sync():
        file_extension = input_file_path.split(".")[-1].lower()
        images = []
        
        if file_extension == 'pdf':
            from magic_pdf.utils.load_image import pdf_to_images
            images = pdf_to_images(input_file_path)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            from PIL import Image
            images = [input_file_path]
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        return images, file_extension
    
    images, file_extension = await asyncio.get_event_loop().run_in_executor(None, load_images_sync)
    
    # Perform recognition
    logger.info(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()
    
    # Prepare instructions for all images
    instructions = [instruction] * len(images)
    
    # Use chat model for recognition
    if supports_async and hasattr(monkey_ocr_model.chat_model, 'async_batch_inference'):
        # Use async batch inference if available
        try:
            responses = await monkey_ocr_model.chat_model.async_batch_inference(images, instructions)
        except Exception as e:
            logger.warning(f"Async batch inference failed: {e}, falling back to sync")
            responses = await asyncio.get_event_loop().run_in_executor(
                None, monkey_ocr_model.chat_model.batch_inference, images, instructions
            )
    else:
        # Use sync batch inference in thread pool
        responses = await asyncio.get_event_loop().run_in_executor(
            None, monkey_ocr_model.chat_model.batch_inference, images, instructions
        )
    
    recognition_time = time.time() - start_time
    logger.info(f"Recognition time: {recognition_time:.2f}s")
    
    # Combine and save results
    def save_results_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response
        
        # Save result with original name (without unique suffix)
        result_filename = f"{name_without_suff}_{task}_result.md"
        md_writer.write(result_filename, combined_result.encode('utf-8'))
        
        return result_filename
    
    result_filename = await asyncio.get_event_loop().run_in_executor(None, save_results_sync)
    
    logger.info(f"Single task recognition completed!")
    logger.info(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")
    
    # Clean up images
    def cleanup_images():
        try:
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
        except Exception as cleanup_error:
            logger.warning(f"Warning: Error during cleanup: {cleanup_error}")
    
    await asyncio.get_event_loop().run_in_executor(None, cleanup_images)
    
    return local_md_dir

async def parse_document_internal(file: UploadFile, split_pages: bool = False):
    """Internal function to parse document with optional page splitting"""
    try:
        if not monkey_ocr_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Validate file type - support both PDF and image files
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext_with_dot = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        
        if file_ext_with_dot not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext_with_dot}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Get original filename without extension
        original_name = '.'.join(file.filename.split('.')[:-1])
        
        # Save uploaded file temporarily with unique name to avoid conflicts
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext_with_dot, prefix=f"upload_{unique_suffix}_") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory with unique name
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_parse_{unique_suffix}_")
            
            # Use optimized async parse function
            result_dir = await async_parse_file(temp_file_path, output_dir, split_pages)
            
            # List generated files
            files = []
            if os.path.exists(result_dir):
                for root, dirs, filenames in os.walk(result_dir):
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), result_dir)
                        files.append(rel_path)
            
            # Create download URL with original filename and timestamp
            suffix = "_split" if split_pages else "_parsed"
            timestamp = int(time.time() * 1000)  # Use milliseconds for better uniqueness
            zip_filename = f"{original_name}{suffix}_{timestamp}_{unique_suffix}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            # Create ZIP file asynchronously
            await create_zip_file_async(result_dir, zip_path, original_name, split_pages)
            
            download_url = f"/static/{zip_filename}"
            
            # Determine file type for response message
            file_type = "PDF" if file_ext_with_dot == '.pdf' else "image"
            parse_type = "with page splitting" if split_pages else "standard"
            
            return ParseResponse(
                success=True,
                message=f"{file_type} parsing ({parse_type}) completed successfully",
                output_dir=result_dir,
                files=files,
                download_url=download_url
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

async def create_zip_file_async(result_dir, zip_path, original_name, split_pages):
    """Create ZIP file asynchronously"""
    def create_zip_sync():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, filenames in os.walk(result_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, result_dir)
                    
                    if split_pages:
                        # For split pages, maintain the page directory structure
                        # but add original name prefix
                        if rel_path.startswith('page_'):
                            # Keep the page structure: page_0/filename -> page_0/original_name_filename
                            parts = rel_path.split('/', 1)
                            if len(parts) == 2:
                                page_dir, filename_part = parts
                                if filename_part.startswith('images/'):
                                    # Handle images: page_0/images/img.jpg -> page_0/images/original_name_img.jpg
                                    img_name = filename_part.replace('images/', '')
                                    new_filename = f"{page_dir}/images/{original_name}_{img_name}"
                                else:
                                    # Handle other files in page directories
                                    new_filename = f"{page_dir}/{original_name}_{filename_part}"
                            else:
                                new_filename = f"{original_name}_{rel_path}"
                        else:
                            new_filename = f"{original_name}_{rel_path}"
                    else:
                        # Handle different file types
                        if filename.endswith('.md'):
                            new_filename = f"{original_name}.md"
                        elif filename.endswith('_content_list.json'):
                            new_filename = f"{original_name}_content_list.json"
                        elif filename.endswith('_middle.json'):
                            new_filename = f"{original_name}_middle.json"
                        elif filename.endswith('_model.pdf'):
                            new_filename = f"{original_name}_model.pdf"
                        elif filename.endswith('_layout.pdf'):
                            new_filename = f"{original_name}_layout.pdf"
                        elif filename.endswith('_spans.pdf'):
                            new_filename = f"{original_name}_spans.pdf"
                        else:
                            # For images and other files, keep relative path structure but rename
                            if 'images/' in rel_path:
                                # Keep images in images subfolder with original name prefix
                                image_name = os.path.basename(rel_path)
                                new_filename = f"images/{original_name}_{image_name}"
                            else:
                                new_filename = f"{original_name}_{filename}"
                    
                    zipf.write(file_path, new_filename)
    
    # Run ZIP creation in thread pool to avoid blocking
    await asyncio.get_event_loop().run_in_executor(None, create_zip_sync)

async def perform_ocr_task(file: UploadFile, task_type: str) -> TaskResponse:
    """Perform OCR task on uploaded file"""
    try:
        if not monkey_ocr_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily with unique name
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, prefix=f"ocr_{unique_suffix}_") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory with unique name
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_{task_type}_{unique_suffix}_")
            
            # Use optimized async single task recognition
            result_dir = await async_single_task_recognition(temp_file_path, output_dir, task_type)
            
            # Read result file
            def read_result_sync():
                result_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{task_type}_result.md')]
                if not result_files:
                    raise Exception("No result file generated")
                
                result_file_path = os.path.join(result_dir, result_files[0])
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            content = await asyncio.get_event_loop().run_in_executor(None, read_result_sync)
            
            return TaskResponse(
                success=True,
                task_type=task_type,
                content=content,
                message=f"{task_type.capitalize()} extraction completed successfully"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"OCR task failed: {str(e)}")
        return TaskResponse(
            success=False,
            task_type=task_type,
            content="",
            message=f"OCR task failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
