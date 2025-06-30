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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tempfile import gettempdir
import zipfile
from loguru import logger
import time

from magic_pdf.model.custom_model import MonkeyOCR
from parse import single_task_recognition, parse_file
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
model_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=4)

def initialize_model():
    """Initialize MonkeyOCR model"""
    global monkey_ocr_model
    if monkey_ocr_model is None:
        config_path = os.getenv("MONKEYOCR_CONFIG", "model_configs.yaml")
        monkey_ocr_model = MonkeyOCR(config_path)
    return monkey_ocr_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    try:
        initialize_model()
        logger.info("✅ MonkeyOCR model initialized successfully")
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
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext_with_dot) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix="monkeyocr_parse_")
            
            # Define a function that uses the model (this will be locked)
            def run_parse_with_model():
                return parse_file(temp_file_path, output_dir, monkey_ocr_model, split_pages)
            
            # Only lock during model inference
            async with model_lock:
                loop = asyncio.get_event_loop()
                result_dir = await loop.run_in_executor(executor, run_parse_with_model)
            
            # List generated files
            files = []
            if os.path.exists(result_dir):
                for root, dirs, filenames in os.walk(result_dir):
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), result_dir)
                        files.append(rel_path)
            
            # Create download URL with original filename
            suffix = "_split" if split_pages else "_parsed"
            zip_filename = f"{original_name}{suffix}_{int(time.time())}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            # Create ZIP file with renamed files
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
                            # Original non-split logic
                            file_ext = os.path.splitext(filename)[1]
                            file_base = os.path.splitext(filename)[0]
                            
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
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download result files"""
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get parsing results by task ID"""
    result_dir = os.path.join(temp_dir, f"monkeyocr_parse_{task_id}")

    if not os.path.exists(result_dir):
        raise HTTPException(status_code=404, detail="Results not found")
    
    files = []
    for root, dirs, filenames in os.walk(result_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), result_dir)
            files.append(rel_path)
    
    return {"files": files, "result_dir": result_dir}

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
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_{task_type}_")
            
            # Define a function that uses the model (this will be locked)
            def run_ocr_with_model():
                return single_task_recognition(
                    temp_file_path,
                    output_dir,
                    monkey_ocr_model,
                    task_type
                )
            
            # Only lock during model inference
            async with model_lock:
                loop = asyncio.get_event_loop()
                result_dir = await loop.run_in_executor(executor, run_ocr_with_model)
            
            # Read result file (can be done in parallel after model finishes)
            result_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{task_type}_result.md')]
            if not result_files:
                raise Exception("No result file generated")
            
            result_file_path = os.path.join(result_dir, result_files[0])
            with open(result_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return TaskResponse(
                success=True,
                task_type=task_type,
                content=content,
                message=f"{task_type.capitalize()} extraction completed successfully"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        return TaskResponse(
            success=False,
            task_type=task_type,
            content="",
            message=f"OCR task failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
