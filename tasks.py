#!/usr/bin/env python3
import os
import tempfile
import requests
import json
import fitz # PyMuPDF
from celery import Celery, Task

# Import the necessary components from the existing MonkeyOCR project
# This demonstrates the principle of treating MonkeyOCR as a library
from magic_pdf.model.custom_model import MonkeyOCR
from parse import parse_file
from monkeyocr.conf.settings import app_settings
from monkeyocr.utils.oss import download_file_from_oss, upload_file_to_oss

# 1. Celery Configuration
# Create the Celery application instance
app = Celery(
    'monkey_ocr_tasks',
    broker=app_settings.celery.broker_url,
    backend=app_settings.celery.result_backend
)

def update_celery_config_from_settings(app, celery_settings):
    """
    Update Celery app config from a Pydantic settings object, using only lowercase keys (Celery 5.x+ requirement).
    """
    config_dict = celery_settings.model_dump()
    app.conf.update(config_dict)  # 只用小写 key，避免新旧混用

# Apply Celery settings from app_settings.celery (全部小写)
update_celery_config_from_settings(app, app_settings.celery)

# Optional: Configure Celery for better production practices
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1, # Ensures a worker only takes one task at a time, important for GPU tasks
    task_acks_late=True, # Acknowledges task after it's completed, not when it's received
)

# 2. Model Loading Strategy: Class-Based Task
# This is the key to loading the model only once per worker process.
class OcrTask(Task):
    """
    A Celery Task that loads the MonkeyOCR model upon first access.
    The model is then available for all subsequent task executions within that worker process.
    """
    _model = None

    @property
    def model(self):
        if self._model is None:
            print("Initializing and loading MonkeyOCR model...")
            # Assuming model_configs.yaml is in the root directory
            config_path = "model_configs.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found at {config_path}")
            self._model = MonkeyOCR(config_path)
            print("MonkeyOCR model loaded successfully.")
        return self._model

# 3. Define the Celery Task
# We register our class-based task with Celery.
@app.task(base=OcrTask, bind=True)
def process_document(self, bucket_name: str, file_key: str):
    """
    A Celery task to process a single PDF document from OSS.

    Args:
        bucket_name (str): The name of the OSS bucket where the PDF is located.
        file_key (str): The key (path) of the PDF file in the bucket.

    Returns:
        dict: A dictionary containing the status and the path to the results.
    """
    print(f"Received task to process document: {bucket_name}/{file_key}")

    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, os.path.basename(file_key))
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Step 1: Download the PDF from OSS
            print(f"Downloading PDF from {bucket_name}/{file_key}...")
            download_file_from_oss(bucket_name, file_key, input_path)
            print(f"PDF downloaded to {input_path}")

            # Step 2: Process the file using the pre-loaded model
            print(f"Starting PDF parsing with MonkeyOCR...")
            result_path = parse_file(
                input_file=input_path,
                output_dir=output_dir,
                MonkeyOCR_model=self.model,  # Access the model from the task instance
                split_pages=False,  # Process as single document
                pred_abandon=False  # Don't predict abandon elements
            )
            print(f"PDF parsing complete. Results are in: {result_path}")

            # Step 3: Upload results to scholardata_bucket
            # parse_file generates multiple files: markdown, PDFs, JSONs, images
            result_files = []
            for root, dirs, files in os.walk(result_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Get relative path from result_path
                    rel_path = os.path.relpath(file_path, result_path)
                    result_files.append((file_path, rel_path))
            
            uploaded_files = []
            for local_file_path, rel_path in result_files:
                upload_key = f"monkey_ocr/{file_key}/{rel_path}"
                print(f"Uploading {local_file_path} to {app_settings.scholardata_bucket_name}/{upload_key}...")
                upload_file_to_oss(app_settings.scholardata_bucket_name, upload_key, local_file_path)
                uploaded_files.append(upload_key)
            print(f"Processed files uploaded: {uploaded_files}")

            return {
                'status': 'success',
                'source_bucket': bucket_name,
                'source_file_key': file_key,
                'uploaded_bucket': app_settings.scholardata_bucket_name,
                'uploaded_keys': uploaded_files,
                'result_path': result_path
            }

        except Exception as e:
            print(f"Error processing {bucket_name}/{file_key}: {e}")
            raise


@app.task(bind=True)
def process_document_metadata(self, bucket_name: str, file_key: str):
    """
    A Celery task to download a PDF, extract its header metadata, and upload it as JSON.

    Args:
        bucket_name (str): The name of the OSS bucket where the PDF is located.
        file_key (str): The key (path) of the PDF file in the bucket.

    Returns:
        dict: A dictionary containing the status and the path to the uploaded metadata JSON.
    """
    print(f"Received task to process metadata for: {bucket_name}/{file_key}")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_pdf_path = os.path.join(temp_dir, os.path.basename(file_key))
        metadata_json_path = os.path.join(temp_dir, f"{os.path.basename(file_key)}.json")

        try:
            # Step 1: Download the PDF from OSS
            print(f"Downloading {file_key} from {bucket_name}...")
            download_file_from_oss(bucket_name, file_key, local_pdf_path)
            print(f"PDF downloaded to {local_pdf_path}")

            # Step 2: Extract PDF header information
            print(f"Extracting metadata from {local_pdf_path}...")
            doc = fitz.open(local_pdf_path)
            metadata = doc.metadata
            doc.close()
            print(f"Extracted metadata: {metadata}")

            # Step 3: Save metadata as JSON
            with open(metadata_json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            print(f"Metadata saved to {metadata_json_path}")

            # Step 4: Upload JSON to scholardata_bucket
            upload_key = f"{app_settings.scholardata_header_prefix}/{os.path.basename(file_key)}.json"
            print(f"Uploading metadata JSON to {app_settings.scholardata_bucket_name}/{upload_key}...")
            upload_file_to_oss(app_settings.scholardata_bucket_name, upload_key, metadata_json_path)
            print(f"Metadata JSON uploaded to {upload_key}")

            return {
                'status': 'success',
                'source_bucket': bucket_name,
                'source_file_key': file_key,
                'uploaded_bucket': app_settings.scholardata_bucket_name,
                'uploaded_key': upload_key
            }

        except Exception as e:
            print(f"Error processing metadata for {bucket_name}/{file_key}: {e}")
            raise

# To run a worker for this:
# celery -A tasks worker --loglevel=info -P solo
# The -P solo (or -P gevent/eventlet) might be needed depending on the environment.
# For GPU tasks, running one worker per machine with -c 1 is recommended.
# celery -A tasks worker --loglevel=info -c 1
