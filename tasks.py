#!/usr/bin/env python3
import os
import tempfile
import requests
from celery import Celery, Task

# Import the necessary components from the existing MonkeyOCR project
# This demonstrates the principle of treating MonkeyOCR as a library
from magic_pdf.model.custom_model import MonkeyOCR
from parse import parse_file

# 1. Celery Configuration
# We will use Redis as the message broker and result backend.
# The broker URL should be configured via environment variables for production.
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create the Celery application instance
app = Celery(
    'monkey_ocr_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

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
    A Celery Task that loads the MonkeyOCR model upon initialization.
    The model is then available for all subsequent task executions within that worker process.
    """
    _model = None

    def __init__(self):
        if self._model is None:
            print("Initializing and loading MonkeyOCR model...")
            # Assuming model_configs.yaml is in the root directory
            config_path = "model_configs.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found at {config_path}")
            self._model = MonkeyOCR(config_path)
            print("MonkeyOCR model loaded successfully.")

    @property
    def model(self):
        return self._model

# 3. Define the Celery Task
# We register our class-based task with Celery.
@app.task(base=OcrTask, bind=True)
def process_document(self, pdf_url):
    """
    A Celery task to process a single PDF document.

    Args:
        pdf_url (str): The URL of the PDF file to process.

    Returns:
        dict: A dictionary containing the status and the path to the results.
    """
    print(f"Received task to process document: {pdf_url}")

    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, 'input.pdf')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Step 1: Download the PDF from the URL
            print(f"Downloading PDF from {pdf_url}...")
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"PDF downloaded to {input_path}")

            # Step 2: Process the file using the pre-loaded model
            # We call the original parse_file function, passing the model instance.
            print(f"Starting PDF parsing with MonkeyOCR...")
            result_path = parse_file(
                input_file=input_path,
                output_dir=output_dir,
                MonkeyOCR_model=self.model, # Access the model from the task instance
                split_pages=False, # Example: you can pass parameters here
                pred_abandon=False
            )
            print(f"PDF parsing complete. Results are in: {result_path}")

            # Step 3: (Placeholder) Upload results to a persistent storage (e.g., S3)
            # In a real application, you would upload the contents of `result_path`
            # to a cloud storage and return the URL.
            # For this example, we'll just list the output files.
            result_files = os.listdir(result_path)
            print(f"Result files: {result_files}")

            # Here, you would implement the upload logic and then clean up.

            return {
                'status': 'success',
                'pdf_url': pdf_url,
                'result_path': result_path, # In real life, this would be an S3 path
                'output_files': result_files
            }

        except Exception as e:
            print(f"Error processing {pdf_url}: {e}")
            # Celery can automatically retry the task if you raise an exception
            raise

# To run a worker for this:
# celery -A tasks worker --loglevel=info -P solo
# The -P solo (or -P gevent/eventlet) might be needed depending on the environment.
# For GPU tasks, running one worker per machine with -c 1 is recommended.
# celery -A tasks worker --loglevel=info -c 1
