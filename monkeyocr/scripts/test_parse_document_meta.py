import os
from tasks import process_document_metadata

# Set up dummy environment variables for local testing if needed
# In a real deployment, these would be set in your environment or .env file
os.environ.setdefault('ALIYUN_ACCESS_KEY', 'your_access_key')
os.environ.setdefault('ALIYUN_ACCESS_SECRET', 'your_access_secret')
os.environ.setdefault('CELERY_BROKER_URL', 'redis://localhost:6379/12')
os.environ.setdefault('CELERY_RESULT_BACKEND', 'redis://localhost:6379/12')

if __name__ == "__main__":
    # Example usage: send a task to process metadata for a dummy file
    dummy_bucket_name = "your-source-bucket" # Replace with a real bucket name
    dummy_file_key = "path/to/your/dummy_document.pdf" # Replace with a real file key

    print(f"Sending task to process metadata for bucket: {dummy_bucket_name}, file_key: {dummy_file_key}")
    task = process_document_metadata.delay(dummy_bucket_name, dummy_file_key)
    print(f"Task sent with ID: {task.id}")
    print("Please ensure a Celery worker is running to process this task.")
    print("Example worker command: celery -A tasks worker --loglevel=info")
