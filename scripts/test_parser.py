import os
import click
from tasks import process_document_metadata, process_document

@click.group()
def cli():
    pass

@cli.command('parse_document_meta')
@click.option('--bucket', default='ivypri-testing', help='S3 bucket name')
@click.option('--file_key', required=True, help='S3 file key')
def parse_document_meta(bucket, file_key):
    print(f"Sending task to process metadata for bucket: {bucket}, file_key: {file_key}")
    task = process_document_metadata.delay(bucket, file_key)
    print(f"Task sent with ID: {task.id}")
    print("Please ensure a Celery worker is running to process this task.")
    print("Example worker command: celery -A tasks worker --loglevel=info")

@cli.command('parse_document')
@click.option('--bucket', default='ivypri-testing', help='S3 bucket name')
@click.option('--file_key', required=True, help='S3 file key')
def parse_document_cmd(bucket, file_key):
    print(f"Sending task to process document for bucket: {bucket}, file_key: {file_key}")
    task = process_document.delay(bucket, file_key)
    print(f"Task sent with ID: {task.id}")
    print("Please ensure a Celery worker is running to process this task.")
    print("Example worker command: celery -A tasks worker --loglevel=info")

if __name__ == "__main__":
    cli()
