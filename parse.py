#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import argparse
import sys
from pathlib import Path
import uuid
from pdf2image import convert_from_path

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.custom_model import MonkeyOCR

# 定义任务指令
TASK_INSTRUCTIONS = {
    'text': 'Please output the text content from the image.',
    'formula': 'Please write out the expression of the formula in the image using LaTeX format.',
    'table': 'Please output the table in the image in LaTeX format.'
}

def single_task_recognition(input_file, output_dir, config_path, task):
    """
    Single task recognition for specific content type
    
    Args:
        input_file: Input file path
        output_dir: Output directory
        config_path: Configuration file path
        task: Task type ('text', 'formula', 'table')
    """
    print(f"Starting single task recognition: {task}")
    print(f"Processing file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Initialize model
    print("Loading model...")
    MonkeyOCR_model = MonkeyOCR(config_path)
    
    # Get filename
    name_without_suff = os.path.basename(input_file).split(".")[0]
    
    # Prepare output directory
    local_md_dir = os.path.join(output_dir, name_without_suff)
    os.makedirs(local_md_dir, exist_ok=True)
    
    print(f"Output dir: {local_md_dir}")
    md_writer = FileBasedDataWriter(local_md_dir)
    
    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])
    
    # Check file type and prepare images
    file_extension = input_file.split(".")[-1].lower()
    images = []
    
    if file_extension == 'pdf':
        print("⚠️  WARNING: PDF input detected for single task recognition.")
        print("⚠️  WARNING: Converting all PDF pages to images for processing.")
        print("⚠️  WARNING: This may take longer and use more resources than image input.")
        print("⚠️  WARNING: Consider using individual images for better performance.")
        
        try:
            # Convert PDF pages to PIL images directly
            print("Converting PDF pages to images...")
            images = convert_from_path(input_file, dpi=150)
            print(f"Converted {len(images)} pages to images")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")
            
    elif file_extension in ['jpg', 'jpeg', 'png']:
        # Load single image
        from PIL import Image
        images = [Image.open(input_file)]
    else:
        raise ValueError(f"Single task recognition supports PDF and image files, got: {file_extension}")
    
    # Start recognition
    print(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()
    
    try:
        # Prepare instructions for all images
        instructions = [instruction] * len(images)
        
        # Use chat model for single task recognition with PIL images directly
        responses = MonkeyOCR_model.chat_model.batch_inference(images, instructions)
        
        recognition_time = time.time() - start_time
        print(f"Recognition time: {recognition_time:.2f}s")
        
        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response
        
        # Save result
        result_filename = f"{name_without_suff}_{task}_result.md"
        md_writer.write(result_filename, combined_result.encode('utf-8'))
        
        print(f"Single task recognition completed!")
        print(f"Task: {task}")
        print(f"Processed {len(images)} image(s)")
        print(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")
        
        return local_md_dir
        
    except Exception as e:
        raise RuntimeError(f"Single task recognition failed: {str(e)}")

def parse_pdf(input_file, output_dir, config_path, task=None):
    """
    Parse PDF file and save results, or perform single task recognition
    
    Args:
        input_file: Input PDF file path
        output_dir: Output directory
        config_path: Configuration file path
        task: Optional task type for single task recognition
    """
    # If task is specified, use single task recognition
    if task:
        return single_task_recognition(input_file, output_dir, config_path, task)
    
    print(f"Starting to parse file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Initialize model
    print("Loading model...")
    MonkeyOCR_model = MonkeyOCR(config_path)
    
    # Get filename
    name_without_suff = os.path.basename(input_file).split(".")[0]
    
    # Prepare output directory
    local_image_dir = os.path.join(output_dir, name_without_suff, "images")
    local_md_dir = os.path.join(output_dir, name_without_suff)
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    
    print(f"Output dir: {local_md_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)
    
    # Read file content
    reader = FileBasedDataReader()
    file_bytes = reader.read(input_file)
    
    # Create dataset instance
    file_extension = input_file.split(".")[-1].lower()
    if file_extension == "pdf":
        ds = PymuDocDataset(file_bytes)
    else:
        ds = ImageDataset(file_bytes)
    
    # Start inference
    print("Performing document parsing...")
    start_time = time.time()
    
    infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model)
    
    # Pipeline processing
    pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=MonkeyOCR_model)
    
    parsing_time = time.time() - start_time
    print(f"Parsing time: {parsing_time:.2f}s")

    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
    
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
    
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
    
    print("Results saved to ", local_md_dir)
    return local_md_dir


def main():
    parser = argparse.ArgumentParser(
        description="PDF Document Parsing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python parse.py input.pdf
  python parse.py input.pdf -o ./output
  python parse.py input.pdf -m /path/to/model -c model_configs.yaml
  python parse.py image.jpg -t text          # Single task: text recognition
  python parse.py image.jpg -t formula       # Single task: formula recognition  
  python parse.py image.jpg -t table         # Single task: table recognition
  python parse.py document.pdf -t text       # Single task: text recognition from all PDF pages (with warning)
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input PDF/image file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="model_configs.yaml",
        help="Configuration file path (default: model_configs.yaml)"
    )
    
    parser.add_argument(
        "-t", "--task",
        choices=['text', 'formula', 'table'],
        help="Single task recognition type (text/formula/table). Supports both image and PDF files."
    )
    
    args = parser.parse_args()
    
    try:
        result_dir = parse_pdf(
            args.input_file,
            args.output,
            args.config,
            args.task
        )
        
        if args.task:
            print(f"\n✅ Single task ({args.task}) recognition completed! Results saved in: {result_dir}")
        else:
            print(f"\n✅ Parsing completed! Results saved in: {result_dir}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()