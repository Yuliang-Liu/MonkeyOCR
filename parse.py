#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import argparse
import sys
import torch.distributed as dist
from pdf2image import convert_from_path

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset, MultiFileDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.custom_model import MonkeyOCR

TASK_INSTRUCTIONS = {
    'text': 'Please output the text content from the image.',
    'formula': 'Please write out the expression of the formula in the image using LaTeX format.',
    'table': 'This is the image of a table. Please output the table in html format.'
}

def parse_folder(folder_path, output_dir, config_path, task=None, split_pages=False, group_size=None, pred_abandon=False):
    """
    Parse all PDF and image files in a folder
    
    Args:
        folder_path: Input folder path
        output_dir: Output directory
        config_path: Configuration file path
        task: Optional task type for single task recognition
        group_size: Number of files to group together by total page count (None means process individually)
    """
    print(f"Starting to parse folder: {folder_path}")
    
    # Record start time for total processing time
    total_start_time = time.time()
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all supported files
    supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
    all_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                all_files.append(file_path)
    
    all_files.sort()
    
    # Initialize model once for all files
    print("Loading model...")
    MonkeyOCR_model = MonkeyOCR(config_path)
    
    successful_files = []
    failed_files = []
    
    if group_size and group_size > 1:
        # Group files by total page count
        print(f"Found {len(all_files)} files to process in groups with max {group_size} total pages")
        
        file_groups = create_file_groups_by_page_count(all_files, group_size)
        print(f"Created {len(file_groups)} file groups")
        
        for i, file_group in enumerate(file_groups, 1):
            print(f"\n{'='*60}")
            print(f"Processing file group {i}/{len(file_groups)} (contains {len(file_group)} files)")
            for file_path in file_group:
                print(f"  - {os.path.basename(file_path)}")
            print(f"{'='*60}")
            
            try:
                if task:
                    result_dir = single_task_recognition_multi_file_group(file_group, output_dir, MonkeyOCR_model, task, folder_path)
                else:
                    result_dir = parse_multi_file_group(file_group, output_dir, MonkeyOCR_model, folder_path, split_pages, pred_abandon)

                successful_files.extend(file_group)
                print(f"✅ Successfully processed file group {i}")
                
            except Exception as e:
                failed_files.extend([(path, str(e)) for path in file_group])
                print(f"❌ Failed to process file group {i}: {str(e)}")
    else:
        # Process files individually
        print(f"Found {len(all_files)} files to process individually:")
        for file_path in all_files:
            print(f"  - {file_path}")
        
        for i, file_path in enumerate(all_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(all_files)}: {os.path.basename(file_path)}")
            print(f"{'='*60}")
            
            try:
                if task:
                    result_dir = single_task_recognition(file_path, output_dir, MonkeyOCR_model, task)
                else:
                    result_dir = parse_file(file_path, output_dir, MonkeyOCR_model, pred_abandon=pred_abandon)
                
                successful_files.append(file_path)
                print(f"✅ Successfully processed: {os.path.basename(file_path)}")
                
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"❌ Failed to process {os.path.basename(file_path)}: {str(e)}")
    
    if not all_files:
        print("No supported files found in the folder.")
        return
    
    # Calculate total processing time
    total_processing_time = time.time() - total_start_time
    
    # Summary
    total_files = len(all_files)
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {total_files}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Total processing time: {total_processing_time:.2f}s")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files:
            print(f"  - {os.path.basename(file_path)}: {error}")
    
    return output_dir

def create_file_groups_by_page_count(file_paths, max_pages_per_group):
    """
    Create file groups based on total page count limit
    
    Args:
        file_paths: List of file paths
        max_pages_per_group: Maximum total pages per group
        
    Returns:
        List of file groups
    """
    import fitz
    
    groups = []
    current_group = []
    current_page_count = 0
    
    for file_path in file_paths:
        try:
            # Get page count for this file
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                with fitz.open(file_path) as doc:
                    file_page_count = len(doc)
            else:
                # Images have 1 page
                file_page_count = 1
            
            # Check if adding this file would exceed the limit
            if current_page_count + file_page_count > max_pages_per_group and current_group:
                # Start a new group
                groups.append(current_group)
                current_group = [file_path]
                current_page_count = file_page_count
            else:
                # Add to current group
                current_group.append(file_path)
                current_page_count += file_page_count
                
        except Exception as e:
            print(f"Warning: Could not determine page count for {file_path}: {e}")
            # Treat as 1 page if we can't determine
            if current_page_count + 1 > max_pages_per_group and current_group:
                groups.append(current_group)
                current_group = [file_path]
                current_page_count = 1
            else:
                current_group.append(file_path)
                current_page_count += 1
    
    # Add the last group if not empty
    if current_group:
        groups.append(current_group)
    
    return groups

def parse_multi_file_group(file_paths, output_dir, MonkeyOCR_model, base_folder_path, split_pages=False, pred_abandon=False):
    """
    Parse a group of mixed PDF and image files using MultiFileDataset
    
    Args:
        file_paths: List of file paths (PDF and images)
        output_dir: Output directory
        MonkeyOCR_model: Pre-initialized model instance
        base_folder_path: Base folder path for maintaining relative structure
        split_pages: Whether to further split each file's results by pages
    """
    print(f"Starting to parse multi-file group with {len(file_paths)} files")
    
    # Read all files and collect extensions
    reader = FileBasedDataReader()
    file_bytes_list = []
    file_extensions = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        file_bytes = reader.read(file_path)
        file_bytes_list.append(file_bytes)
        
        # Extract file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        file_extensions.append(file_ext)
    
    # Create MultiFileDataset with file extensions
    ds = MultiFileDataset(file_bytes_list, file_extensions)
    
    # Start inference with split_files=True to get individual file results
    print("Performing document parsing on multi-file group...")
    start_time = time.time()

    infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model, split_files=True, split_pages=split_pages, pred_abandon=pred_abandon)

    # Process each file result separately using original file names
    for file_idx, (file_infer_result, file_path) in enumerate(zip(infer_result, file_paths)):
        # Get original file name without extension
        file_name = '.'.join(os.path.basename(file_path).split(".")[:-1])
        
        # Maintain relative path structure from base folder
        rel_path = os.path.relpath(os.path.dirname(file_path), base_folder_path)
        
        # Create output directory for this specific file
        if rel_path == '.':
            file_local_md_dir = os.path.join(output_dir, file_name)
        else:
            file_local_md_dir = os.path.join(output_dir, rel_path, file_name)
        
        file_local_image_dir = os.path.join(file_local_md_dir, "images")
        image_dir = os.path.basename(file_local_image_dir)
        
        # Create file-specific directories
        os.makedirs(file_local_image_dir, exist_ok=True)
        os.makedirs(file_local_md_dir, exist_ok=True)
        
        print(f"Processing file {file_idx + 1}/{len(infer_result)}: {file_name} - Output dir: {file_local_md_dir}")
        
        # Handle split_pages case where file_infer_result might be a list
        if isinstance(file_infer_result, list):
            # Process each page result separately for this file
            for page_idx, page_infer_result in enumerate(file_infer_result):
                page_dir_name = f"page_{page_idx}"
                page_local_image_dir = os.path.join(file_local_md_dir, page_dir_name, "images")
                page_local_md_dir = os.path.join(file_local_md_dir, page_dir_name)
                page_image_dir = os.path.basename(page_local_image_dir)
                
                # Create page-specific directories
                os.makedirs(page_local_image_dir, exist_ok=True)
                os.makedirs(page_local_md_dir, exist_ok=True)
                
                # Create page-specific writers
                page_image_writer = FileBasedDataWriter(page_local_image_dir)
                page_md_writer = FileBasedDataWriter(page_local_md_dir)
                
                # Pipeline processing for this page
                page_pipe_result = page_infer_result.pipe_ocr_mode(page_image_writer, MonkeyOCR_model=MonkeyOCR_model)
                
                # Save page-specific results
                page_infer_result.draw_model(os.path.join(page_local_md_dir, f"{file_name}_page_{page_idx}_model.pdf"))
                page_pipe_result.draw_layout(os.path.join(page_local_md_dir, f"{file_name}_page_{page_idx}_layout.pdf"))
                page_pipe_result.draw_span(os.path.join(page_local_md_dir, f"{file_name}_page_{page_idx}_spans.pdf"))
                page_pipe_result.dump_md(page_md_writer, f"{file_name}_page_{page_idx}.md", page_image_dir)
                page_pipe_result.dump_content_list(page_md_writer, f"{file_name}_page_{page_idx}_content_list.json", page_image_dir)
                page_pipe_result.dump_middle_json(page_md_writer, f'{file_name}_page_{page_idx}_middle.json')
        else:
            # Create file-specific writers
            file_image_writer = FileBasedDataWriter(file_local_image_dir)
            file_md_writer = FileBasedDataWriter(file_local_md_dir)
            
            # Pipeline processing for this file
            file_pipe_result = file_infer_result.pipe_ocr_mode(file_image_writer, MonkeyOCR_model=MonkeyOCR_model)
            
            # Save file-specific results using original file name
            file_infer_result.draw_model(os.path.join(file_local_md_dir, f"{file_name}_model.pdf"))
            file_pipe_result.draw_layout(os.path.join(file_local_md_dir, f"{file_name}_layout.pdf"))
            file_pipe_result.draw_span(os.path.join(file_local_md_dir, f"{file_name}_spans.pdf"))
            file_pipe_result.dump_md(file_md_writer, f"{file_name}.md", image_dir)
            file_pipe_result.dump_content_list(file_md_writer, f"{file_name}_content_list.json", image_dir)
            file_pipe_result.dump_middle_json(file_md_writer, f'{file_name}_middle.json')

    parsing_time = time.time() - start_time
    print(f"Parsing and saving time: {parsing_time:.2f}s")
    
    print(f"All {len(infer_result)} files processed and saved in separate directories")
    
    # Return the base directory containing all individual file results
    return output_dir

def single_task_recognition_multi_file_group(file_paths, output_dir, MonkeyOCR_model, task, base_folder_path):
    """
    Single task recognition for a group of mixed PDF and image files
    
    Args:
        file_paths: List of file paths (PDF and images)
        output_dir: Output directory
        MonkeyOCR_model: Pre-initialized model instance
        task: Task type ('text', 'formula', 'table')
        base_folder_path: Base folder path for maintaining relative structure
    """
    print(f"Starting single task recognition: {task} for multi-file group with {len(file_paths)} files")
    
    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])
    
    # Process each file separately for single task recognition
    for file_idx, file_path in enumerate(file_paths):
        file_name = '.'.join(os.path.basename(file_path).split(".")[:-1])
        
        # Maintain relative path structure from base folder
        rel_path = os.path.relpath(os.path.dirname(file_path), base_folder_path)
        if rel_path == '.':
            local_md_dir = os.path.join(output_dir, file_name)
        else:
            local_md_dir = os.path.join(output_dir, rel_path, file_name)
        
        os.makedirs(local_md_dir, exist_ok=True)
        
        print(f"Processing file {file_idx + 1}/{len(file_paths)}: {file_name} - Output dir: {local_md_dir}")
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # Load images for this file
        file_extension = file_path.split(".")[-1].lower()
        images = []
        
        if file_extension == 'pdf':
            try:
                # Convert PDF pages to PIL images directly
                print(f"Converting PDF pages to images for {file_name}...")
                images = convert_from_path(file_path, dpi=150)
                print(f"Converted {len(images)} pages to images")
            except Exception as e:
                raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")
        elif file_extension in ['jpg', 'jpeg', 'png']:
            # Load single image
            from PIL import Image
            images = [Image.open(file_path)]
        else:
            print(f"Skipping unsupported file: {file_path}")
            continue
        
        # Start recognition for this file
        print(f"Performing {task} recognition on {len(images)} image(s) from {file_name}...")
        start_time = time.time()
        
        try:
            # Prepare instructions for all images
            instructions = [instruction] * len(images)
            
            # Use chat model for single task recognition with PIL images directly
            responses = MonkeyOCR_model.chat_model.batch_inference(images, instructions)
            
            recognition_time = time.time() - start_time
            print(f"Recognition time for {file_name}: {recognition_time:.2f}s")
            
            # Combine results
            combined_result = responses[0]
            for i, response in enumerate(responses):
                if i > 0:
                    combined_result = combined_result + "\n\n" + response
            
            # Save result
            result_filename = f"{file_name}_{task}_result.md"
            md_writer.write(result_filename, combined_result.encode('utf-8'))
            
            print(f"File {file_name} {task} recognition completed!")
            print(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")
            
            # Clean up resources for this file
            try:
                for img in images:
                    if hasattr(img, 'close'):
                        img.close()
            except Exception as cleanup_error:
                print(f"Warning: Error during cleanup for {file_name}: {cleanup_error}")
                
        except Exception as e:
            raise RuntimeError(f"Single task recognition failed for {file_name}: {str(e)}")
    
    return output_dir

def single_task_recognition(input_file, output_dir, MonkeyOCR_model, task):
    """
    Single task recognition for specific content type
    
    Args:
        input_file: Input file path
        output_dir: Output directory
        MonkeyOCR_model: Pre-initialized model instance
        task: Task type ('text', 'formula', 'table')
    """
    print(f"Starting single task recognition: {task}")
    print(f"Processing file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])
    
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
        
        # Clean up resources
        try:
            # Give some time for async tasks to complete
            time.sleep(0.5)
            
            # Close images if they were opened
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
                    
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")
        
        return local_md_dir
        
    except Exception as e:
        raise RuntimeError(f"Single task recognition failed: {str(e)}")

def parse_file(input_file, output_dir, MonkeyOCR_model, split_pages=False, pred_abandon=False):
    """
    Parse PDF or image and save results
    
    Args:
        input_file: Input PDF or image file path
        output_dir: Output directory
        MonkeyOCR_model: Pre-initialized model instance
        split_pages: Whether to split result by pages
    """
    print(f"Starting to parse file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])
    
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
    
    infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model, split_pages=split_pages, pred_abandon=pred_abandon)
    
    # Check if infer_result is a list type
    if isinstance(infer_result, list):
        print(f"Processing {len(infer_result)} pages separately...")
        
        # Process each page result separately
        for page_idx, page_infer_result in enumerate(infer_result):
            page_dir_name = f"page_{page_idx}"
            page_local_image_dir = os.path.join(output_dir, name_without_suff, page_dir_name, "images")
            page_local_md_dir = os.path.join(output_dir, name_without_suff, page_dir_name)
            page_image_dir = os.path.basename(page_local_image_dir)
            
            # Create page-specific directories
            os.makedirs(page_local_image_dir, exist_ok=True)
            os.makedirs(page_local_md_dir, exist_ok=True)
            
            # Create page-specific writers
            page_image_writer = FileBasedDataWriter(page_local_image_dir)
            page_md_writer = FileBasedDataWriter(page_local_md_dir)
            
            print(f"Processing page {page_idx} - Output dir: {page_local_md_dir}")
            
            # Pipeline processing for this page
            page_pipe_result = page_infer_result.pipe_ocr_mode(page_image_writer, MonkeyOCR_model=MonkeyOCR_model)
            
            # Save page-specific results
            page_infer_result.draw_model(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_model.pdf"))
            page_pipe_result.draw_layout(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_layout.pdf"))
            page_pipe_result.draw_span(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_spans.pdf"))
            page_pipe_result.dump_md(page_md_writer, f"{name_without_suff}_page_{page_idx}.md", page_image_dir)
            page_pipe_result.dump_content_list(page_md_writer, f"{name_without_suff}_page_{page_idx}_content_list.json", page_image_dir)
            page_pipe_result.dump_middle_json(page_md_writer, f'{name_without_suff}_page_{page_idx}_middle.json')
        
        print(f"All {len(infer_result)} pages processed and saved in separate subdirectories")
    else:
        print("Processing as single result...")
        
        # Pipeline processing for single result
        pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=MonkeyOCR_model)
        
        # Save single result (original logic)
        infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
        
        pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

        pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
        
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

        pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

    parsing_time = time.time() - start_time
    print(f"Parsing and saving time: {parsing_time:.2f}s")
    
    print("Results saved to ", local_md_dir)
    return local_md_dir

def main():
    parser = argparse.ArgumentParser(
        description="PDF Document Parsing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Single file processing
  python parse.py input.pdf                           # Parse single PDF file
  python parse.py input.pdf -o ./output               # Parse with custom output dir
  python parse.py input.pdf -s                        # Parse PDF with page splitting
  python parse.py image.jpg                           # Parse single image file
  
  # Single task recognition
  python parse.py image.jpg -t text                   # Text recognition from image
  python parse.py image.jpg -t formula                # Formula recognition from image
  python parse.py image.jpg -t table                  # Table recognition from image
  python parse.py document.pdf -t text                # Text recognition from all PDF pages
  
  # Folder processing (all files individually)
  python parse.py /path/to/folder                     # Parse all files in folder
  python parse.py /path/to/folder -s                  # Parse with page splitting
  python parse.py /path/to/folder -t text             # Single task recognition for all files
  
  # Multi-file grouping (batch processing by page count)
  python parse.py /path/to/folder -g 5                # Group files with max 5 total pages
  python parse.py /path/to/folder -g 10 -s            # Group files with page splitting
  python parse.py /path/to/folder -g 8 -t text        # Group files for single task recognition
  
  # Advanced configurations
  python parse.py input.pdf -c model_configs.yaml     # Custom model configuration
  python parse.py /path/to/folder -g 15 -s -o ./out   # Group files, split pages, custom output
  python parse.py input.pdf --pred-abandon            # Enable predicting abandon elements
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Input PDF/image file path or folder path"
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

    parser.add_argument(
        "-s", "--split_pages",
        action='store_true',
        help="Split the output of PDF pages into separate ones (default: False)"
    )
    
    parser.add_argument(
        "-g", "--group-size",
        type=int,
        help="Maximum total page count per group when processing folders (applies to all file types)"
    )

    parser.add_argument(
        "--pred-abandon",
        action='store_true',
        help="Enable predicting abandon elements like footer and header (default: False)"
    )
    
    args = parser.parse_args()
    
    MonkeyOCR_model = None
    
    try:
        # Check if input path is a directory or file
        if os.path.isdir(args.input_path):
            # Process folder
            result_dir = parse_folder(
                folder_path = args.input_path,
                output_dir = args.output,
                config_path = args.config,
                task = args.task,
                split_pages = args.split_pages,
                group_size = args.group_size,
                pred_abandon = args.pred_abandon
            )
            
            if args.task:
                if args.group_size:
                    print(f"\n✅ Folder processing with single task ({args.task}) recognition and image grouping (size: {args.group_size}) completed! Results saved in: {result_dir}")
                else:
                    print(f"\n✅ Folder processing with single task ({args.task}) recognition completed! Results saved in: {result_dir}")
            else:
                if args.group_size:
                    print(f"\n✅ Folder processing with image grouping (size: {args.group_size}) completed! Results saved in: {result_dir}")
                else:
                    print(f"\n✅ Folder processing completed! Results saved in: {result_dir}")
        elif os.path.isfile(args.input_path):
            # Process single file - initialize model for single file processing
            print("Loading model...")
            MonkeyOCR_model = MonkeyOCR(args.config)
            
            if args.task:
                result_dir = single_task_recognition(
                    input_file = args.input_path,
                    output_dir = args.output,
                    MonkeyOCR_model = MonkeyOCR_model,
                    task = args.task
                )
                print(f"\n✅ Single task ({args.task}) recognition completed! Results saved in: {result_dir}")
            else:
                result_dir = parse_file(
                    input_file = args.input_path,
                    output_dir = args.output,
                    MonkeyOCR_model = MonkeyOCR_model,
                    split_pages = args.split_pages,
                    pred_abandon = args.pred_abandon
                )
                print(f"\n✅ Parsing completed! Results saved in: {result_dir}")
        else:
            raise FileNotFoundError(f"Input path does not exist: {args.input_path}")
            
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up resources
        try:
            if MonkeyOCR_model is not None:
                # Clean up model resources if needed
                if hasattr(MonkeyOCR_model, 'chat_model') and hasattr(MonkeyOCR_model.chat_model, 'close'):
                    MonkeyOCR_model.chat_model.close()
                    
            # Give time for async tasks to complete before exiting
            time.sleep(1.0)
            
            if dist.is_initialized():
                dist.destroy_process_group()
                
        except Exception as cleanup_error:
            print(f"Warning: Error during final cleanup: {cleanup_error}")


if __name__ == "__main__":
    main()
