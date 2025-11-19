import anyio
import gradio as gr
import os
import base64
from magic_pdf.utils.load_image import pdf_to_images
import re
import zipfile
import subprocess
import tempfile
import uuid
import json

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.model_manager import model_manager
from PIL import Image
from loguru import logger

def load_i18n(lang='en'):
    i18n_dir = os.path.join(os.path.dirname(__file__), 'i18n')
    i18n_file = os.path.join(i18n_dir, f'{lang}.json')
    try:
        with open(i18n_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        with open(os.path.join(i18n_dir, 'en.json'), 'r', encoding='utf-8') as f:
            return json.load(f)

def get_instructions(texts):
    return [
        (texts['prompt_text_content_label'], texts['prompt_text_content']),
        (texts['prompt_formula_label'], texts['prompt_formula']),
        (texts['prompt_table_html_label'], texts['prompt_table_html']),
        (texts['prompt_table_latex_label'], texts['prompt_table_latex'])
    ]


def render_latex_table_to_image(latex_content, temp_dir):
    """
    Render LaTeX table to image and return base64 encoding
    """
    try:
        # Use regex to extract tabular environment content
        pattern = r"(\\begin\{tabular\}.*?\\end\{tabular\})"
        matches = re.findall(pattern, latex_content, re.DOTALL)
        
        if matches:
            # If complete tabular environment found, use the first one
            table_content = matches[0]
        elif '\\begin{tabular}' in latex_content:
            # If only start tag without end tag, add end tag
            if '\\end{tabular}' not in latex_content:
                table_content = latex_content + '\n\\end{tabular}'
            else:
                table_content = latex_content
        else:
            # If no tabular environment, might be table content that needs wrapping
            return latex_content  # Return original content without rendering
        
        # Build complete LaTeX document, consistent with reference code format
        full_latex = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{bm}
\usepackage{multirow}
\usepackage{array}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{makecell}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tabular}
\begin{document}
""" + table_content + r"""
\end{document}
"""
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        tex_path = os.path.join(temp_dir, f"table_{unique_id}.tex")
        pdf_path = os.path.join(temp_dir, f"table_{unique_id}.pdf")
        png_path = os.path.join(temp_dir, f"table_{unique_id}.png")
        
        # Write tex file
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(full_latex)
        
        # Call pdflatex to generate PDF, add more detailed error handling
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_path], 
            timeout=20,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # If compilation fails, output error info and return original content
            print(f"LaTeX compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"LaTeX content: {table_content}")
            return f"<pre>{latex_content}</pre>"  # Return original content as preformatted text
        
        # Check if PDF file is generated
        if not os.path.exists(pdf_path):
            print(f"PDF file not generated: {pdf_path}")
            return f"<pre>{latex_content}</pre>"
        
        # Convert PDF to PNG image
        images = pdf_to_images(pdf_path)
        images[0].save(png_path, "PNG")
        
        # Read image and convert to base64
        with open(png_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        
        # Clean up temporary files
        for file_path in [tex_path, pdf_path, png_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        # Clean up possible auxiliary files
        for ext in ['.aux', '.log', '.fls', '.fdb_latexmk']:
            aux_file = os.path.join(temp_dir, f"table_{unique_id}{ext}")
            if os.path.exists(aux_file):
                os.remove(aux_file)
        
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;height:auto;">'
        
    except subprocess.TimeoutExpired:
        print("LaTeX compilation timeout")
        return f"<pre>{latex_content}</pre>"
    except Exception as e:
        print(f"LaTeX rendering error: {e}")
        return f"<pre>{latex_content}</pre>"  # If rendering fails, return original content as preformatted text

async def parse_pdf_and_return_results(pdf_file):
    if pdf_file is None:
        return (
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False)  # Hide parsing prompt
        )
    parent_path = os.path.dirname(pdf_file)
    full_name = os.path.basename(pdf_file)
    name = '.'.join(full_name.split(".")[:-1])
    local_image_dir, local_md_dir = parent_path+"/markdown/images", parent_path+"/markdown"
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)   
    reader1 = FileBasedDataReader(parent_path)
    data_bytes = reader1.read(full_name)
    if full_name.split(".")[-1] in ['jpg', 'jpeg', 'png']:
        ds = ImageDataset(data_bytes)
    else:
        ds = PymuDocDataset(data_bytes)
    MonkeyOCR_model = model_manager.get_model()
    async with model_manager.get_model_lock():
        infer_result = await anyio.to_thread.run_sync(
            lambda: ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model)
        )
        pipe_result = await anyio.to_thread.run_sync(
            lambda: infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=MonkeyOCR_model)
        )
    layout_pdf_path = os.path.join(parent_path, f"{name}_layout.pdf")
    pipe_result.draw_layout(layout_pdf_path)
    pipe_result.dump_md(md_writer, f"{name}.md", image_dir)
    md_content_ori = FileBasedDataReader(local_md_dir).read(f"{name}.md").decode("utf-8")
    
    # Create temporary directory for LaTeX rendering
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process HTML-wrapped LaTeX tables
        def replace_html_latex_table(match):
            html_content = match.group(1)
            # Check if contains \begin{tabular}
            if '\\begin{tabular}' in html_content:
                return render_latex_table_to_image(html_content, temp_dir)
            else:
                return match.group(0)  # Keep original
        
        # Use regex to replace LaTeX tables wrapped in <html>...</html>
        md_content = re.sub(r'<html>(.*?)</html>', replace_html_latex_table, md_content_ori, flags=re.DOTALL)
        
        # Convert local image links in markdown to base64 encoded HTML
        def replace_image_with_base64(match):
            img_path = match.group(1)
            # Handle relative paths
            if not os.path.isabs(img_path):
                full_img_path = os.path.join(local_md_dir, img_path)
            else:
                full_img_path = img_path
            
            try:
                if os.path.exists(full_img_path):
                    with open(full_img_path, "rb") as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    # Get file extension to determine MIME type
                    ext = os.path.splitext(full_img_path)[1].lower()
                    mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else f"image/{ext[1:]}"
                    return f'<img src="data:{mime_type};base64,{img_base64}" style="max-width:100%;height:auto;">'
                else:
                    return match.group(0)  # If file not found, keep original
            except Exception:
                return match.group(0)  # If error, keep original
        
        # Use regex to replace markdown image syntax ![alt](path)
        md_content = re.sub(r'!\[.*?\]\(([^)]+)\)', replace_image_with_base64, md_content)
        
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Create zip file
    zip_path = os.path.join(parent_path, f"{name}_markdown.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Traverse local_md_dir folder, add all files to zip
        for root, dirs, files in os.walk(local_md_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path, maintain folder structure
                arcname = os.path.relpath(file_path, local_md_dir)
                zipf.write(file_path, arcname)
    
    return (
        md_content_ori,
        md_content,
        gr.update(value=layout_pdf_path, visible=True),
        gr.update(value=zip_path, visible=True),
    )

async def chat_with_image(message, pdf_file, texts):
    """Chat with the uploaded image"""
    if pdf_file is None:
        return texts['error_no_file_uploaded']

    base_dir = os.path.dirname(pdf_file)
    file_ext = pdf_file.split(".")[-1].lower()
    if file_ext not in ['jpg', 'jpeg', 'png', 'pdf']:
        return texts['error_no_file_uploaded']

    try:
        MonkeyOCR_model = model_manager.get_model()
        if file_ext in ['jpg', 'jpeg', 'png']:
            image_path = pdf_file
            async with model_manager.get_model_lock():
                response = await anyio.to_thread.run_sync(
                    lambda: MonkeyOCR_model.chat_model.batch_inference([image_path], [message])[0]
                )
        else:
            response = texts['error_pdf_chat_not_supported']
        file_writer = FileBasedDataWriter(base_dir)
        md_name = f"chat_response_{uuid.uuid4().hex}.md"
        file_writer.write(md_name, response.encode('utf-8'))
        return response, response, gr.update(value=None, visible=True), gr.update(value=os.path.join(base_dir, md_name), visible=True)
    except Exception as e:
        response = f"{texts['error_chat_processing']}{str(e)}"
        return response, response, gr.update(value=None, visible=True), gr.update(value=None, visible=True)

# Global cache: store images of each page
pdf_cache = {
    "images": [],
    "current_page": 0,
    "total_pages": 0,
}

def load_file(file, texts):
    if file.endswith('.pdf'):
        pages = pdf_to_images(file)
    else:
        image = Image.open(file)
        pages = [image]
    pdf_cache["images"] = pages
    pdf_cache["current_page"] = 0
    pdf_cache["total_pages"] = len(pages)
    return pages[0], f"<div id='page_info_box'>1{texts['page_separator']}{len(pages)}</div>"

def turn_page(direction, texts):
    if not pdf_cache["images"]:
        return None, f"<div id='page_info_box'>0{texts['page_separator']}0</div>"

    if direction == "prev":
        pdf_cache["current_page"] = max(0, pdf_cache["current_page"] - 1)
    elif direction == "next":
        pdf_cache["current_page"] = min(pdf_cache["total_pages"] - 1, pdf_cache["current_page"] + 1)

    index = pdf_cache["current_page"]
    return pdf_cache["images"][index], f"<div id='page_info_box'>{index + 1}{texts['page_separator']}{pdf_cache['total_pages']}</div>"

# Global variables to store parsed result file paths
layout_pdf_path = None
markdown_zip_path = None

def download_layout_pdf():
    if layout_pdf_path and os.path.exists(layout_pdf_path):
        return layout_pdf_path
    return None

def download_markdown_zip():
    if markdown_zip_path and os.path.exists(markdown_zip_path):
        return markdown_zip_path
    return None

async def parse_and_update_view(pdf_file, texts):
    """Parse PDF and update view"""

    if pdf_file is None:
        return (
            gr.update(),
            texts['error_please_upload_pdf'],
            texts['error_please_upload_pdf'],
            f"<div id='page_info_box'>0{texts['page_separator']}0</div>",
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=True),
        )

    try:
        # Call the original parsing function
        md_content_ori, md_content, layout_pdf_update, zip_update = await parse_pdf_and_return_results(pdf_file)

        # Update global variables
        layout_pdf_path = layout_pdf_update['value']
        markdown_zip_path = zip_update['value']

        # Load parsed layout PDF for preview
        if layout_pdf_path and os.path.exists(layout_pdf_path):
            pages = pdf_to_images(layout_pdf_path)
            pdf_cache["images"] = pages
            pdf_cache["current_page"] = 0
            pdf_cache["total_pages"] = len(pages)
            preview_image = pages[0]
            page_info = f"<div id='page_info_box'>1{texts['page_separator']}{len(pages)}</div>"
        else:
            preview_image = None
            page_info = f"<div id='page_info_box'>0{texts['page_separator']}0</div>"

        return (
            preview_image,
            md_content,
            md_content_ori,
            page_info,
            layout_pdf_update,
            zip_update,
        )
    except:
        logger.warning(texts['warning_parse_failed_switching_chat'])
        md_content_ori, md_content, layout_pdf_update, zip_update = chat_with_image(texts['prompt_text_content'], pdf_file, texts)
        return (
            gr.update(),
            md_content,
            md_content_ori,
            f"<div id='page_info_box'>1{texts['page_separator']}1</div>",
            layout_pdf_update,
            zip_update,
        )

def clear_all(texts):
    """Clear all inputs and outputs"""
    pdf_cache["images"] = []
    pdf_cache["current_page"] = 0
    pdf_cache["total_pages"] = 0
    return (
        None,
        None,
        texts['please_click_parse'],
        texts['waiting_for_parsing'],
        f"<div id='page_info_box'>0{texts['page_separator']}0</div>",
        gr.update(value=None, visible=True),
        gr.update(value=None, visible=True),
    )

def switch_language(lang):
    texts = load_i18n(lang)
    instructions = get_instructions(texts)
    return (
        texts,
        gr.update(choices=instructions, value=instructions[0], label=texts['select_prompt']),
        gr.update(value=f"<div style=\"display: flex; align-items: center; justify-content: center; margin-bottom: 20px;\"><h1 style=\"margin: 0; font-size: 2em;\">{texts['title']}</h1></div>"),
        gr.update(value=f"### {texts['upload_section']}"),
        gr.update(label=texts['select_file']),
        gr.update(value=f"### {texts['actions_section']}"),
        gr.update(value=texts['parse_button']),
        gr.update(value=texts['chat_button']),
        gr.update(value=texts['clear_button']),
        gr.update(value=f"### {texts['file_preview']}"),
        gr.update(value=texts['prev_page']),
        gr.update(value=texts['next_page']),
        gr.update(value=f"### {texts['result_display']}"),
        gr.update(label=texts['markdown_render_preview']),
        gr.update(label=texts['markdown_raw_text']),
        gr.update(value=texts['please_click_parse']),
        gr.update(value=texts['waiting_for_parsing']),
        gr.update(label=texts['download_pdf_layout']),
        gr.update(label=texts['download_markdown']),
        gr.update(label=texts['language'])
    )

css = """
#page_info_html {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;  /* Ensure consistent height with button row */
    margin: 0 12px;  /* Increase left and right margin for centering */
}

#page_info_box {
    padding: 8px 20px;
    font-size: 16px;
    border: 1px solid #bbb;
    border-radius: 8px;
    background-color: #f8f8f8;
    text-align: center;
    min-width: 80px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

#markdown_output {
    min-height: 800px;
    overflow: auto;
}

footer {
    visibility: hidden;
}
"""

def create_gradio_app():
    with gr.Blocks(theme="ocean", css=css, title='MonkeyOCR') as demo:
        texts = load_i18n('en')
        instructions = get_instructions(texts)
        i18n_texts = gr.State(load_i18n('en'))

        title_html = gr.HTML(f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">{texts['title']}</h1>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, variant="compact"):
                upload_section = gr.Markdown(f"### {texts['upload_section']}")
                pdf_input = gr.File(label=texts['select_file'], type="filepath", file_types=[".pdf", ".jpg", ".jpeg", ".png"], show_label=True)
                chat_input = gr.Dropdown(label=texts['select_prompt'], choices=instructions, value=instructions[0], show_label=True, multiselect=False, visible=True)
                actions_section = gr.Markdown(f"### {texts['actions_section']}")
                parse_button = gr.Button(texts['parse_button'], variant="primary")
                chat_button = gr.Button(texts['chat_button'], variant="secondary")
                clear_button = gr.Button(texts['clear_button'], variant="huggingface")
                
                lang_switch = gr.Dropdown(
                    label=texts['language'],
                    choices=[("English", "en"), ("中文", "zh")],
                    value="en",
                    show_label=True
                )

            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    with gr.Column(scale=3):
                        file_preview_section = gr.Markdown(f"### {texts['file_preview']}")
                        pdf_view = gr.Image(label=texts['pdf_preview'], visible=True, height=800, show_label=False)
                        with gr.Row():
                            prev_btn = gr.Button(texts['prev_page'])
                            page_info = gr.HTML(value=f"<div id='page_info_box'>0{texts['page_separator']}0</div>", elem_id="page_info_html")
                            next_btn = gr.Button(texts['next_page'])
                    with gr.Column(scale=3):
                        result_display_section = gr.Markdown(f"### {texts['result_display']}")
                        with gr.Tabs(elem_id="markdown_tabs"):
                            tab_render = gr.TabItem(texts['markdown_render_preview'])
                            with tab_render:
                                md_view = gr.Markdown(value=texts['please_click_parse'], label=texts['markdown_render_preview'], max_height=600, latex_delimiters=[
                                    {"left": "$$", "right": "$$", "display": True},
                                    {"left": "$", "right": "$", "display": False},
                                ], show_copy_button=False, elem_id="markdown_output")
                            tab_raw = gr.TabItem(texts['markdown_raw_text'])
                            with tab_raw:
                                md_raw = gr.Textbox(value=texts['waiting_for_parsing'], label=texts['markdown_raw_text'], max_lines=100, lines=38, show_copy_button=True, elem_id="markdown_output", show_label=False)
                with gr.Row():
                    with gr.Column(scale=3):
                        pdf_download_button = gr.DownloadButton(texts['download_pdf_layout'], visible=True)
                    with gr.Column(scale=3):
                        md_download_button = gr.DownloadButton(texts['download_markdown'], visible=True)

        lang_switch.change(
            fn=switch_language,
            inputs=lang_switch,
            outputs=[
                i18n_texts, chat_input, title_html, upload_section, pdf_input,
                actions_section, parse_button, chat_button, clear_button,
                file_preview_section, prev_btn, next_btn, result_display_section,
                tab_render, tab_raw, md_view, md_raw,
                pdf_download_button, md_download_button, lang_switch
            ]
        )

        pdf_input.upload(
            fn=load_file,
            inputs=[pdf_input, i18n_texts],
            outputs=[pdf_view, page_info]
        )

        prev_btn.click(
            fn=lambda texts: turn_page("prev", texts),
            inputs=i18n_texts,
            outputs=[pdf_view, page_info],
            show_progress=False
        )
        next_btn.click(
            fn=lambda texts: turn_page("next", texts),
            inputs=i18n_texts,
            outputs=[pdf_view, page_info],
            show_progress=False
        )

        parse_button.click(
            fn=parse_and_update_view,
            inputs=[pdf_input, i18n_texts],
            outputs=[pdf_view, md_view, md_raw, page_info, pdf_download_button, md_download_button],
            show_progress=True,
            show_progress_on=[md_view, md_raw]
        )

        chat_button.click(
            fn=chat_with_image,
            inputs=[chat_input, pdf_input, i18n_texts],
            outputs=[md_view, md_raw, pdf_download_button, md_download_button],
            show_progress=True,
            show_progress_on=[md_view, md_raw]
        )

        clear_button.click(
            fn=clear_all,
            inputs=i18n_texts,
            outputs=[pdf_input, pdf_view, md_view, md_raw, page_info, pdf_download_button, md_download_button],
            show_progress=False
        )
    return demo

if __name__ == '__main__':
    model_manager.initialize_model()

    demo = create_gradio_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, debug=True)
