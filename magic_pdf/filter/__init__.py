from magic_pdf.config.drop_reason import DropReason
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.filter.pdf_classify_by_type import classify as do_classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan


def classify(pdf_bytes: bytes) -> SupportedPdfParseMethod:
    """Determine whether it's text PDF or OCR PDF based on PDF metadata."""
    pdf_meta = pdf_meta_scan(pdf_bytes)
    if pdf_meta.get('_need_drop', False):  # If returned flag indicates need to drop, throw exception
        raise Exception(f"pdf meta_scan need_drop,reason is {pdf_meta['_drop_reason']}")
    else:
        is_encrypted = pdf_meta['is_encrypted']
        is_needs_password = pdf_meta['is_needs_password']
        if is_encrypted or is_needs_password:  # Encrypted, password-required, no pages - don't process
            raise Exception(f'pdf meta_scan need_drop,reason is {DropReason.ENCRYPTED}')
        else:
            is_text_pdf, results = do_classify(
                pdf_meta['total_page'],
                pdf_meta['page_width_pts'],
                pdf_meta['page_height_pts'],
                pdf_meta['image_info_per_page'],
                pdf_meta['text_len_per_page'],
                pdf_meta['imgs_per_page'],
                pdf_meta['text_layout_per_page'],
                pdf_meta['invalid_chars'],
            )
            if is_text_pdf:
                return SupportedPdfParseMethod.TXT
            else:
                return SupportedPdfParseMethod.OCR
