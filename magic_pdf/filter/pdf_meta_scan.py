from collections import Counter

import fitz
from loguru import logger

from magic_pdf.config.drop_reason import DropReason
from magic_pdf.libs.commons import get_top_percent_list, mymax
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.pdf_check import detect_invalid_chars_by_pymupdf, detect_invalid_chars

scan_max_page = 50
junk_limit_min = 10


def calculate_max_image_area_per_page(result: list, page_width_pts, page_height_pts):
    max_image_area_per_page = [
        mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz])
        for page_img_sz in result
    ]
    page_area = int(page_width_pts) * int(page_height_pts)
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.6]
    return max_image_area_per_page


def process_image(page, junk_img_bojids=[]):
    page_result = []
    items = page.get_images()
    dedup = set()
    for img in items:

        img_bojid = img[
            0
        ]
        if img_bojid in junk_img_bojids:
            continue
        recs = page.get_image_rects(img, transform=True)
        if recs:
            rec = recs[0][0]
            x0, y0, x1, y1 = map(int, rec)
            width = x1 - x0
            height = y1 - y0
            if (
                x0,
                y0,
                x1,
                y1,
                img_bojid,
            ) in dedup:
                continue
            if not all(
                [width, height]
            ):
                continue
            dedup.add((x0, y0, x1, y1, img_bojid))
            page_result.append([x0, y0, x1, y1, img_bojid])
    return page_result


def get_image_info(doc: fitz.Document, page_width_pts, page_height_pts) -> list:

    img_bojid_counter = Counter(img[0] for page in doc for img in page.get_images())


    junk_limit = max(len(doc) * 0.5, junk_limit_min)

    junk_img_bojids = [
        img_bojid
        for img_bojid, count in img_bojid_counter.items()
        if count >= junk_limit
    ]






    imgs_len_list = [len(page.get_images()) for page in doc]

    special_limit_pages = 10


    result = []
    break_loop = False
    for i, page in enumerate(doc):
        if break_loop:
            break
        if i >= special_limit_pages:
            break
        page_result = process_image(
            page
        )
        result.append(page_result)
        for item in result:
            if not any(
                item
            ):
                if (
                    max(imgs_len_list) == min(imgs_len_list)
                    and max(imgs_len_list) >= junk_limit_min
                ):
                    junk_img_bojids = []
                else:
                    pass
                break_loop = True
                break
    if not break_loop:

        top_eighty_percent = get_top_percent_list(imgs_len_list, 0.8)

        if len(set(top_eighty_percent)) == 1 and max(imgs_len_list) >= junk_limit_min:

            # if max(imgs_len_list) == min(imgs_len_list) and max(imgs_len_list) >= junk_limit_min:


            max_image_area_per_page = calculate_max_image_area_per_page(
                result, page_width_pts, page_height_pts
            )
            if (
                len(max_image_area_per_page) < 0.8 * special_limit_pages
            ):
                junk_img_bojids = []
            else:
                pass
        else:
            junk_img_bojids = []


    result = []
    for i, page in enumerate(doc):
        if i >= scan_max_page:
            break
        page_result = process_image(page, junk_img_bojids)
        # logger.info(f"page {i} img_len: {len(page_result)}")
        result.append(page_result)

    return result, junk_img_bojids


def get_pdf_page_size_pts(doc: fitz.Document):
    page_cnt = len(doc)
    l: int = min(page_cnt, 50)

    page_width_list = []
    page_height_list = []
    for i in range(l):
        page = doc[i]
        page_rect = page.rect
        page_width_list.append(page_rect.width)
        page_height_list.append(page_rect.height)

    page_width_list.sort()
    page_height_list.sort()

    median_width = page_width_list[len(page_width_list) // 2]
    median_height = page_height_list[len(page_height_list) // 2]

    return median_width, median_height


def get_pdf_textlen_per_page(doc: fitz.Document):
    text_len_lst = []
    for page in doc:

        # text_block = page.get_text("blocks")

        # text_block = page.get_text("words")
        # text_block_len = sum([len(t[4]) for t in text_block])

        text_block = page.get_text('text')
        text_block_len = len(text_block)
        # logger.info(f"page {page.number} text_block_len: {text_block_len}")
        text_len_lst.append(text_block_len)

    return text_len_lst


def get_pdf_text_layout_per_page(doc: fitz.Document):
    text_layout_list = []

    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break

        vertical_count = 0
        horizontal_count = 0
        text_dict = page.get_text('dict')
        if 'blocks' in text_dict:
            for block in text_dict['blocks']:
                if 'lines' in block:
                    for line in block['lines']:

                        x0, y0, x1, y1 = line['bbox']

                        width = x1 - x0
                        height = y1 - y0

                        area = width * height
                        font_sizes = []
                        for span in line['spans']:
                            if 'size' in span:
                                font_sizes.append(span['size'])
                        if len(font_sizes) > 0:
                            average_font_size = sum(font_sizes) / len(font_sizes)
                        else:
                            average_font_size = (
                                10
                            )
                        if (
                            area <= average_font_size**2
                        ):
                            continue
                        else:
                            if 'wmode' in line:
                                if line['wmode'] == 1:
                                    vertical_count += 1
                                elif line['wmode'] == 0:
                                    horizontal_count += 1


                        #         dir_value = line['dir']
                        #         cosine, sine = dir_value

                        #         angle = math.degrees(math.acos(cosine))
                        #

                        #         if abs(angle - 0) < 0.01 or abs(angle - 180) < 0.01:
                        #             # line_text = ' '.join(span['text'] for span in line['spans'])
                        #             # print('This line is horizontal:', line_text)
                        #             horizontal_count += 1

                        #         elif abs(angle - 90) < 0.01 or abs(angle - 270) < 0.01:
                        #             # line_text = ' '.join(span['text'] for span in line['spans'])
                        #             # print('This line is vertical:', line_text)
                        #             vertical_count += 1
        # print(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")

        if vertical_count == 0 and horizontal_count == 0:
            text_layout_list.append('unknow')
            continue
        else:
            if vertical_count > horizontal_count:
                text_layout_list.append('vertical')
            else:
                text_layout_list.append('horizontal')
        # logger.info(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")
    return text_layout_list


class PageSvgsTooManyError(Exception):
    def __init__(self, message='Page SVGs are too many'):
        self.message = message
        super().__init__(self.message)


def get_svgs_per_page(doc: fitz.Document):
    svgs_len_list = []
    for page_id, page in enumerate(doc):
        # svgs = page.get_drawings()
        svgs = page.get_cdrawings()
        len_svgs = len(svgs)
        if len_svgs >= 3000:
            raise PageSvgsTooManyError()
        else:
            svgs_len_list.append(len_svgs)
        # logger.info(f"page_id: {page_id}, svgs_len: {len(svgs)}")
    return svgs_len_list


def get_imgs_per_page(doc: fitz.Document):
    imgs_len_list = []
    for page_id, page in enumerate(doc):
        imgs = page.get_images()
        imgs_len_list.append(len(imgs))
        # logger.info(f"page_id: {page}, imgs_len: {len(imgs)}")

    return imgs_len_list


def get_language(doc: fitz.Document):
    language_lst = []
    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break

        text_block = page.get_text('text')
        page_language = detect_lang(text_block)
        language_lst.append(page_language)

        # logger.info(f"page_id: {page_id}, page_language: {page_language}")


    count_dict = Counter(language_lst)

    language = max(count_dict, key=count_dict.get)
    return language


def check_invalid_chars(pdf_bytes):
    # return detect_invalid_chars_by_pymupdf(pdf_bytes)
    return detect_invalid_chars(pdf_bytes)


def pdf_meta_scan(pdf_bytes: bytes):
    doc = fitz.open('pdf', pdf_bytes)
    is_needs_password = doc.needs_pass
    is_encrypted = doc.is_encrypted
    total_page = len(doc)
    if total_page == 0:
        logger.warning(f'drop this pdf, drop_reason: {DropReason.EMPTY_PDF}')
        result = {'_need_drop': True, '_drop_reason': DropReason.EMPTY_PDF}
        return result
    else:
        page_width_pts, page_height_pts = get_pdf_page_size_pts(doc)
        # logger.info(f"page_width_pts: {page_width_pts}, page_height_pts: {page_height_pts}")

        # svgs_per_page = get_svgs_per_page(doc)
        # logger.info(f"svgs_per_page: {svgs_per_page}")
        imgs_per_page = get_imgs_per_page(doc)
        # logger.info(f"imgs_per_page: {imgs_per_page}")

        image_info_per_page, junk_img_bojids = get_image_info(
            doc, page_width_pts, page_height_pts
        )
        # logger.info(f"image_info_per_page: {image_info_per_page}, junk_img_bojids: {junk_img_bojids}")
        text_len_per_page = get_pdf_textlen_per_page(doc)
        # logger.info(f"text_len_per_page: {text_len_per_page}")
        text_layout_per_page = get_pdf_text_layout_per_page(doc)
        # logger.info(f"text_layout_per_page: {text_layout_per_page}")
        text_language = get_language(doc)
        # logger.info(f"text_language: {text_language}")
        invalid_chars = check_invalid_chars(pdf_bytes)
        # logger.info(f"invalid_chars: {invalid_chars}")


        res = {
            'is_needs_password': is_needs_password,
            'is_encrypted': is_encrypted,
            'total_page': total_page,
            'page_width_pts': int(page_width_pts),
            'page_height_pts': int(page_height_pts),
            'image_info_per_page': image_info_per_page,
            'text_len_per_page': text_len_per_page,
            'text_layout_per_page': text_layout_per_page,
            'text_language': text_language,
            # "svgs_per_page": svgs_per_page,
            'imgs_per_page': imgs_per_page,
            'junk_img_bojids': junk_img_bojids,
            'invalid_chars': invalid_chars,
            'metadata': doc.metadata,
        }
        # logger.info(json.dumps(res, ensure_ascii=False))
        return res


if __name__ == '__main__':
    pass


    # "D:\project/20231108code-clean\pdf_cost_time\scihub\scihub_86800000\libgen.scimag86880000-86880999.zip_10.1021/acsami.1c03109.s002.pdf"
    # "D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_18600000/libgen.scimag18645000-18645999.zip_10.1021/om3006239.pdf"
    # file_content = read_file("D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_31000000/libgen.scimag31098000-31098999.zip_10.1109/isit.2006.261791.pdf","")  # noqa: E501

    # doc = fitz.open("pdf", file_content)
    # text_layout_lst = get_pdf_text_layout_per_page(doc)
    # print(text_layout_lst)
