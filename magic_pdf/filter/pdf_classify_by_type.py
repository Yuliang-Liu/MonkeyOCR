"""
Classify PDF as text version or scanned version based on results from meta_scan.
Definition criteria:
1. What PDFs are text PDFs - meeting any of the following conditions:
  1. Randomly sample N pages, if any page has more than 100 text characters
  2. As long as there exists a page with 0 images
2. What are scanned PDFs - meeting any of the following conditions:
  1. ~~80% of pages have the same maximum image size and area exceeds 0.6 of page area~~
  2. Most pages have equal text length.
"""
import json
import sys
from collections import Counter

import click
import numpy as np
from loguru import logger

from magic_pdf.libs.commons import mymax, get_top_percent_list
from magic_pdf.filter.pdf_meta_scan import scan_max_page, junk_limit_min

TEXT_LEN_THRESHOLD = 100
AVG_TEXT_LEN_THRESHOLD = 100
TEXT_LEN_SAMPLE_RATIO = 0.1  # Sample 0.1 of pages for text length statistics

# A solution for merging images, combining certain special scanned version split images into one complete image
def merge_images(image_list, page_width, page_height, max_offset=5, max_gap=2):
    # First remove all overlapping bbox image data through set
    image_list_result = []
    for page_images in image_list:
        page_result = []
        dedup = set()
        for img in page_images:
            x0, y0, x1, y1, img_bojid = img
            if (x0, y0, x1, y1) in dedup:  # Some duplicate bboxes may appear, no need to repeat, need to remove
                continue
            else:
                dedup.add((x0, y0, x1, y1))
                page_result.append([x0, y0, x1, y1, img_bojid])
        image_list_result.append(page_result)

    # Next, merge images on the same page that can be stitched together
    merged_images = []
    for page_images in image_list_result:
        if not page_images:
            continue

        # First sort images on the same page from top to bottom, left to right
        page_images.sort(key=lambda img: (img[1], img[0]))

        merged = [page_images[0]]

        for img in page_images[1:]:
            x0, y0, x1, y1, imgid = img

            last_img = merged[-1]
            last_x0, last_y0, last_x1, last_y1, last_imgid = last_img

            # A single image width or height covering more than 90% of page width/height is a prerequisite for stitching
            full_width = abs(x1 - x0) >= page_width * 0.9
            full_height = abs(y1 - y0) >= page_height * 0.9

            # If width meets standard, check if can stitch vertically
            if full_width:
                # Vertical stitching needs two prerequisites: left and right boundaries can't offset more than max_offset, first image's bottom boundary and second image's top boundary can't offset more than max_gap
                close1 = (last_x0 - max_offset) <= x0 <= (last_x0 + max_offset) and (last_x1 - max_offset) <= x1 <= (
                            last_x1 + max_offset) and (last_y1 - max_gap) <= y0 <= (last_y1 + max_gap)

            # If height meets standard, check if can stitch horizontally
            if full_height:
                # Horizontal stitching needs two prerequisites: top and bottom boundaries can't offset more than max_offset, first image's right boundary and second image's left boundary can't offset more than max_gap
                close2 = (last_y0 - max_offset) <= y0 <= (last_y0 + max_offset) and (last_y1 - max_offset) <= y1 <= (
                            last_y1 + max_offset) and (last_x1 - max_gap) <= x0 <= (last_x1 + max_gap)

            # Check if the image can be merged with the last image
            if (full_width and close1) or (full_height and close2):
                # Merge the image with the last image
                merged[-1] = [min(x0, last_x0), min(y0, last_y0),
                              max(x1, last_x1), max(y1, last_y1), imgid]
            else:
                # Add the image as a new image
                merged.append(img)

        merged_images.append(merged)

    return merged_images


def classify_by_area(total_page: int, page_width, page_height, img_sz_list, text_len_list: list):
    """
    Returns False if 80% of pages have the same maximum image size and area exceeds 0.6 of page area, otherwise returns True
    """
    # # Only one page without images means it's a text PDF. But also needs to meet one condition, that is, there can't be any text on the page. Some scanned PDFs have blank pages with neither images nor text.
    # if any([len(img_sz) == 0 for img_sz in img_sz_list]):  # Contains pages without images
    #     # Now find the index of these pages
    #     empty_page_index = [i for i, img_sz in enumerate(img_sz_list) if len(img_sz) == 0]
    #     # Then check if there is any text on these pages
    #     text_len_at_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in empty_page_index and text_len > 0]
    #     if len(text_len_at_page_idx) > TEXT_LEN_THRESHOLD:  # No images, but has text, indicating it might be a text version, if no text then can't be determined, left for next step, now requires the text on this page to exceed a certain threshold
    #         return True

    # Remove images that appear more than 10 times by objid, these are hidden transparent layers with same id
    # First count occurrences of each id - optimized to avoid nested list comprehensions
    objid_cnt = Counter()
    for page_img_sz in img_sz_list:
        for _, _, _, _, objid in page_img_sz:
            objid_cnt[objid] += 1
    
    # Then remove those appearing more than 10 times
    if total_page >= scan_max_page:  # New meta_scan only scans first scan_max_page pages, when page count > scan_max_page, treat total_page as scan_max_page
        total_page = scan_max_page

    repeat_threshold = 2  # Set bad_image threshold to 2
    # repeat_threshold = min(2, total_page)  # When total_page is 1, repeat_threshold is 1, will cause misjudgment making all img become bad_img
    bad_image_objid = {objid for objid, cnt in objid_cnt.items() if cnt >= repeat_threshold}
    # bad_image_page_idx = [i for i, page_img_sz in enumerate(img_sz_list) if any([objid in bad_image_objid for _, _, _, _, objid in page_img_sz])]
    # text_len_at_bad_image_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in bad_image_page_idx and text_len > 0]

    # Special case: a text PDF covers each page with a huge transparent image, huge means image covers more than 90% of page area
    # fake_image_ids = [objid for objid in bad_image_objid if
    #                   any([abs((x1 - x0) * (y1 - y0) / page_width * page_height) > 0.9 for images in img_sz_list for
    #                        x0, y0, x1, y1, _ in images])]  # Original code, any inside always true, reason？？？
    # fake_image_ids = [objid for objid in bad_image_objid for images in img_sz_list for x0, y0, x1, y1, img_id in images
    #                   if img_id == objid and abs((x1 - x0) * (y1 - y0)) / (page_width * page_height) > 0.9]

    # if len(fake_image_ids) > 0 and any([l > TEXT_LEN_THRESHOLD for l in text_len_at_bad_image_page_idx]):  # These transparent images' pages have text greater than threshold
    #     return True

    # Filter out repeatedly appearing images - optimized to avoid rebuilding lists unnecessarily
    img_sz_list = [[img_sz for img_sz in page_img_sz if img_sz[-1] not in bad_image_objid] for page_img_sz in
                   img_sz_list]

    # Some scanned versions split one page image into many, need to stitch images first then calculate
    img_sz_list = merge_images(img_sz_list, page_width, page_height)

    # Calculate maximum image area per page, then calculate ratio of this area to page area
    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in
                               img_sz_list]
    page_area = page_width * page_height
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.5]

    if len(max_image_area_per_page) >= 0.5 * total_page:  # Threshold changed from 0.8 to 0.5, adapt to cases like 2 out of 3 pages and 1 out of 2 pages
        # Prerequisite for this condition is removing repeatedly appearing images. These are hidden transparent layers with same id
        return False
    else:
        return True


def classify_by_text_len(text_len_list: list, total_page: int):
    """
    Randomly sample 10% of pages, if less than 5 pages, take all pages.
    Check text length on pages, if any page has text length > TEXT_LEN_THRESHOLD, then it's text PDF
    """
    select_page_cnt = int(total_page * TEXT_LEN_SAMPLE_RATIO)  # Select 10% of pages
    if select_page_cnt < 5:
        select_page_cnt = total_page

    # # Excluding first and last 10 pages
    # if total_page > 20:  # If total pages > 20
    #     page_range = list(range(10, total_page - 10))  # From 11th page to the last 11th page
    # else:
    #     page_range = list(range(total_page))  # Otherwise select all pages
    # page_num = np.random.choice(page_range, min(select_page_cnt, len(page_range)), replace=False)
    # Excluding first and last 10 pages is awkward for PDFs with only 21, 22 pages, if the selected middle 1-2 pages happen to have no text, easy to misjudge, with avg_words rule, this rule can be ignored
    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    text_len_lst = [text_len_list[i] for i in page_num]
    is_text_pdf = any([text_len > TEXT_LEN_THRESHOLD for text_len in text_len_lst])
    return is_text_pdf


def classify_by_avg_words(text_len_list: list):
    """
    Supplementary rule: if average words per page < AVG_TEXT_LEN_THRESHOLD, not text PDF
    Mainly for various image collections
    """
    sum_words = sum(text_len_list)
    count_of_numbers = len(text_len_list)
    if count_of_numbers == 0:
        is_text_pdf = False
    else:
        avg_words = round(sum_words / count_of_numbers)
        if avg_words > AVG_TEXT_LEN_THRESHOLD:
            is_text_pdf = True
        else:
            is_text_pdf = False

    return is_text_pdf


def classify_by_img_num(img_sz_list: list, img_num_list: list):
    """
    Supplementary rule: there's a type of scanned PDF that puts all scanned pages on each page, which gets deduplicated during metascan,
    characteristic of this PDF's metascan result is all empty elements in img_sz_list, each page in img_num_list has large and same count
    """
    # Calculate number of non-empty elements in img_sz_list
    count_img_sz_list_not_none = sum(1 for item in img_sz_list if item)
    # Get top 80% elements
    top_eighty_percent = get_top_percent_list(img_num_list, 0.8)
    # Non-empty elements in img_sz_list <= 1, top 80% elements are all equal, and max value >= junk_limit_min
    if count_img_sz_list_not_none <= 1 and len(set(top_eighty_percent)) == 1 and max(img_num_list) >= junk_limit_min:
        return False  # If meets this condition, definitely not text PDF
    else:
        return True  # If doesn't meet these three conditions, might be text PDF, judge by other rules


def classify_by_text_layout(text_layout_per_page: list):
    """
    Judge if text layout is mainly vertical.

    Args:
        text_layout_per_page (list): Text layout list, each element represents text layout of one page,
                                     'vertical' means vertical layout, 'horizontal' means horizontal layout.

    Returns:
        bool: If text layout is mainly vertical, return False; otherwise return True.
    """
    # Count vertical layouts in text_layout_per_page
    count_vertical = sum(1 for item in text_layout_per_page if item == 'vertical')
    # Count horizontal layouts in text_layout_per_page
    count_horizontal = sum(1 for item in text_layout_per_page if item == 'horizontal')
    # Calculate ratio of vertical layouts in text_layout_per_page
    known_layout_cnt = count_vertical + count_horizontal
    if known_layout_cnt != 0:
        ratio = count_vertical / known_layout_cnt
        if ratio >= 0.5:  # Threshold set to 0.5, adapt to cases like 2 out of 3 pages and 1 out of 2 pages
            return False  # Text layout is mainly vertical, consider not text PDF
        else:
            return True  # Text layout is mainly horizontal, consider text PDF
    else:
        return False  # Text layout unknown, default consider not text PDF


def classify_by_img_narrow_strips(page_width, page_height, img_sz_list):
    """
    Judge if a page consists of narrow strips, two conditions:
    1. Image width or height reaches 90% of page width or height, and long side needs to be multiple times longer than short side
    2. 80% or more of all images on the entire page meet condition 1

    Args:
        page_width (float): Page width
        page_height (float): Page height
        img_sz_list (list): Image size list, each element is a tuple representing image rectangle area and size, format (x0, y0, x1, y1, size), where (x0, y0) is top-left corner coordinate, (x1, y1) is bottom-right corner coordinate, size is image size

    Returns:
        bool: If ratio of pages meeting conditions < 0.5, return True, otherwise return False
    """

    def is_narrow_strip(img):
        x0, y0, x1, y1, _ = img
        width, height = x1 - x0, y1 - y0
        return any([
            # Image width >= 90% of page width, and width >= 4 times height
            width >= page_width * 0.9 and width >= height * 4,
            # Image height >= 90% of page height, and height >= 4 times width
            height >= page_height * 0.9 and height >= width * 4,
        ])

    # Initialize count of pages meeting conditions
    narrow_strip_pages_count = 0

    # Traverse all pages
    for page_img_list in img_sz_list:
        # Ignore empty pages
        if not page_img_list:
            continue

        # Calculate total number of images on page
        total_images = len(page_img_list)

        # Calculate number of narrow strip images on page
        narrow_strip_images_count = 0
        for img in page_img_list:
            if is_narrow_strip(img):
                narrow_strip_images_count += 1
        # If narrow strip image count < 5, skip
        if narrow_strip_images_count < 5:
            continue
        else:
            # If narrow strip image ratio >= 0.8, increase count of pages meeting conditions
            if narrow_strip_images_count / total_images >= 0.8:
                narrow_strip_pages_count += 1

    # Calculate ratio of pages meeting conditions
    narrow_strip_pages_ratio = narrow_strip_pages_count / len(img_sz_list)

    return narrow_strip_pages_ratio < 0.5


def classify(total_page: int, page_width, page_height, img_sz_list: list, text_len_list: list, img_num_list: list,
             text_layout_list: list, invalid_chars: bool):
    """
    Image and page length units here are pts
    """
    results = {
        'by_image_area': classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list),
        'by_text_len': classify_by_text_len(text_len_list, total_page),
        'by_avg_words': classify_by_avg_words(text_len_list),
        'by_img_num': classify_by_img_num(img_sz_list, img_num_list),
        'by_text_layout': classify_by_text_layout(text_layout_list),
        'by_img_narrow_strips': classify_by_img_narrow_strips(page_width, page_height, img_sz_list),
        'by_invalid_chars': invalid_chars,
    }

    if all(results.values()):
        return True, results
    elif not any(results.values()):
        return False, results
    else:
        logger.warning(
            f"pdf is not classified by area and text_len, by_image_area: {results['by_image_area']},"
            f" by_text: {results['by_text_len']}, by_avg_words: {results['by_avg_words']}, by_img_num: {results['by_img_num']},"
            f" by_text_layout: {results['by_text_layout']}, by_img_narrow_strips: {results['by_img_narrow_strips']},"
            f" by_invalid_chars: {results['by_invalid_chars']}",
            file=sys.stderr)  # Use this situation to quickly find which PDFs are special, and fix classification algorithm accordingly
        return False, results


@click.command()
@click.option("--json-file", type=str, help="PDF information")
def main(json_file):
    if json_file is None:
        print("json_file is None", file=sys.stderr)
        exit(0)
    try:
        with open(json_file, "r") as f:
            for l in f:
                if l.strip() == "":
                    continue
                o = json.loads(l)
                total_page = o["total_page"]
                page_width = o["page_width_pts"]
                page_height = o["page_height_pts"]
                img_sz_list = o["image_info_per_page"]
                text_len_list = o['text_len_per_page']
                text_layout_list = o['text_layout_per_page']
                pdf_path = o['pdf_path']
                is_encrypted = o['is_encrypted']
                is_needs_password = o['is_needs_password']
                if is_encrypted or total_page == 0 or is_needs_password:  # Encrypted, password-required, no pages - don't process
                    continue
                tag = classify(total_page, page_width, page_height, img_sz_list, text_len_list, text_layout_list)
                o['is_text_pdf'] = tag
                print(json.dumps(o, ensure_ascii=False))
    except Exception as e:
        print("ERROR: ", e, file=sys.stderr)