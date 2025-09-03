class DropReason:
    TEXT_BLOCK_HOR_OVERLAP = 'text_block_horizontal_overlap'
    USEFUL_BLOCK_HOR_OVERLAP = (
        'useful_block_horizontal_overlap'
    )
    COMPLICATED_LAYOUT = 'complicated_layout'
    TOO_MANY_LAYOUT_COLUMNS = 'too_many_layout_columns'
    COLOR_BACKGROUND_TEXT_BOX = 'color_background_text_box'
    HIGH_COMPUTATIONAL_lOAD_BY_IMGS = (
        'high_computational_load_by_imgs'
    )
    HIGH_COMPUTATIONAL_lOAD_BY_SVGS = (
        'high_computational_load_by_svgs'
    )
    HIGH_COMPUTATIONAL_lOAD_BY_TOTAL_PAGES = 'high_computational_load_by_total_pages'
    MISS_DOC_LAYOUT_RESULT = 'missing doc_layout_result'
    Exception = '_exception'
    ENCRYPTED = 'encrypted'
    EMPTY_PDF = 'total_page=0'
    NOT_IS_TEXT_PDF = 'not_is_text_pdf'
    DENSE_SINGLE_LINE_BLOCK = 'dense_single_line_block'
    TITLE_DETECTION_FAILED = 'title_detection_failed'
    TITLE_LEVEL_FAILED = (
        'title_level_failed'
    )
    PARA_SPLIT_FAILED = 'para_split_failed'
    PARA_MERGE_FAILED = 'para_merge_failed'
    NOT_ALLOW_LANGUAGE = 'not_allow_language'
    SPECIAL_PDF = 'special_pdf'
    PSEUDO_SINGLE_COLUMN = 'pseudo_single_column'
    CAN_NOT_DETECT_PAGE_LAYOUT = 'can_not_detect_page_layout'
    NEGATIVE_BBOX_AREA = 'negative_bbox_area'
    OVERLAP_BLOCKS_CAN_NOT_SEPARATION = (
        'overlap_blocks_can_t_separation'
    )
