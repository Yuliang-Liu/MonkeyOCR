"""Custom fields for span dimension."""
# Whether span is merged across pages
CROSS_PAGE = 'cross_page'

"""
Custom fields for block dimension
"""
# Whether lines in block are deleted
LINES_DELETED = 'lines_deleted'

# table recognition max time default value
TABLE_MAX_TIME_VALUE = 400

# pp_table_result_max_length
TABLE_MAX_LEN = 480

# table master structure dict
TABLE_MASTER_DICT = 'table_master_structure_dict.txt'

# table master dir
TABLE_MASTER_DIR = 'table_structure_tablemaster_infer/'

# pp detect model dir
DETECT_MODEL_DIR = 'ch_PP-OCRv4_det_infer'

# pp rec model dir
REC_MODEL_DIR = 'ch_PP-OCRv4_rec_infer'

# pp rec char dict path
REC_CHAR_DICT = 'ppocr_keys_v1.txt'

# pp rec copy rec directory
PP_REC_DIRECTORY = '.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer'

# pp rec copy det directory
PP_DET_DIRECTORY = '.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer'


class MODEL_NAME:
    DocLayout_YOLO = 'doclayout_yolo'
    PaddleXLayoutModel = 'PP-DocLayout_plus-L'
    PP_DoclayoutV2 = 'PP-DocLayoutV2'

PARSE_TYPE_TXT = 'txt'
PARSE_TYPE_OCR = 'ocr'

