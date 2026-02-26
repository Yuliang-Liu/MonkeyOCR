import torch
from loguru import logger

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.model_list import AtomicModel


def doclayout_yolo_model_init(weight, device='cpu'):
    from magic_pdf.model.sub_modules.layout.doclayout_yolo.DocLayoutYOLO import \
        DocLayoutYOLOModel
    if str(device).startswith("npu"):
        device = torch.device(device)
    model = DocLayoutYOLOModel(weight, device)
    return model


def paddex_layout_model_init(device: str, model_dir: str = None, model_name: str = MODEL_NAME.PaddleXLayoutModel):
    from magic_pdf.model.sub_modules.layout.paddlex_layout.PaddleXLayoutModel import \
        PaddleXLayoutModelWrapper
    model = PaddleXLayoutModelWrapper(model_name=model_name, device=device, model_dir=model_dir)
    return model


def ocr_model_init(det_db_box_thresh=0.3,
                   lang=None,
                   use_dilation=True,
                   det_db_unclip_ratio=1.8,
                   ):
    from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
    if lang is not None and lang != '':
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    else:
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    return model


def mfd_model_init(weight, device='cpu'):
    from magic_pdf.model.sub_modules.mfd.yolo_v8 import YOLOv8MFDModel
    if str(device).startswith('npu'):
        device = torch.device(device)
    mfd_model = YOLOv8MFDModel(weight, device)
    return mfd_model


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):

        layout_model_name = kwargs.get('layout_model_name', None)

        if atom_model_name in [AtomicModel.Layout]:
            key = (atom_model_name, layout_model_name)
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]


def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        if kwargs.get('layout_model_name') == MODEL_NAME.DocLayout_YOLO:
            atom_model = doclayout_yolo_model_init(
                kwargs.get('doclayout_yolo_weights'),
                kwargs.get('device')
            )
        elif kwargs.get('layout_model_name') in [MODEL_NAME.PaddleXLayoutModel, MODEL_NAME.PP_DoclayoutV2]:
            atom_model = paddex_layout_model_init(
                model_name=kwargs.get('layout_model_name'),
                model_dir=kwargs.get('paddlexlayout_model_dir'),
                device=kwargs.get('device')
            )
        else:
            logger.error('layout model name not allowed')
            exit(1)
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
    else:
        logger.error('model name not allowed')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model
