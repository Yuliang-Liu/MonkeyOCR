import os

from loguru import logger

from magic_pdf.libs.commons import parse_bucket_key
import yaml


CONFIG_FILE_NAME = os.getenv('MONKEYOCR_MODEL_CONFIGS', 'model_configs.yaml')

def get_base_directory(path):
    return os.path.dirname(os.path.dirname(os.path.dirname(path)))

def get_current_file_parent_parent_dir():
    current_file = os.path.abspath(__file__)
    return get_base_directory(current_file)


def read_config():
    config_file = os.path.join(get_current_file_parent_parent_dir(), 'model_configs.yaml')

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} not found')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_s3_config(bucket_name: str):
    config = read_config()

    bucket_info = config.get('bucket_info')
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info['[default]']
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f'ak, sk or endpoint not found in {CONFIG_FILE_NAME}')

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def get_local_models_dir():
    config = read_config()
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use '/tmp/models' as default")
        return '/tmp/models'
    else:
        return models_dir


def get_local_layoutreader_model_dir():
    config = read_config()
    layoutreader_model_dir = config.get('layoutreader-model-dir')
    if layoutreader_model_dir is None or not os.path.exists(layoutreader_model_dir):
        home_dir = os.path.expanduser('~')
        layoutreader_at_modelscope_dir_path = os.path.join(home_dir, '.cache/modelscope/hub/ppaanngggg/layoutreader')
        logger.warning(f"'layoutreader-model-dir' not exists, use {layoutreader_at_modelscope_dir_path} as default")
        return layoutreader_at_modelscope_dir_path
    else:
        return layoutreader_model_dir


def get_device():
    config = read_config()
    device = config.get('device')
    if device is None:
        logger.warning(f"'device' not found in {CONFIG_FILE_NAME}, use 'cpu' as default")
        return 'cpu'
    else:
        return device


if __name__ == '__main__':
    ak, sk, endpoint = get_s3_config('llm-raw')
