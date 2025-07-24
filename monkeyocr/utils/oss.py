import oss2
from monkeyocr.conf.settings import app_settings

def _get_oss_bucket(bucket_name: str):
    auth = oss2.AuthV4(
        app_settings.aliyun_access_key, app_settings.aliyun_access_secret
    )
    return oss2.Bucket(
        auth,
        endpoint=app_settings.oss_endpoint,
        bucket_name=bucket_name,
        region=app_settings.oss_region,
    )

def download_file_from_oss(bucket_name: str, file_key: str, local_path: str):
    bucket = _get_oss_bucket(bucket_name)
    bucket.get_object_to_file(file_key, local_path)

def upload_file_to_oss(bucket_name: str, file_key: str, local_path: str):
    bucket = _get_oss_bucket(bucket_name)
    bucket.put_object_from_file(file_key, local_path)

def test_oss_connection(bucket_name: str, file_key: str, endpoint: str | None = None, region: str | None = None):
    auth = oss2.AuthV4(
        app_settings.aliyun_access_key, app_settings.aliyun_access_secret
    )
    endpoint = endpoint or app_settings.oss_endpoint
    region = region or app_settings.oss_region
    bucket = oss2.Bucket(
        auth,
        endpoint=endpoint,
        bucket_name=bucket_name,
        region=region,
    )
    object = bucket.get_object(file_key)
    return object


    