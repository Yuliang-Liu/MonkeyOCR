import logging
import os
from functools import cached_property
from typing_extensions import Annotated, Self

import oss2
import sentry_sdk
from pydantic import BaseModel, Field, model_validator

from .base import BaseAppSettings

__all__ = [
    "app_settings",
]

logger = logging.getLogger(__name__)


class SentrySettings(BaseModel):
    dsn: str = ""
    enable_tracing: bool = False
    sample_rate: float = 1


class CelerySettings(BaseModel):
    """
    https://docs.celeryproject.org/en/stable/userguide/configuration.html
    """

    broker_url: str = "redis://localhost:6379/12"
    broker_connection_retry_on_startup: bool = False
    
    # 默认队列改为 unobpi
    task_default_queue: str = "unobpi"
    
    task_soft_time_limit: int = 20 * 60
    task_time_limit: int = 25 * 60
    task_acks_late: bool = True
    result_backend: str = "redis://localhost:6379/12"
    result_extended: bool = True
    result_expires: int = 1500
    worker_send_task_events: bool = True
    worker_prefetch_multiplier: int = 2
    worker_max_tasks_per_child: int = 100
    worker_deduplicate_successful_tasks: bool = True
    worker_log_format: str = (
        "%(asctime)s[%(levelname)-8s][%(process)d]%(threadName)s-%(thread)d | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )


class AppSettings(BaseAppSettings):
    environment: str = "development"
    sentry: SentrySettings = SentrySettings()
    celery: CelerySettings = CelerySettings()

    aliyun_account_id: str = "1992993760666663"
    aliyun_access_key: str = "invalid_key"
    aliyun_access_secret: str = "invalid_secret"

    oss_endpoint: str = "oss-cn-shanghai.aliyuncs.com"
    oss_region: str = "cn-shanghai"
    translated_pdf_bucket_name: str = "ivyscholardata-testing"
    scholardata_bucket_name: str = "ivyscholardata-testing" # 请根据实际情况替换
    scholardata_header_prefix: str = "monkey_ocr/file_header"

    @cached_property
    def translated_pdf_bucket(self):
        auth = oss2.AuthV4(self.aliyun_access_key, self.aliyun_access_secret)
        return oss2.Bucket(
            auth,
            endpoint=self.oss_endpoint,
            bucket_name=self.translated_pdf_bucket_name,
            region=self.oss_region,
        )

    @cached_property
    def scholardata_bucket(self):
        auth = oss2.AuthV4(self.aliyun_access_key, self.aliyun_access_secret)
        return oss2.Bucket(
            auth,
            endpoint=self.oss_endpoint,
            bucket_name=self.scholardata_bucket_name,
            region=self.oss_region,
        )

    @model_validator(mode="after")
    def setup_sentry(self) -> Self:
        logger.debug("Setting up sentry on environment: %s", self.environment)
        sentry_sdk.init(
            dsn=self.sentry.dsn,
            debug=False,
            environment=self.environment,
            enable_tracing=self.sentry.enable_tracing,
            sample_rate=self.sentry.sample_rate,
            send_default_pii=True,
        )
        return self


app_settings = AppSettings()
