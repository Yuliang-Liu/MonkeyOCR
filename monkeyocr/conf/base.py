import os
import sys
import enum
import logging
from typing import Type, Tuple, Self

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

__all__ = ["TEST", "LogLevel", "BaseAppSettings"]

TEST = bool(
    any(filter(lambda v: "pytest" in v, sys.argv)) or os.getenv("PYTEST_XDIST_WORKER")
)

# 只使用 settings.yaml
YAML_FILE = ["settings.yaml"] if not TEST else []


class Environments(str, enum.Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_development(self) -> bool:
        return self == Environments.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self == Environments.PRODUCTION


class LogLevel(enum.Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class BaseAppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=YAML_FILE,
        extra="ignore",
        env_nested_delimiter="__",
    )
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # https://docs.pydantic.dev/latest/concepts/pydantic_settings/#customise-settings-sources
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    environment: Environments = Environments.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    log_format: str = (
        "%(asctime)s[%(levelname)-8s][%(process)d]%(threadName)s-%(thread)d | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    @model_validator(mode="after")
    def setup(self) -> Self:
        logging.basicConfig(level=self.log_level.value, format=self.log_format)
        return self
