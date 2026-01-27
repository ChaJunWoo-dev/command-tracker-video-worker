from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AWSConfig:
    access_key_id: str
    secret_access_key: str
    region: str
    bucket_name: str

    @classmethod
    def from_env(cls) -> AWSConfig:
        return cls(
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ.get("AWS_REGION", "ap-northeast-2"),
            bucket_name=os.environ["S3_BUCKET_NAME"]
        )


@dataclass(frozen=True)
class RabbitMQEnvConfig:
    host: str
    heart_beat: int

    @classmethod
    def from_env(cls) -> RabbitMQEnvConfig:
        return cls(
            host=os.environ["MQ_HOST"],
            heart_beat=int(os.environ.get("MQ_HEART_BEAT", "60"))
        )


@dataclass(frozen=True)
class AppConfig:
    aws: AWSConfig
    rabbitmq: RabbitMQEnvConfig

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            aws=AWSConfig.from_env(),
            rabbitmq=RabbitMQEnvConfig.from_env()
        )


_config = None

def get_config() -> AppConfig:
    global _config
    if _config is None:
        try:
            _config = AppConfig.from_env()
        except KeyError as e:
            raise RuntimeError(f"환경 변수 누락: {e}")

    return _config
