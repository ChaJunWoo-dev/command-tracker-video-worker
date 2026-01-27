import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from aio_pika import IncomingMessage

from infra.s3_client import S3Client
from infra.ffmpeg_client import FFmpegClient
from infra.rabbitmq_client import RabbitMQClient
from infra.temp_storage import TempStorage
from ai.detector import PersonDetector
from ai.pose_estimator import PoseEstimator
from worker.analysis_worker import run_analysis
from config.settings import get_config
from config.constants import RabbitMQConfig, S3Config, ErrorCode, Messages
from config.exceptions import AppError
from pathlib import Path

config = get_config()


async def on_message(
    msg: IncomingMessage,
    rabbitmq: RabbitMQClient,
    pool: ThreadPoolExecutor,
    detector: PersonDetector,
    pose_estimator: PoseEstimator,
):
    async with msg.process():
        data = json.loads(msg.body.decode())

        file_name = data["filename"]
        file_path = Path(file_name)
        job_id = file_path.stem
        start = data["trimStart"]
        end = data["trimEnd"]

        storage = TempStorage()
        ffmpeg = FFmpegClient()

        async with S3Client() as s3, storage.job_dir(job_id) as job:
            ext = file_path.suffix[1:]
            input_path = job / f"raw.{ext}"
            output_path = job / f"cut.{ext}"
            final_path = job / f"final.{ext}"
            original_s3_key = f"{S3Config.ORIGINAL_PREFIX}/{file_name}"
            processed_s3_key = f"{S3Config.PROCESSED_PREFIX}/{file_name}"

            try:
                try:
                    await s3.download_file(original_s3_key, str(input_path), config.aws.bucket_name)
                except Exception as e:
                    raise AppError(ErrorCode.DOWNLOAD_FAILED, Messages.Error.DOWNLOAD_FAILED)

                try:
                    await ffmpeg.cut(input_path, output_path, start, end)
                except Exception as e:
                    raise AppError(ErrorCode.CUT_FAILED, Messages.Error.CUT_FAILED)

                try:
                    loop = asyncio.get_running_loop()
                    overlays = await loop.run_in_executor(
                        pool,
                        run_analysis,
                        detector,
                        pose_estimator,
                        output_path,
                        data["character"],
                        data["position"],
                        job,
                    )
                except AppError:
                    raise
                except Exception as e:
                    raise AppError(ErrorCode.ANALYZE_FAILED, Messages.Error.ANALYZE_FAILED)

                try:
                    await ffmpeg.overlay_images(output_path, final_path, overlays)
                except Exception as e:
                    raise AppError(ErrorCode.CUT_FAILED, Messages.Error.CUT_FAILED)

                try:
                    await s3.upload_file(str(final_path), processed_s3_key, config.aws.bucket_name)
                except Exception as e:
                    raise AppError(ErrorCode.UPLOAD_FAILED, Messages.Error.UPLOAD_FAILED)

                try:
                    message = {
                        "email": data["email"],
                        "key": processed_s3_key
                    }
                    await rabbitmq.publish(message, RabbitMQConfig.VIDEO_RESULT)
                except Exception as e:
                    print(e)

            except AppError as e:
                message = { "email": data["email"], "detail": e.detail }
                try:
                    await rabbitmq.publish(message, RabbitMQConfig.VIDEO_RESULT)
                except Exception as e:
                    print(f"Failed to publish result: {e}")
