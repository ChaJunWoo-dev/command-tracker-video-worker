from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from infra.rabbitmq_client import RabbitMQClient
from worker.handlers import on_message
from ai.detector import PersonDetector
from ai.pose_estimator import PoseEstimator
from config.constants import RabbitMQConfig


async def main():
    shutdown_event = asyncio.Event()

    def signal_handler(*_):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    detector = PersonDetector()
    pose_estimator = PoseEstimator()
    pool = ThreadPoolExecutor(max_workers=2)

    async with RabbitMQClient() as rabbitmq:
        await rabbitmq.consume(
            RabbitMQConfig.VIDEO_PROCESS,
            partial(
                on_message,
                rabbitmq=rabbitmq,
                pool=pool,
                detector=detector,
                pose_estimator=pose_estimator,
            )
        )
        await shutdown_event.wait()

    pool.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
