from pathlib import Path
from typing import List

from ai.detector import PersonDetector
from ai.pose_estimator import PoseEstimator
from services.video_analyzer import VideoAnalyzer
from services.icon_composer import IconComposer
from services.command_service import MotionRecognizer
from config.constants import ErrorCode, Messages
from config.exceptions import AppError


def run_analysis(
    detector: PersonDetector,
    pose_estimator: PoseEstimator,
    video_path: Path,
    character: str,
    position: str,
    job_dir: Path,
) -> List[dict]:
    """영상 분석 + 아이콘 합성 수행"""
    analyzer = VideoAnalyzer(detector, pose_estimator, position)
    motion_recognizer = MotionRecognizer(character, position)
    icon_composer = IconComposer()

    overlays = []
    for result in analyzer.analyze(video_path, motion_recognizer):
        if result["command"]:
            inputs = motion_recognizer.get_input(result["command"])
            image_path = job_dir / f"{result['frame_idx']}.png"
            icon_composer.compose(inputs, image_path)

            overlays.append({
                "frame": result["frame_idx"],
                "image_path": image_path,
            })

    if not overlays:
        raise AppError(ErrorCode.NO_SUBTITLE, Messages.Error.NO_SUBTITLE)

    return overlays
