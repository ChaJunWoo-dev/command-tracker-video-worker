import cv2
from pathlib import Path
from typing import Generator

from ai.detector import PersonDetector
from ai.pose_estimator import PoseEstimator
from services.command_service import MotionRecognizer


class VideoAnalyzer:
    def __init__(
        self,
        detector: PersonDetector,
        pose_estimator: PoseEstimator,
        position: str = "left",
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.position = position

    def analyze(
        self,
        video_path: Path,
        motion_recognizer: MotionRecognizer
    ) -> Generator[dict, None, None]:
        """
        영상을 분석하여 프레임별 poses와 발동된 커맨드를 반환
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                bboxes = self.detector.detect(frame, max_persons=2)
                target_bbox = self.select_target_bbox(bboxes)

                command = None
                if target_bbox is not None:
                    poses = self.pose_estimator.estimate(frame, target_bbox)
                    if len(poses) > 0:
                        command = motion_recognizer.extract(poses[0])

                yield {
                    "frame_idx": frame_idx,
                    "command": command,
                }

                frame_idx += 1
        finally:
            cap.release()

    def select_target_bbox(self, bboxes):
        """position에 따라 분석할 캐릭터의 bbox 선택"""
        if len(bboxes) == 0:
            return None
        if len(bboxes) == 1:
            return bboxes[:1]

        centers = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
        if self.position == "left":
            idx = centers.index(min(centers))
        else:
            idx = centers.index(max(centers))

        return bboxes[idx:idx+1]
