from pathlib import Path

from jumpbot.cv.types import FramePose, Landmark

LANDMARK_INDEXES = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot": 31,
    "right_foot": 32,
}


def extract_pose(video_path: Path) -> tuple[list[FramePose], float, int]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on optional native packages
        raise RuntimeError("Install JumpBot with the 'cv' extra") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS metadata")

    frames: list[FramePose] = []
    frame_index = 0
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as estimator:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            image_height, image_width = image.shape[:2]
            result = estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                points = {}
                for name, index in LANDMARK_INDEXES.items():
                    raw = result.pose_landmarks.landmark[index]
                    points[name] = Landmark(
                        x_px=float(raw.x * image_width),
                        y_px=float(raw.y * image_height),
                        visibility=float(raw.visibility),
                    )
                frames.append(FramePose(frame_index, frame_index / fps, points))
            frame_index += 1
    capture.release()
    return frames, fps, declared_frames or frame_index
