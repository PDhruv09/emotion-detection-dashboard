import os
import urllib.request

import numpy as np
from PIL import Image


FACE_LANDMARK_COUNT = 478
FACE_FEATURE_SIZE = FACE_LANDMARK_COUNT * 3
FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_TASK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")
)


class FaceLandmarkDetector:
    def __init__(self, backend, detector):
        self.backend = backend
        self.detector = detector

    def detect_landmarks(self, image_np):
        if self.backend == "solutions":
            result = self.detector.process(image_np)
            if not result.multi_face_landmarks:
                return None
            return result.multi_face_landmarks[0].landmark

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def ensure_face_landmarker_task(task_path=DEFAULT_TASK_PATH):
    if os.path.exists(task_path):
        return task_path

    os.makedirs(os.path.dirname(task_path), exist_ok=True)
    urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, task_path)
    return task_path


def create_face_mesh(task_path=DEFAULT_TASK_PATH):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for the landmark model. Install it with `pip install mediapipe`."
        ) from exc

    if hasattr(mp, "solutions"):
        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return FaceLandmarkDetector("solutions", detector)

    task_path = ensure_face_landmarker_task(task_path)
    base_options = mp.tasks.BaseOptions(model_asset_path=task_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return FaceLandmarkDetector("tasks", detector)


def extract_face_landmarks(image_path, face_mesh):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    landmarks = face_mesh.detect_landmarks(image_np)

    if not landmarks:
        return np.zeros(FACE_FEATURE_SIZE, dtype=np.float32), False

    coords = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)

    if coords.shape[0] < FACE_LANDMARK_COUNT:
        pad = np.zeros((FACE_LANDMARK_COUNT - coords.shape[0], 3), dtype=np.float32)
        coords = np.vstack([coords, pad])
    elif coords.shape[0] > FACE_LANDMARK_COUNT:
        coords = coords[:FACE_LANDMARK_COUNT]

    present = np.any(coords != 0, axis=1)
    if np.any(present):
        visible_coords = coords[present]
        center = visible_coords[:, :2].mean(axis=0)
        scale = np.ptp(visible_coords[:, :2], axis=0).max()
        if scale < 1e-6:
            scale = 1.0
        coords[:, 0] = (coords[:, 0] - center[0]) / scale
        coords[:, 1] = (coords[:, 1] - center[1]) / scale
        coords[:, 2] = coords[:, 2] / scale

    return coords.reshape(-1).astype(np.float32), True
