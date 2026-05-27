import os
import urllib.request

import numpy as np
from PIL import Image


FACE_LANDMARK_COUNT = 478
RAW_FACE_FEATURE_SIZE = FACE_LANDMARK_COUNT * 3
GEOMETRY_FEATURE_VERSION = "geometry_v2"
FACE_FEATURE_VERSION = "geometry_blend_v3"
FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_TASK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")
)

REGION_INDICES = {
    "face": [10, 152, 234, 454, 93, 323, 58, 288],
    "left_eye": [33, 133, 145, 153, 154, 155, 158, 159, 160, 161],
    "right_eye": [263, 362, 373, 374, 380, 381, 385, 386, 387, 388],
    "left_brow": [46, 52, 53, 63, 65, 66, 70, 105, 107],
    "right_brow": [276, 282, 283, 293, 295, 296, 300, 334, 336],
    "mouth": [0, 13, 14, 17, 37, 39, 40, 61, 78, 82, 87, 91, 181, 291, 308, 312, 317, 321, 405],
    "nose": [1, 2, 4, 5, 6, 45, 98, 168, 195, 275, 327],
    "jaw": [58, 93, 132, 136, 148, 149, 150, 152, 176, 234, 288, 323, 361, 365, 377, 378, 379, 454],
}

GEOMETRY_PAIRS = [
    (234, 454), (10, 152), (93, 323), (58, 288),
    (33, 133), (159, 145), (160, 144), (158, 153),
    (263, 362), (386, 374), (387, 373), (385, 380),
    (105, 159), (70, 145), (334, 386), (300, 374),
    (61, 291), (13, 14), (0, 17), (78, 308), (82, 87), (312, 317),
    (61, 13), (291, 13), (61, 14), (291, 14),
    (1, 13), (1, 14), (1, 61), (1, 291),
    (33, 61), (263, 291), (33, 1), (263, 1),
    (152, 14), (10, 13), (234, 61), (454, 291),
]

ANGLE_TRIPLETS = [
    (61, 13, 291), (61, 14, 291), (33, 1, 263), (234, 1, 454),
    (105, 159, 145), (334, 386, 374), (13, 1, 14), (10, 1, 152),
]

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


def geometry_feature_size():
    region_features = len(REGION_INDICES) * 9
    pair_features = len(GEOMETRY_PAIRS) * 4
    angle_features = len(ANGLE_TRIPLETS)
    ratio_features = 8
    return RAW_FACE_FEATURE_SIZE + region_features + pair_features + angle_features + ratio_features


def feature_size_for_version(feature_version):
    if feature_version == "raw":
        return RAW_FACE_FEATURE_SIZE
    if feature_version == GEOMETRY_FEATURE_VERSION:
        return geometry_feature_size()
    if feature_version == FACE_FEATURE_VERSION:
        return geometry_feature_size() + len(BLENDSHAPE_NAMES)
    raise ValueError(f"Unsupported landmark feature version: {feature_version}")


FACE_FEATURE_SIZE = feature_size_for_version(FACE_FEATURE_VERSION)


class FaceLandmarkDetector:
    def __init__(self, backend, detector):
        self.backend = backend
        self.detector = detector

    def detect_landmarks(self, image_np):
        landmarks, _ = self.detect_face_features(image_np)
        return landmarks

    def detect_face_features(self, image_np):
        if self.backend == "solutions":
            result = self.detector.process(image_np)
            if not result.multi_face_landmarks:
                return None, None
            return result.multi_face_landmarks[0].landmark, None

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None, None
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
        return result.face_landmarks[0], blendshapes

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
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return FaceLandmarkDetector("tasks", detector)


def point_distance(coords, start, end):
    delta = coords[start] - coords[end]
    return float(np.linalg.norm(delta[:2]))


def point_angle(coords, left, center, right):
    a = coords[left, :2] - coords[center, :2]
    b = coords[right, :2] - coords[center, :2]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.arccos(cosine) / np.pi)


def build_geometry_features(coords):
    features = [coords.reshape(-1)]

    region_values = []
    for indices in REGION_INDICES.values():
        region = coords[indices]
        region_values.extend(region.mean(axis=0))
        region_values.extend(region.std(axis=0))
        region_values.append(np.ptp(region[:, 0]))
        region_values.append(np.ptp(region[:, 1]))
        region_values.append(np.ptp(region[:, 2]))
    features.append(np.asarray(region_values, dtype=np.float32))

    pair_values = []
    for start, end in GEOMETRY_PAIRS:
        delta = coords[start] - coords[end]
        pair_values.append(np.linalg.norm(delta[:2]))
        pair_values.append(abs(delta[0]))
        pair_values.append(abs(delta[1]))
        pair_values.append(abs(delta[2]))
    features.append(np.asarray(pair_values, dtype=np.float32))

    angle_values = [point_angle(coords, left, center, right) for left, center, right in ANGLE_TRIPLETS]
    features.append(np.asarray(angle_values, dtype=np.float32))

    face_width = max(point_distance(coords, 234, 454), 1e-6)
    face_height = max(point_distance(coords, 10, 152), 1e-6)
    mouth_width = point_distance(coords, 61, 291)
    mouth_open = point_distance(coords, 13, 14)
    left_eye_open = point_distance(coords, 159, 145)
    right_eye_open = point_distance(coords, 386, 374)
    left_brow_eye = point_distance(coords, 105, 159)
    right_brow_eye = point_distance(coords, 334, 386)
    ratio_values = np.asarray(
        [
            face_width / face_height,
            mouth_width / face_width,
            mouth_open / face_height,
            mouth_open / max(mouth_width, 1e-6),
            left_eye_open / face_height,
            right_eye_open / face_height,
            left_brow_eye / face_height,
            right_brow_eye / face_height,
        ],
        dtype=np.float32,
    )
    features.append(ratio_values)

    return np.concatenate(features).astype(np.float32)


def build_blendshape_features(blendshapes):
    scores = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
    if not blendshapes:
        return scores

    name_to_index = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}
    for category in blendshapes:
        index = name_to_index.get(category.category_name)
        if index is not None:
            scores[index] = category.score
    return scores


def extract_face_landmarks(image_path, face_mesh, feature_version=FACE_FEATURE_VERSION):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    landmarks, blendshapes = face_mesh.detect_face_features(image_np)

    if not landmarks:
        return np.zeros(feature_size_for_version(feature_version), dtype=np.float32), False

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

    if feature_version == "raw":
        return coords.reshape(-1).astype(np.float32), True
    if feature_version == GEOMETRY_FEATURE_VERSION:
        return build_geometry_features(coords), True
    if feature_version != FACE_FEATURE_VERSION:
        raise ValueError(f"Unsupported landmark feature version: {feature_version}")

    return np.concatenate(
        [build_geometry_features(coords), build_blendshape_features(blendshapes)]
    ).astype(np.float32), True
