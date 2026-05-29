import os
import subprocess
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_landmark_mlp import CLASS_NAMES, canonical_class_name
from utils.landmark_features import FACE_FEATURE_VERSION, create_face_mesh, extract_face_landmarks


OUTPUT_PATH = os.path.join("assets", "validation_landmark_features.npz")
VALIDATION_ROOT = os.path.join("data", "images", "validation")


def tracked_validation_images():
    result = subprocess.run(
        ["git", "ls-files", VALIDATION_ROOT.replace("\\", "/")],
        check=True,
        capture_output=True,
        text=True,
    )
    image_paths = []
    for line in result.stdout.splitlines():
        if os.path.splitext(line)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image_paths.append(line)
    return sorted(image_paths)


def label_for_path(path):
    folder_name = path.replace("\\", "/").split("/")[-2]
    class_name = canonical_class_name(folder_name)
    return CLASS_NAMES.index(class_name)


def main():
    image_paths = tracked_validation_images()
    if not image_paths:
        raise RuntimeError(f"No tracked validation images found under {VALIDATION_ROOT}")

    features = []
    labels = []
    detected = []

    with create_face_mesh() as face_mesh:
        for image_path in image_paths:
            feature_vector, was_detected = extract_face_landmarks(
                image_path,
                face_mesh,
                feature_version=FACE_FEATURE_VERSION,
            )
            features.append(feature_vector)
            labels.append(label_for_path(image_path))
            detected.append(was_detected)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        paths=np.array(image_paths),
        features=np.stack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        detected=np.array(detected, dtype=bool),
        feature_version=np.array(FACE_FEATURE_VERSION),
    )
    print(f"Wrote {OUTPUT_PATH} with {len(image_paths)} samples")


if __name__ == "__main__":
    main()
