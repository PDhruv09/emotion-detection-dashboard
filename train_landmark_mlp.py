import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.landmark_mlp import LandmarkEmotionMLP
from utils.landmark_features import FACE_FEATURE_SIZE, create_face_mesh, extract_face_landmarks


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_NAME_ALIASES = {"disgut": "disgust"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical_class_name(name):
    return CLASS_NAME_ALIASES.get(name.lower(), name.lower())


def iter_image_samples(data_dir):
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        class_name = canonical_class_name(folder_name)
        if class_name not in class_to_index:
            print(f"Skipping unknown class folder: {folder_path}")
            continue

        for root, _, filenames in os.walk(folder_path):
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    yield os.path.join(root, filename), class_to_index[class_name]


def build_feature_dataset(data_dir, face_mesh, include_missing=False):
    features, labels = [], []
    skipped = 0

    for image_path, label in iter_image_samples(data_dir):
        feature_vector, detected = extract_face_landmarks(image_path, face_mesh)
        if not detected and not include_missing:
            skipped += 1
            continue
        features.append(feature_vector)
        labels.append(label)

    if not features:
        raise RuntimeError(f"No usable face landmarks found in {data_dir}")

    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y), CLASS_NAMES, skipped


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total * 100


def main():
    parser = argparse.ArgumentParser(description="Train a MediaPipe face-landmark emotion classifier.")
    parser.add_argument("--data-dir", default="data/images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--output", default="saved_models/landmark_mlp.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    with create_face_mesh() as face_mesh:
        train_dataset, class_names, train_skipped = build_feature_dataset(
            train_dir, face_mesh, include_missing=args.include_missing
        )
        val_dataset, val_class_names, val_skipped = build_feature_dataset(
            val_dir, face_mesh, include_missing=args.include_missing
        )

    if class_names != val_class_names:
        raise RuntimeError(f"Class mismatch: train={class_names}, validation={val_class_names}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LandmarkEmotionMLP(
        input_size=FACE_FEATURE_SIZE,
        num_classes=len(class_names),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)} (skipped {train_skipped} without detected faces)")
    print(f"Validation samples: {len(val_dataset)} (skipped {val_skipped} without detected faces)")

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total * 100
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "feature_size": FACE_FEATURE_SIZE,
                    "hidden_size": args.hidden_size,
                    "dropout": args.dropout,
                    "best_val_acc": val_acc,
                },
                args.output,
            )
            best_val_acc = val_acc

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
