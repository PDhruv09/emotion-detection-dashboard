import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.landmark_mlp import LandmarkEmotionMLP
from utils.landmark_features import (
    FACE_FEATURE_SIZE,
    FACE_FEATURE_VERSION,
    GEOMETRY_FEATURE_VERSION,
    create_face_mesh,
    extract_face_landmarks,
)


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_NAME_ALIASES = {"disgut": "disgust"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LandmarkTensorDataset(Dataset):
    def __init__(self, features, labels, noise_std=0.0):
        self.features = features
        self.labels = labels
        self.noise_std = noise_std

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        features = self.features[index]
        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std
        return features, self.labels[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_feature_dataset(data_dir, face_mesh, include_missing=False, feature_version=FACE_FEATURE_VERSION):
    features, labels = [], []
    skipped = 0

    for image_path, label in iter_image_samples(data_dir):
        feature_vector, detected = extract_face_landmarks(
            image_path, face_mesh, feature_version=feature_version
        )
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


def cache_path_for(cache_dir, split_name, include_missing, feature_version):
    suffix = "with_missing" if include_missing else "detected_only"
    return os.path.join(cache_dir, f"landmark_features_{split_name}_{suffix}_{feature_version}.npz")


def load_or_build_feature_dataset(
    data_dir,
    split_name,
    face_mesh,
    include_missing=False,
    cache_dir=None,
    feature_version=FACE_FEATURE_VERSION,
):
    if cache_dir:
        cache_path = cache_path_for(cache_dir, split_name, include_missing, feature_version)
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            x = torch.tensor(cached["features"], dtype=torch.float32)
            y = torch.tensor(cached["labels"], dtype=torch.long)
            class_names = cached["class_names"].tolist()
            skipped = int(cached["skipped"])
            return TensorDataset(x, y), class_names, skipped

    dataset, class_names, skipped = build_feature_dataset(
        data_dir,
        face_mesh,
        include_missing=include_missing,
        feature_version=feature_version,
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        x, y = dataset.tensors
        np.savez_compressed(
            cache_path,
            features=x.numpy(),
            labels=y.numpy(),
            class_names=np.array(class_names),
            skipped=np.array(skipped),
        )

    return dataset, class_names, skipped


def standardize_datasets(train_dataset, val_dataset):
    train_x, train_y = train_dataset.tensors
    val_x, val_y = val_dataset.tensors

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y), mean.squeeze(0), std.squeeze(0)


def compute_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = labels.numel() / (counts.clamp_min(1.0) * num_classes)
    return weights.to(device), counts.long().tolist()


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument(
        "--feature-version",
        default=FACE_FEATURE_VERSION,
        choices=["raw", GEOMETRY_FEATURE_VERSION, FACE_FEATURE_VERSION],
    )
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--output", default="saved_models/landmark_mlp.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    with create_face_mesh() as face_mesh:
        train_dataset, class_names, train_skipped = load_or_build_feature_dataset(
            train_dir,
            "train",
            face_mesh,
            include_missing=args.include_missing,
            cache_dir=args.cache_dir,
            feature_version=args.feature_version,
        )
        val_dataset, val_class_names, val_skipped = load_or_build_feature_dataset(
            val_dir,
            "validation",
            face_mesh,
            include_missing=args.include_missing,
            cache_dir=args.cache_dir,
            feature_version=args.feature_version,
        )

    if class_names != val_class_names:
        raise RuntimeError(f"Class mismatch: train={class_names}, validation={val_class_names}")

    train_dataset, val_dataset, feature_mean, feature_std = standardize_datasets(
        train_dataset, val_dataset
    )
    train_x, train_y = train_dataset.tensors
    class_weights, class_counts = compute_class_weights(train_y, len(class_names), device)

    train_loader = DataLoader(
        LandmarkTensorDataset(train_x, train_y, noise_std=args.noise_std),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = LandmarkEmotionMLP(
        input_size=train_x.shape[1],
        num_classes=len(class_names),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=None if args.no_class_weights else class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Classes: {class_names}")
    print(f"Class counts: {dict(zip(class_names, class_counts))}")
    print(f"Training samples: {len(train_dataset)} (skipped {train_skipped} without detected faces)")
    print(f"Validation samples: {len(val_dataset)} (skipped {val_skipped} without detected faces)")
    print(f"Device: {device}")

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
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}\n")

        improved = val_acc > best_val_acc + args.min_delta
        if improved:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "feature_size": train_x.shape[1],
                    "feature_version": args.feature_version,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "architecture": "modern",
                    "best_val_acc": val_acc,
                    "best_val_loss": val_loss,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                },
                args.output,
            )
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
