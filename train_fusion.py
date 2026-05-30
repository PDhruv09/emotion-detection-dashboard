import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from models.fusion import FusionEmotionNet
from train_landmark_mlp import CLASS_NAMES, canonical_class_name
from utils.landmark_features import (
    FACE_FEATURE_VERSION,
    GEOMETRY_FEATURE_VERSION,
    create_face_mesh,
    extract_face_landmarks,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FusionDataset(Dataset):
    def __init__(self, paths, labels, landmarks, transform=None, landmark_noise_std=0.0):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.landmarks = torch.tensor(landmarks, dtype=torch.float32)
        self.transform = transform
        self.landmark_noise_std = landmark_noise_std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        landmarks = self.landmarks[index]
        if self.landmark_noise_std > 0:
            landmarks = landmarks + torch.randn_like(landmarks) * self.landmark_noise_std

        return image, landmarks, self.labels[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def fusion_cache_path(cache_dir, split_name, include_missing, feature_version):
    suffix = "with_missing" if include_missing else "detected_only"
    return os.path.join(cache_dir, f"fusion_features_{split_name}_{suffix}_{feature_version}.npz")


def load_or_build_fusion_cache(
    data_dir,
    split_name,
    face_mesh,
    cache_dir,
    include_missing=False,
    feature_version=FACE_FEATURE_VERSION,
):
    cache_path = fusion_cache_path(cache_dir, split_name, include_missing, feature_version)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        return {
            "paths": cached["paths"].tolist(),
            "features": cached["features"].astype(np.float32),
            "labels": cached["labels"].astype(np.int64),
            "detected": cached["detected"].astype(bool),
            "skipped": int(cached["skipped"]),
        }

    paths, features, labels, detected_flags = [], [], [], []
    skipped = 0
    for image_path, label in iter_image_samples(data_dir):
        feature_vector, detected = extract_face_landmarks(
            image_path,
            face_mesh,
            feature_version=feature_version,
        )
        if not detected and not include_missing:
            skipped += 1
            continue

        paths.append(image_path)
        features.append(feature_vector)
        labels.append(label)
        detected_flags.append(detected)

    if not paths:
        raise RuntimeError(f"No usable fusion samples found in {data_dir}")

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        cache_path,
        paths=np.array(paths),
        features=np.stack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        detected=np.array(detected_flags, dtype=bool),
        skipped=np.array(skipped),
        feature_version=np.array(feature_version),
    )

    return {
        "paths": paths,
        "features": np.stack(features).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "detected": np.array(detected_flags, dtype=bool),
        "skipped": skipped,
    }


def limit_cache(cache, max_samples):
    if max_samples is None or max_samples <= 0 or len(cache["paths"]) <= max_samples:
        return cache

    return {
        "paths": cache["paths"][:max_samples],
        "features": cache["features"][:max_samples],
        "labels": cache["labels"][:max_samples],
        "detected": cache["detected"][:max_samples],
        "skipped": cache["skipped"],
    }


def standardize_features(train_features, val_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (
        (train_features - mean).astype(np.float32),
        (val_features - mean).astype(np.float32),
        mean.squeeze(0).astype(np.float32),
        std.squeeze(0).astype(np.float32),
    )


def get_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def compute_class_weights(labels, num_classes, device):
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = labels.shape[0] / (counts.clamp_min(1.0) * num_classes)
    return weights.to(device), counts.long().tolist()


def build_sampler(labels):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=len(CLASS_NAMES)).float()
    sample_weights = 1.0 / counts.clamp_min(1.0)[labels_tensor]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels_out = [], []

    with torch.no_grad():
        for images, landmarks, labels in dataloader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().tolist())
            labels_out.extend(labels.cpu().tolist())

    return total_loss / total, correct / total * 100, preds, labels_out


def main():
    parser = argparse.ArgumentParser(description="Train an image + MediaPipe landmark fusion classifier.")
    parser.add_argument("--data-dir", default="data/images")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--feature-version", default=FACE_FEATURE_VERSION, choices=["raw", GEOMETRY_FEATURE_VERSION, FACE_FEATURE_VERSION])
    parser.add_argument("--output", default="saved_models/fusion_emotion.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-backbone", default="efficientnet_b0", choices=["efficientnet_b0", "convnext_tiny", "resnet18"])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--landmark-hidden", type=int, default=256)
    parser.add_argument("--fusion-hidden", type=int, default=512)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--landmark-noise-std", type=float, default=0.01)
    parser.add_argument("--freeze-image-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--no-sampler", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    with create_face_mesh() as face_mesh:
        train_cache = load_or_build_fusion_cache(
            train_dir,
            "train",
            face_mesh,
            args.cache_dir,
            include_missing=args.include_missing,
            feature_version=args.feature_version,
        )
        val_cache = load_or_build_fusion_cache(
            val_dir,
            "validation",
            face_mesh,
            args.cache_dir,
            include_missing=args.include_missing,
            feature_version=args.feature_version,
        )

    train_cache = limit_cache(train_cache, args.max_samples_per_split)
    val_cache = limit_cache(val_cache, args.max_samples_per_split)

    train_features, val_features, feature_mean, feature_std = standardize_features(
        train_cache["features"],
        val_cache["features"],
    )
    train_transform, val_transform = get_transforms(args.image_size)

    train_dataset = FusionDataset(
        train_cache["paths"],
        train_cache["labels"],
        train_features,
        transform=train_transform,
        landmark_noise_std=args.landmark_noise_std,
    )
    val_dataset = FusionDataset(
        val_cache["paths"],
        val_cache["labels"],
        val_features,
        transform=val_transform,
    )

    sampler = None if args.no_sampler else build_sampler(train_cache["labels"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    class_weights, class_counts = compute_class_weights(train_cache["labels"], len(CLASS_NAMES), device)
    model = FusionEmotionNet(
        landmark_size=train_features.shape[1],
        num_classes=len(CLASS_NAMES),
        image_backbone=args.image_backbone,
        landmark_hidden=args.landmark_hidden,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    model.set_image_backbone_trainable(args.freeze_image_epochs <= 0)
    criterion = nn.CrossEntropyLoss(
        weight=None if args.no_class_weights else class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        [
            {"params": model.image_backbone.parameters(), "lr": args.backbone_learning_rate},
            {"params": model.landmark_encoder.parameters(), "lr": args.learning_rate},
            {"params": model.classifier.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Classes: {CLASS_NAMES}")
    print(f"Class counts: {dict(zip(CLASS_NAMES, class_counts))}")
    print(f"Training samples: {len(train_dataset)} (skipped {train_cache['skipped']} without detected faces)")
    print(f"Validation samples: {len(val_dataset)} (skipped {val_cache['skipped']} without detected faces)")
    print(f"Device: {device}")
    print(f"Backbone: {args.image_backbone}")
    print(f"Feature version: {args.feature_version} ({train_features.shape[1]} features)")

    for epoch in range(args.epochs):
        if epoch == args.freeze_image_epochs:
            model.set_image_backbone_trainable(True)
            print("Unfroze image backbone.")

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, landmarks, labels in train_loader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total * 100
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[1]["lr"]

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}\n")

        if val_acc > best_val_acc:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "feature_size": train_features.shape[1],
                    "feature_version": args.feature_version,
                    "feature_mean": torch.tensor(feature_mean),
                    "feature_std": torch.tensor(feature_std),
                    "image_backbone": args.image_backbone,
                    "image_size": args.image_size,
                    "landmark_hidden": args.landmark_hidden,
                    "fusion_hidden": args.fusion_hidden,
                    "dropout": args.dropout,
                    "best_val_acc": val_acc,
                    "best_val_loss": val_loss,
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
