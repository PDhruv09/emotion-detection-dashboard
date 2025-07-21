# train_resnet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import EmotionResNet
from utils.preprocessing import get_dataloaders

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
learning_rate = 1e-4
batch_size = 64
num_epochs = 10

# Load data
train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)

# Load model
model = EmotionResNet(num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_acc = 0.0
os.makedirs("saved_models", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total * 100

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total * 100

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), "saved_models/resnet.pth")
        best_val_acc = val_acc
