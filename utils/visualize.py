import matplotlib.pyplot as plt
import numpy as np
import torch
import streamlit as st
from utils.evaluation import evaluate
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(model, dataloader, device, class_names):
    _, preds, labels = evaluate(model, dataloader, device)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt.gcf())


def show_misclassified(model, dataloader, device, class_names, max_images=12):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), preds[i].item(), labels[i].item()))
                if len(misclassified) >= max_images:
                    break
            if len(misclassified) >= max_images:
                break

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, (img, pred, label) in zip(axes.flatten(), misclassified):
        img = img.squeeze() * 0.5 + 0.5  # unnormalize
        ax.imshow(img.numpy(), cmap='gray')
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}")
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)
