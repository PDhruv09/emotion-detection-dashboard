# dashboard/app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cnn import EmotionCNN
from models.landmark_mlp import LandmarkEmotionMLP
from models.resnet import EmotionResNet
from utils.visualize import plot_confusion_matrix, show_misclassified
from utils.landmark_features import FACE_FEATURE_SIZE, create_face_mesh, extract_face_landmarks
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Facial Emotion Dashboard", layout="wide")
st.title("😊 Emotion Detection Model Dashboard")

# Emotion class labels
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LANDMARK_MODEL = "MediaPipe Landmarks"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation data
@st.cache_data
def load_data():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.ImageFolder("data/images/validation", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

test_loader = load_data()

# Load model
@st.cache_resource
def load_model(model_type):
    if model_type == "CNN":
        model = EmotionCNN(num_classes=7).to(device)
        model.load_state_dict(torch.load("saved_models/cnn.pth", map_location=device))
    elif model_type == "ResNet":
        model = EmotionResNet(num_classes=7).to(device)
        model.load_state_dict(torch.load("saved_models/resnet.pth", map_location=device))
    else:
        checkpoint_path = "saved_models/landmark_mlp.pth"
        if not os.path.exists(checkpoint_path):
            return None
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = LandmarkEmotionMLP(
            input_size=checkpoint.get("feature_size", FACE_FEATURE_SIZE),
            num_classes=len(checkpoint.get("class_names", class_names)),
            hidden_size=checkpoint.get("hidden_size", 256),
            dropout=checkpoint.get("dropout", 0.3),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_face_mesh():
    return create_face_mesh()


def predict_landmark_image(model, image_path):
    face_mesh = load_face_mesh()
    feature_vector, detected = extract_face_landmarks(image_path, face_mesh)
    features = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs, detected


def evaluate_landmark_model(model, dataset):
    preds, labels = [], []
    detected_count = 0

    for image_path, label in dataset.samples:
        probs, detected = predict_landmark_image(model, image_path)
        preds.append(int(np.argmax(probs)))
        labels.append(label)
        detected_count += int(detected)

    return preds, labels, detected_count


def plot_landmark_confusion_matrix(model, dataset):
    preds, labels, detected_count = evaluate_landmark_model(model, dataset)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({detected_count}/{len(dataset)} faces detected)")
    st.pyplot(fig)


def show_landmark_misclassified(model, dataset, max_images=12):
    misclassified = []

    for image_path, label in dataset.samples:
        probs, detected = predict_landmark_image(model, image_path)
        pred = int(np.argmax(probs))
        if pred != label:
            misclassified.append((image_path, pred, label, detected))
        if len(misclassified) >= max_images:
            break

    if not misclassified:
        st.success("No misclassified images found in this validation sample.")
        return

    cols = 4
    rows = int(np.ceil(len(misclassified) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (image_path, pred, label, detected) in zip(axes, misclassified):
        image = plt.imread(image_path)
        ax.imshow(image, cmap="gray")
        detected_text = "" if detected else "\nNo face detected"
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}{detected_text}")
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

# Sidebar controls
model_choice = st.sidebar.selectbox("Choose Model", ["CNN", "ResNet", LANDMARK_MODEL])
model = load_model(model_choice)

if model is None:
    st.warning(
        "The MediaPipe landmark model has not been trained yet. "
        "Run `python train_landmark_mlp.py` to create `saved_models/landmark_mlp.pth`."
    )
    st.stop()

st.sidebar.markdown("Use the sidebar to explore predictions made by different models.")

# Use session state for toggles
if "show_cm" not in st.session_state:
    st.session_state.show_cm = False
if "show_errors" not in st.session_state:
    st.session_state.show_errors = False

if st.sidebar.button("🔄 Show Confusion Matrix"):
    st.session_state.show_cm = not st.session_state.show_cm
    st.session_state.show_errors = False

if st.sidebar.button("❌ Show Misclassified Images"):
    st.session_state.show_errors = not st.session_state.show_errors
    st.session_state.show_cm = False

# Show confusion matrix or errors
if st.session_state.show_cm:
    st.subheader(f"Confusion Matrix for {model_choice}")
    if model_choice == LANDMARK_MODEL:
        plot_landmark_confusion_matrix(model, test_loader.dataset)
    else:
        plot_confusion_matrix(model, test_loader, device, class_names)
elif st.session_state.show_errors:
    st.subheader(f"Misclassified Images for {model_choice}")
    if model_choice == LANDMARK_MODEL:
        show_landmark_misclassified(model, test_loader.dataset)
    else:
        show_misclassified(model, test_loader, device, class_names)
else:
    # Inspect single prediction
    st.markdown("<h2 style='margin-bottom:5px;'>🔍 Inspect a Specific Test Image</h2>", unsafe_allow_html=True)
    col_idx, col_dummy = st.columns([1, 4])
    with col_idx:
        st.markdown("<h4 style='margin-bottom:5px;'>Image index:</h4>", unsafe_allow_html=True)
        index = st.number_input("", min_value=0, max_value=len(test_loader.dataset)-1, value=0, step=1, label_visibility="collapsed")
        st.markdown(f"<span style='font-size: 18px;'>/{len(test_loader.dataset)-1}</span>", unsafe_allow_html=True)

    dataset = test_loader.dataset
    image, label = dataset[int(index)]
    if model_choice == LANDMARK_MODEL:
        image_path = dataset.samples[int(index)][0]
        probs, detected = predict_landmark_image(model, image_path)
        if not detected:
            st.warning("MediaPipe did not detect a face in this image, so the model used a zero landmark vector.")
    else:
        input_image = image.unsqueeze(0).to(device)
        output = model(input_image)
        probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
    pred_label = np.argmax(probs)

    col_img = st.columns([1, 2, 1])[1]  # center image
    with col_img:
        unnorm = image * 0.5 + 0.5
        st.image(unnorm.squeeze().numpy(), width=500)
        st.markdown(f"<h3>True Label: {class_names[label]}  |  Predicted Label: {class_names[pred_label]}</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.barh(class_names, probs)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Display model performance chart
    st.markdown("<h2 style='margin-bottom:5px;'>📊 Model Accuracy Comparison</h2>", unsafe_allow_html=True)
    model_names = ["CNN", "ResNet", "MediaPipe"]
    test_accuracies = [51.36, 58.65, 0.0]  # Replace MediaPipe after training
    fig2, ax2 = plt.subplots()
    ax2.bar(model_names, test_accuracies, color=["skyblue", "lightgreen", "lightcoral"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Test Accuracy (%)")
    st.pyplot(fig2)
