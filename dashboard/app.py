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
from models.resnet import EmotionResNet
from utils.visualize import plot_confusion_matrix, show_misclassified

st.set_page_config(page_title="Facial Emotion Dashboard", layout="wide")
st.title("😊 Emotion Detection Model Dashboard")

# Emotion class labels
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

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
    test_dataset = datasets.ImageFolder("data/Images/validation", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

test_loader = load_data()

# Load model
@st.cache_resource
def load_model(model_type):
    if model_type == "CNN":
        model = EmotionCNN(num_classes=7).to(device)
        model.load_state_dict(torch.load("saved_models/cnn.pth", map_location=device))
    else:
        model = EmotionResNet(num_classes=7).to(device)
        model.load_state_dict(torch.load("saved_models/resnet.pth", map_location=device))
    model.eval()
    return model

# Sidebar controls
model_choice = st.sidebar.selectbox("Choose Model", ["CNN", "ResNet"])
model = load_model(model_choice)

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
    plot_confusion_matrix(model, test_loader, device, class_names)
elif st.session_state.show_errors:
    st.subheader(f"Misclassified Images for {model_choice}")
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
    model_names = ["CNN", "ResNet"]
    test_accuracies = [51.36, 58.65]  # Replace with your actual best results
    fig2, ax2 = plt.subplots()
    ax2.bar(model_names, test_accuracies, color=["skyblue", "lightgreen"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Test Accuracy (%)")
    st.pyplot(fig2)
