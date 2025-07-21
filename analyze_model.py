import torch
from models.cnn import EmotionCNN
from models.resnet import EmotionResNet
from utils.preprocessing import get_dataloaders
from utils.evaluation import evaluate

model_type = "resnet"  # or "cnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "resnet":
    model = EmotionResNet().to(device)
    model.load_state_dict(torch.load("saved_models/resnet.pth", map_location=device))
else:
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("saved_models/cnn.pth", map_location=device))

_, val_loader, class_names = get_dataloaders()
accuracy, _, _ = evaluate(model, val_loader, device)
print(f"{model_type.upper()} Accuracy: {accuracy:.2f}%")