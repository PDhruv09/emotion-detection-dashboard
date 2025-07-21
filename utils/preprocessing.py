# # utils/preprocessing.py
# import os
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# def get_dataloaders(data_dir='data/Images', batch_size=64):
#     # Augmentation for training set
#     train_transform = transforms.Compose([
#         transforms.Grayscale(),  # Convert to 1 channel
#         transforms.Resize((48, 48)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     val_transform = transforms.Compose([
#         transforms.Grayscale(),  # Convert to 1 channel
#         transforms.Resize((48, 48)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     train_path = os.path.join(data_dir, 'train')
#     val_path = os.path.join(data_dir, 'validation')

#     train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
#     val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, train_dataset.classes

# utils/preprocessing.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data/Images', batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes
