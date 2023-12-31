import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vit_b_16
from tqdm import tqdm

from model import ViT

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5
batch_size = 128
learning_rate = 0.0002

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# CIFAR-10 dataset (train:50000, test:10000)
train_dataset = CIFAR10(root="../data", train=True, download=True, transform=transform)
test_dataset = CIFAR10(root="../data", train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Models
# 1. handcraft ViT
# print("Handcraft ViT:")
# model = ViT(image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072)
# model.to(device)

# 2. Torch ViT
# print("Torch ViT")
# model = vit_b_16()
# # print(model)
# model.heads.head = nn.Linear(model.heads.head.in_features, 10)
# # print(model)
# model.to(device)

# 3. Torch ViT
print("Torch Pretrained ViT")
model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, 10)
model.to(device)


# load model checkpoint
# model.load_state_dict(torch.load('./checkpoint/ViT.ckpt'))
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}")


# torch.save(model.state_dict(), './checkpoint/ViT.ckpt')
# model.load_state_dict(torch.load('./checkpoint/ViT.ckpt'))


# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")
