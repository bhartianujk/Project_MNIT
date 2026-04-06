import os
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3

CIFAR_ANIMAL_CLASSES = {
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse"
}

ORIG_TO_NEW = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5
}

IDX_TO_CLASS = {
    0: "bird",
    1: "cat",
    2: "deer",
    3: "dog",
    4: "frog",
    5: "horse"
}


class AnimalCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def filter_indices(targets):
    indices = []
    for i, label in enumerate(targets):
        if label in CIFAR_ANIMAL_CLASSES:
            indices.append(i)
    return indices


def remap_dataset_targets(dataset):
    dataset.targets = [ORIG_TO_NEW[t] for t in dataset.targets if t in ORIG_TO_NEW]


transform = transforms.ToTensor()

full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_indices = filter_indices(full_train.targets)
test_indices = filter_indices(full_test.targets)

train_subset = Subset(full_train, train_indices)
test_subset = Subset(full_test, test_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

model = AnimalCnn().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        remapped = torch.tensor([ORIG_TO_NEW[int(x)] for x in labels], dtype=torch.long)
        images = images.to(DEVICE)
        remapped = remapped.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, remapped)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total += remapped.size(0)
        correct += (preds == remapped).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        remapped = torch.tensor([ORIG_TO_NEW[int(x)] for x in labels], dtype=torch.long)
        images = images.to(DEVICE)
        remapped = remapped.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total += remapped.size(0)
        correct += (preds == remapped).sum().item()

test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")

os.makedirs("saved_model_demo", exist_ok=True)
torch.save(model.state_dict(), "saved_model_demo/cifar10_animals.pt")

with open("saved_model_demo/class_names.json", "w") as f:
    json.dump(IDX_TO_CLASS, f)

os.makedirs("inference_images", exist_ok=True)

saved_per_class = defaultdict(int)
needed_per_class = 2

for idx in test_indices:
    image, label = full_test[idx]
    class_name = CIFAR_ANIMAL_CLASSES[int(label)]

    if saved_per_class[class_name] < needed_per_class:
        file_index = saved_per_class[class_name] + 1
        save_image(image, f"inference_images/{class_name}_{file_index}.png")
        saved_per_class[class_name] += 1

    if all(v >= needed_per_class for v in saved_per_class.values()) and len(saved_per_class) == 6:
        break

print("Saved weights to saved_model_demo/cifar10_animals.pt")
print("Saved class names to saved_model_demo/class_names.json")
print("Saved separate inference images to inference_images/")