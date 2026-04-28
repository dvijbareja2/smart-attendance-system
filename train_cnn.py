# ============================================================
#  train_cnn.py  -  CNN model training with epochs
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATASET_PATH

# ── Dataset ───────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.data = []
        self.labels = []
        self.name_map = {}
        self.transform = transform

        for idx, person in enumerate(sorted(os.listdir(dataset_path))):
            folder = os.path.join(dataset_path, person)
            if not os.path.isdir(folder):
                continue
            self.name_map[idx] = person
            for img_file in os.listdir(folder):
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (100, 100))
                img = cv2.equalizeHist(img)
                self.data.append(img)
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # add channel dim
        label = self.labels[idx]
        return img, label


# ── CNN Architecture ──────────────────────────────────────

class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 50x50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 25x25

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 12x12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Training ──────────────────────────────────────────────

def train_cnn(dataset_path=DATASET_PATH, epochs=20, save_path="cnn_model.pth"):

    dataset = FaceDataset(dataset_path)
    num_classes = len(dataset.name_map)
    print(f"Classes: {dataset.name_map}")
    print(f"Total images: {len(dataset)}")

    # Split 80% train, 20% test
    split = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=16, shuffle=False)

    model     = FaceCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    print(f"\nTraining CNN for {epochs} epochs...\n")

    for epoch in range(epochs):

        # ── Train phase ──
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total * 100

        # ── Test phase ──
        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss = running_loss / len(test_loader)
        test_acc  = correct / total * 100

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}%  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.1f}%")

    # ── Save model ──
    torch.save({
        "model_state": model.state_dict(),
        "name_map": dataset.name_map,
        "num_classes": num_classes
    }, save_path)
    print(f"\nCNN model saved -> {save_path}")

    # ── Save graphs ──
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs_range, test_losses,  label="Test Loss",  color="red")
    ax1.set_title("CNN - Loss vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs_range, train_accs, label="Train Accuracy", color="blue")
    ax2.plot(epochs_range, test_accs,  label="Test Accuracy",  color="red")
    ax2.set_title("CNN - Accuracy vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("cnn_training_graph.png")
    plt.close()
    print("CNN training graph saved -> cnn_training_graph.png")

    return model, dataset.name_map


if __name__ == "__main__":
    train_cnn()