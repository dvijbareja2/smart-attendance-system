# ============================================================
#  compare_models.py  -  Compare all 3 models
# ============================================================

import cv2
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from preprocessing import load_dataset
from config import DATASET_PATH
from train_cnn import FaceCNN, FaceDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

print("Loading dataset...")

# ── Load data for MSE/LBPH ────────────────────────────────
faces, names = load_dataset(DATASET_PATH)
name_list = list(set(names))
labels = np.array([name_list.index(n) for n in names])

split = int(len(faces) * 0.8)
train_faces, test_faces = faces[:split], faces[split:]
train_labels, test_labels = labels[:split], labels[split:]

print(f"Loaded {len(faces)} images for {name_list}")
print()

# ── MODEL 1: MSE ──────────────────────────────────────────
print("Testing Model 1: MSE...")

def mse_predict(train_faces, train_labels, test_face):
    best_score = float("inf")
    best_label = -1
    for face, label in zip(train_faces, train_labels):
        score = float(np.mean((face.astype(np.float32) - test_face.astype(np.float32)) ** 2))
        if score < best_score:
            best_score = score
            best_label = label
    return best_label

mse_correct = sum(1 for face, label in zip(test_faces, test_labels)
                  if mse_predict(train_faces, train_labels, face) == label)
mse_accuracy = mse_correct / len(test_faces) * 100
print(f"MSE Accuracy: {mse_accuracy:.1f}%")

# ── MODEL 2: LBPH ─────────────────────────────────────────
print("Testing Model 2: LBPH...")

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(train_faces, train_labels)

lbph_correct = sum(1 for face, label in zip(test_faces, test_labels)
                   if lbph.predict(face)[0] == label)
lbph_accuracy = lbph_correct / len(test_faces) * 100
print(f"LBPH Accuracy: {lbph_accuracy:.1f}%")

# ── MODEL 3: EIGENFACES ───────────────────────────────────
print("Testing Model 3: Eigenfaces...")

eigen = cv2.face.EigenFaceRecognizer_create()
eigen.train(train_faces, train_labels)

eigen_correct = sum(1 for face, label in zip(test_faces, test_labels)
                    if eigen.predict(face)[0] == label)
eigen_accuracy = eigen_correct / len(test_faces) * 100
print(f"Eigenfaces Accuracy: {eigen_accuracy:.1f}%")

# ── MODEL 4: SVM ──────────────────────────────────────────
print("Testing Model 4: SVM...")

with open("svm_model.pkl", "rb") as f:
    svm_data = pickle.load(f)
svm_model = svm_data["model"]
le = svm_data["label_encoder"]

flat_train = [f.flatten() for f in train_faces]
flat_test  = [f.flatten() for f in test_faces]

train_labels_str = [name_list[l] for l in train_labels]
test_labels_str  = [name_list[l] for l in test_labels]

svm_preds = svm_model.predict(flat_test)
svm_accuracy = accuracy_score(le.transform(test_labels_str), svm_preds) * 100
print(f"SVM Accuracy: {svm_accuracy:.1f}%")

# ── MODEL 5: CNN ──────────────────────────────────────────
print("Testing Model 5: CNN...")

checkpoint = torch.load("cnn_model.pth", map_location="cpu")
cnn_name_map = checkpoint["name_map"]
num_classes  = checkpoint["num_classes"]

cnn = FaceCNN(num_classes)
cnn.load_state_dict(checkpoint["model_state"])
cnn.eval()

dataset = FaceDataset(DATASET_PATH)
split_idx = int(0.8 * len(dataset))
_, test_set = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])
test_loader = DataLoader(test_set, batch_size=16)

cnn_correct, cnn_total = 0, 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        outputs = cnn(imgs)
        _, predicted = torch.max(outputs, 1)
        cnn_correct += (predicted == lbls).sum().item()
        cnn_total += lbls.size(0)

cnn_accuracy = cnn_correct / cnn_total * 100
print(f"CNN Accuracy: {cnn_accuracy:.1f}%")

# ── RESULTS ───────────────────────────────────────────────
print()
print("=" * 45)
print("FINAL RESULTS")
print("=" * 45)
print(f"Model 1 - MSE:        {mse_accuracy:.1f}%")
print(f"Model 2 - LBPH:       {lbph_accuracy:.1f}%")
print(f"Model 3 - Eigenfaces: {eigen_accuracy:.1f}%")
print(f"Model 4 - SVM:        {svm_accuracy:.1f}%")
print(f"Model 5 - CNN:        {cnn_accuracy:.1f}%")
print()

scores = {"MSE": mse_accuracy, "LBPH": lbph_accuracy,
          "Eigenfaces": eigen_accuracy, "SVM": svm_accuracy, "CNN": cnn_accuracy}
best = max(scores, key=scores.get)
print(f"BEST MODEL: {best} with {scores[best]:.1f}% accuracy")
print("=" * 45)

# ── Comparison Graph ──────────────────────────────────────
models = list(scores.keys())
accs   = list(scores.values())
colors = ["#e74c3c", "#e74c3c", "#e74c3c", "#2ecc71", "#2ecc71"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Model Comparison - Accuracy (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim([0, 110])
ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()
print("Comparison graph saved -> model_comparison.png")