# ============================================================
#  train_svm.py  -  SVM model training
# ============================================================

import cv2
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import DATASET_PATH

def load_faces(dataset_path):
    faces, names = [], []
    for person in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, person)
        if not os.path.isdir(folder):
            continue
        for img_file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            img = cv2.equalizeHist(img)
            faces.append(img.flatten())
            names.append(person)
    return np.array(faces), np.array(names)

def train_svm(dataset_path=DATASET_PATH, save_path="svm_model.pkl"):
    print("Loading dataset...")
    X, y = load_faces(dataset_path)
    print(f"Total images: {len(X)}, Classes: {list(set(y))}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    train_accs, test_accs = [], []
    c_values = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]

    print("\nTraining SVM across C values...\n")
    for c in c_values:
        svm = SVC(kernel="rbf", C=c, probability=True)
        svm.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, svm.predict(X_train)) * 100
        test_acc  = accuracy_score(y_test,  svm.predict(X_test))  * 100
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"C={c:6}  Train Acc: {train_acc:.1f}%  Test Acc: {test_acc:.1f}%")

    best_svm = SVC(kernel="rbf", C=10, probability=True)
    best_svm.fit(X_train, y_train)
    final_test_acc = accuracy_score(y_test, best_svm.predict(X_test)) * 100
    print(f"\nFinal SVM Test Accuracy: {final_test_acc:.1f}%")

    with open(save_path, "wb") as f:
        pickle.dump({"model": best_svm, "label_encoder": le}, f)
    print(f"SVM model saved -> {save_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(len(c_values)), train_accs, marker="o", label="Train Accuracy", color="blue")
    ax1.plot(range(len(c_values)), test_accs,  marker="o", label="Test Accuracy",  color="red")
    ax1.set_xticks(range(len(c_values)))
    ax1.set_xticklabels([str(c) for c in c_values], rotation=45)
    ax1.set_title("SVM - Accuracy vs C value")
    ax1.set_xlabel("C value")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.bar(["Train", "Test"], [train_accs[-3], test_accs[-3]], color=["blue", "red"])
    ax2.set_title("SVM - Final Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig("svm_training_graph.png")
    plt.close()
    print("SVM training graph saved -> svm_training_graph.png")

    return best_svm, le

if __name__ == "__main__":
    train_svm()