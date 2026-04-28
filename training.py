# ============================================================
#  training.py  -  Model training code and architecture
# ============================================================
#
#  Three models are trained and compared:
#  1. LBPH    - Local Binary Pattern Histograms (OpenCV)
#  2. SVM     - Support Vector Machine (scikit-learn)
#  3. CNN     - Convolutional Neural Network (PyTorch)
#
#  Run this file directly to train all models and
#  generate comparison graphs.
# ============================================================

from train_cnn import train_cnn
from train_svm import train_svm
from compare_models import *
from logger import log_info


if __name__ == "__main__":
    log_info("Training CNN...")
    train_cnn()

    log_info("Training SVM...")
    train_svm()

    log_info("All models trained. Running comparison...")
    log_info("Check cnn_training_graph.png, svm_training_graph.png, model_comparison.png")