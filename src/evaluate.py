import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Cần thiết cho CNN
import tensorflow as tf

# Cần thiết cho RF
import joblib
import cv2

# Import logic trích xuất feature từ train_rf (nếu cần)
try:
    from train_rf import extract_all_features, get_class_names, collect_image_paths, build_feature_matrix
except ImportError:
    # Nếu chạy từ root folder, cần copy logic hoặc import đúng đường dẫn
    import sys
    sys.path.append("src")
    from train_rf import extract_all_features, get_class_names, collect_image_paths, build_feature_matrix

def plot_cm(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def evaluate_cnn(args):
    print("\n--- ĐÁNH GIÁ MODEL CNN ---")
    model_path = Path(args.model_dir) / "cnn" / "cnn_model.keras"
    class_path = Path(args.model_dir) / "cnn" / "class_names_cnn.json"
    
    with open(class_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    model = tf.keras.models.load_model(model_path)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(args.data_dir) / "test",
        image_size=(224, 224), # Khớp với train_cnn
        batch_size=32,
        shuffle=False
    )
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, class_names, title="CNN Confusion Matrix")

def evaluate_rf(args):
    print("\n--- ĐÁNH GIÁ MODEL RANDOM FOREST ---")
    model_path = Path(args.model_dir) / "rf" / "random_forest_pipeline.pkl"
    class_path = Path(args.model_dir) / "rf" / "class_names_rf.json"
    
    with open(class_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    pipeline = joblib.load(model_path)
    
    test_dir = Path(args.data_dir) / "test"
    test_paths, y_true = collect_image_paths(test_dir, class_names)
    
    # Do RF dùng feature handcrafted nên phải trích xuất lại cho tập test
    X_test, y_true_clean, _ = build_feature_matrix(test_paths, y_true, img_size=128, split_name="test")
    
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_true_clean, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true_clean, y_pred)
    plot_cm(cm, class_names, title="Random Forest Confusion Matrix")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["cnn", "rf"], required=True, help="Loại model muốn đánh giá")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()
    
    if args.type == "cnn":
        evaluate_cnn(args)
    else:
        evaluate_rf(args)