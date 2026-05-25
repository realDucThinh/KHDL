import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train basic CNN image classifier")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models/cnn")
    parser.add_argument("--report_dir", type=str, default="reports/cnn")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_datasets(args):
    data_dir = Path(args.data_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục val: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    image_size = (args.img_size, args.img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
        seed=args.seed,
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def build_cnn_model(img_size: int, num_classes: int, lr: float):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.15, 0.15),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="Basic_CNN_Classifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history, report_dir: Path):
    history_dict = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["accuracy"], label="train_accuracy")
    plt.plot(history_dict["val_accuracy"], label="val_accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "cnn_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="train_loss")
    plt.plot(history_dict["val_loss"], label="val_loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "cnn_loss.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, report_dir: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(report_dir / "cnn_confusion_matrix.png", dpi=200)
    plt.close()


def evaluate_model(model, test_ds, class_names, report_dir: Path):
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(report_dir / "cnn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        report_dir / "cnn_classification_report.csv",
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        report_dir / "cnn_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    plot_confusion_matrix(cm, class_names, report_dir)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print("\n========== CNN TEST RESULT ==========")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(report_text)


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)

    ensure_dir(model_dir)
    ensure_dir(report_dir)

    train_ds, val_ds, test_ds, class_names = load_datasets(args)

    with open(model_dir / "class_names_cnn.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model = build_cnn_model(
        img_size=args.img_size,
        num_classes=len(class_names),
        lr=args.lr,
    )

    model.summary()

    best_model_path = model_dir / "cnn_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_training_history(history, report_dir)

    model = tf.keras.models.load_model(best_model_path)

    evaluate_model(model, test_ds, class_names, report_dir)

    print(f"\nĐã lưu model CNN tại: {best_model_path}")
    print(f"Đã lưu báo cáo tại: {report_dir}")


if __name__ == "__main__":
    main()