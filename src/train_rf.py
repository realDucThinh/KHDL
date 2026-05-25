import argparse
import json
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops
    from skimage.feature import local_binary_pattern


warnings.filterwarnings("ignore")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest image classifier with handcrafted features"
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models/rf")
    parser.add_argument("--report_dir", type=str, default="reports/rf")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Dùng grid nhỏ để chạy nhanh hơn",
    )

    return parser.parse_args()


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image_unicode(path: Path):
    """
    Dùng np.fromfile + cv2.imdecode để đọc được cả path có Unicode trên Windows.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")

    return image


def get_class_names(train_dir: Path):
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    if not class_names:
        raise ValueError(f"Không tìm thấy class nào trong: {train_dir}")

    return class_names


def collect_image_paths(split_dir: Path, class_names):
    image_paths = []
    labels = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Thiếu thư mục class '{class_name}' trong {split_dir}")

        for path in class_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
                labels.append(label_idx)

    return image_paths, np.array(labels, dtype=np.int64)


def extract_color_histogram(image_bgr, bins=(8, 8, 8)):
    """
    HSV Color Histogram:
    - H range: 0-180 trong OpenCV
    - S range: 0-256
    - V range: 0-256
    Output: 8*8*8 = 512 features
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(bins),
        ranges=[0, 180, 0, 256, 0, 256],
    )

    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def extract_haralick_features(image_bgr):
    """
    GLCM / Haralick features:
    - contrast
    - dissimilarity
    - homogeneity
    - energy
    - correlation
    - ASM

    Với mỗi property, lấy mean và std qua nhiều distance/angle.
    Output: 6 * 2 = 12 features
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    features = []

    for prop in properties:
        values = graycoprops(glcm, prop)
        features.append(values.mean())
        features.append(values.std())

    return np.array(features, dtype=np.float32)


def extract_lbp_features(image_bgr, p=24, r=3):
    """
    LBP - Local Binary Pattern:
    - P = 24
    - R = 3
    - method = uniform
    Output: P + 2 = 26 features
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P=p, R=r, method="uniform")

    n_bins = p + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_bins + 1),
        range=(0, n_bins),
    )

    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)

    return hist


def extract_hu_moments(image_bgr):
    """
    Hu Moments:
    - 7 features hình dạng
    - Dùng log transform để ổn định giá trị
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()

    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu_log.astype(np.float32)


def extract_all_features(image_path: Path, img_size: int):
    image = read_image_unicode(image_path)

    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )

    color_features = extract_color_histogram(image)
    haralick_features = extract_haralick_features(image)
    lbp_features = extract_lbp_features(image)
    hu_features = extract_hu_moments(image)

    features = np.concatenate(
        [
            color_features,
            haralick_features,
            lbp_features,
            hu_features,
        ]
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features.astype(np.float32)


def build_feature_matrix(image_paths, labels, img_size: int, split_name: str):
    X = []
    y = []
    failed = []

    total = len(image_paths)

    print(f"\nĐang trích xuất đặc trưng cho tập {split_name}: {total} ảnh")

    for idx, (path, label) in enumerate(zip(image_paths, labels), start=1):
        try:
            features = extract_all_features(path, img_size)
            X.append(features)
            y.append(label)
        except Exception as e:
            failed.append((str(path), str(e)))

        if idx % 200 == 0 or idx == total:
            print(f"{split_name}: {idx}/{total}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if failed:
        print(f"\nCó {len(failed)} ảnh lỗi ở tập {split_name}.")
        for item in failed[:10]:
            print("Lỗi:", item)

    return X, y, failed


def get_feature_names():
    names = []

    for i in range(8 * 8 * 8):
        names.append(f"hsv_hist_{i}")

    haralick_props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    for prop in haralick_props:
        names.append(f"haralick_{prop}_mean")
        names.append(f"haralick_{prop}_std")

    for i in range(26):
        names.append(f"lbp_{i}")

    for i in range(7):
        names.append(f"hu_moment_{i + 1}")

    return names


def build_param_grid(quick: bool):
    if quick:
        return {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [20, None],
            "rf__min_samples_split": [2],
            "rf__min_samples_leaf": [1],
            "rf__max_features": ["sqrt"],
        }

    return {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2"],
    }


def plot_confusion_matrix(cm, class_names, report_dir: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(report_dir / "rf_confusion_matrix.png", dpi=200)
    plt.close()


def plot_roc_curves(model, X_test, y_test, class_names, report_dir: Path):
    num_classes = len(class_names)

    if num_classes < 2:
        return

    y_score = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

    plt.figure(figsize=(9, 7))

    auc_rows = []

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        auc_rows.append(
            {
                "class": class_name,
                "auc": roc_auc,
            }
        )

        plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("Random Forest ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(report_dir / "rf_roc_auc.png", dpi=200)
    plt.close()

    pd.DataFrame(auc_rows).to_csv(
        report_dir / "rf_auc_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_feature_importance(model, report_dir: Path, top_k=30):
    rf = model.named_steps["rf"]
    importances = rf.feature_importances_

    feature_names = get_feature_names()

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(
        report_dir / "rf_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_df = df.head(top_k).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Top {top_k} Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(report_dir / "rf_feature_importance.png", dpi=200)
    plt.close()


def evaluate_model(model, X_test, y_test, class_names, report_dir: Path):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(report_dir / "rf_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        report_dir / "rf_classification_report.csv",
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        report_dir / "rf_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    plot_confusion_matrix(cm, class_names, report_dir)
    plot_roc_curves(model, X_test, y_test, class_names, report_dir)
    plot_feature_importance(model, report_dir)

    print("\n========== RANDOM FOREST TEST RESULT ==========")
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(report_text)

    return acc


def main():
    args = parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)

    ensure_dir(model_dir)
    ensure_dir(report_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục val: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    class_names = get_class_names(train_dir)

    with open(model_dir / "class_names_rf.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_paths, y_train_raw = collect_image_paths(train_dir, class_names)
    val_paths, y_val_raw = collect_image_paths(val_dir, class_names)
    test_paths, y_test_raw = collect_image_paths(test_dir, class_names)

    X_train, y_train, failed_train = build_feature_matrix(
        train_paths,
        y_train_raw,
        args.img_size,
        "train",
    )

    X_val, y_val, failed_val = build_feature_matrix(
        val_paths,
        y_val_raw,
        args.img_size,
        "val",
    )

    X_test, y_test, failed_test = build_feature_matrix(
        test_paths,
        y_test_raw,
        args.img_size,
        "test",
    )

    print("\nKích thước feature:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    test_fold = np.concatenate(
        [
            np.full(len(y_train), -1, dtype=np.int64),
            np.zeros(len(y_val), dtype=np.int64),
        ]
    )

    predefined_split = PredefinedSplit(test_fold=test_fold)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    random_state=args.seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = build_param_grid(args.quick)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=predefined_split,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print("\nBắt đầu GridSearchCV cho Random Forest...")
    grid_search.fit(X_train_val, y_train_val)

    best_model = grid_search.best_estimator_

    print("\n========== BEST RF PARAMS ==========")
    print(grid_search.best_params_)
    print(f"Best validation accuracy: {grid_search.best_score_:.4f}")

    pd.DataFrame(grid_search.cv_results_).to_csv(
        report_dir / "rf_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_acc = evaluate_model(
        best_model,
        X_test,
        y_test,
        class_names,
        report_dir,
    )

    joblib.dump(best_model, model_dir / "random_forest_pipeline.pkl")
    joblib.dump(best_model.named_steps["rf"], model_dir / "random_forest.pkl")
    joblib.dump(best_model.named_steps["scaler"], model_dir / "scaler.pkl")

    summary = {
        "best_params": grid_search.best_params_,
        "best_validation_accuracy": float(grid_search.best_score_),
        "test_accuracy": float(test_acc),
        "num_classes": len(class_names),
        "class_names": class_names,
        "img_size": args.img_size,
        "feature_count": int(X_train.shape[1]),
        "failed_train": len(failed_train),
        "failed_val": len(failed_val),
        "failed_test": len(failed_test),
    }

    with open(report_dir / "rf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu Random Forest pipeline tại: {model_dir / 'random_forest_pipeline.pkl'}")
    print(f"Đã lưu Random Forest riêng tại: {model_dir / 'random_forest.pkl'}")
    print(f"Đã lưu scaler tại: {model_dir / 'scaler.pkl'}")
    print(f"Đã lưu báo cáo tại: {report_dir}")


if __name__ == "__main__":
    main()