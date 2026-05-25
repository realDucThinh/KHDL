import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Thư mục chứa các class gốc, ví dụ: dataset_raw",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Thư mục output sau khi chia, mặc định là data",
    )

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def copy_images(image_paths, output_class_dir):
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        shutil.copy2(image_path, output_class_dir / image_path.name)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy input_dir: {input_dir}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio phải bằng 1.0")

    random.seed(args.seed)

    class_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        raise ValueError("Không tìm thấy thư mục class nào.")

    print("Các class tìm thấy:")
    for class_dir in class_dirs:
        print("-", class_dir.name)

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_paths = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(image_paths)

        total = len(image_paths)

        train_count = int(total * args.train_ratio)
        val_count = int(total * args.val_ratio)

        train_images = image_paths[:train_count]
        val_images = image_paths[train_count:train_count + val_count]
        test_images = image_paths[train_count + val_count:]

        copy_images(train_images, output_dir / "train" / class_name)
        copy_images(val_images, output_dir / "val" / class_name)
        copy_images(test_images, output_dir / "test" / class_name)

        print(
            f"{class_name}: "
            f"total={total}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print("\nChia dữ liệu xong.")
    print(f"Output tại: {output_dir}")


if __name__ == "__main__":
    main()