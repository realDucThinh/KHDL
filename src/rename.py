import argparse
import sys
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


def normalize_prefix(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    return cleaned.rstrip("_")


def rename_images_in_folder(folder: Path, prefix: str, dry_run: bool) -> int:
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return 1

    files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]
    files.sort(key=lambda x: x.name.lower())

    if not files:
        print("No images found.")
        return 0

    print(f"Found {len(files)} images in {folder}.")

    if dry_run:
        for i, file_path in enumerate(files, 1):
            new_name = f"{prefix}_{i}{file_path.suffix.lower()}"
            print(f"{file_path.name} -> {new_name}")
        print(f"Dry run: {len(files)} files.")
        return 0

    temp_renames = []
    for i, file_path in enumerate(files, 1):
        temp_name = f"__temp_{i}__{file_path.name}"
        temp_path = folder / temp_name
        try:
            file_path.rename(temp_path)
            temp_renames.append((temp_path, file_path.suffix.lower()))
        except Exception as exc:
            print(f"Error during temp rename of {file_path.name}: {exc}")

    for i, (temp_path, ext) in enumerate(temp_renames, 1):
        new_name = f"{prefix}_{i}{ext}"
        final_path = folder / new_name
        try:
            temp_path.rename(final_path)
        except Exception as exc:
            print(f"Error during final rename to {new_name}: {exc}")

    print(f"Renamed {len(temp_renames)} files.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename images in one or more folders with a prefix"
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Folders to rename (default: img when omitted)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for renamed files (default: folder name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files",
    )
    args = parser.parse_args()

    if not args.input:
        raw_prefix = input("Nhap xxx (vi du: hoa): ")
        prefix = normalize_prefix(raw_prefix)
        if not prefix:
            print("Prefix khong hop le.")
            sys.exit(1)
        folder = Path("img").resolve()
        raise SystemExit(rename_images_in_folder(folder, prefix, args.dry_run))

    status = 0
    for input_path in args.input:
        folder = Path(input_path).expanduser().resolve()
        prefix = normalize_prefix(args.prefix) if args.prefix else normalize_prefix(folder.name)
        if not prefix:
            print(f"Prefix khong hop le cho folder: {folder}")
            status = 1
            continue
        result = rename_images_in_folder(folder, prefix, args.dry_run)
        if result != 0:
            status = result
    raise SystemExit(status)
