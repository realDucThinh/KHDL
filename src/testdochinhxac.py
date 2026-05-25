import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

try:
	import tensorflow as tf
except ImportError:
	tf = None

try:
	import joblib
except ImportError:
	joblib = None

sys.path.append(str(Path(__file__).parent))

try:
	from train_rf import collect_image_paths, build_feature_matrix
except ImportError:
	collect_image_paths = None
	build_feature_matrix = None


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate a trained model and plot per-class precision/recall/F1."
	)
	parser.add_argument("--type", choices=["cnn", "rf"], required=True)
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--model_dir", type=str, default="models")
	parser.add_argument("--report_dir", type=str, default="reports")
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--rf_img_size", type=int, default=128)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--no_show", action="store_true")
	return parser.parse_args()


def ensure_dir(path: Path):
	path.mkdir(parents=True, exist_ok=True)


def load_class_names(path: Path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def print_report_table(report_dict, class_names):
	df = pd.DataFrame(report_dict).transpose()
	ordered_rows = class_names + ["accuracy", "macro avg", "weighted avg"]
	df = df.loc[ordered_rows]
	print("\nDetailed classification report:")
	print(df.round(4).to_string())
	return df


def plot_prf_per_class(df, class_names, title, output_path, show):
	metrics = ["precision", "recall", "f1-score"]
	class_df = df.loc[class_names, metrics]

	x = np.arange(len(class_names))
	width = 0.25

	plt.figure(figsize=(12, 6))
	# Grouped bars for precision/recall/f1
	plt.bar(x - width, class_df["precision"], width, label="Precision")
	plt.bar(x, class_df["recall"], width, label="Recall")
	plt.bar(x + width, class_df["f1-score"], width, label="F1-score")

	plt.title(title)
	plt.xlabel("Class")
	plt.ylabel("Score")
	plt.xticks(x, class_names, rotation=45, ha="right")
	plt.ylim(0.0, 1.05)
	plt.legend()
	plt.tight_layout()

	ensure_dir(output_path.parent)
	plt.savefig(output_path, dpi=200)
	print(f"Saved chart: {output_path}")
	if show:
		plt.show()
	plt.close()


def resolve_cnn_paths(model_dir: Path):
	path1 = model_dir / "cnn" / "cnn_model.keras"
	class1 = model_dir / "cnn" / "class_names_cnn.json"
	if path1.exists() and class1.exists():
		return path1, class1

	path2 = model_dir / "cnn_model.keras"
	class2 = model_dir / "class_names_cnn.json"
	return path2, class2


def resolve_rf_paths(model_dir: Path):
	path1 = model_dir / "rf" / "random_forest_pipeline.pkl"
	class1 = model_dir / "rf" / "class_names_rf.json"
	if path1.exists() and class1.exists():
		return path1, class1

	path2 = model_dir / "random_forest_pipeline.pkl"
	class2 = model_dir / "class_names_rf.json"
	return path2, class2


def _wrap_layer_class(layer_cls):
	class CompatLayer(layer_cls):
		@classmethod
		def from_config(cls, config):
			config.pop("quantization_config", None)
			return super().from_config(config)

	return CompatLayer


def load_cnn_model(model_path: Path):
	try:
		return tf.keras.models.load_model(model_path, compile=False)
	except Exception as exc:
		if "quantization_config" not in str(exc):
			raise

		layer_names = [
			"Conv2D",
			"MaxPooling2D",
			"Flatten",
			"Dense",
			"Dropout",
			"Rescaling",
			"RandomRotation",
			"RandomTranslation",
			"RandomZoom",
			"RandomFlip",
		]
		custom_objects = {}
		for name in layer_names:
			layer_cls = getattr(tf.keras.layers, name, None)
			if layer_cls is not None:
				custom_objects[name] = _wrap_layer_class(layer_cls)

		return tf.keras.models.load_model(
			model_path,
			custom_objects=custom_objects,
			compile=False,
		)


def evaluate_cnn(args):
	if tf is None:
		raise RuntimeError("TensorFlow is not available. Install tensorflow first.")

	model_path, class_path = resolve_cnn_paths(Path(args.model_dir))
	if not model_path.exists() or not class_path.exists():
		raise FileNotFoundError("CNN model or class_names file not found.")

	class_names = load_class_names(class_path)
	model = load_cnn_model(model_path)

	test_dir = Path(args.data_dir) / "test"
	test_ds = tf.keras.utils.image_dataset_from_directory(
		test_dir,
		image_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		label_mode="int",
		shuffle=False,
		class_names=class_names,
	)

	y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
	y_prob = model.predict(test_ds, verbose=1)
	y_pred = np.argmax(y_prob, axis=1)

	report_dict = classification_report(
		y_true,
		y_pred,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)

	df = print_report_table(report_dict, class_names)
	output_path = Path(args.report_dir) / "cnn" / "cnn_prf_per_class.png"
	plot_prf_per_class(
		df,
		class_names,
		"Precision / Recall / F1-score per class (CNN)",
		output_path,
		show=not args.no_show,
	)


def evaluate_rf(args):
	if joblib is None:
		raise RuntimeError("joblib is not available. Install joblib first.")
	if collect_image_paths is None or build_feature_matrix is None:
		raise RuntimeError("RF helper functions are not available.")

	model_path, class_path = resolve_rf_paths(Path(args.model_dir))
	if not model_path.exists() or not class_path.exists():
		raise FileNotFoundError("RF model or class_names file not found.")

	class_names = load_class_names(class_path)
	pipeline = joblib.load(model_path)

	test_dir = Path(args.data_dir) / "test"
	test_paths, y_true_raw = collect_image_paths(test_dir, class_names)
	X_test, y_true, _ = build_feature_matrix(
		test_paths,
		y_true_raw,
		img_size=args.rf_img_size,
		split_name="test",
	)

	y_pred = pipeline.predict(X_test)

	report_dict = classification_report(
		y_true,
		y_pred,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)

	df = print_report_table(report_dict, class_names)
	output_path = Path(args.report_dir) / "rf" / "rf_prf_per_class.png"
	plot_prf_per_class(
		df,
		class_names,
		"Precision / Recall / F1-score per class (RF)",
		output_path,
		show=not args.no_show,
	)


def main():
	args = parse_args()

	if args.type == "cnn":
		evaluate_cnn(args)
	else:
		evaluate_rf(args)


if __name__ == "__main__":
	main()
