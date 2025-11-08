from pathlib import Path

from ultralytics import YOLO


def convert_model_to_onnx(model_path: Path):
	"""Convert a single .pt model to ONNX format"""
	print(f"\nConverting {model_path}...")

	try:
		# Load the model
		model = YOLO(str(model_path))

		# Export to ONNX
		onnx_path = model.export(format="onnx", simplify=True)
		print(f"✓ Successfully converted to {onnx_path}")
		return True
	except Exception as e:
		print(f"✗ Failed to convert {model_path}: {e}")
		return False


def main():
	# Define the base directory
	models_dir = Path("models")

	# Find all .pt model files
	model_files = list(models_dir.rglob("*.pt"))

	if not model_files:
		print("No .pt model files found!")
		return

	print(f"Found {len(model_files)} .pt model files")
	print("=" * 60)

	# Convert each model
	successful = 0
	failed = 0

	for model_path in model_files:
		if convert_model_to_onnx(model_path):
			successful += 1
		else:
			failed += 1

	# Summary
	print("\n" + "=" * 60)
	print("Conversion complete!")
	print(f"✓ Successful: {successful}")
	print(f"✗ Failed: {failed}")
	print(f"Total: {len(model_files)}")


if __name__ == "__main__":
	main()
