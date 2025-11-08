"""Generate board detection training data (boards on backgrounds)."""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def place_board_on_background(
	board: np.ndarray,
	background: np.ndarray,
	scale_range: tuple[float, float] = (0.3, 0.9),
) -> tuple[np.ndarray, np.ndarray]:
	"""Place board on background at random position."""
	bg_h, bg_w = background.shape[:2]

	scale = random.uniform(*scale_range)
	new_size = int(min(bg_h, bg_w) * scale)

	board_resized = cv2.resize(board, (new_size, new_size))

	max_x = bg_w - new_size
	max_y = bg_h - new_size

	if max_x <= 0 or max_y <= 0:
		bg_size = int(new_size / scale_range[0])
		background = cv2.resize(background, (bg_size, bg_size))
		bg_h, bg_w = background.shape[:2]
		max_x = bg_w - new_size
		max_y = bg_h - new_size

	x_offset = random.randint(0, max(0, max_x))
	y_offset = random.randint(0, max(0, max_y))

	composite = background.copy()
	composite[y_offset : y_offset + new_size, x_offset : x_offset + new_size] = (
		board_resized
	)

	corners = np.float32(
		[
			[x_offset, y_offset],
			[x_offset + new_size - 1, y_offset],
			[x_offset + new_size - 1, y_offset + new_size - 1],
			[x_offset, y_offset + new_size - 1],
		]
	)

	return composite, corners


def generate_board_detection_sample(
	board_image: np.ndarray,
	background_image: np.ndarray,
	output_size: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
	"""Generate training sample with board on background."""
	background = cv2.resize(background_image, (output_size, output_size))
	composite, corners = place_board_on_background(board_image, background)

	if composite.shape[0] != output_size or composite.shape[1] != output_size:
		scale_x = output_size / composite.shape[1]
		scale_y = output_size / composite.shape[0]
		corners[:, 0] *= scale_x
		corners[:, 1] *= scale_y
		composite = cv2.resize(composite, (output_size, output_size))

	normalized_corners = corners / output_size

	# Stronger augmentation for better generalization
	if random.random() < 0.7:  # Increased probability
		# Brightness/contrast variation
		composite = cv2.convertScaleAbs(
			composite, alpha=random.uniform(0.6, 1.4), beta=random.randint(-20, 20)
		)

	# Add noise
	if random.random() < 0.3:
		noise = np.random.normal(0, random.randint(5, 15), composite.shape).astype(
			np.int16
		)
		composite = np.clip(composite.astype(np.int16) + noise, 0, 255).astype(np.uint8)

	# Slight blur
	if random.random() < 0.2:
		composite = cv2.GaussianBlur(composite, (3, 3), 0)

	return composite, normalized_corners


def corners_to_yolo_bbox(corners: np.ndarray) -> tuple:
	"""Convert 4 corners to YOLO bounding box (center_x, center_y, width, height)."""
	x_coords = [corners[i][0] for i in range(4)]
	y_coords = [corners[i][1] for i in range(4)]

	min_x = min(x_coords)
	max_x = max(x_coords)
	min_y = min(y_coords)
	max_y = max(y_coords)

	center_x = (min_x + max_x) / 2
	center_y = (min_y + max_y) / 2
	width = max_x - min_x
	height = max_y - min_y

	return center_x, center_y, width, height


def generate_backgrounds(output_dir: Path, count: int = 100):
	"""Generate diverse synthetic backgrounds mimicking real screenshots."""
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Generating {count} background images...")

	for i in tqdm(range(count), desc="Generating backgrounds"):
		bg_type = random.choice(["solid", "gradient", "textured", "window", "mixed"])

		if bg_type == "solid":
			# Solid colors with slight noise (common for browser backgrounds)
			color = np.random.randint(200, 256, 3)  # Lighter colors
			bg = np.ones((640, 640, 3), dtype=np.uint8) * color
			noise = np.random.normal(0, 5, (640, 640, 3)).astype(np.int16)
			bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

		elif bg_type == "gradient":
			# Multi-directional gradients
			direction = random.choice(["horizontal", "vertical", "diagonal", "radial"])
			color1 = np.random.randint(180, 256, 3)
			color2 = np.random.randint(180, 256, 3)

			if direction == "horizontal":
				gradient = np.linspace(0, 1, 640).reshape(1, -1)
				gradient = np.repeat(gradient, 640, axis=0)
			elif direction == "vertical":
				gradient = np.linspace(0, 1, 640).reshape(-1, 1)
				gradient = np.repeat(gradient, 640, axis=1)
			elif direction == "diagonal":
				x = np.linspace(0, 1, 640)
				y = np.linspace(0, 1, 640)
				xx, yy = np.meshgrid(x, y)
				gradient = (xx + yy) / 2
			else:  # radial
				x = np.linspace(-1, 1, 640)
				y = np.linspace(-1, 1, 640)
				xx, yy = np.meshgrid(x, y)
				gradient = np.sqrt(xx**2 + yy**2)
				gradient = gradient / gradient.max()

			bg = np.zeros((640, 640, 3), dtype=np.uint8)
			for c in range(3):
				bg[:, :, c] = (
					gradient * color2[c] + (1 - gradient) * color1[c]
				).astype(np.uint8)

		elif bg_type == "textured":
			# Perlin-like noise texture
			base_color = np.random.randint(200, 256, 3)
			bg = np.ones((640, 640, 3), dtype=np.uint8) * base_color

			# Multi-scale noise
			for scale in [8, 16, 32]:
				small = np.random.randint(
					0, 256, (640 // scale, 640 // scale, 3), dtype=np.uint8
				)
				large = cv2.resize(small, (640, 640), interpolation=cv2.INTER_LINEAR)
				# Manual blend to avoid dtype issues
				bg = (
					bg.astype(np.float32) * 0.7 + large.astype(np.float32) * 0.3
				).astype(np.uint8)

		elif bg_type == "window":
			# Simulate windowed interfaces with rectangles
			base_color = np.random.randint(220, 256, 3)
			bg = np.ones((640, 640, 3), dtype=np.uint8)
			bg[:, :, 0] = base_color[0]
			bg[:, :, 1] = base_color[1]
			bg[:, :, 2] = base_color[2]

			# Add 2-4 random rectangles (simulating windows/panels)
			num_rects = random.randint(2, 4)
			for _ in range(num_rects):
				x1 = random.randint(0, 500)
				y1 = random.randint(0, 500)
				w = random.randint(100, 400)
				h = random.randint(100, 400)
				color = tuple(np.random.randint(180, 256, 3).tolist())
				cv2.rectangle(bg, (x1, y1), (x1 + w, y1 + h), color, -1)

				# Add border
				border_color = tuple(np.random.randint(100, 200, 3).tolist())
				cv2.rectangle(bg, (x1, y1), (x1 + w, y1 + h), border_color, 2)

		else:  # mixed
			# Combination of techniques
			bg = np.random.randint(200, 256, (640, 640, 3), dtype=np.uint8)

			# Add some geometric shapes
			for _ in range(random.randint(1, 3)):
				shape_type = random.choice(["rect", "circle"])
				color = tuple(np.random.randint(150, 256, 3).tolist())

				if shape_type == "rect":
					x1, y1 = random.randint(0, 500), random.randint(0, 500)
					x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
					cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)
				else:
					center = (random.randint(100, 540), random.randint(100, 540))
					radius = random.randint(30, 150)
					cv2.circle(bg, center, radius, color, -1)

			# Add noise
			noise = np.random.normal(0, 10, (640, 640, 3)).astype(np.int16)
			bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

		# Ensure uint8 before any operations
		bg = bg.astype(np.uint8)

		# Optional: Add slight blur to make it more realistic
		if random.random() < 0.3:
			bg = cv2.GaussianBlur(bg, (5, 5), 0)

		cv2.imwrite(str(output_dir / f"bg_{i:04d}.png"), bg)


def main():
	parser = argparse.ArgumentParser(description="Generate board detection dataset")
	parser.add_argument(
		"--board-dir",
		type=str,
		default="./assets/boards",
		help="Directory with clean board images",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="yolo_board",
		help="Output directory (YOLO format)",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=5000,
		help="Number of samples (split 80/20 train/val)",
	)
	parser.add_argument(
		"--bg-count",
		type=int,
		default=500,
		help="Number of backgrounds (more = better diversity)",
	)

	args = parser.parse_args()

	board_dir = Path(args.board_dir)
	output_dir = Path(args.output)

	# Remove existing data first
	if output_dir.exists():
		print(f"Removing existing data at {output_dir}/")
		shutil.rmtree(output_dir)

	# Create YOLO directory structure
	for split in ["train", "val"]:
		(output_dir / split / "images").mkdir(parents=True, exist_ok=True)
		(output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

	# Generate backgrounds
	bg_dir = Path("temp_backgrounds")
	if not bg_dir.exists() or len(list(bg_dir.glob("*.png"))) < args.bg_count:
		if bg_dir.exists():
			shutil.rmtree(bg_dir)
		generate_backgrounds(bg_dir, args.bg_count)

	board_files = list(board_dir.glob("*.png"))
	if len(board_files) == 0:
		print(f"No board images found in {board_dir}")
		print("Run: uv run gen_pieces_data.py first to generate boards")
		return

	print(f"Found {len(board_files)} board images")

	bg_files = list(bg_dir.glob("*.png"))
	print(f"Found {len(bg_files)} background images")

	print(f"\nGenerating {args.count} training samples...")

	# Split backgrounds and boards FIRST to ensure no overlap
	# This prevents train/val from having similar-looking images
	random.shuffle(board_files)
	random.shuffle(bg_files)

	# Use 80% of boards/backgrounds for train, 20% for val
	board_split_idx = int(len(board_files) * 0.8)
	bg_split_idx = int(len(bg_files) * 0.8)

	train_boards = board_files[:board_split_idx]
	val_boards = board_files[board_split_idx:]
	train_bgs = bg_files[:bg_split_idx]
	val_bgs = bg_files[bg_split_idx:]

	print(f"Train: {len(train_boards)} boards, {len(train_bgs)} backgrounds")
	print(f"Val: {len(val_boards)} boards, {len(val_bgs)} backgrounds")

	# Generate train samples (80% of total)
	train_count = int(args.count * 0.8)
	val_count = args.count - train_count

	train_samples = []
	for i in tqdm(range(train_count), desc="Generating train samples"):
		board_file = random.choice(train_boards)
		bg_file = random.choice(train_bgs)

		board = cv2.imread(str(board_file))
		background = cv2.imread(str(bg_file))

		if board is None or background is None:
			continue

		image, corners = generate_board_detection_sample(board, background)
		train_samples.append((image, corners))

	# Generate val samples (20% of total)
	val_samples = []
	for i in tqdm(range(val_count), desc="Generating val samples"):
		board_file = random.choice(val_boards)
		bg_file = random.choice(val_bgs)

		board = cv2.imread(str(board_file))
		background = cv2.imread(str(bg_file))

		if board is None or background is None:
			continue

		image, corners = generate_board_detection_sample(board, background)
		val_samples.append((image, corners))

	print("\nSaving YOLO format...")
	print(f"  Train: {len(train_samples)} images")
	print(f"  Val: {len(val_samples)} images")

	# Save train samples
	for idx, (image, corners) in enumerate(
		tqdm(train_samples, desc="Processing train")
	):
		image_name = f"sample_{idx:06d}.png"
		image_path = output_dir / "train" / "images" / image_name

		# Ensure image is valid before writing
		if image is None or not isinstance(image, np.ndarray):
			print(f"Warning: Invalid image at train index {idx}, skipping")
			continue

		# Ensure uint8 dtype
		if image.dtype != np.uint8:
			image = np.clip(image, 0, 255).astype(np.uint8)

		# Ensure correct shape (H, W, 3)
		if len(image.shape) != 3 or image.shape[2] != 3:
			print(
				f"Warning: Invalid image shape {image.shape} at train index {idx}, skipping"
			)
			continue

		success = cv2.imwrite(str(image_path), image)
		if not success:
			print(f"Warning: Failed to write image to {image_path}")
			continue

		# Convert corners to YOLO bbox
		center_x, center_y, width, height = corners_to_yolo_bbox(corners)

		label_path = output_dir / "train" / "labels" / f"sample_{idx:06d}.txt"
		with open(label_path, "w") as f:
			f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

	# Save val samples
	for idx, (image, corners) in enumerate(tqdm(val_samples, desc="Processing val")):
		image_name = f"sample_{idx:06d}.png"
		image_path = output_dir / "val" / "images" / image_name

		# Ensure image is valid before writing
		if image is None or not isinstance(image, np.ndarray):
			print(f"Warning: Invalid image at val index {idx}, skipping")
			continue

		success = cv2.imwrite(str(image_path), image)
		if not success:
			print(f"Warning: Failed to write image to {image_path}")
			continue

		# Convert corners to YOLO bbox
		center_x, center_y, width, height = corners_to_yolo_bbox(corners)

		label_path = output_dir / "val" / "labels" / f"sample_{idx:06d}.txt"
		with open(label_path, "w") as f:
			f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

	# Create data.yaml
	yaml_content = f"""# Board Detection Dataset
path: {output_dir.absolute().as_posix()}
train: train/images
val: val/images

# Classes
names:
  0: chessboard
"""

	with open(output_dir / "data.yaml", "w") as f:
		f.write(yaml_content)

	# Clean up temp backgrounds
	shutil.rmtree(bg_dir)

	print(f"\nYOLO dataset created at {output_dir}")
	print(f"Train: {len(train_samples)} images")
	print(f"Val: {len(val_samples)} images")


if __name__ == "__main__":
	main()
