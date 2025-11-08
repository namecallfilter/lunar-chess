"""Generate piece detection training data using local assets."""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Board colors (light, dark) - various theme colors
BOARD_THEMES = [
	((240, 217, 181), (181, 136, 99)),  # Classic brown
	((238, 238, 210), (118, 150, 86)),  # Green
	((222, 227, 230), (140, 162, 173)),  # Blue
	((255, 206, 158), (209, 139, 71)),  # Wood
	((235, 236, 208), (115, 149, 82)),  # Tournament
	((240, 240, 240), (120, 120, 120)),  # Gray
	((255, 228, 181), (205, 133, 63)),  # Tan
	((250, 235, 215), (139, 90, 43)),  # Marble
	((173, 216, 230), (100, 149, 237)),  # Sky blue
	((255, 218, 185), (160, 82, 45)),  # Peach
	((245, 222, 179), (139, 69, 19)),  # Wheat
	((230, 230, 250), (147, 112, 219)),  # Purple
]


def random_fen_generation() -> str:
	"""Generate a random FEN string for a chess position."""
	pieces = ["r", "n", "b", "q", "k", "p", "R", "N", "B", "Q", "K", "P"]
	fen = []

	for _ in range(8):
		row = [None] * 8
		for i in range(8):
			if random.random() < 0.5:
				row[i] = random.choice(pieces)

		row_str = ""
		empty_count = 0
		for piece in row:
			if piece is not None:
				if empty_count > 0:
					row_str += str(empty_count)
					empty_count = 0
				row_str += piece
			else:
				empty_count += 1

		if empty_count > 0:
			row_str += str(empty_count)

		fen.append(row_str)

	fen_string = "/".join(fen)
	fen_string += " w KQkq - 0 1"
	return fen_string


def generate_empty_board_fen() -> str:
	"""Generate a FEN string for an empty or nearly empty board (negative sample)."""
	# Decide how empty the board should be
	board_type = random.choice(
		[
			"completely_empty",
			"very_sparse",
			"sparse",
			"hard_negative_midgame",  # NEW: Complex positions that look busy but will have blank labels
		]
	)

	if board_type == "completely_empty":
		# Completely empty board
		return "8/8/8/8/8/8/8/8 w KQkq - 0 1"

	pieces = ["r", "n", "b", "q", "k", "p", "R", "N", "B", "Q", "K", "P"]
	fen = []

	# Determine piece density
	if board_type == "very_sparse":
		piece_probability = 0.05  # ~4 pieces on average
	elif board_type == "sparse":
		piece_probability = 0.15  # ~10 pieces on average
	else:  # hard_negative_midgame
		# Create a complex mid-game position (many pieces)
		# This will force the model to learn precise shapes, not just "busy = pieces"
		piece_probability = 0.35  # ~23 pieces (realistic mid-game)

	for _ in range(8):
		row = [None] * 8
		for i in range(8):
			if random.random() < piece_probability:
				row[i] = random.choice(pieces)

		row_str = ""
		empty_count = 0
		for piece in row:
			if piece is not None:
				if empty_count > 0:
					row_str += str(empty_count)
					empty_count = 0
				row_str += piece
			else:
				empty_count += 1

		if empty_count > 0:
			row_str += str(empty_count)

		fen.append(row_str)

	fen_string = "/".join(fen)
	fen_string += " w KQkq - 0 1"
	return fen_string


def generate_royalty_focused_fen() -> str:
	"""Generate a FEN with focus on Kings and Queens in confusing positions."""
	fen = []

	# Place Kings and Queens with high probability
	# Also include Bishops as they're often confused with royalty
	royalty_pieces = ["k", "q", "b", "K", "Q", "B"]
	support_pieces = ["r", "n", "p", "R", "N", "P"]

	for _ in range(8):
		row = [None] * 8
		for i in range(8):
			if random.random() < 0.4:  # 40% chance of a piece
				# 70% chance it's royalty/bishop, 30% support pieces
				if random.random() < 0.7:
					row[i] = random.choice(royalty_pieces)
				else:
					row[i] = random.choice(support_pieces)

		row_str = ""
		empty_count = 0
		for piece in row:
			if piece is not None:
				if empty_count > 0:
					row_str += str(empty_count)
					empty_count = 0
				row_str += piece
			else:
				empty_count += 1

		if empty_count > 0:
			row_str += str(empty_count)

		fen.append(row_str)

	fen_string = "/".join(fen)
	fen_string += " w KQkq - 0 1"
	return fen_string


def fen_to_grid(fen: str) -> list[int]:
	"""Convert FEN to grid of piece IDs (0=empty, 1-12=pieces)."""
	piece_to_id = {
		"r": 1,
		"n": 2,
		"b": 3,
		"q": 4,
		"k": 5,
		"p": 6,
		"R": 7,
		"N": 8,
		"B": 9,
		"Q": 10,
		"K": 11,
		"P": 12,
	}

	board_fen = fen.split()[0]
	grid = []

	for row in board_fen.split("/"):
		for char in row:
			if char.isdigit():
				grid.extend([0] * int(char))
			else:
				grid.append(piece_to_id[char])

	return grid


def fen_to_piece_list(fen: str) -> list:
	"""Convert FEN to list of (piece, row, col)."""
	piece_map = {
		"r": "bR",
		"n": "bN",
		"b": "bB",
		"q": "bQ",
		"k": "bK",
		"p": "bP",
		"R": "wR",
		"N": "wN",
		"B": "wB",
		"Q": "wQ",
		"K": "wK",
		"P": "wP",
	}

	board_fen = fen.split()[0]
	pieces = []

	for row_idx, row in enumerate(board_fen.split("/")):
		col_idx = 0
		for char in row:
			if char.isdigit():
				col_idx += int(char)
			else:
				pieces.append((piece_map[char], row_idx, col_idx))
				col_idx += 1

	return pieces


def get_available_piece_sets(assets_dir: Path) -> list:
	"""Get all available piece sets from assets directory."""
	pieces_dir = assets_dir / "pieces"
	if not pieces_dir.exists():
		return []

	# Blacklist of themes to exclude
	blacklist = ["blindfold"]  # Themes that hide pieces

	# Find directories that have piece PNGs
	piece_sets = []
	for theme_dir in pieces_dir.iterdir():
		if theme_dir.is_dir() and (theme_dir / "wK.png").exists():
			# Skip blacklisted themes
			if theme_dir.name.lower() not in blacklist:
				piece_sets.append(theme_dir.name)

	return sorted(piece_sets)


def add_gray_dots_to_board(board: np.ndarray, num_dots: int = None) -> np.ndarray:
	"""Add gray dots to random squares to create hard negatives for false positives.

	These dots simulate artifacts/shadows that the model might confuse with pieces.
	CRITICAL: Caller must NOT add labels for these dots - that's the whole point!
	"""
	board_copy = board.copy()
	square_size = board.shape[0] // 8

	# Random number of dots (1-5 if not specified)
	if num_dots is None:
		num_dots = random.randint(1, 5)

	for _ in range(num_dots):
		# Pick a random square
		row = random.randint(0, 7)
		col = random.randint(0, 7)

		# Calculate square center
		center_y = row * square_size + square_size // 2
		center_x = col * square_size + square_size // 2

		# Random dot properties
		dot_radius = random.randint(square_size // 8, square_size // 4)
		# Gray color with some variation (120-160 range)
		gray_value = random.randint(120, 160)
		dot_color = (gray_value, gray_value, gray_value)

		# Add some randomness to position (not perfectly centered)
		offset_x = random.randint(-square_size // 6, square_size // 6)
		offset_y = random.randint(-square_size // 6, square_size // 6)

		# Draw the dot with some transparency
		overlay = board_copy.copy()
		cv2.circle(overlay, (center_x + offset_x, center_y + offset_y),
				   dot_radius, dot_color, -1)

		# Blend it in (50-80% opacity)
		alpha = random.uniform(0.5, 0.8)
		cv2.addWeighted(overlay, alpha, board_copy, 1 - alpha, 0, board_copy)

	return board_copy


def draw_chess_board(
	fen: str,
	piece_set: str,
	board_theme: tuple,
	assets_dir: Path,
	board_size: int = 640,
) -> np.ndarray:
	"""Draw a chess board with pieces."""
	square_size = board_size // 8
	light_color, dark_color = board_theme

	# Create board
	board = np.zeros((board_size, board_size, 3), dtype=np.uint8)

	# Draw squares
	for row in range(8):
		for col in range(8):
			color = light_color if (row + col) % 2 == 0 else dark_color
			y1 = row * square_size
			y2 = (row + 1) * square_size
			x1 = col * square_size
			x2 = (col + 1) * square_size
			board[y1:y2, x1:x2] = color

	# Place pieces
	pieces = fen_to_piece_list(fen)
	piece_dir = assets_dir / "pieces" / piece_set

	for piece_name, row, col in pieces:
		piece_path = piece_dir / f"{piece_name}.png"

		if not piece_path.exists():
			continue

		piece_img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
		if piece_img is None:
			continue

		# Ensure we have at least 3 channels
		if len(piece_img.shape) < 3:
			continue

		# Resize to square size
		piece_img = cv2.resize(piece_img, (square_size, square_size))

		# Place piece on board
		y = row * square_size
		x = col * square_size

		try:
			if (
				len(piece_img.shape) == 3 and piece_img.shape[2] == 4
			):  # Has alpha channel
				alpha = piece_img[:, :, 3:4] / 255.0
				piece_rgb = piece_img[:, :, :3]
				board[y : y + square_size, x : x + square_size] = (
					alpha * piece_rgb
					+ (1 - alpha) * board[y : y + square_size, x : x + square_size]
				)
			else:  # No alpha channel or grayscale
				if len(piece_img.shape) == 2:  # Grayscale
					piece_img = cv2.cvtColor(piece_img, cv2.COLOR_GRAY2BGR)
				board[y : y + square_size, x : x + square_size] = piece_img[:, :, :3]
		except Exception:
			# Skip problematic pieces
			continue

	return board


def generate_training_sample(
	assets_dir: Path,
	piece_sets: list,
	output_size: int = 640,
	is_negative: bool = False,
	mode: str = "normal",
) -> tuple[np.ndarray, str]:
	"""Generate a single training sample.

	Args:
	    assets_dir: Path to assets directory
	    piece_sets: List of available piece sets
	    output_size: Output image size
	    is_negative: If True, generates empty/sparse boards (negative samples)
	    mode: Generation mode - "normal", "hard_negative_dots", or "royalty_focus"
	"""
	# Random selections based on mode
	if mode == "hard_negative_dots":
		# Generate empty/sparse board and ADD gray dots (but don't label them!)
		fen = generate_empty_board_fen()
	elif mode == "royalty_focus":
		# Generate board with lots of Kings, Queens, and Bishops
		fen = generate_royalty_focused_fen()
	elif is_negative:
		fen = generate_empty_board_fen()
	else:
		fen = random_fen_generation()

	piece_set = random.choice(piece_sets)
	board_theme = random.choice(BOARD_THEMES)

	# Draw board
	board = draw_chess_board(fen, piece_set, board_theme, assets_dir, output_size)

	# Validate board before processing
	if board is None or not isinstance(board, np.ndarray):
		return None, fen

	# Add gray dots for hard negative mining
	if mode == "hard_negative_dots":
		board = add_gray_dots_to_board(board)
		# CRITICAL: We keep the FEN as-is (empty/sparse), so dots won't be labeled

	# Optional: slight brightness/contrast variation
	if random.random() < 0.3:
		alpha = random.uniform(0.85, 1.15)  # Contrast
		beta = random.uniform(-10, 10)  # Brightness
		board = cv2.convertScaleAbs(board, alpha=alpha, beta=beta)

	return board, fen


def save_yolo_sample(img: np.ndarray, fen: str, split_dir: Path, index: int):
	"""Save image and YOLO label."""
	# Validate image
	if img is None or not isinstance(img, np.ndarray):
		print(f"Warning: Invalid image at index {index}, skipping")
		return

	# Ensure uint8
	if img.dtype != np.uint8:
		img = np.clip(img, 0, 255).astype(np.uint8)

	# Save image to YOLO directory
	img_name = f"sample_{index:06d}.png"
	yolo_img_path = split_dir / "images" / img_name

	success = cv2.imwrite(str(yolo_img_path), img)
	if not success:
		print(f"Warning: Failed to write image to {yolo_img_path}")
		return

	# Create YOLO label
	grid = fen_to_grid(fen)
	labels = []
	square_w = 1.0 / 8
	square_h = 1.0 / 8

	for idx, piece_id in enumerate(grid):
		if piece_id == 0:
			continue

		row = idx // 8
		col = idx % 8

		center_x = (col + 0.5) / 8
		center_y = (row + 0.5) / 8

		yolo_class = piece_id - 1

		label = (
			f"{yolo_class} {center_x:.6f} {center_y:.6f} {square_w:.6f} {square_h:.6f}"
		)
		labels.append(label)

	yolo_label_file = split_dir / "labels" / f"sample_{index:06d}.txt"
	with open(yolo_label_file, "w") as f:
		if labels:
			f.write("\n".join(labels) + "\n")
		# else: create empty file for negative samples (CRITICAL for YOLO to learn what NOT to detect)


def main():
	parser = argparse.ArgumentParser(
		description="Generate chess piece detection dataset from local assets"
	)
	parser.add_argument(
		"--assets",
		type=str,
		default="./assets",
		help="Assets directory containing pieces",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="yolo_pieces",
		help="Output directory (YOLO format)",
	)
	parser.add_argument(
		"-c",
		"--count",
		type=int,
		default=1000,
		help="Number of boards to generate (with pieces)",
	)
	parser.add_argument(
		"--negative-ratio",
		type=float,
		default=0.15,
		help="Ratio of negative samples (empty/sparse boards). Default 0.15 = 15%% of total (recommended 10-15%%)",
	)
	parser.add_argument(
		"--hard-negative-dots",
		type=int,
		default=500,
		help="Number of hard negative samples with gray dots (recommended ~500)",
	)
	parser.add_argument(
		"--royalty-focus",
		type=int,
		default=1000,
		help="Number of royalty-focused samples (Kings/Queens/Bishops, recommended ~1000)",
	)
	parser.add_argument(
		"--save-boards",
		action="store_true",
		help="Save clean boards to assets/boards directory",
	)

	args = parser.parse_args()
	assets_dir = Path(args.assets)
	output_dir = Path(args.output)
	count = args.count
	negative_ratio = args.negative_ratio
	hard_negative_dots_count = args.hard_negative_dots
	royalty_focus_count = args.royalty_focus

	# Calculate how many negative samples to generate
	total_samples = int(count / (1 - negative_ratio))
	negative_count = int(total_samples * negative_ratio)
	positive_count = count

	print(f"Generating {positive_count} positive samples (boards with pieces)")
	print(f"Generating {negative_count} negative samples (empty/sparse boards)")
	print(f"Generating {hard_negative_dots_count} hard negative samples (gray dots - unlabeled)")
	print(f"Generating {royalty_focus_count} royalty-focused samples (Kings/Queens/Bishops)")
	print(f"Total samples: {total_samples + hard_negative_dots_count + royalty_focus_count}")

	# Get available piece sets
	piece_sets = get_available_piece_sets(assets_dir)

	if not piece_sets:
		print(f"Error: No piece sets found in {assets_dir / 'pieces'}")
		print("Run: uv run get_pieces.py first to download pieces")
		return

	print(f"Found {len(piece_sets)} piece sets: {', '.join(piece_sets[:5])}...")

	# Remove existing data
	if output_dir.exists():
		print(f"Removing existing data at {output_dir}/")
		shutil.rmtree(output_dir)

	# Create YOLO directory structure
	for split in ["train", "val"]:
		(output_dir / split / "images").mkdir(parents=True, exist_ok=True)
		(output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

	# Optionally save clean boards to assets
	boards_dir = None
	if args.save_boards:
		boards_dir = assets_dir / "boards"
		if boards_dir.exists():
			shutil.rmtree(boards_dir)
		boards_dir.mkdir(parents=True, exist_ok=True)
		print(f"Will save clean boards to {boards_dir}")

	print("\nGenerating samples...")

	# Split piece sets FIRST to ensure no overlap between train/val
	random.shuffle(piece_sets)
	piece_set_split_idx = max(1, int(len(piece_sets) * 0.8))  # At least 1 set for val
	train_piece_sets = piece_sets[:piece_set_split_idx]
	val_piece_sets = piece_sets[piece_set_split_idx:]

	print(f"Train: {len(train_piece_sets)} piece sets")
	print(f"Val: {len(val_piece_sets)} piece sets")

	# Calculate train/val split (80/20)
	train_positive_count = int(positive_count * 0.8)
	val_positive_count = positive_count - train_positive_count
	train_negative_count = int(negative_count * 0.8)
	val_negative_count = negative_count - train_negative_count
	train_hard_negative_count = int(hard_negative_dots_count * 0.8)
	val_hard_negative_count = hard_negative_dots_count - train_hard_negative_count
	train_royalty_count = int(royalty_focus_count * 0.8)
	val_royalty_count = royalty_focus_count - train_royalty_count

	print(
		f"\nTrain: {train_positive_count} positive + {train_negative_count} negative + {train_hard_negative_count} hard-neg-dots + {train_royalty_count} royalty = {train_positive_count + train_negative_count + train_hard_negative_count + train_royalty_count} total"
	)
	print(
		f"Val: {val_positive_count} positive + {val_negative_count} negative + {val_hard_negative_count} hard-neg-dots + {val_royalty_count} royalty = {val_positive_count + val_negative_count + val_hard_negative_count + val_royalty_count} total"
	)

	# Generate train samples
	train_samples = []

	# Positive samples (with pieces)
	for i in tqdm(range(train_positive_count), desc="Generating train positive"):
		board, fen = generate_training_sample(
			assets_dir, train_piece_sets, is_negative=False
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"train_pos_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		train_samples.append((board, fen))

	# Negative samples (empty/sparse boards)
	for i in tqdm(range(train_negative_count), desc="Generating train negative"):
		board, fen = generate_training_sample(
			assets_dir, train_piece_sets, is_negative=True
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"train_neg_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		train_samples.append((board, fen))

	# Hard negative samples (gray dots - UNLABELED)
	for i in tqdm(range(train_hard_negative_count), desc="Generating train hard-neg-dots"):
		board, fen = generate_training_sample(
			assets_dir, train_piece_sets, mode="hard_negative_dots"
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"train_hard_neg_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		train_samples.append((board, fen))

	# Royalty-focused samples (Kings/Queens/Bishops)
	for i in tqdm(range(train_royalty_count), desc="Generating train royalty-focus"):
		board, fen = generate_training_sample(
			assets_dir, train_piece_sets, mode="royalty_focus"
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"train_royalty_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		train_samples.append((board, fen))

	# Generate val samples
	val_samples = []

	# Positive samples (with pieces)
	for i in tqdm(range(val_positive_count), desc="Generating val positive"):
		board, fen = generate_training_sample(
			assets_dir, val_piece_sets, is_negative=False
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"val_pos_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		val_samples.append((board, fen))

	# Negative samples (empty/sparse boards)
	for i in tqdm(range(val_negative_count), desc="Generating val negative"):
		board, fen = generate_training_sample(
			assets_dir, val_piece_sets, is_negative=True
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"val_neg_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		val_samples.append((board, fen))

	# Hard negative samples (gray dots - UNLABELED)
	for i in tqdm(range(val_hard_negative_count), desc="Generating val hard-neg-dots"):
		board, fen = generate_training_sample(
			assets_dir, val_piece_sets, mode="hard_negative_dots"
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"val_hard_neg_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		val_samples.append((board, fen))

	# Royalty-focused samples (Kings/Queens/Bishops)
	for i in tqdm(range(val_royalty_count), desc="Generating val royalty-focus"):
		board, fen = generate_training_sample(
			assets_dir, val_piece_sets, mode="royalty_focus"
		)

		# Skip invalid boards
		if board is None or not isinstance(board, np.ndarray):
			continue

		# Optionally save clean board to assets
		if boards_dir:
			board_path = boards_dir / f"val_royalty_{i:06d}.png"
			cv2.imwrite(str(board_path), board)

		val_samples.append((board, fen))

	print("Converting to YOLO format...")
	print(f"  Train: {len(train_samples)} images")
	print(f"  Val: {len(val_samples)} images")

	# Save train samples
	for idx, (img, fen) in enumerate(tqdm(train_samples, desc="Processing train")):
		save_yolo_sample(img, fen, output_dir / "train", idx)

	# Save val samples
	for idx, (img, fen) in enumerate(tqdm(val_samples, desc="Processing val")):
		save_yolo_sample(img, fen, output_dir / "val", idx)

	# Create data.yaml
	yaml_content = f"""# Piece Detection Dataset
path: {output_dir.absolute().as_posix()}
train: train/images
val: val/images

# Classes (only actual pieces, empty squares not included)
names:
  0: r   # black rook
  1: n   # black knight
  2: b   # black bishop
  3: q   # black queen
  4: k   # black king
  5: p   # black pawn
  6: R   # white rook
  7: N   # white knight
  8: B   # white bishop
  9: Q   # white queen
  10: K  # white king
  11: P  # white pawn
"""

	with open(output_dir / "data.yaml", "w") as f:
		f.write(yaml_content)

	print(f"\nYOLO dataset created at {output_dir}")
	print(f"Train: {len(train_samples)} images")
	print(f"Val: {len(val_samples)} images")


if __name__ == "__main__":
	main()
