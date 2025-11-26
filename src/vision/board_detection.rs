use image::GrayImage;

use crate::{
	chess::BOARD_SIZE,
	model::detected::{DetectedBoard, Rect},
};

#[derive(Debug, Copy, Clone)]
struct BoardDetectionParams {
	min_square_size: usize,
	min_runs: usize,
	variance_tolerance: f32,
	expansion_tolerance: f32,
	board_ratio_tolerance: f32,
	min_board_screen_ratio: f32,
	min_grid_confidence: f32,
	square_aspect_min: f32,
	square_aspect_max: f32,
}

impl BoardDetectionParams {
	const DEFAULT: Self = Self {
		min_square_size: 4,
		min_runs: 5,
		variance_tolerance: 0.30,
		expansion_tolerance: 0.25,
		board_ratio_tolerance: 0.12,
		min_board_screen_ratio: 0.10,
		min_grid_confidence: 0.75,
		square_aspect_min: 0.85,
		square_aspect_max: 1.15,
	};
}

static PARAMS: BoardDetectionParams = BoardDetectionParams::DEFAULT;

#[derive(Debug, Clone, Copy)]
struct Run {
	len: usize,
	x_pos: usize,
}

struct VerticalScanResult {
	max_height: f32,
	y_start: f32,
}

struct HorizontalPattern {
	start_x: usize,
	end_x: usize,
	avg_square_width: f32,
	square_count: usize,
}

struct BoardCandidate {
	board: DetectedBoard,
	score: f32,
}

struct PatternSpan {
	start_x: usize,
	end_x: usize,
	avg_width: f32,
	count: usize,
}

#[derive(Debug, Copy, Clone)]
enum Axis {
	Horizontal,
	Vertical,
}

#[derive(Debug, Copy, Clone)]
struct ImageView<'a> {
	raw: &'a [u8],
	width: usize,
	height: usize,
}

impl<'a> ImageView<'a> {
	fn from_gray_image(image: &'a GrayImage) -> Self {
		Self {
			raw: image.as_raw(),
			width: image.width() as usize,
			height: image.height() as usize,
		}
	}

	fn pixel_at(&self, x: usize, y: usize) -> u8 {
		self.raw[y * self.width + x]
	}

	fn column_pixel(&self, x: usize, y: usize) -> u8 {
		self.raw[y * self.width + x]
	}
}

struct EdgeSearchParams {
	start_x: f32,
	start_y: f32,
	perpendicular_length: f32,
	axis: Axis,
	search_distance: f32,
}

pub fn detect_board_scanline(image: &GrayImage) -> Option<DetectedBoard> {
	let img = ImageView::from_gray_image(image);
	let step = 8.max(img.height / 100);

	let mut best_candidate: Option<BoardCandidate> = None;

	for y in (0..img.height).step_by(step) {
		if let Some(candidate) = process_scanline(&img, y)
			&& best_candidate
				.as_ref()
				.is_none_or(|b| candidate.score > b.score)
		{
			best_candidate = Some(candidate);
		}
	}

	best_candidate.map(|c| c.board)
}

fn process_scanline(img: &ImageView, y: usize) -> Option<BoardCandidate> {
	let row_start = y * img.width;
	let row_pixels = &img.raw[row_start..row_start + img.width];
	let h_runs = get_run_lengths_with_pos(row_pixels)?;

	let pattern = find_horizontal_pattern(&h_runs)?;

	let total_width = pattern.end_x - pattern.start_x;
	if (total_width as f32) < (img.width as f32 * PARAMS.min_board_screen_ratio) {
		return None;
	}

	let vertical = scan_vertical_columns(img, &pattern)?;

	let board_ratio = (total_width as f32) / vertical.max_height;
	if (1.0 - board_ratio).abs() >= PARAMS.board_ratio_tolerance {
		return None;
	}

	let grid_confidence = verify_grid_pattern(
		img,
		pattern.start_x as f32,
		vertical.y_start,
		total_width as f32,
		vertical.max_height,
	);

	if grid_confidence < PARAMS.min_grid_confidence {
		return None;
	}

	let score = calculate_board_score(
		total_width as f32,
		board_ratio,
		grid_confidence,
		pattern.square_count,
	);

	let initial_rect = Rect::new(
		pattern.start_x as f32,
		vertical.y_start,
		total_width as f32,
		vertical.max_height,
	);

	let refined_rect = refine_board_corners(img, initial_rect);

	Some(BoardCandidate {
		board: DetectedBoard::new(refined_rect),
		score,
	})
}

fn find_horizontal_pattern(runs: &[Run]) -> Option<HorizontalPattern> {
	let span = find_and_expand_pattern(runs)?;
	Some(HorizontalPattern {
		start_x: span.start_x,
		end_x: span.end_x,
		avg_square_width: span.avg_width,
		square_count: span.count,
	})
}

fn scan_vertical_columns(
	img: &ImageView, pattern: &HorizontalPattern,
) -> Option<VerticalScanResult> {
	let total_width = pattern.end_x - pattern.start_x;
	let scan_percentages = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85];

	let mut max_valid_height = 0.0;
	let mut best_y_start = 0.0;
	let mut valid_columns = 0;

	for &pct in &scan_percentages {
		let check_x = pattern.start_x + (total_width as f32 * pct) as usize;
		if check_x >= img.width {
			continue;
		}

		let v_runs = get_column_run_lengths(img, check_x);
		let Some(v_runs) = v_runs else { continue };

		if let Some(span) = find_and_expand_pattern(&v_runs) {
			let col_height = (span.end_x - span.start_x) as f32;
			let square_ratio = pattern.avg_square_width / span.avg_width;

			if (PARAMS.square_aspect_min..=PARAMS.square_aspect_max).contains(&square_ratio) {
				valid_columns += 1;
				if col_height > max_valid_height {
					max_valid_height = col_height;
					best_y_start = span.start_x as f32;
				}
			}
		}
	}

	if valid_columns >= 2 {
		Some(VerticalScanResult {
			max_height: max_valid_height,
			y_start: best_y_start,
		})
	} else {
		None
	}
}

fn get_column_run_lengths(img: &ImageView, x: usize) -> Option<Vec<Run>> {
	if img.height == 0 {
		return None;
	}

	let mut runs = Vec::with_capacity(64);
	let mut current_run = 1usize;
	let mut last_val = img.column_pixel(x, 0);

	for y in 1..img.height {
		let val = img.column_pixel(x, y);
		let diff = (val as i16 - last_val as i16).abs();
		if diff > RUN_TRANSITION_THRESHOLD {
			if current_run >= PARAMS.min_square_size {
				runs.push(Run {
					len: current_run,
					x_pos: y,
				});
			}
			current_run = 0;
			last_val = val;
		} else {
			current_run += 1;
		}
	}
	if current_run >= PARAMS.min_square_size {
		runs.push(Run {
			len: current_run,
			x_pos: img.height,
		});
	}

	Some(runs)
}

fn calculate_board_score(
	total_width: f32, board_ratio: f32, grid_confidence: f32, square_count: usize,
) -> f32 {
	let squareness_factor = 1.0 - (1.0 - board_ratio).abs();
	let eight_by_eight_bonus = if square_count == BOARD_SIZE { 2.0 } else { 0.5 };
	total_width * grid_confidence * squareness_factor * eight_by_eight_bonus
}

const RUN_TRANSITION_THRESHOLD: i16 = 25;

fn get_run_lengths_with_pos(pixels: &[u8]) -> Option<Vec<Run>> {
	let first_pixel = *pixels.first()?;

	let mut runs = Vec::with_capacity(64);
	let mut current_run = 1usize;
	let mut last_val = first_pixel;

	for (i, &val) in pixels.iter().enumerate().skip(1) {
		let diff = (val as i16 - last_val as i16).abs();
		if diff > RUN_TRANSITION_THRESHOLD {
			if current_run >= PARAMS.min_square_size {
				runs.push(Run {
					len: current_run,
					x_pos: i,
				});
			}
			current_run = 0;
			last_val = val;
		} else {
			current_run += 1;
		}
	}
	if current_run >= PARAMS.min_square_size {
		runs.push(Run {
			len: current_run,
			x_pos: pixels.len(),
		});
	}
	Some(runs)
}

fn find_and_expand_pattern(runs: &[Run]) -> Option<PatternSpan> {
	if runs.len() < PARAMS.min_runs {
		return None;
	}

	let (start_idx, seed_avg) = find_seed_pattern(runs)?;

	let (current_start, current_end) = expand_pattern(runs, start_idx, seed_avg);

	let (final_start, final_end) = select_best_subsequence(runs, current_start, current_end)?;

	let final_start_x = runs[final_start]
		.x_pos
		.saturating_sub(runs[final_start].len);

	let final_end_x = runs[final_end].x_pos;
	let final_count = final_end - final_start + 1;

	let full_seq_sum: usize = runs[final_start..=final_end].iter().map(|r| r.len).sum();
	let final_avg = full_seq_sum as f32 / final_count as f32;

	Some(PatternSpan {
		start_x: final_start_x,
		end_x: final_end_x,
		avg_width: final_avg,
		count: final_count,
	})
}

fn find_seed_pattern(runs: &[Run]) -> Option<(usize, f32)> {
	for i in 0..=runs.len().saturating_sub(PARAMS.min_runs) {
		let window = &runs[i..i + PARAMS.min_runs];

		let (min_len, max_len, sum) = window
			.iter()
			.map(|r| r.len)
			.fold((usize::MAX, 0usize, 0usize), |(min, max, sum), len| {
				(min.min(len), max.max(len), sum + len)
			});

		if min_len == usize::MAX {
			continue;
		}

		let avg_len = sum as f32 / PARAMS.min_runs as f32;

		if (max_len as f32 - min_len as f32) < (avg_len * PARAMS.variance_tolerance) {
			return Some((i, avg_len));
		}
	}
	None
}

fn expand_pattern(runs: &[Run], start_idx: usize, seed_avg: f32) -> (usize, usize) {
	let mut current_start = start_idx;
	let mut current_end = start_idx + PARAMS.min_runs - 1;

	while current_start > 0 {
		let prev_run = &runs[current_start - 1];
		if (prev_run.len as f32 - seed_avg).abs() < (seed_avg * PARAMS.expansion_tolerance) {
			current_start -= 1;
		} else {
			break;
		}
	}

	while current_end < runs.len() - 1 {
		let next_run = &runs[current_end + 1];
		if (next_run.len as f32 - seed_avg).abs() < (seed_avg * PARAMS.expansion_tolerance) {
			current_end += 1;
		} else {
			break;
		}
	}

	(current_start, current_end)
}

fn select_best_subsequence(runs: &[Run], start: usize, end: usize) -> Option<(usize, usize)> {
	let count = end - start + 1;
	if count <= BOARD_SIZE {
		return Some((start, end));
	}

	let mut best_sub_start = start;
	let mut min_variance = f32::MAX;

	for i in start..=(end.saturating_sub(BOARD_SIZE - 1)) {
		let window = &runs[i..i + BOARD_SIZE];

		let (min_l, max_l) = window
			.iter()
			.map(|r| r.len)
			.fold((usize::MAX, 0usize), |(min, max), len| {
				(min.min(len), max.max(len))
			});

		if min_l == usize::MAX {
			continue;
		}

		let variance = max_l as f32 - min_l as f32;

		if variance < min_variance {
			min_variance = variance;
			best_sub_start = i;
		}
	}

	Some((best_sub_start, best_sub_start + BOARD_SIZE - 1))
}

const GRID_DIFF_THRESHOLD: i16 = 15;

fn verify_grid_pattern(img: &ImageView, x: f32, y: f32, width: f32, height: f32) -> f32 {
	let cell_width = width / BOARD_SIZE as f32;
	let cell_height = height / BOARD_SIZE as f32;

	let mut cell_values = [[0u8; BOARD_SIZE]; BOARD_SIZE];

	for (row, cell_row) in cell_values.iter_mut().enumerate() {
		for (col, cell) in cell_row.iter_mut().enumerate() {
			let center_x = (x + (col as f32 + 0.5) * cell_width) as usize;
			let center_y = (y + (row as f32 + 0.5) * cell_height) as usize;

			if center_x >= img.width || center_y >= img.height {
				return 0.0;
			}

			*cell = img.pixel_at(center_x, center_y);
		}
	}

	let pattern_confidence = calculate_pattern_confidence(&cell_values);
	let adjacent_confidence = calculate_adjacent_confidence(&cell_values);

	(pattern_confidence + adjacent_confidence) / 2.0
}

fn calculate_pattern_confidence(cell_values: &[[u8; BOARD_SIZE]; BOARD_SIZE]) -> f32 {
	let mut all_values: Vec<u8> = cell_values.iter().flatten().copied().collect();
	all_values.sort_unstable();
	let median = all_values[BOARD_SIZE * BOARD_SIZE / 2];

	let mut pattern_a_valid = 0;
	let mut pattern_b_valid = 0;

	for (row, cell_row) in cell_values.iter().enumerate() {
		for (col, &val) in cell_row.iter().enumerate() {
			let is_light = val > median;
			let expect_light_a = (row + col) % 2 == 0;
			let expect_light_b = (row + col) % 2 == 1;

			if is_light == expect_light_a {
				pattern_a_valid += 1;
			}
			if is_light == expect_light_b {
				pattern_b_valid += 1;
			}
		}
	}

	let best_valid = pattern_a_valid.max(pattern_b_valid);
	best_valid as f32 / (BOARD_SIZE * BOARD_SIZE) as f32
}

fn calculate_adjacent_confidence(cell_values: &[[u8; BOARD_SIZE]; BOARD_SIZE]) -> f32 {
	let mut adjacent_differ = 0;
	let mut adjacent_total = 0;

	for cell_row in cell_values {
		for col in 0..(BOARD_SIZE - 1) {
			let diff = (cell_row[col] as i16 - cell_row[col + 1] as i16).abs();
			if diff > GRID_DIFF_THRESHOLD {
				adjacent_differ += 1;
			}
			adjacent_total += 1;
		}
	}

	for row in 0..(BOARD_SIZE - 1) {
		for (col, &val) in cell_values[row].iter().enumerate() {
			let diff = (val as i16 - cell_values[row + 1][col] as i16).abs();
			if diff > GRID_DIFF_THRESHOLD {
				adjacent_differ += 1;
			}
			adjacent_total += 1;
		}
	}

	adjacent_differ as f32 / adjacent_total as f32
}

const CORNER_SEARCH_MARGIN: f32 = 0.05;
const EDGE_GRADIENT_THRESHOLD: i16 = 20;
const EDGE_SAMPLE_POSITIONS: [f32; 5] = [0.2, 0.35, 0.5, 0.65, 0.8];

fn refine_board_corners(img: &ImageView, rect: Rect) -> Rect {
	let margin_x = rect.width() * CORNER_SEARCH_MARGIN;
	let margin_y = rect.height() * CORNER_SEARCH_MARGIN;

	let left = refine_edge_with_samples(
		img,
		EdgeSearchParams {
			start_x: rect.x(),
			start_y: rect.y(),
			perpendicular_length: rect.height(),
			axis: Axis::Horizontal,
			search_distance: margin_x,
		},
		-1.0,
	);

	let right = refine_edge_with_samples(
		img,
		EdgeSearchParams {
			start_x: rect.x() + rect.width(),
			start_y: rect.y(),
			perpendicular_length: rect.height(),
			axis: Axis::Horizontal,
			search_distance: margin_x,
		},
		1.0,
	);

	let top = refine_edge_with_samples(
		img,
		EdgeSearchParams {
			start_x: rect.x(),
			start_y: rect.y(),
			perpendicular_length: rect.width(),
			axis: Axis::Vertical,
			search_distance: margin_y,
		},
		-1.0,
	);

	let bottom = refine_edge_with_samples(
		img,
		EdgeSearchParams {
			start_x: rect.x(),
			start_y: rect.y() + rect.height(),
			perpendicular_length: rect.width(),
			axis: Axis::Vertical,
			search_distance: margin_y,
		},
		1.0,
	);

	let new_x = left;
	let new_y = top;
	let new_width = right - left;
	let new_height = bottom - top;

	if new_width <= 0.0 || new_height <= 0.0 {
		return rect;
	}

	let aspect_ratio = new_width / new_height;
	if !(0.9..=1.1).contains(&aspect_ratio) {
		let size = new_width.min(new_height);
		let center_x = new_x + new_width / 2.0;
		let center_y = new_y + new_height / 2.0;
		return Rect::new(center_x - size / 2.0, center_y - size / 2.0, size, size);
	}

	Rect::new(new_x, new_y, new_width, new_height)
}

fn refine_edge_with_samples(img: &ImageView, params: EdgeSearchParams, dir_sign: f32) -> f32 {
	let mut positions: Vec<f32> = Vec::with_capacity(EDGE_SAMPLE_POSITIONS.len());

	for &pct in &EDGE_SAMPLE_POSITIONS {
		let (sample_x, sample_y) = match params.axis {
			Axis::Horizontal => (
				params.start_x,
				params.start_y + params.perpendicular_length * pct,
			),
			Axis::Vertical => (
				params.start_x + params.perpendicular_length * pct,
				params.start_y,
			),
		};

		let pos = refine_edge_position(
			img,
			sample_x,
			sample_y,
			params.axis,
			dir_sign,
			params.search_distance,
		);
		positions.push(pos);
	}

	if positions.is_empty() {
		return match params.axis {
			Axis::Horizontal => params.start_x,
			Axis::Vertical => params.start_y,
		};
	}

	positions.sort_by(|a, b| a.total_cmp(b));
	positions[positions.len() / 2]
}

fn refine_edge_position(
	img: &ImageView, start_x: f32, start_y: f32, axis: Axis, dir_sign: f32, search_distance: f32,
) -> f32 {
	let steps = search_distance as usize;
	let default_position = match axis {
		Axis::Horizontal => start_x,
		Axis::Vertical => start_y,
	};

	for i in 0..steps {
		let offset = i as f32;

		let (x_pos, y_pos) = match axis {
			Axis::Horizontal => (start_x + dir_sign * offset, start_y),
			Axis::Vertical => (start_x, start_y + dir_sign * offset),
		};
		if let Some(pos) = check_gradient_at(img, x_pos, y_pos, axis) {
			return pos;
		}

		if i > 0 {
			let (x_neg, y_neg) = match axis {
				Axis::Horizontal => (start_x - dir_sign * offset, start_y),
				Axis::Vertical => (start_x, start_y - dir_sign * offset),
			};
			if let Some(pos) = check_gradient_at(img, x_neg, y_neg, axis) {
				return pos;
			}
		}
	}

	default_position
}

fn check_gradient_at(img: &ImageView, x: f32, y: f32, axis: Axis) -> Option<f32> {
	let xi = x as usize;
	let yi = y as usize;

	if xi < 1 || xi >= img.width - 1 || yi < 1 || yi >= img.height - 1 {
		return None;
	}

	let gradient = match axis {
		Axis::Horizontal => compute_horizontal_gradient(img, xi, yi),
		Axis::Vertical => compute_vertical_gradient(img, xi, yi),
	};

	if gradient > EDGE_GRADIENT_THRESHOLD {
		Some(match axis {
			Axis::Horizontal => x,
			Axis::Vertical => y,
		})
	} else {
		None
	}
}

fn compute_horizontal_gradient(img: &ImageView, x: usize, y: usize) -> i16 {
	let left = img.pixel_at(x - 1, y) as i16;
	let right = img.pixel_at(x + 1, y) as i16;
	(right - left).abs()
}

fn compute_vertical_gradient(img: &ImageView, x: usize, y: usize) -> i16 {
	let top = img.pixel_at(x, y - 1) as i16;
	let bottom = img.pixel_at(x, y + 1) as i16;
	(bottom - top).abs()
}
