use image::GrayImage;

use crate::model::detected::DetectedBoard;

const MIN_SQUARE_SIZE: usize = 4;
const MIN_RUNS: usize = 5;

const VARIANCE_TOLERANCE: f32 = 0.30;
const EXPANSION_TOLERANCE: f32 = 0.25;
const BOARD_RATIO_TOLERANCE: f32 = 0.20;
const MIN_BOARD_SCREEN_RATIO: f32 = 0.10;

#[derive(Debug, Clone, Copy)]
struct Run {
	len: usize,
	x_pos: usize,
}

pub fn detect_board_scanline(image: &GrayImage) -> Option<DetectedBoard> {
	let width = image.width() as usize;
	let height = image.height() as usize;

	let step = 8.max(height / 100);

	let mut best_candidate: Option<DetectedBoard> = None;
	let mut best_score = 0.0;

	for y in (0..height).step_by(step) {
		let row_start = y * width;
		let row_pixels = &image.as_raw()[row_start..row_start + width];

		let h_runs = get_run_lengths_with_pos(row_pixels);

		if let Some((start_x, end_x, avg_sq_w, sq_count_h)) = find_and_expand_pattern(&h_runs) {
			let total_width = end_x - start_x;
			if (total_width as f32) < (width as f32 * MIN_BOARD_SCREEN_RATIO) {
				continue;
			}

			let mut max_valid_height = 0.0;
			let mut best_y_start = 0.0;
			let mut valid_columns = 0;

			let scan_percentages = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85];

			for &pct in &scan_percentages {
				let check_x = start_x + (total_width as f32 * pct) as usize;
				if check_x >= width {
					continue;
				}

				let col_pixels: Vec<u8> = (0..height)
					.map(|y_idx| image.as_raw()[y_idx * width + check_x])
					.collect();

				let v_runs = get_run_lengths_with_pos(&col_pixels);

				if let Some((start_y, end_y, avg_sq_h, _)) = find_and_expand_pattern(&v_runs) {
					let col_height = (end_y - start_y) as f32;
					let square_ratio = avg_sq_w / avg_sq_h;

					if (0.7..=1.4).contains(&square_ratio) {
						valid_columns += 1;

						if col_height > max_valid_height {
							max_valid_height = col_height;
							best_y_start = start_y as f32;
						}
					}
				}
			}

			if valid_columns >= 2 {
				let board_ratio = (total_width as f32) / max_valid_height;

				if (1.0 - board_ratio).abs() < BOARD_RATIO_TOLERANCE {
					let squareness_bonus = if sq_count_h == 8 { 1000.0 } else { 0.0 };
					let score = (total_width as f32) + squareness_bonus;

					if best_candidate.is_none() || score > best_score {
						best_score = score;
						best_candidate = Some(DetectedBoard {
							x: start_x as f32,
							y: best_y_start,
							width: total_width as f32,
							height: max_valid_height,
							playing_as_white: true,
						});
					}
				}
			}
		}
	}

	best_candidate
}

fn get_run_lengths_with_pos(pixels: &[u8]) -> Vec<Run> {
	let mut runs = Vec::with_capacity(64);
	let mut current_run = 0;
	let mut last_val = pixels[0];
	let threshold = 25;

	for (i, &val) in pixels.iter().enumerate().skip(1) {
		let diff = (val as i16 - last_val as i16).abs();
		if diff > threshold {
			if current_run >= MIN_SQUARE_SIZE {
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
	if current_run >= MIN_SQUARE_SIZE {
		runs.push(Run {
			len: current_run,
			x_pos: pixels.len(),
		});
	}
	runs
}

fn find_and_expand_pattern(runs: &[Run]) -> Option<(usize, usize, f32, usize)> {
	if runs.len() < MIN_RUNS {
		return None;
	}

	let mut seed_idx = None;
	let mut seed_avg = 0.0;

	for i in 0..=runs.len() - MIN_RUNS {
		let window = &runs[i..i + MIN_RUNS];

		let min_len = window.iter().map(|r| r.len).min().unwrap();
		let max_len = window.iter().map(|r| r.len).max().unwrap();
		let avg_len = window.iter().map(|r| r.len).sum::<usize>() as f32 / MIN_RUNS as f32;

		if (max_len as f32 - min_len as f32) < (avg_len * VARIANCE_TOLERANCE) {
			seed_idx = Some(i);
			seed_avg = avg_len;
			break;
		}
	}

	let start_idx = seed_idx?;
	let mut current_start = start_idx;
	let mut current_end = start_idx + MIN_RUNS - 1;

	while current_start > 0 {
		let prev_run = &runs[current_start - 1];
		if (prev_run.len as f32 - seed_avg).abs() < (seed_avg * EXPANSION_TOLERANCE) {
			current_start -= 1;
		} else {
			break;
		}
	}

	while current_end < runs.len() - 1 {
		let next_run = &runs[current_end + 1];
		if (next_run.len as f32 - seed_avg).abs() < (seed_avg * EXPANSION_TOLERANCE) {
			current_end += 1;
		} else {
			break;
		}
	}

	let count = current_end - current_start + 1;
	if count > 8 {
		let mut best_sub_start = current_start;
		let mut min_variance = f32::MAX;

		for i in current_start..=(current_end - 7) {
			let window = &runs[i..i + 8];
			let max_l = window.iter().map(|r| r.len).max().unwrap() as f32;
			let min_l = window.iter().map(|r| r.len).min().unwrap() as f32;
			let variance = max_l - min_l;

			if variance < min_variance {
				min_variance = variance;
				best_sub_start = i;
			}
		}
		current_start = best_sub_start;
		current_end = best_sub_start + 7;
	}

	let final_start_x = if current_start == 0 {
		runs[0].x_pos.saturating_sub(runs[0].len)
	} else {
		runs[current_start - 1].x_pos
	};

	let final_end_x = runs[current_end].x_pos;
	let final_count = current_end - current_start + 1;

	let full_seq_sum: usize = runs[current_start..=current_end]
		.iter()
		.map(|r| r.len)
		.sum();
	let final_avg = full_seq_sum as f32 / final_count as f32;

	Some((final_start_x, final_end_x, final_avg, final_count))
}
