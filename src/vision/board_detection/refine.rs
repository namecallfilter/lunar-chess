use image::{RgbImage, RgbaImage, imageops::blur};
use tracing::{debug, trace};

use crate::{
	model::detected::{DetectedBoard, Rect},
	vision::board_detection::{
		edges::{EdgeMap, sobel_edge_detection},
		grid::{
			cluster_lines_full, compute_vertical_span, filter_horizontals_by_vertical_span,
			find_best_grid,
		},
		hough::{HoughAccumulator, HoughLine, detect_peaks, hough_voting},
		scoring::calculate_grid_score,
	},
};

const BELL_FLATTENING_LEVEL: f32 = 1.0;
const EDGE_THRESHOLD: u16 = 40;

pub fn detect_board_hough(image: &RgbaImage) -> Option<DetectedBoard> {
	let rgb_image = RgbImage::from_fn(image.width(), image.height(), |x, y| {
		let p = image.get_pixel(x, y);
		image::Rgb([p[0], p[1], p[2]])
	});

	let width = rgb_image.width() as usize;
	let height = rgb_image.height() as usize;

	if width == 0 || height == 0 {
		return None;
	}

	let blurred_image = blur(&rgb_image, BELL_FLATTENING_LEVEL);

	let edge_map = sobel_edge_detection(&blurred_image, EDGE_THRESHOLD);

	let mut accumulator = HoughAccumulator::new(width, height)?;
	hough_voting(&edge_map, &mut accumulator);

	let vote_threshold = ((width.min(height) / 10).max(20)) as u32;
	let max_lines = 200;

	let lines_raw = detect_peaks(&accumulator, vote_threshold, max_lines);
	debug!(
		"Detected {} raw Hough lines (threshold: {})",
		lines_raw.len(),
		vote_threshold
	);

	if lines_raw.is_empty() {
		return None;
	}

	let rect = extract_rect_from_lines_with_segments(&lines_raw, &edge_map)?;

	let refined_rect = refine_rect_with_edges(rect, &edge_map);
	debug!("Refined rect from {:?} to {:?}", rect, refined_rect);

	let rect = refined_rect;

	let aspect_ratio = rect.width() / rect.height();
	if !(0.7..=1.3).contains(&aspect_ratio) {
		debug!("Board rejected due to aspect ratio: {:.2}", aspect_ratio);
		return None;
	}

	let min_dimension = (width.min(height) as f32) * 0.1;
	if rect.width() < min_dimension || rect.height() < min_dimension {
		debug!(
			"Board rejected due to size: {:.0}x{:.0} < min {:.0}",
			rect.width(),
			rect.height(),
			min_dimension
		);
		return None;
	}

	Some(DetectedBoard::new(rect))
}

fn extract_rect_from_lines_with_segments(lines: &[HoughLine], edge_map: &EdgeMap) -> Option<Rect> {
	if lines.is_empty() {
		trace!("No lines to extract rect from");
		return None;
	}

	let (mut vertical_lines, mut horizontal_lines) = cluster_lines_full(lines);

	trace!(
		"Clustered lines: {} vertical, {} horizontal",
		vertical_lines.len(),
		horizontal_lines.len()
	);

	vertical_lines.sort_by(|a, b| a.rho.total_cmp(&b.rho));
	horizontal_lines.sort_by(|a, b| a.rho.total_cmp(&b.rho));

	let v_grid = match find_best_grid(&vertical_lines, None) {
		Some(g) => g,
		None => {
			debug!("Failed to find vertical grid pattern");
			return None;
		}
	};

	let v_span = compute_vertical_span(&vertical_lines, edge_map, &v_grid);

	let (mut horizontal_candidates, expected_h_spacing) = if let Some(v_span) = v_span {
		let filtered =
			filter_horizontals_by_vertical_span(&horizontal_lines, edge_map, v_span, &v_grid);

		let measured_height = v_span.1 - v_span.0;
		let spacing_from_span = measured_height / 8.0;

		(filtered, spacing_from_span)
	} else {
		(horizontal_lines, v_grid.spacing)
	};

	horizontal_candidates.sort_by(|a, b| a.rho.total_cmp(&b.rho));

	let h_grid = match find_best_grid(&horizontal_candidates, Some(expected_h_spacing)) {
		Some(g) => g,
		None => {
			debug!("Failed to find horizontal grid pattern");
			return None;
		}
	};

	let spacing_diff = (v_grid.spacing - h_grid.spacing).abs();
	if spacing_diff > v_grid.spacing * 0.1 {
		debug!(
			"Grid spacing mismatch: v={:.1}, h={:.1}, diff={:.1}",
			v_grid.spacing, h_grid.spacing, spacing_diff
		);
		return None;
	}

	let mut width = v_grid.end - v_grid.start;
	let mut height = h_grid.end - h_grid.start;
	let mut x = v_grid.start;
	let mut y = h_grid.start;

	let v_intervals = ((v_grid.end - v_grid.start) / v_grid.spacing).round() as usize;
	let h_intervals = ((h_grid.end - h_grid.start) / h_grid.spacing).round() as usize;

	if v_intervals == h_intervals {
		let avg_size = (width + height) / 2.0;
		let center_x = x + width / 2.0;
		let center_y = y + height / 2.0;

		width = avg_size;
		height = avg_size;
		x = center_x - width / 2.0;
		y = center_y - height / 2.0;
	} else if v_intervals == 7 && h_intervals == 8 {
		debug!("Fixing 7x8 grid by guessing missing vertical line");
		let cand_w = width + v_grid.spacing;

		let rect_right = Rect::new(x, y, cand_w, height);
		let rect_left = Rect::new(x - v_grid.spacing, y, cand_w, height);

		let score_right = calculate_grid_score(&rect_right, edge_map);
		let score_left = calculate_grid_score(&rect_left, edge_map);

		debug!(
			"  Right extension score: {:.3}, Left: {:.3}",
			score_right, score_left
		);

		if score_right >= score_left {
			width = cand_w;
		} else {
			x -= v_grid.spacing;
			width = cand_w;
		}
	} else if v_intervals == 8 && h_intervals == 7 {
		debug!("Fixing 8x7 grid by guessing missing horizontal line");
		let cand_h = height + h_grid.spacing;

		let rect_bottom = Rect::new(x, y, width, cand_h);
		let rect_top = Rect::new(x, y - h_grid.spacing, width, cand_h);

		let score_bottom = calculate_grid_score(&rect_bottom, edge_map);
		let score_top = calculate_grid_score(&rect_top, edge_map);

		debug!(
			"  Bottom extension score: {:.3}, Top: {:.3}",
			score_bottom, score_top
		);

		if score_bottom >= score_top {
			height = cand_h;
		} else {
			y -= h_grid.spacing;
			height = cand_h;
		}
	}

	Some(Rect::new(x.max(0.0), y.max(0.0), width, height))
}

const MAX_ITERS_PER_STEP: usize = 10;
const MIN_SCORE_DELTA: f32 = 1e-4;

fn refine_rect_with_edges(initial_rect: Rect, edge_map: &EdgeMap) -> Rect {
	let mut best = initial_rect;
	let mut best_score = calculate_grid_score(&best, edge_map);

	let steps = [4.0_f32, 2.0_f32, 1.0_f32, 0.5_f32];

	let init_aspect = if initial_rect.height() > 0.0 {
		initial_rect.width() / initial_rect.height()
	} else {
		0.0
	};

	let keep_square = (0.95..=1.05).contains(&init_aspect);

	let map_w = edge_map.width() as f32;
	let map_h = edge_map.height() as f32;

	for &step in &steps {
		let mut made_improvement_this_step = true;
		let mut iter_count = 0usize;

		while made_improvement_this_step && iter_count < MAX_ITERS_PER_STEP {
			iter_count += 1;
			made_improvement_this_step = false;

			for &dx in &[-step, 0.0_f32, step] {
				for &dy in &[-step, 0.0_f32, step] {
					if keep_square {
						for &d_size in &[-step, 0.0_f32, step] {
							if dx == 0.0 && dy == 0.0 && d_size == 0.0 {
								continue;
							}

							let mut cand_x = (best.x() + dx).max(0.0);
							let mut cand_y = (best.y() + dy).max(0.0);
							let mut cand_w = (best.width() + d_size).max(1.0);
							let mut cand_h = (best.height() + d_size).max(1.0);

							cand_w = cand_w.clamp(1.0, map_w - cand_x);
							cand_h = cand_h.clamp(1.0, map_h - cand_y);

							cand_x = cand_x.clamp(0.0, (map_w - cand_w).max(0.0));
							cand_y = cand_y.clamp(0.0, (map_h - cand_h).max(0.0));

							let cand = Rect::new(cand_x, cand_y, cand_w, cand_h);
							let score = calculate_grid_score(&cand, edge_map);

							if score > best_score + MIN_SCORE_DELTA {
								best_score = score;
								best = cand;
								made_improvement_this_step = true;
								break;
							}
						}
					} else {
						for &dw in &[-step, 0.0_f32, step] {
							for &dh in &[-step, 0.0_f32, step] {
								if dx == 0.0 && dy == 0.0 && dw == 0.0 && dh == 0.0 {
									continue;
								}

								let mut cand_x = (best.x() + dx).max(0.0);
								let mut cand_y = (best.y() + dy).max(0.0);
								let mut cand_w = (best.width() + dw).max(1.0);
								let mut cand_h = (best.height() + dh).max(1.0);

								cand_w = cand_w.clamp(1.0, map_w - cand_x);
								cand_h = cand_h.clamp(1.0, map_h - cand_y);

								cand_x = cand_x.clamp(0.0, (map_w - cand_w).max(0.0));
								cand_y = cand_y.clamp(0.0, (map_h - cand_h).max(0.0));

								let cand = Rect::new(cand_x, cand_y, cand_w, cand_h);
								let score = calculate_grid_score(&cand, edge_map);

								if score > best_score + MIN_SCORE_DELTA {
									best_score = score;
									best = cand;
									made_improvement_this_step = true;
									break;
								}
							}
							if made_improvement_this_step {
								break;
							}
						}
					}
					if made_improvement_this_step {
						break;
					}
				}
				if made_improvement_this_step {
					break;
				}
			}
		}
	}

	best
}
