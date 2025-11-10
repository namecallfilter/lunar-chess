use image::{GrayImage, Rgb, RgbImage, RgbaImage};

use super::{
	edge_detection::EdgeDetector,
	grayscale::to_grayscale,
	line_detection::{HoughLine, LineDetector},
};
use crate::{config::CONFIG, drawing::DetectedBoard};

#[derive(Debug, Clone)]
pub struct Grid {
	pub horizontal_lines: Vec<HoughLine>,
	pub vertical_lines: Vec<HoughLine>,
	pub corners: [(f32, f32); 4],
}

impl Grid {
	pub fn is_valid_chessboard(&self) -> bool {
		self.horizontal_lines.len() >= 9 && self.vertical_lines.len() >= 9
	}

	pub fn bounding_box(&self) -> (f32, f32, f32, f32) {
		let xs: Vec<f32> = self.corners.iter().map(|(x, _)| *x).collect();
		let ys: Vec<f32> = self.corners.iter().map(|(_, y)| *y).collect();

		let min_x = xs.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
		let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
		let min_y = ys.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
		let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));

		(min_x, min_y, max_x - min_x, max_y - min_y)
	}

	pub fn to_detected_board(&self, confidence: f32) -> DetectedBoard {
		let (x, y, width, height) = self.bounding_box();

		DetectedBoard {
			x,
			y,
			width,
			height,
			confidence,
			playing_as_white: true,
		}
	}
}

pub fn detect_board_algorithmic(image: &RgbaImage) -> Option<DetectedBoard> {
	let width = image.width() as usize;
	let height = image.height() as usize;

	let gray = to_grayscale(image);

	if CONFIG.debugging.save_images {
		save_grayscale_debug(&gray, width, height, "debug_01_grayscale.png");
	}

	let edge_detector = EdgeDetector::new(width, height);
	let strong_edge_threshold = CONFIG.detection.edge_threshold * 3.0;
	let edges = edge_detector.simple_edges(&gray, strong_edge_threshold);

	let dilated_edges = dilate_edges(&edges, width, height, 1);

	if CONFIG.debugging.save_images {
		save_edges_debug(&dilated_edges, width, height, "debug_02_edges_strong.png");
	}

	let line_detector = LineDetector::new(width, height);
	let (horizontal_lines, vertical_lines) =
		line_detector.detect_orthogonal_lines(&dilated_edges, CONFIG.detection.hough_threshold);

	tracing::debug!(
		"Detected {} horizontal and {} vertical lines",
		horizontal_lines.len(),
		vertical_lines.len()
	);

	let h_clustered = line_detector.cluster_lines(&horizontal_lines, 50.0);
	let v_clustered = line_detector.cluster_lines(&vertical_lines, 50.0);

	tracing::debug!(
		"After clustering: {} horizontal and {} vertical lines",
		h_clustered.len(),
		v_clustered.len()
	);

	if CONFIG.debugging.save_images {
		save_lines_debug(
			image,
			&h_clustered,
			&[],
			width,
			height,
			"debug_03a_horizontal_lines.png",
		);
		save_lines_debug(
			image,
			&[],
			&v_clustered,
			width,
			height,
			"debug_03b_vertical_lines.png",
		);
	}

	let grid = find_chess_grid(&h_clustered, &v_clustered, width, height, image)?;

	if CONFIG.debugging.save_images {
		save_grid_debug(image, &grid, "debug_04_grid.png");
	}

	if grid.is_valid_chessboard() {
		let confidence = calculate_grid_confidence(&grid, width, height);
		Some(grid.to_detected_board(confidence))
	} else {
		tracing::debug!("Grid validation failed: not a valid 8x8 chessboard");
		None
	}
}

fn find_chess_grid(
	horizontal: &[HoughLine], vertical: &[HoughLine], width: usize, height: usize,
	image: &RgbaImage,
) -> Option<Grid> {
	let h_grid_result = find_evenly_spaced_lines(horizontal, 9, true, width, height);
	let v_grid_result_initial = if let Some(ref h_grid) = h_grid_result {
		let h_span = h_grid.last()?.rho - h_grid.first()?.rho;
		find_evenly_spaced_lines_with_target_span(vertical, 9, false, width, height, h_span)
	} else {
		None
	};

	if let (Some(h_grid), Some(v_grid)) = (h_grid_result, v_grid_result_initial) {
		let h_span = h_grid.last()?.rho - h_grid.first()?.rho;
		let v_span = v_grid.last()?.rho - v_grid.first()?.rho;
		let size_ratio = h_span.max(v_span) / h_span.min(v_span);

		tracing::debug!(
			"Complete grid found: h_span={:.1}, v_span={:.1}, ratio={:.2}",
			h_span,
			v_span,
			size_ratio
		);

		if size_ratio <= 1.10 {
			return build_grid(h_grid, v_grid, image, width, height);
		}
	}

	tracing::debug!("Attempting to find best 9x9 grid from all detected lines...");

	let mut best_grid: Option<(Vec<HoughLine>, Vec<HoughLine>, f32)> = None;
	let mut best_combined_score = f32::MIN;

	if let Some(h_candidates) = get_all_grid_subsets(horizontal, 9, true) {
		for (h_grid, h_score) in h_candidates {
			let h_span = h_grid.last()?.rho - h_grid.first()?.rho;

			if let Some(v_candidates) = get_all_grid_subsets(vertical, 9, false) {
				for (v_grid, v_score) in &v_candidates {
					let v_span = v_grid.last()?.rho - v_grid.first()?.rho;
					let size_ratio = h_span.max(v_span) / h_span.min(v_span);

					if size_ratio <= 1.10 {
						let combined_score = h_score + v_score;

						if combined_score > best_combined_score {
							best_combined_score = combined_score;
							best_grid = Some((h_grid.clone(), v_grid.clone(), combined_score));

							tracing::debug!(
								"Found square grid candidate: h_span={:.1}, v_span={:.1}, ratio={:.2}, score={:.3}",
								h_span,
								v_span,
								size_ratio,
								combined_score
							);
						}
					}
				}
			}
		}
	}

	if let Some((h_grid, v_grid, score)) = best_grid {
		let h_span = h_grid.last()?.rho - h_grid.first()?.rho;
		let v_span = v_grid.last()?.rho - v_grid.first()?.rho;

		tracing::debug!(
			"Selected best square grid: h_span={:.1}, v_span={:.1}, score={:.3}",
			h_span,
			v_span,
			score
		);

		build_grid(h_grid, v_grid, image, width, height)
	} else {
		tracing::debug!("No square grid found from subsets");
		None
	}
}

fn build_grid(
	h_grid: Vec<HoughLine>, v_grid: Vec<HoughLine>, image: &RgbaImage, width: usize, height: usize,
) -> Option<Grid> {
	if CONFIG.debugging.save_images {
		let vis_img = image.clone();
		save_lines_debug(
			&vis_img,
			&h_grid,
			&v_grid,
			width,
			height,
			"debug_03c_selected_grid_lines.png",
		);
	}

	let corners = calculate_corners(&h_grid, &v_grid)?;

	let (_, _, board_width, board_height) = get_bbox_from_corners(&corners);

	let aspect_ratio = board_width / board_height;
	if !(0.8..=1.2).contains(&aspect_ratio) {
		tracing::debug!("Aspect ratio validation failed: {}", aspect_ratio);
		return None;
	}

	Some(Grid {
		horizontal_lines: h_grid,
		vertical_lines: v_grid,
		corners,
	})
}

fn find_evenly_spaced_lines(
	lines: &[HoughLine], n: usize, is_horizontal: bool, _width: usize, _height: usize,
) -> Option<Vec<HoughLine>> {
	if lines.len() < n {
		tracing::debug!(
			"Not enough {} lines: need {}, have {}",
			if is_horizontal {
				"horizontal"
			} else {
				"vertical"
			},
			n,
			lines.len()
		);
		return None;
	}

	let mut sorted_lines = lines.to_vec();
	sorted_lines.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

	tracing::debug!(
		"Finding {} evenly-spaced {} lines from {} candidates",
		n,
		if is_horizontal {
			"horizontal"
		} else {
			"vertical"
		},
		sorted_lines.len()
	);

	let mut lines_by_strength = sorted_lines.clone();
	lines_by_strength.sort_by(|a, b| b.votes.cmp(&a.votes));

	let vote_threshold_ratio = if sorted_lines.len() <= n + 5 {
		0.65
	} else {
		0.75
	};

	let vote_threshold = if let Some(strongest) = lines_by_strength.first() {
		(strongest.votes as f32 * vote_threshold_ratio) as usize
	} else {
		return None;
	};

	let mut strong_lines: Vec<HoughLine> = sorted_lines
		.iter()
		.filter(|line| line.votes >= vote_threshold)
		.copied()
		.collect();

	strong_lines.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

	tracing::debug!(
		"Filtered to {} strong lines (vote threshold: {} @ {:.0}%)",
		strong_lines.len(),
		vote_threshold,
		vote_threshold_ratio * 100.0
	);

	if strong_lines.len() < n {
		tracing::debug!(
			"Not enough strong lines: need {}, have {}",
			n,
			strong_lines.len()
		);
		return None;
	}

	let min_rho = strong_lines.first()?.rho;
	let max_rho = strong_lines.last()?.rho;
	let total_range = max_rho - min_rho;

	tracing::debug!(
		"Rho range: {:.1} to {:.1} (span: {:.1})",
		min_rho,
		max_rho,
		total_range
	);

	let expected_spacing = total_range / (n - 1) as f32;
	let min_spacing = expected_spacing * 0.5;

	let segment_size = total_range / (n - 1) as f32;
	let mut selected_lines: Vec<HoughLine> = Vec::new();

	for i in 0..n {
		let target_rho = min_rho + i as f32 * segment_size;
		let tolerance = segment_size * 0.6;

		let best_in_segment = strong_lines
			.iter()
			.filter(|line| {
				let distance_to_target = (line.rho - target_rho).abs();
				if distance_to_target > tolerance {
					return false;
				}

				for selected in &selected_lines {
					if (line.rho - selected.rho).abs() < min_spacing {
						return false;
					}
				}

				true
			})
			.max_by_key(|line| line.votes);

		if let Some(line) = best_in_segment {
			selected_lines.push(*line);
		} else {
			tracing::debug!(
				"No valid line found near target rho {:.1} (±{:.1}, min_spacing: {:.1})",
				target_rho,
				tolerance,
				min_spacing
			);
			return None;
		}
	}

	selected_lines.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

	if selected_lines.len() != n {
		tracing::debug!("Selected {} lines, but need {}", selected_lines.len(), n);
		return None;
	}

	let score = calculate_spacing_score(&selected_lines);

	tracing::debug!(
		"Selected evenly-spaced set: score={:.2}, rho values: {:?}",
		score,
		selected_lines
			.iter()
			.map(|l| l.rho as i32)
			.collect::<Vec<_>>()
	);

	Some(selected_lines)
}

fn find_evenly_spaced_lines_with_target_span(
	lines: &[HoughLine], n: usize, is_horizontal: bool, _width: usize, _height: usize,
	target_span: f32,
) -> Option<Vec<HoughLine>> {
	if lines.len() < n {
		return None;
	}

	let mut sorted_lines = lines.to_vec();
	sorted_lines.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

	let mut lines_by_strength = sorted_lines.clone();
	lines_by_strength.sort_by(|a, b| b.votes.cmp(&a.votes));

	let vote_threshold_ratio = if sorted_lines.len() <= n + 5 {
		0.65
	} else {
		0.75
	};
	let vote_threshold = if let Some(strongest) = lines_by_strength.first() {
		(strongest.votes as f32 * vote_threshold_ratio) as usize
	} else {
		return None;
	};

	let strong_lines: Vec<HoughLine> = sorted_lines
		.iter()
		.filter(|line| line.votes >= vote_threshold)
		.copied()
		.collect();

	if strong_lines.len() < n {
		return None;
	}

	let mut best_set: Option<Vec<HoughLine>> = None;
	let mut best_span_error = f32::MAX;

	for start_idx in 0..=(strong_lines.len().saturating_sub(n)) {
		let candidate_set: Vec<HoughLine> = strong_lines[start_idx..start_idx + n].to_vec();

		let span = candidate_set.last().unwrap().rho - candidate_set.first().unwrap().rho;
		let span_error = (span - target_span).abs();

		let spacing_score = calculate_spacing_score(&candidate_set);

		let total_error = span_error + spacing_score;

		if total_error < best_span_error {
			best_span_error = total_error;
			best_set = Some(candidate_set);
		}
	}

	if let Some(selected) = &best_set {
		let span = selected.last()?.rho - selected.first()?.rho;
		tracing::debug!(
			"Selected {} lines with span {:.1} (target: {:.1}, error: {:.1})",
			if is_horizontal {
				"horizontal"
			} else {
				"vertical"
			},
			span,
			target_span,
			(span - target_span).abs()
		);
	}

	best_set
}

fn calculate_spacing_score(lines: &[HoughLine]) -> f32 {
	if lines.len() < 2 {
		return f32::MAX;
	}

	let mut spacings = Vec::new();
	for i in 0..lines.len() - 1 {
		spacings.push((lines[i + 1].rho - lines[i].rho).abs());
	}

	let mean_spacing = spacings.iter().sum::<f32>() / spacings.len() as f32;

	spacings
		.iter()
		.map(|s| (s - mean_spacing).powi(2))
		.sum::<f32>()
		/ spacings.len() as f32
}

fn find_best_evenly_spaced_grid(
	lines: &[HoughLine], target_count: usize,
) -> Option<Vec<HoughLine>> {
	if lines.len() < target_count {
		return None;
	}

	let mut best_grid: Option<(Vec<HoughLine>, f32)> = None;

	for start_idx in 0..lines.len() {
		for end_idx in (start_idx + target_count - 1)..lines.len() {
			let span = lines[end_idx].rho - lines[start_idx].rho;
			let expected_spacing = span / (target_count - 1) as f32;

			if expected_spacing < 100.0 || expected_spacing > 250.0 {
				continue;
			}

			let mut selected = vec![lines[start_idx]];

			for i in 1..(target_count - 1) {
				let target_rho = lines[start_idx].rho + (expected_spacing * i as f32);

				let closest_line = lines[start_idx..=end_idx]
					.iter()
					.filter(|line| !selected.contains(line))
					.min_by_key(|line| ((line.rho - target_rho).abs() * 1000.0) as i32);

				if let Some(line) = closest_line {
					selected.push(*line);
				} else {
					break;
				}
			}

			if selected.len() == target_count - 1 {
				selected.push(lines[end_idx]);

				selected.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

				let spacing_quality = calculate_spacing_score(&selected);
				let score = 1.0 / (1.0 + spacing_quality);

				if best_grid.is_none() || score > best_grid.as_ref().unwrap().1 {
					best_grid = Some((selected.clone(), score));
				}
			}
		}
	}

	best_grid.map(|(grid, _)| grid)
}

fn get_all_grid_subsets(
	lines: &[HoughLine], target_count: usize, is_horizontal: bool,
) -> Option<Vec<(Vec<HoughLine>, f32)>> {
	if lines.len() < target_count {
		tracing::debug!(
			"Not enough {} lines: need {}, have {}",
			if is_horizontal {
				"horizontal"
			} else {
				"vertical"
			},
			target_count,
			lines.len()
		);
		return None;
	}

	let mut sorted_lines = lines.to_vec();
	sorted_lines.sort_by(|a, b| a.rho.partial_cmp(&b.rho).unwrap());

	let mut votes_for_percentile: Vec<usize> = sorted_lines.iter().map(|l| l.votes).collect();
	votes_for_percentile.sort();
	let vote_threshold = votes_for_percentile[votes_for_percentile.len() / 2];

	let strong_lines: Vec<HoughLine> = sorted_lines
		.into_iter()
		.filter(|line| line.votes >= vote_threshold)
		.collect();

	let max_isolation_distance = 250.0;
	let filtered_lines: Vec<HoughLine> = strong_lines
		.iter()
		.filter(|line| {
			strong_lines.iter().any(|other| {
				if line.rho == other.rho {
					return false;
				}
				(line.rho - other.rho).abs() <= max_isolation_distance
			})
		})
		.copied()
		.collect();

	if filtered_lines.len() < target_count {
		tracing::debug!(
			"Not enough {} lines after filtering: need {}, have {} (removed {} weak and {} outliers)",
			if is_horizontal {
				"horizontal"
			} else {
				"vertical"
			},
			target_count,
			filtered_lines.len(),
			lines.len() - strong_lines.len(),
			strong_lines.len() - filtered_lines.len()
		);
		return None;
	}

	let strong_lines = filtered_lines;

	let mut all_subsets: Vec<(Vec<HoughLine>, f32)> = Vec::new();

	for start_idx in 0..=(strong_lines.len() - target_count) {
		let subset: Vec<HoughLine> = strong_lines[start_idx..start_idx + target_count].to_vec();

		let votes: Vec<usize> = subset.iter().map(|l| l.votes).collect();
		let median_votes = {
			let mut sorted_votes = votes.clone();
			sorted_votes.sort();
			sorted_votes[sorted_votes.len() / 2]
		};
		let first_line_votes = subset.first().unwrap().votes;
		let last_line_votes = subset.last().unwrap().votes;

		if (first_line_votes as f32) < (median_votes as f32 * 0.70)
			|| (last_line_votes as f32) < (median_votes as f32 * 0.70)
		{
			tracing::trace!(
				"Subset [{}-{}]: rejected - edge lines too weak (first={}, last={}, median={})",
				start_idx,
				start_idx + target_count - 1,
				first_line_votes,
				last_line_votes,
				median_votes
			);
			continue;
		}

		let spacing_quality = calculate_spacing_score(&subset);
		let total_votes: usize = subset.iter().map(|l| l.votes).sum();
		let span = subset.last().unwrap().rho - subset.first().unwrap().rho;
		let avg_spacing = span / (target_count - 1) as f32;

		let vote_score = (total_votes as f32 / 10000.0).min(1.0);
		let spacing_uniformity_score = 1.0 / (1.0 + spacing_quality / 100.0);

		let expected_spacing_score = 1.0 / (1.0 + ((avg_spacing - 153.0).abs() / 20.0));

		let score =
			vote_score * 0.15 + spacing_uniformity_score * 0.15 + expected_spacing_score * 0.7;

		tracing::trace!(
			"Subset [{}-{}]: span={:.1}, spacing={:.1}, votes={}, quality={:.2}, score={:.3}",
			start_idx,
			start_idx + target_count - 1,
			span,
			avg_spacing,
			total_votes,
			spacing_quality,
			score
		);

		all_subsets.push((subset, score));
	}

	if strong_lines.len() >= target_count {
		tracing::debug!(
			"Trying grid-fitting approach for {} direction",
			if is_horizontal {
				"horizontal"
			} else {
				"vertical"
			}
		);

		if let Some(grid_subset) = find_best_evenly_spaced_grid(&strong_lines, target_count) {
			let spacing_quality = calculate_spacing_score(&grid_subset);
			let total_votes: usize = grid_subset.iter().map(|l| l.votes).sum();
			let span = grid_subset.last().unwrap().rho - grid_subset.first().unwrap().rho;
			let avg_spacing = span / (target_count - 1) as f32;

			let vote_score = (total_votes as f32 / 10000.0).min(1.0);
			let spacing_uniformity_score = 1.0 / (1.0 + spacing_quality / 100.0);
			let expected_spacing_score = 1.0 / (1.0 + ((avg_spacing - 153.0).abs() / 20.0));
			let score =
				vote_score * 0.15 + spacing_uniformity_score * 0.15 + expected_spacing_score * 0.7;

			tracing::debug!(
				"Grid-fitting found subset: span={:.1}, spacing={:.1}, votes={}, quality={:.2}, score={:.3}",
				span,
				avg_spacing,
				total_votes,
				spacing_quality,
				score
			);

			all_subsets.push((grid_subset, score));
		}
	}

	all_subsets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

	if !all_subsets.is_empty() {
		Some(all_subsets)
	} else {
		None
	}
}

fn calculate_corners(horizontal: &[HoughLine], vertical: &[HoughLine]) -> Option<[(f32, f32); 4]> {
	if horizontal.is_empty() || vertical.is_empty() {
		return None;
	}

	let top = horizontal.first()?;
	let bottom = horizontal.last()?;
	let left = vertical.first()?;
	let right = vertical.last()?;

	let top_left = top.intersect(left)?;
	let top_right = top.intersect(right)?;
	let bottom_left = bottom.intersect(left)?;
	let bottom_right = bottom.intersect(right)?;

	Some([top_left, top_right, bottom_left, bottom_right])
}

fn get_bbox_from_corners(corners: &[(f32, f32); 4]) -> (f32, f32, f32, f32) {
	let xs: Vec<f32> = corners.iter().map(|(x, _)| *x).collect();
	let ys: Vec<f32> = corners.iter().map(|(_, y)| *y).collect();

	let min_x = xs.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
	let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
	let min_y = ys.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
	let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));

	(min_x, min_y, max_x - min_x, max_y - min_y)
}

fn calculate_grid_confidence(grid: &Grid, _width: usize, _height: usize) -> f32 {
	let mut confidence: f32 = 0.9;

	if grid.horizontal_lines.len() == 9 {
		confidence += 0.05;
	}
	if grid.vertical_lines.len() == 9 {
		confidence += 0.05;
	}

	let h_score = calculate_spacing_score(&grid.horizontal_lines);
	let v_score = calculate_spacing_score(&grid.vertical_lines);

	let avg_spacing = (h_score + v_score) / 2.0;
	if avg_spacing < 10.0 {
		confidence += 0.1;
	} else if avg_spacing < 50.0 {
		confidence += 0.05;
	}

	confidence.min(1.0)
}

fn save_grayscale_debug(gray: &[u8], width: usize, height: usize, filename: &str) {
	if let Some(img) = GrayImage::from_raw(width as u32, height as u32, gray.to_vec()) {
		if let Err(e) = img.save(filename) {
			tracing::warn!("Failed to save debug image {}: {}", filename, e);
		} else {
			tracing::info!("Saved debug image: {}", filename);
		}
	}
}

fn save_edges_debug(edges: &[bool], width: usize, height: usize, filename: &str) {
	let mut img = GrayImage::new(width as u32, height as u32);
	for y in 0..height {
		for x in 0..width {
			let idx = y * width + x;
			let value = if edges[idx] { 255 } else { 0 };
			img.put_pixel(x as u32, y as u32, image::Luma([value]));
		}
	}

	if let Err(e) = img.save(filename) {
		tracing::warn!("Failed to save debug image {}: {}", filename, e);
	} else {
		tracing::info!("Saved debug image: {}", filename);
	}
}

fn save_lines_debug(
	original: &RgbaImage, horizontal: &[HoughLine], vertical: &[HoughLine], width: usize,
	height: usize, filename: &str,
) {
	tracing::info!(
		"Drawing {} horizontal and {} vertical lines for debug visualization",
		horizontal.len(),
		vertical.len()
	);

	let mut img = RgbImage::new(width as u32, height as u32);

	for y in 0..height {
		for x in 0..width {
			let pixel = original.get_pixel(x as u32, y as u32);
			img.put_pixel(x as u32, y as u32, Rgb([pixel[0], pixel[1], pixel[2]]));
		}
	}

	for (i, line) in horizontal.iter().enumerate() {
		tracing::debug!(
			"H-Line {}: rho={:.1}, theta={:.3} rad ({:.1}°), votes={}",
			i,
			line.rho,
			line.theta,
			line.theta.to_degrees(),
			line.votes
		);
		draw_line_on_image(&mut img, line, Rgb([255, 0, 0]), width, height);
	}

	for (i, line) in vertical.iter().enumerate() {
		tracing::debug!(
			"V-Line {}: rho={:.1}, theta={:.3} rad ({:.1}°), votes={}",
			i,
			line.rho,
			line.theta,
			line.theta.to_degrees(),
			line.votes
		);
		draw_line_on_image(&mut img, line, Rgb([0, 255, 0]), width, height);
	}

	if let Err(e) = img.save(filename) {
		tracing::warn!("Failed to save debug image {}: {}", filename, e);
	} else {
		tracing::info!("Saved debug image: {}", filename);
	}
}

fn save_grid_debug(original: &RgbaImage, grid: &Grid, filename: &str) {
	let width = original.width() as usize;
	let height = original.height() as usize;
	let mut img = RgbImage::new(width as u32, height as u32);

	for y in 0..height {
		for x in 0..width {
			let pixel = original.get_pixel(x as u32, y as u32);
			img.put_pixel(x as u32, y as u32, Rgb([pixel[0], pixel[1], pixel[2]]));
		}
	}

	for line in &grid.horizontal_lines {
		draw_line_on_image(&mut img, line, Rgb([0, 255, 255]), width, height);
	}
	for line in &grid.vertical_lines {
		draw_line_on_image(&mut img, line, Rgb([0, 255, 255]), width, height);
	}

	let corner_size = 10;
	for &(x, y) in &grid.corners {
		let cx = x as i32;
		let cy = y as i32;

		for dy in -corner_size..=corner_size {
			for dx in -corner_size..=corner_size {
				let px = cx + dx;
				let py = cy + dy;

				if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
					img.put_pixel(px as u32, py as u32, Rgb([255, 0, 255]));
				}
			}
		}
	}

	if let Err(e) = img.save(filename) {
		tracing::warn!("Failed to save debug image {}: {}", filename, e);
	} else {
		tracing::info!("Saved debug image: {}", filename);
	}
}

fn draw_line_on_image(
	img: &mut RgbImage, line: &HoughLine, color: Rgb<u8>, width: usize, height: usize,
) {
	let cos_theta = line.theta.cos();
	let sin_theta = line.theta.sin();

	let is_more_vertical =
		(line.theta - std::f32::consts::FRAC_PI_2).abs() < std::f32::consts::FRAC_PI_4;

	for thickness in -1..=1 {
		if is_more_vertical {
			for y in 0..height {
				let yf = y as f32;

				if cos_theta.abs() > 0.01 {
					let x = (line.rho - yf * sin_theta) / cos_theta + thickness as f32;
					let xi = x.round() as i32;
					if xi >= 0 && xi < width as i32 {
						img.put_pixel(xi as u32, y as u32, color);
					}
				}
			}
		} else {
			for x in 0..width {
				let xf = x as f32;

				if sin_theta.abs() > 0.01 {
					let y = (line.rho - xf * cos_theta) / sin_theta + thickness as f32;
					let yi = y.round() as i32;
					if yi >= 0 && yi < height as i32 {
						img.put_pixel(x as u32, yi as u32, color);
					}
				}
			}
		}
	}
}

fn dilate_edges(edges: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
	let mut dilated = vec![false; edges.len()];

	for y in 0..height {
		for x in 0..width {
			let idx = y * width + x;

			if edges[idx] {
				dilated[idx] = true;
				continue;
			}

			let mut found_edge = false;
			'outer: for dy in -(radius as isize)..=(radius as isize) {
				for dx in -(radius as isize)..=(radius as isize) {
					let ny = y as isize + dy;
					let nx = x as isize + dx;

					if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
						let neighbor_idx = (ny as usize) * width + (nx as usize);
						if edges[neighbor_idx] {
							found_edge = true;
							break 'outer;
						}
					}
				}
			}

			dilated[idx] = found_edge;
		}
	}

	dilated
}
