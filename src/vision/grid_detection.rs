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

	if CONFIG.debugging.save_images {
		save_edges_debug(&edges, width, height, "debug_02_edges_strong.png");
	}

	let line_detector = LineDetector::new(width, height);
	let (horizontal_lines, vertical_lines) =
		line_detector.detect_orthogonal_lines(&edges, CONFIG.detection.hough_threshold);

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
	let h_grid = find_evenly_spaced_lines(horizontal, 9, true, width, height)?;

	let h_span = h_grid.last()?.rho - h_grid.first()?.rho;
	let expected_square_size = h_span;

	let v_grid = find_evenly_spaced_lines_with_target_span(
		vertical,
		9,
		false,
		width,
		height,
		expected_square_size,
	)?;

	let v_span = v_grid.last()?.rho - v_grid.first()?.rho;
	let size_ratio = h_span.max(v_span) / h_span.min(v_span);

	tracing::debug!(
		"Grid spans: h={:.1}, v={:.1}, ratio={:.2}",
		h_span,
		v_span,
		size_ratio
	);

	if size_ratio > 1.10 {
		tracing::debug!("Grid is not square enough (max ratio: 1.10)");
		return None;
	}

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

	// if board_width < config.min_board_size
	// 	|| board_height < config.min_board_size
	// 	|| board_width > config.max_board_size
	// 	|| board_height > config.max_board_size
	// {
	// 	tracing::debug!(
	// 		"Board size validation failed: {}x{} (min: {}, max: {})",
	// 		board_width,
	// 		board_height,
	// 		config.min_board_size,
	// 		config.max_board_size
	// 	);
	// 	return None;
	// }

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
