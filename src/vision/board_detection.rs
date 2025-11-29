use std::{cmp::Ordering, f64::consts::PI};

use image::GrayImage;

use crate::model::detected::{DetectedBoard, Rect};

const EPSILON: f32 = 1e-6;

const SOBEL_GX: [[i16; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
const SOBEL_GY: [[i16; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

const EDGE_THRESHOLD: u16 = 100;
const THETA_BINS: usize = 180;
const NMS_WINDOW_RHO: usize = 10;
const NMS_WINDOW_THETA: usize = 5;
const SEGMENT_GAP_TOLERANCE: usize = 5;
const MIN_SEGMENT_FRACTION: f32 = 0.1;
const VERTICAL_THETA_TOLERANCE: usize = 10;
const HORIZONTAL_THETA_TOLERANCE: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PixelIdx(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PixelCount(usize);

#[derive(Debug, Clone, Copy)]
struct PixelF(f32);

#[derive(Debug, Clone, Copy)]
struct VoteCount(u32);

#[derive(Debug, Clone, Copy)]
struct Rho(f32);

#[derive(Debug, Clone, Copy)]
struct ThetaDeg(usize);

#[derive(Debug, Clone, Copy)]
struct GridSpacing(f32);

#[derive(Debug, Clone, Copy)]
struct GridPosition(f32);

#[derive(Debug, Clone, Copy)]
struct PixelPoint {
	x: PixelF,
	y: PixelF,
}

impl PixelPoint {
	fn new(x: f32, y: f32) -> Self {
		Self {
			x: PixelF(x),
			y: PixelF(y),
		}
	}
}

struct EdgeMap {
	data: Vec<bool>,
	width: PixelCount,
	height: PixelCount,
}

impl EdgeMap {
	fn new(width: usize, height: usize) -> Self {
		Self {
			data: vec![false; width * height],
			width: PixelCount(width),
			height: PixelCount(height),
		}
	}

	#[inline]
	fn set_edge(&mut self, x: PixelIdx, y: PixelIdx) {
		if x.0 < self.width.0 && y.0 < self.height.0 {
			self.data[y.0 * self.width.0 + x.0] = true;
		}
	}

	#[inline]
	fn is_edge(&self, x: PixelIdx, y: PixelIdx) -> bool {
		if x.0 < self.width.0 && y.0 < self.height.0 {
			self.data[y.0 * self.width.0 + x.0]
		} else {
			false
		}
	}

	fn width(&self) -> usize {
		self.width.0
	}

	fn height(&self) -> usize {
		self.height.0
	}
}

#[derive(Debug, Clone, Copy)]
struct HoughLine {
	rho: Rho,
	theta: ThetaDeg,
	votes: VoteCount,
}

#[derive(Debug, Clone, Copy)]
struct MeasuredSegment {
	start: PixelPoint,
	end: PixelPoint,
	length_px: f32,
}

struct HoughAccumulator {
	data: Vec<u32>,
	rho_bins: usize,
	theta_bins: usize,
	max_rho: f32,
	sin_table: [f32; THETA_BINS],
	cos_table: [f32; THETA_BINS],
}

impl HoughAccumulator {
	fn new(image_width: usize, image_height: usize) -> Self {
		let max_rho =
			((image_width * image_width + image_height * image_height) as f64).sqrt() as f32;
		let rho_bins = (2.0 * max_rho).ceil() as usize + 1;

		let mut sin_table = [0.0f32; THETA_BINS];
		let mut cos_table = [0.0f32; THETA_BINS];

		for theta_deg in 0..THETA_BINS {
			let theta_rad = (theta_deg as f64) * PI / 180.0;
			sin_table[theta_deg] = theta_rad.sin() as f32;
			cos_table[theta_deg] = theta_rad.cos() as f32;
		}

		Self {
			data: vec![0u32; rho_bins * THETA_BINS],
			rho_bins,
			theta_bins: THETA_BINS,
			max_rho,
			sin_table,
			cos_table,
		}
	}

	#[inline]
	fn rho_to_index(&self, rho: Rho) -> usize {
		let idx = ((rho.0 + self.max_rho) as isize).max(0) as usize;
		idx.min(self.rho_bins - 1)
	}

	#[inline]
	fn index_to_rho(&self, index: usize) -> Rho {
		Rho(index as f32 - self.max_rho)
	}

	#[inline]
	fn vote(&mut self, rho: Rho, theta: ThetaDeg) {
		let rho_idx = self.rho_to_index(rho);
		let idx = theta.0 * self.rho_bins + rho_idx;
		if idx < self.data.len() {
			// safe increment
			self.data[idx] = self.data[idx].saturating_add(1);
		}
	}

	#[inline]
	fn get_votes(&self, rho_idx: usize, theta: ThetaDeg) -> VoteCount {
		let idx = theta.0 * self.rho_bins + rho_idx;
		if idx < self.data.len() {
			VoteCount(self.data[idx])
		} else {
			VoteCount(0)
		}
	}

	fn trig_for(&self, theta: ThetaDeg) -> (f32, f32) {
		(self.cos_table[theta.0], self.sin_table[theta.0])
	}
}

fn sobel_edge_detection(image: &GrayImage, threshold: u16) -> EdgeMap {
	let width = image.width() as usize;
	let height = image.height() as usize;
	let raw = image.as_raw();

	let mut edge_map = EdgeMap::new(width, height);

	for y in 1..height.saturating_sub(1) {
		for x in 1..width.saturating_sub(1) {
			let mut gx: i32 = 0;
			let mut gy: i32 = 0;

			for ky in 0..3 {
				for kx in 0..3 {
					let px = x + kx - 1;
					let py = y + ky - 1;
					let pixel = raw[py * width + px] as i32;

					gx += pixel * SOBEL_GX[ky][kx] as i32;
					gy += pixel * SOBEL_GY[ky][kx] as i32;
				}
			}

			let magnitude = ((gx * gx + gy * gy) as f64).sqrt() as u16;

			if magnitude > threshold {
				edge_map.set_edge(PixelIdx(x), PixelIdx(y));
			}
		}
	}

	edge_map
}

fn hough_voting(edge_map: &EdgeMap, accumulator: &mut HoughAccumulator) {
	let width = edge_map.width();
	let height = edge_map.height();

	for y in 0..height {
		for x in 0..width {
			if edge_map.is_edge(PixelIdx(x), PixelIdx(y)) {
				for theta_deg in 0..THETA_BINS {
					let theta = ThetaDeg(theta_deg);
					let (cos_t, sin_t) = accumulator.trig_for(theta);
					let rho_val = (x as f32) * cos_t + (y as f32) * sin_t;
					accumulator.vote(Rho(rho_val), theta);
				}
			}
		}
	}
}

fn detect_peaks(
	accumulator: &HoughAccumulator, vote_threshold: VoteCount, max_lines: usize,
) -> Vec<HoughLine> {
	let mut peaks: Vec<HoughLine> = Vec::new();

	for theta_deg in 0..accumulator.theta_bins {
		let theta = ThetaDeg(theta_deg);
		for rho_idx in 0..accumulator.rho_bins {
			let votes = accumulator.get_votes(rho_idx, theta);
			if votes.0 < vote_threshold.0 {
				continue;
			}

			let mut is_max = true;
			'nms: for dt in 0..=NMS_WINDOW_THETA {
				for dr in 0..=NMS_WINDOW_RHO {
					if dt == 0 && dr == 0 {
						continue;
					}

					let checks = [
						(rho_idx.wrapping_add(dr), theta_deg.wrapping_add(dt)),
						(rho_idx.wrapping_sub(dr), theta_deg.wrapping_add(dt)),
						(rho_idx.wrapping_add(dr), theta_deg.wrapping_sub(dt)),
						(rho_idx.wrapping_sub(dr), theta_deg.wrapping_sub(dt)),
					];

					for (r, t) in checks {
						if r < accumulator.rho_bins && t < accumulator.theta_bins {
							let other_votes = accumulator.get_votes(r, ThetaDeg(t));
							if other_votes.0 > votes.0 {
								is_max = false;
								break 'nms;
							}
						}
					}
				}
			}

			if is_max {
				peaks.push(HoughLine {
					rho: accumulator.index_to_rho(rho_idx),
					theta,
					votes,
				});
			}
		}
	}

	peaks.sort_by(|a, b| b.votes.0.cmp(&a.votes.0));
	peaks.truncate(max_lines);

	peaks
}

fn measure_hough_line_segment(
	edge_map: &EdgeMap, line: &HoughLine, gap_tol: usize,
) -> Option<MeasuredSegment> {
	let width = edge_map.width();
	let height = edge_map.height();

	if width == 0 || height == 0 {
		return None;
	}

	let theta_rad = (line.theta.0 as f64) * PI / 180.0;
	let cos_t = theta_rad.cos() as f32;
	let sin_t = theta_rad.sin() as f32;
	let rho = line.rho.0;

	let mut intersections: Vec<PixelPoint> = Vec::with_capacity(4);

	// x = 0
	if sin_t.abs() > EPSILON {
		let y = rho / sin_t;
		if (0.0..(height as f32)).contains(&y) {
			intersections.push(PixelPoint::new(0.0, y));
		}
	}

	// x = width-1
	if sin_t.abs() > EPSILON {
		let x = (width - 1) as f32;
		let y = (rho - x * cos_t) / sin_t;
		if (0.0..(height as f32)).contains(&y) {
			intersections.push(PixelPoint::new(x, y));
		}
	}

	// y = 0
	if cos_t.abs() > EPSILON {
		let x = rho / cos_t;
		if (0.0..(width as f32)).contains(&x) {
			intersections.push(PixelPoint::new(x, 0.0));
		}
	}

	// y = height-1
	if cos_t.abs() > EPSILON {
		let y = (height - 1) as f32;
		let x = (rho - y * sin_t) / cos_t;
		if (0.0..(width as f32)).contains(&x) {
			intersections.push(PixelPoint::new(x, y));
		}
	}

	if intersections.len() < 2 {
		return None;
	}

	let (p0, p1) = find_most_distant_points(&intersections);

	let (dx, dy) = (p1.x.0 - p0.x.0, p1.y.0 - p0.y.0);
	let steps = (dx.abs().max(dy.abs()) as usize).max(1);

	let mut edge_samples: Vec<(usize, PixelPoint)> = Vec::with_capacity(steps + 1);

	for i in 0..=steps {
		let t = i as f32 / steps as f32;
		let x = p0.x.0 + dx * t;
		let y = p0.y.0 + dy * t;
		let xi = x.round() as usize;
		let yi = y.round() as usize;

		if xi < width && yi < height && edge_map.is_edge(PixelIdx(xi), PixelIdx(yi)) {
			edge_samples.push((i, PixelPoint::new(x, y)));
		}
	}

	if edge_samples.is_empty() {
		return None;
	}

	let (best_start_idx, best_end_idx) = find_longest_run_with_gaps(&edge_samples, gap_tol);

	if best_start_idx >= edge_samples.len() || best_end_idx >= edge_samples.len() {
		return None;
	}

	let start = edge_samples[best_start_idx].1;
	let end = edge_samples[best_end_idx].1;

	let length_px = {
		let dx = end.x.0 - start.x.0;
		let dy = end.y.0 - start.y.0;
		(dx * dx + dy * dy).sqrt()
	};

	let potential_length = ((p1.x.0 - p0.x.0).powi(2) + (p1.y.0 - p0.y.0).powi(2)).sqrt();
	if length_px < potential_length * MIN_SEGMENT_FRACTION {
		return None;
	}

	Some(MeasuredSegment {
		start,
		end,
		length_px,
	})
}

fn find_most_distant_points(points: &[PixelPoint]) -> (PixelPoint, PixelPoint) {
	debug_assert!(points.len() >= 2);
	let mut max_dist_sq = 0.0f32;
	let mut best_pair = (points[0], points[1]);

	for i in 0..points.len() {
		for j in (i + 1)..points.len() {
			let dx = points[j].x.0 - points[i].x.0;
			let dy = points[j].y.0 - points[i].y.0;
			let dist_sq = dx * dx + dy * dy;
			if dist_sq > max_dist_sq {
				max_dist_sq = dist_sq;
				best_pair = (points[i], points[j]);
			}
		}
	}

	best_pair
}

fn find_longest_run_with_gaps(samples: &[(usize, PixelPoint)], gap_tol: usize) -> (usize, usize) {
	if samples.is_empty() {
		return (0, 0);
	}

	let mut best_start = 0usize;
	let mut best_end = 0usize;
	let mut best_length = 0usize;

	let mut run_start = 0usize;
	let mut run_end = 0usize;

	for i in 1..samples.len() {
		let gap = samples[i].0 - samples[i - 1].0;

		if gap <= gap_tol + 1 {
			run_end = i;
		} else {
			let run_length = samples[run_end].0 - samples[run_start].0;
			if run_length > best_length {
				best_length = run_length;
				best_start = run_start;
				best_end = run_end;
			}
			run_start = i;
			run_end = i;
		}
	}

	let run_length = samples[run_end].0 - samples[run_start].0;
	if run_length > best_length {
		best_start = run_start;
		best_end = run_end;
	}

	(best_start, best_end)
}

fn cluster_lines_full(lines: &[HoughLine]) -> (Vec<HoughLine>, Vec<HoughLine>) {
	let mut vertical_lines: Vec<HoughLine> = Vec::new();
	let mut horizontal_lines: Vec<HoughLine> = Vec::new();

	for line in lines {
		let theta = line.theta.0;
		if theta <= VERTICAL_THETA_TOLERANCE || theta >= (180 - VERTICAL_THETA_TOLERANCE) {
			vertical_lines.push(*line);
		} else if (90 - HORIZONTAL_THETA_TOLERANCE..=90 + HORIZONTAL_THETA_TOLERANCE)
			.contains(&theta)
		{
			horizontal_lines.push(*line);
		}
	}

	(vertical_lines, horizontal_lines)
}

#[derive(Debug, Clone, Copy)]
struct GridPattern {
	start: GridPosition,
	end: GridPosition,
	spacing: GridSpacing,
}

fn compute_vertical_span(
	vertical_lines: &[HoughLine], edge_map: &EdgeMap, v_grid: &GridPattern,
) -> Option<(f32, f32)> {
	let mut y_starts: Vec<f32> = Vec::new();
	let mut y_ends: Vec<f32> = Vec::new();

	let tolerance = (v_grid.spacing.0 * 0.05).min(8.0);

	for line in vertical_lines {
		if line.rho.0 < v_grid.start.0 - tolerance || line.rho.0 > v_grid.end.0 + tolerance {
			continue;
		}

		if let Some(segment) = measure_hough_line_segment(edge_map, line, SEGMENT_GAP_TOLERANCE) {
			let y_min = segment.start.y.0.min(segment.end.y.0);
			let y_max = segment.start.y.0.max(segment.end.y.0);

			let expected_height = v_grid.spacing.0 * 8.0;
			if segment.length_px >= expected_height * 0.4 {
				y_starts.push(y_min);
				y_ends.push(y_max);
			}
		}
	}

	if y_starts.len() < 3 {
		return None;
	}

	y_starts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
	y_ends.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

	let median_y_start = y_starts[y_starts.len() / 2];
	let median_y_end = y_ends[y_ends.len() / 2];

	Some((median_y_start, median_y_end))
}

fn filter_horizontals_by_vertical_span(
	horizontal_lines: &[HoughLine], edge_map: &EdgeMap, v_span: (f32, f32), v_grid: &GridPattern,
) -> Vec<f32> {
	let (v_y_start, v_y_end) = v_span;
	let mut filtered_rhos: Vec<f32> = Vec::new();
	let margin = v_grid.spacing.0 * 0.25;

	for line in horizontal_lines {
		if line.rho.0 < v_y_start - margin || line.rho.0 > v_y_end + margin {
			continue;
		}

		if let Some(segment) = measure_hough_line_segment(edge_map, line, SEGMENT_GAP_TOLERANCE) {
			let x_min = segment.start.x.0.min(segment.end.x.0);
			let x_max = segment.start.x.0.max(segment.end.x.0);

			let overlap_start = x_min.max(v_grid.start.0);
			let overlap_end = x_max.min(v_grid.end.0);
			let overlap = (overlap_end - overlap_start).max(0.0);

			let v_width = v_grid.end.0 - v_grid.start.0;
			let overlap_ratio = if v_width > 0.0 {
				overlap / v_width
			} else {
				0.0
			};

			if overlap_ratio >= 0.66 {
				filtered_rhos.push(line.rho.0);
			}
		}
	}

	filtered_rhos
}

fn extract_rect_from_lines_with_segments(lines: &[HoughLine], edge_map: &EdgeMap) -> Rect {
	if lines.is_empty() {
		return Rect::new(0.0, 0.0, 0.0, 0.0);
	}

	let (vertical_lines, horizontal_lines) = cluster_lines_full(lines);

	let mut vertical_rhos: Vec<f32> = vertical_lines.iter().map(|l| l.rho.0).collect();

	let v_grid = find_best_grid(&mut vertical_rhos, None);

	let Some(v_grid) = v_grid else {
		return Rect::new(0.0, 0.0, 0.0, 0.0);
	};

	let v_span = compute_vertical_span(&vertical_lines, edge_map, &v_grid);

	let (mut horizontal_rhos, expected_h_spacing) = if let Some(v_span) = v_span {
		let filtered =
			filter_horizontals_by_vertical_span(&horizontal_lines, edge_map, v_span, &v_grid);

		let measured_height = v_span.1 - v_span.0;
		let spacing_from_span = measured_height / 8.0;

		(filtered, spacing_from_span)
	} else {
		(
			horizontal_lines.iter().map(|l| l.rho.0).collect(),
			v_grid.spacing.0,
		)
	};

	let h_grid = find_best_grid(&mut horizontal_rhos, Some(expected_h_spacing));

	let Some(h_grid) = h_grid else {
		return Rect::new(0.0, 0.0, 0.0, 0.0);
	};

	let mut width = v_grid.end.0 - v_grid.start.0;
	let mut height = h_grid.end.0 - h_grid.start.0;
	let mut x = v_grid.start.0;
	let mut y = h_grid.start.0;

	let v_intervals = ((v_grid.end.0 - v_grid.start.0) / v_grid.spacing.0).round() as usize;
	let h_intervals = ((h_grid.end.0 - h_grid.start.0) / h_grid.spacing.0).round() as usize;

	if v_intervals == h_intervals {
		let avg_size = (width + height) / 2.0;
		let center_x = x + width / 2.0;
		let center_y = y + height / 2.0;

		width = avg_size;
		height = avg_size;
		x = center_x - width / 2.0;
		y = center_y - height / 2.0;
	}

	Rect::new(x.max(0.0), y.max(0.0), width, height)
}

fn find_best_grid(rhos: &mut [f32], expected_spacing: Option<f32>) -> Option<GridPattern> {
	if rhos.len() < 4 {
		return None;
	}
	rhos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

	let mut deduped: Vec<f32> = Vec::new();
	for &rho in rhos.iter() {
		if deduped.is_empty() || (rho - deduped.last().unwrap()).abs() > 5.0 {
			deduped.push(rho);
		}
	}

	if deduped.len() < 4 {
		return None;
	}

	let mut best_grid: Option<GridPattern> = None;
	let mut max_score = -f32::INFINITY;
	let target_count = 9;

	for i in 0..deduped.len() {
		for j in (i + 1)..deduped.len() {
			let diff = deduped[j] - deduped[i];

			if diff < 20.0 {
				continue;
			}

			if let Some(target) = expected_spacing
				&& (diff - target).abs() > target * 0.05
			{
				continue;
			}

			let spacing = diff;
			let (matched_count, first_match, last_match) =
				count_grid_matches(&deduped, deduped[i], spacing);

			if matched_count < 7 {
				continue;
			}
			if matched_count > 11 {
				continue;
			}

			let count_diff = (matched_count as f32 - target_count as f32).abs();
			let count_score = 20.0 - (count_diff * 5.0);

			let spacing_score = if let Some(target) = expected_spacing {
				let error_pct = (spacing - target).abs() / target;
				(1.0 - error_pct) * 200.0
			} else {
				0.0
			};

			let span = last_match - first_match;
			let span_ratio = span / spacing;

			if span_ratio > 10.5 {
				continue;
			}

			let ideal_span = 8.0;
			let span_diff = (span_ratio - ideal_span).abs();
			let compactness_score = (25.0 - span_diff * 15.0).max(0.0);

			let score = count_score + spacing_score + compactness_score;

			if score > max_score {
				max_score = score;
				best_grid = Some(GridPattern {
					start: GridPosition(first_match),
					end: GridPosition(last_match),
					spacing: GridSpacing(spacing),
				});
			}
		}
	}

	best_grid
}

fn count_grid_matches(rhos: &[f32], start: f32, spacing: f32) -> (usize, f32, f32) {
	let mut matches = 0;
	let mut first_pos = start;
	let mut last_pos = start;
	let mut found_any = false;

	let tolerance = (spacing * 0.05).min(8.0);

	for k in -8..=12 {
		let target = start + (k as f32 * spacing);

		let found = rhos.iter().any(|&r| (r - target).abs() < tolerance);

		if found {
			matches += 1;
			if !found_any {
				first_pos = target;
				found_any = true;
			} else if target < first_pos {
				first_pos = target;
			}
			if target > last_pos {
				last_pos = target;
			}
		}
	}

	if !found_any {
		return (0, start, start);
	}

	(matches, first_pos, last_pos)
}

pub fn detect_board_hough(image: &GrayImage) -> Option<DetectedBoard> {
	let width = image.width() as usize;
	let height = image.height() as usize;

	if width == 0 || height == 0 {
		return None;
	}

	let edge_map = sobel_edge_detection(image, EDGE_THRESHOLD);

	let mut accumulator = HoughAccumulator::new(width, height);
	hough_voting(&edge_map, &mut accumulator);

	let vote_threshold = VoteCount(((width.min(height) / 10).max(20)) as u32);
	let max_lines = 200;

	let lines_raw = detect_peaks(&accumulator, vote_threshold, max_lines);
	if lines_raw.is_empty() {
		return None;
	}

	let rect = extract_rect_from_lines_with_segments(&lines_raw, &edge_map);
	if rect.is_empty() {
		return None;
	}

	let aspect_ratio = rect.width() / rect.height();
	if !(0.7..=1.3).contains(&aspect_ratio) {
		return None;
	}

	let min_dimension = (width.min(height) as f32) * 0.1;
	if rect.width() < min_dimension || rect.height() < min_dimension {
		return None;
	}

	Some(DetectedBoard::new(rect))
}
