use std::f64::consts::PI;

use image::{RgbImage, RgbaImage, imageops::blur};
use tracing::{debug, trace};

use crate::model::detected::{DetectedBoard, Rect};

const BELL_FLATTENING_LEVEL: f32 = 1.0;

const EPSILON: f32 = 1e-6;

const SOBEL_GX: [[i16; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
const SOBEL_GY: [[i16; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

const EDGE_THRESHOLD: u16 = 40;
const THETA_BINS: usize = 180;
const NMS_WINDOW_RHO: usize = 10;
const NMS_WINDOW_THETA: usize = 5;
const SEGMENT_GAP_TOLERANCE: usize = 15;
const MIN_SEGMENT_FRACTION: f32 = 0.1;
const VERTICAL_THETA_TOLERANCE: usize = 10;
const HORIZONTAL_THETA_TOLERANCE: usize = 10;

type Rho = f32;
type Theta = usize;
type Votes = u32;

#[derive(Debug, Clone, Copy)]
struct Point {
	x: f32,
	y: f32,
}

impl Point {
	fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}
}

struct EdgeMap {
	data: Vec<u8>, // 0 or 1
	width: usize,
	height: usize,
}

impl EdgeMap {
	fn new(width: usize, height: usize) -> Self {
		Self {
			data: vec![0; width * height],
			width,
			height,
		}
	}

	#[inline]
	fn idx(&self, x: usize, y: usize) -> usize {
		y * self.width + x
	}

	#[inline]
	fn set_edge(&mut self, x: usize, y: usize) {
		if x < self.width && y < self.height {
			let idx = self.idx(x, y);
			self.data[idx] = 1;
		}
	}

	#[inline]
	fn is_edge(&self, x: usize, y: usize) -> bool {
		if x < self.width && y < self.height {
			self.data[self.idx(x, y)] != 0
		} else {
			false
		}
	}

	fn width(&self) -> usize {
		self.width
	}

	fn height(&self) -> usize {
		self.height
	}
}

#[derive(Debug, Clone, Copy)]
struct HoughLine {
	rho: Rho,
	theta: Theta,
	votes: Votes,
}

#[derive(Debug, Clone, Copy)]
struct MeasuredSegment {
	start: Point,
	end: Point,
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
	fn new(image_width: usize, image_height: usize) -> Option<Self> {
		let max_rho =
			((image_width * image_width + image_height * image_height) as f64).sqrt() as f32;
		let rho_bins = (2.0 * max_rho).ceil() as usize + 1;

		// Guard against OOM
		let approx_size =
			(rho_bins as u64) * (THETA_BINS as u64) * std::mem::size_of::<u32>() as u64;
		const MAX_ACC_BYTES: u64 = 300_000_000; // 300 MB
		if approx_size > MAX_ACC_BYTES {
			return None;
		}

		let mut sin_table = [0.0f32; THETA_BINS];
		let mut cos_table = [0.0f32; THETA_BINS];

		for theta_deg in 0..THETA_BINS {
			let theta_rad = (theta_deg as f64) * PI / 180.0;
			sin_table[theta_deg] = theta_rad.sin() as f32;
			cos_table[theta_deg] = theta_rad.cos() as f32;
		}

		Some(Self {
			data: vec![0u32; rho_bins * THETA_BINS],
			rho_bins,
			theta_bins: THETA_BINS,
			max_rho,
			sin_table,
			cos_table,
		})
	}

	#[inline]
	fn rho_to_index(&self, rho: Rho) -> usize {
		let offset_rho = rho + self.max_rho;
		let idx = offset_rho.round() as isize;
		idx.clamp(0, (self.rho_bins as isize) - 1) as usize
	}

	#[inline]
	fn index_to_rho(&self, index: usize) -> Rho {
		index as f32 - self.max_rho
	}

	#[inline]
	fn vote(&mut self, rho: Rho, theta: Theta) {
		let rho_idx = self.rho_to_index(rho);
		if theta < self.theta_bins {
			let idx = theta * self.rho_bins + rho_idx;
			if idx < self.data.len() {
				self.data[idx] = self.data[idx].saturating_add(1);
			}
		}
	}

	#[inline]
	fn get_votes(&self, rho_idx: usize, theta: Theta) -> Votes {
		if theta < self.theta_bins && rho_idx < self.rho_bins {
			let idx = theta * self.rho_bins + rho_idx;
			if idx < self.data.len() {
				return self.data[idx];
			}
		}
		0
	}

	fn trig_for(&self, theta: Theta) -> (f32, f32) {
		if theta < self.theta_bins {
			(self.cos_table[theta], self.sin_table[theta])
		} else {
			debug_assert!(false, "theta out of bounds");
			(0.0, 0.0) // Should not happen if used correctly
		}
	}
}

fn sobel_edge_detection(image: &RgbImage, threshold: u16) -> EdgeMap {
	let width = image.width() as usize;
	let height = image.height() as usize;
	let raw = image.as_raw();

	// Safety check: 3 bytes per pixel
	debug_assert_eq!(raw.len(), width * height * 3, "Image must be RGB format");

	let mut edge_map = EdgeMap::new(width, height);

	for y in 1..height.saturating_sub(1) {
		for x in 1..width.saturating_sub(1) {
			let mut gx_r: i32 = 0;
			let mut gy_r: i32 = 0;
			let mut gx_g: i32 = 0;
			let mut gy_g: i32 = 0;
			let mut gx_b: i32 = 0;
			let mut gy_b: i32 = 0;

			for ky in 0..3 {
				for kx in 0..3 {
					let px = x + kx - 1;
					let py = y + ky - 1;

					let pixel_idx = (py * width + px) * 3;

					let r = raw[pixel_idx] as i32;
					let g = raw[pixel_idx + 1] as i32;
					let b = raw[pixel_idx + 2] as i32;

					let kernel_x = SOBEL_GX[ky][kx] as i32;
					let kernel_y = SOBEL_GY[ky][kx] as i32;

					gx_r += r * kernel_x;
					gy_r += r * kernel_y;
					gx_g += g * kernel_x;
					gy_g += g * kernel_y;
					gx_b += b * kernel_x;
					gy_b += b * kernel_y;
				}
			}

			let mag_r = ((gx_r * gx_r + gy_r * gy_r) as f64).sqrt() as u16;
			let mag_g = ((gx_g * gx_g + gy_g * gy_g) as f64).sqrt() as u16;
			let mag_b = ((gx_b * gx_b + gy_b * gy_b) as f64).sqrt() as u16;

			let max_magnitude = mag_r.max(mag_g).max(mag_b);

			if max_magnitude > threshold {
				edge_map.set_edge(x, y);
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
			if edge_map.is_edge(x, y) {
				for theta in 0..THETA_BINS {
					let (cos_t, sin_t) = accumulator.trig_for(theta);
					let rho_val = (x as f32) * cos_t + (y as f32) * sin_t;
					accumulator.vote(rho_val, theta);
				}
			}
		}
	}
}

fn detect_peaks(
	accumulator: &HoughAccumulator, vote_threshold: Votes, max_lines: usize,
) -> Vec<HoughLine> {
	let mut peaks: Vec<HoughLine> = Vec::new();

	for theta in 0..accumulator.theta_bins {
		for rho_idx in 0..accumulator.rho_bins {
			let votes = accumulator.get_votes(rho_idx, theta);
			if votes < vote_threshold {
				continue;
			}

			let mut is_max = true;
			'nms: for dt in 0..=NMS_WINDOW_THETA {
				for dr in 0..=NMS_WINDOW_RHO {
					if dt == 0 && dr == 0 {
						continue;
					}

					let check_offsets = [
						(dr as isize, dt as isize),
						(-(dr as isize), dt as isize),
						(dr as isize, -(dt as isize)),
						(-(dr as isize), -(dt as isize)),
					];

					for (r_off, t_off) in check_offsets {
						let r_check = (rho_idx as isize) + r_off;
						let t_check = (theta as isize) + t_off;

						if r_check >= 0 && r_check < (accumulator.rho_bins as isize) {
							let t_wrapped_usize =
								t_check.rem_euclid(accumulator.theta_bins as isize) as usize;
							debug_assert!(t_wrapped_usize < accumulator.theta_bins);
							let r_idx = r_check as usize;

							let other_votes = accumulator.get_votes(r_idx, t_wrapped_usize);
							if other_votes > votes {
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

	peaks.sort_by(|a, b| b.votes.cmp(&a.votes));
	peaks.truncate(max_lines);

	peaks
}

fn measure_hough_line_segment(
	edge_map: &EdgeMap, line: &HoughLine, gap_tol: usize, intersections: &mut Vec<Point>,
	edge_samples: &mut Vec<(usize, Point)>,
) -> Option<MeasuredSegment> {
	let width = edge_map.width();
	let height = edge_map.height();

	if width == 0 || height == 0 {
		return None;
	}

	intersections.clear();
	edge_samples.clear();

	let theta_rad = (line.theta as f64) * PI / 180.0;
	let cos_t = theta_rad.cos() as f32;
	let sin_t = theta_rad.sin() as f32;
	let rho = line.rho;

	// x = 0
	if sin_t.abs() > EPSILON {
		let y = rho / sin_t;
		if (0.0..(height as f32)).contains(&y) {
			intersections.push(Point::new(0.0, y));
		}
	}

	// x = width-1
	if sin_t.abs() > EPSILON {
		let x = (width - 1) as f32;
		let y = (rho - x * cos_t) / sin_t;
		if (0.0..(height as f32)).contains(&y) {
			intersections.push(Point::new(x, y));
		}
	}

	// y = 0
	if cos_t.abs() > EPSILON {
		let x = rho / cos_t;
		if (0.0..(width as f32)).contains(&x) {
			intersections.push(Point::new(x, 0.0));
		}
	}

	// y = height-1
	if cos_t.abs() > EPSILON {
		let y = (height - 1) as f32;
		let x = (rho - y * sin_t) / cos_t;
		if (0.0..(width as f32)).contains(&x) {
			intersections.push(Point::new(x, y));
		}
	}

	if intersections.len() < 2 {
		return None;
	}

	let (p0, p1) = find_most_distant_points(intersections);

	if (p0.x - p1.x).abs() < EPSILON && (p0.y - p1.y).abs() < EPSILON {
		return None;
	}

	let (dx, dy) = (p1.x - p0.x, p1.y - p0.y);
	let steps = (dx.abs().max(dy.abs()).ceil() as usize).max(1);

	for i in 0..=steps {
		let t = i as f32 / steps as f32;
		let x = p0.x + dx * t;
		let y = p0.y + dy * t;

		if x >= 0.0 && y >= 0.0 {
			let xi = x.round() as usize;
			let yi = y.round() as usize;

			if xi < width && yi < height && edge_map.is_edge(xi, yi) {
				edge_samples.push((i, Point::new(x, y)));
			}
		}
	}

	if edge_samples.is_empty() {
		return None;
	}

	let (best_start_idx, best_end_idx) = find_longest_run_with_gaps(edge_samples, gap_tol)?;

	if best_start_idx >= edge_samples.len() || best_end_idx >= edge_samples.len() {
		return None;
	}

	let start = edge_samples[best_start_idx].1;
	let end = edge_samples[best_end_idx].1;

	let length_px = {
		let dx = end.x - start.x;
		let dy = end.y - start.y;
		(dx * dx + dy * dy).sqrt()
	};

	let potential_length = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();
	if length_px < potential_length * MIN_SEGMENT_FRACTION {
		return None;
	}

	Some(MeasuredSegment {
		start,
		end,
		length_px,
	})
}

fn find_most_distant_points(points: &[Point]) -> (Point, Point) {
	debug_assert!(points.len() >= 2);
	let mut max_dist_sq = 0.0f32;
	let mut best_pair = (points[0], points[1]);

	for i in 0..points.len() {
		for j in (i + 1)..points.len() {
			let dx = points[j].x - points[i].x;
			let dy = points[j].y - points[i].y;
			let dist_sq = dx * dx + dy * dy;
			if dist_sq > max_dist_sq {
				max_dist_sq = dist_sq;
				best_pair = (points[i], points[j]);
			}
		}
	}

	best_pair
}

fn find_longest_run_with_gaps(
	samples: &[(usize, Point)], gap_tol: usize,
) -> Option<(usize, usize)> {
	if samples.is_empty() {
		return None;
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

	Some((best_start, best_end))
}

fn cluster_lines_full(lines: &[HoughLine]) -> (Vec<HoughLine>, Vec<HoughLine>) {
	let mut vertical_lines: Vec<HoughLine> = Vec::new();
	let mut horizontal_lines: Vec<HoughLine> = Vec::new();

	for line in lines {
		let theta = line.theta;
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
	start: f32,
	end: f32,
	spacing: f32,
}

fn compute_vertical_span(
	vertical_lines: &[HoughLine], edge_map: &EdgeMap, v_grid: &GridPattern,
) -> Option<(f32, f32)> {
	let mut y_starts: Vec<f32> = Vec::new();
	let mut y_ends: Vec<f32> = Vec::new();

	let tolerance = (v_grid.spacing * 0.25).min(30.0);

	let mut intersections = Vec::with_capacity(4);
	let mut edge_samples = Vec::with_capacity(1000);

	for line in vertical_lines {
		if line.rho < v_grid.start - tolerance || line.rho > v_grid.end + tolerance {
			continue;
		}

		if let Some(segment) = measure_hough_line_segment(
			edge_map,
			line,
			SEGMENT_GAP_TOLERANCE,
			&mut intersections,
			&mut edge_samples,
		) {
			let y_min = segment.start.y.min(segment.end.y);
			let y_max = segment.start.y.max(segment.end.y);

			let expected_height = v_grid.spacing * 8.0;
			if segment.length_px >= expected_height * 0.4 {
				y_starts.push(y_min);
				y_ends.push(y_max);
			}
		}
	}

	if y_starts.len() < 3 {
		return None;
	}

	y_starts.sort_by(|a, b| a.total_cmp(b));
	y_ends.sort_by(|a, b| a.total_cmp(b));

	let median_y_start = y_starts[y_starts.len() / 2];
	let median_y_end = y_ends[y_ends.len() / 2];

	Some((median_y_start, median_y_end))
}

fn filter_horizontals_by_vertical_span(
	horizontal_lines: &[HoughLine], edge_map: &EdgeMap, v_span: (f32, f32), v_grid: &GridPattern,
) -> Vec<HoughLine> {
	let (v_y_start, v_y_end) = v_span;
	let mut filtered_lines: Vec<HoughLine> = Vec::new();
	let margin = v_grid.spacing * 0.25;

	let mut intersections = Vec::with_capacity(4);
	let mut edge_samples = Vec::with_capacity(1000);

	for line in horizontal_lines {
		if line.rho < v_y_start - margin || line.rho > v_y_end + margin {
			continue;
		}

		if let Some(segment) = measure_hough_line_segment(
			edge_map,
			line,
			SEGMENT_GAP_TOLERANCE,
			&mut intersections,
			&mut edge_samples,
		) {
			let x_min = segment.start.x.min(segment.end.x);
			let x_max = segment.start.x.max(segment.end.x);

			let overlap_start = x_min.max(v_grid.start);
			let overlap_end = x_max.min(v_grid.end);
			let overlap = (overlap_end - overlap_start).max(0.0);

			let v_width = v_grid.end - v_grid.start;
			let overlap_ratio = if v_width > 0.0 {
				overlap / v_width
			} else {
				0.0
			};

			if overlap_ratio >= 0.66 {
				filtered_lines.push(*line);
			}
		}
	}

	filtered_lines
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

fn find_best_grid(lines: &[HoughLine], expected_spacing: Option<f32>) -> Option<GridPattern> {
	if lines.len() < 4 {
		trace!("Not enough lines to search for grid ({} < 4)", lines.len());
		return None;
	}

	let mut deduped: Vec<HoughLine> = Vec::new();
	if let Some(first) = lines.first() {
		let mut current_group_best = *first;

		for line in lines.iter().skip(1) {
			if (line.rho - current_group_best.rho).abs() < 5.0 {
				if line.votes > current_group_best.votes {
					current_group_best = *line;
				}
			} else {
				deduped.push(current_group_best);
				current_group_best = *line;
			}
		}
		deduped.push(current_group_best);
	}

	trace!("Deduped lines: {} -> {}", lines.len(), deduped.len());

	if tracing::enabled!(tracing::Level::TRACE) {
		for (i, line) in deduped.iter().enumerate() {
			trace!("  [{}] rho={:.1} votes={}", i, line.rho, line.votes);
		}
	}

	if deduped.len() < 4 {
		trace!("Not enough lines after dedup ({} < 4)", deduped.len());
		return None;
	}

	let mut best_grid: Option<GridPattern> = None;
	let mut max_score = -f32::INFINITY;

	// Ideal chess grid is 9 lines (8 squares).
	let target_count = 9;

	let max_votes = deduped.iter().map(|l| l.votes).max().unwrap_or(1) as f32;

	for i in 0..deduped.len() {
		for j in (i + 1)..deduped.len() {
			let diff = deduped[j].rho - deduped[i].rho;

			if diff < 8.0 {
				continue;
			}

			if let Some(target) = expected_spacing
				&& (diff - target).abs() > target * 0.25
			{
				continue;
			}

			let spacing = diff;
			let (mut matched_lines, mut total_votes) =
				count_grid_matches(&deduped, deduped[i].rho, spacing);

			let mut matched_count = matched_lines.len();

			if matched_count > target_count {
				let mut trimmed = false;

				while matched_count > target_count {
					let first = matched_lines.first().unwrap();
					let last = matched_lines.last().unwrap();

					if first.votes < last.votes {
						total_votes -= first.votes;
						matched_lines.remove(0);
					} else {
						total_votes -= last.votes;
						matched_lines.pop();
					}
					trimmed = true;
					matched_count = matched_lines.len();
				}

				if trimmed {
					total_votes = matched_lines.iter().map(|l| l.votes).sum();
				}
			}

			if !(7..=13).contains(&matched_count) {
				continue;
			}

			let first_match = matched_lines.first().unwrap().rho;
			let last_match = matched_lines.last().unwrap().rho;

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

			if span_ratio > 8.6 {
				continue;
			}
			if span_ratio < 4.0 {
				continue;
			}

			let ideal_span = 8.0;
			let span_diff = (span_ratio - ideal_span).abs();
			let compactness_score = (25.0 - span_diff * 40.0).max(0.0);

			let avg_votes = total_votes as f32 / matched_count as f32;
			let strength_score = (avg_votes / max_votes) * 50.0;

			let score = count_score + spacing_score + compactness_score + strength_score;

			trace!(
				"  Grid candidate: start={:.1} space={:.1} count={} score={:.1} (C={:.1} S={:.1} Comp={:.1} Str={:.1})",
				first_match,
				spacing,
				matched_count,
				score,
				count_score,
				spacing_score,
				compactness_score,
				strength_score
			);

			if score > max_score {
				max_score = score;
				best_grid = Some(GridPattern {
					start: first_match,
					end: last_match,
					spacing,
				});
			}
		}
	}

	if best_grid.is_none() {
		trace!("No valid grid pattern found. Max score: {:.1}", max_score);
	}

	best_grid
}

fn count_grid_matches(lines: &[HoughLine], start: f32, spacing: f32) -> (Vec<HoughLine>, u32) {
	let mut matches = Vec::new();
	let mut total_votes = 0;

	let tolerance = (spacing * 0.25).min(30.0);

	for k in -8..=12 {
		let target = start + (k as f32 * spacing);

		let best_match = lines
			.iter()
			.filter(|l| (l.rho - target).abs() <= tolerance)
			.max_by_key(|l| l.votes);

		if let Some(line) = best_match {
			matches.push(*line);
			total_votes += line.votes;
		}
	}

	matches.sort_by(|a, b| a.rho.total_cmp(&b.rho));

	(matches, total_votes)
}

const GRID_LINES: usize = 9;

fn calculate_grid_score(rect: &Rect, edge_map: &EdgeMap) -> f32 {
	let mut hits: usize = 0;
	let mut total: usize = 0;
	let cols = (GRID_LINES - 1) as f32;
	let rows = (GRID_LINES - 1) as f32;

	let x0 = rect.x();
	let y0 = rect.y();
	let w = rect.width();
	let h = rect.height();

	if w <= 0.0 || h <= 0.0 {
		return 0.0;
	}

	let map_w = edge_map.width();
	let map_h = edge_map.height();

	let max_samples = 400usize;
	let vert_len = (h as usize).max(1);
	let hor_len = (w as usize).max(1);
	let v_stride = if vert_len > max_samples {
		((vert_len as f32 / max_samples as f32).ceil() as usize).max(1)
	} else {
		1
	};
	let h_stride = if hor_len > max_samples {
		((hor_len as f32 / max_samples as f32).ceil() as usize).max(1)
	} else {
		1
	};

	for i in 0..GRID_LINES {
		let fx = x0 + (i as f32) * (w / cols);
		let xi = fx.round() as isize;
		let mut j = 0usize;
		while j < vert_len {
			let yi = (y0.round() as isize) + (j as isize);
			if xi >= 0 && yi >= 0 {
				let xu = xi as usize;
				let yu = yi as usize;
				if xu < map_w && yu < map_h {
					total += 1;
					if edge_map.is_edge(xu, yu) {
						hits += 1;
					}
				}
			}
			j = j.saturating_add(v_stride);
		}
	}

	for i in 0..GRID_LINES {
		let fy = y0 + (i as f32) * (h / rows);
		let yi = fy.round() as isize;
		let mut j = 0usize;
		while j < hor_len {
			let xi = (x0.round() as isize) + (j as isize);
			if xi >= 0 && yi >= 0 {
				let xu = xi as usize;
				let yu = yi as usize;
				if xu < map_w && yu < map_h {
					total += 1;
					if edge_map.is_edge(xu, yu) {
						hits += 1;
					}
				}
			}
			j = j.saturating_add(h_stride);
		}
	}

	if total == 0 {
		0.0
	} else {
		(hits as f32) / (total as f32)
	}
}

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

pub struct BoardStabilizer {
	smoothed_rect: Option<Rect>,
	alpha: f32,
}

impl BoardStabilizer {
	pub fn new(smoothing_factor: f32) -> Self {
		Self {
			smoothed_rect: None,
			alpha: smoothing_factor.clamp(0.0, 1.0),
		}
	}

	pub fn update(&mut self, new_detection: Option<DetectedBoard>) -> Option<DetectedBoard> {
		match (self.smoothed_rect, new_detection) {
			(Some(prev), Some(curr)) => {
				if (prev.x() - curr.rect.x()).abs() > 50.0
					|| (prev.width() - curr.rect.width()).abs() > 50.0
				{
					self.smoothed_rect = Some(curr.rect);
				} else {
					let new_x = prev.x() + (curr.rect.x() - prev.x()) * self.alpha;
					let new_y = prev.y() + (curr.rect.y() - prev.y()) * self.alpha;
					let new_w = prev.width() + (curr.rect.width() - prev.width()) * self.alpha;
					let new_h = prev.height() + (curr.rect.height() - prev.height()) * self.alpha;

					self.smoothed_rect = Some(Rect::new(new_x, new_y, new_w, new_h));
				}
			}
			(None, Some(curr)) => {
				self.smoothed_rect = Some(curr.rect);
			}
			(_, None) => {
				self.smoothed_rect = None;
			}
		}

		self.smoothed_rect.map(DetectedBoard::new)
	}
}
