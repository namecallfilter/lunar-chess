use std::f64::consts::PI;

use crate::vision::board_detection::{
	edges::EdgeMap,
	geometry::{EPSILON, Point, find_most_distant_points},
	hough::HoughLine,
};

pub const SEGMENT_GAP_TOLERANCE: usize = 15;
const MIN_SEGMENT_FRACTION: f32 = 0.1;

#[derive(Debug, Clone, Copy)]
pub struct MeasuredSegment {
	pub start: Point,
	pub end: Point,
	pub length_px: f32,
}

pub fn measure_hough_line_segment(
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

pub fn find_longest_run_with_gaps(
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
