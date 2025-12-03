use tracing::trace;

use crate::vision::board_detection::{
	edges::EdgeMap,
	hough::HoughLine,
	segments::{SEGMENT_GAP_TOLERANCE, measure_hough_line_segment},
};

const VERTICAL_THETA_TOLERANCE: usize = 10;
const HORIZONTAL_THETA_TOLERANCE: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct GridPattern {
	pub start: f32,
	pub end: f32,
	pub spacing: f32,
}

pub fn cluster_lines_full(lines: &[HoughLine]) -> (Vec<HoughLine>, Vec<HoughLine>) {
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

pub fn find_best_grid(lines: &[HoughLine], expected_spacing: Option<f32>) -> Option<GridPattern> {
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
			let (matched_lines_full, _) = count_grid_matches(&deduped, deduped[i].rho, spacing);

			let total_matched_count = matched_lines_full.len();

			if total_matched_count < 7 {
				continue;
			}

			let windows_count = if total_matched_count > target_count {
				total_matched_count - target_count + 1
			} else {
				1
			};

			for w_idx in 0..windows_count {
				let count = if total_matched_count > target_count {
					target_count
				} else {
					total_matched_count
				};

				let start_idx = if total_matched_count > target_count {
					w_idx
				} else {
					0
				};

				let matched_lines = &matched_lines_full[start_idx..start_idx + count];
				let matched_count = matched_lines.len();

				let total_votes: u32 = matched_lines.iter().map(|l| l.votes).sum();

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

				let avg_votes = total_votes as f32 / matched_count as f32;
				let strength_score = (avg_votes / max_votes) * 50.0;

				let mut points = Vec::with_capacity(matched_count);
				for line in matched_lines {
					let k = ((line.rho - first_match) / spacing).round();
					points.push((k, line.rho));
				}

				let n = points.len() as f32;
				let sum_x: f32 = points.iter().map(|(x, _)| x).sum();
				let sum_y: f32 = points.iter().map(|(_, y)| y).sum();
				let sum_xy: f32 = points.iter().map(|(x, y)| x * y).sum();
				let sum_xx: f32 = points.iter().map(|(x, _)| x * x).sum();

				let denominator = n * sum_xx - sum_x * sum_x;
				let (refined_spacing, refined_start) = if denominator.abs() > 1e-6 {
					let slope = (n * sum_xy - sum_x * sum_y) / denominator;
					let intercept = (sum_y - slope * sum_x) / n;
					(slope, intercept)
				} else {
					(spacing, first_match)
				};

				let mut total_error_sq = 0.0;
				for (k, rho) in points {
					let target = refined_start + k * refined_spacing;
					let error = (rho - target).abs();
					total_error_sq += error * error;
				}

				let rmse = (total_error_sq / n).sqrt();
				let alignment_score = (1.0 - rmse / 15.0).max(0.0) * 20.0;

				let span = last_match - first_match;
				let span_ratio = span / refined_spacing;

				if !(4.0..=8.6).contains(&span_ratio) {
					continue;
				}

				let ideal_span = 8.0;
				let span_diff = (span_ratio - ideal_span).abs();
				let compactness_score = (25.0 - span_diff * 40.0).max(0.0);

				let variance_score = if matched_count > 2 {
					let mut gaps = Vec::with_capacity(matched_count - 1);

					for k in 0..matched_count - 1 {
						let gap = matched_lines[k + 1].rho - matched_lines[k].rho;
						gaps.push(gap);
					}

					let mean_gap: f32 = gaps.iter().sum::<f32>() / gaps.len() as f32;
					let variance: f32 = gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f32>()
						/ gaps.len() as f32;
					let std_dev = variance.sqrt();

					if mean_gap > 0.0 {
						let cv = std_dev / mean_gap;
						(1.0 - cv * 10.0).max(0.0) * 20.0
					} else {
						0.0
					}
				} else {
					0.0
				};

				let score = count_score
					+ spacing_score + compactness_score
					+ strength_score
					+ variance_score
					+ alignment_score;

				trace!(
					"  Grid candidate (w={}): start={:.1} space={:.1} count={} score={:.1} (C={:.1} S={:.1} Comp={:.1} Str={:.1} Var={:.1} Align={:.1})",
					w_idx,
					first_match,
					spacing,
					matched_count,
					score,
					count_score,
					spacing_score,
					compactness_score,
					strength_score,
					variance_score,
					alignment_score
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
	}

	if best_grid.is_none() {
		trace!("No valid grid pattern found. Max score: {:.1}", max_score);
	}

	best_grid
}

pub fn count_grid_matches(lines: &[HoughLine], start: f32, spacing: f32) -> (Vec<HoughLine>, u32) {
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

pub fn compute_vertical_span(
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

pub fn filter_horizontals_by_vertical_span(
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
