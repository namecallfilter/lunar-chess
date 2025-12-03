use std::f64::consts::PI;

use crate::vision::board_detection::edges::EdgeMap;

pub type Rho = f32;
pub type Theta = usize;
pub type Votes = u32;

pub const THETA_BINS: usize = 180;
const NMS_WINDOW_RHO: usize = 10;
const NMS_WINDOW_THETA: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct HoughLine {
	pub rho: Rho,
	pub theta: Theta,
	pub votes: Votes,
}

pub struct HoughAccumulator {
	pub data: Vec<u32>,
	pub rho_bins: usize,
	pub theta_bins: usize,
	pub max_rho: f32,
	pub sin_table: [f32; THETA_BINS],
	pub cos_table: [f32; THETA_BINS],
}

impl HoughAccumulator {
	pub fn new(image_width: usize, image_height: usize) -> Option<Self> {
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
	pub fn rho_to_index(&self, rho: Rho) -> usize {
		let offset_rho = rho + self.max_rho;
		let idx = offset_rho.round() as isize;
		idx.clamp(0, (self.rho_bins as isize) - 1) as usize
	}

	#[inline]
	pub fn index_to_rho(&self, index: usize) -> Rho {
		index as f32 - self.max_rho
	}

	#[inline]
	pub fn vote(&mut self, rho: Rho, theta: Theta) {
		let rho_idx = self.rho_to_index(rho);

		if theta < self.theta_bins {
			let idx = theta * self.rho_bins + rho_idx;
			if idx < self.data.len() {
				self.data[idx] = self.data[idx].saturating_add(1);
			}
		}
	}

	#[inline]
	pub fn get_votes(&self, rho_idx: usize, theta: Theta) -> Votes {
		if theta < self.theta_bins && rho_idx < self.rho_bins {
			let idx = theta * self.rho_bins + rho_idx;
			if idx < self.data.len() {
				return self.data[idx];
			}
		}

		0
	}

	pub fn trig_for(&self, theta: Theta) -> (f32, f32) {
		if theta < self.theta_bins {
			(self.cos_table[theta], self.sin_table[theta])
		} else {
			debug_assert!(false, "theta out of bounds");
			(0.0, 0.0) // Should not happen if used correctly
		}
	}
}

pub fn hough_voting(edge_map: &EdgeMap, accumulator: &mut HoughAccumulator) {
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

pub fn detect_peaks(
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
