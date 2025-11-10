use std::f32::consts::PI;

const ANGLE_TOLERANCE: f32 = PI / 60.0;
const HORIZONTAL_MIN: f32 = PI / 60.0;
const HORIZONTAL_MAX: f32 = 59.0 * PI / 60.0;
const INTERSECTION_EPSILON: f32 = 0.001;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLine {
	pub rho: f32,
	pub theta: f32,
	pub votes: usize,
}

impl HoughLine {
	#[inline]
	pub fn is_horizontal(&self) -> bool {
		let angle = self.theta.abs();
		!(HORIZONTAL_MIN..=HORIZONTAL_MAX).contains(&angle)
	}

	#[inline]
	pub fn is_vertical(&self) -> bool {
		(self.theta - PI / 2.0).abs() < ANGLE_TOLERANCE
	}

	pub fn intersect(&self, other: &HoughLine) -> Option<(f32, f32)> {
		let (cos1, sin1) = (self.theta.cos(), self.theta.sin());
		let (cos2, sin2) = (other.theta.cos(), other.theta.sin());

		let denom = cos1 * sin2 - sin1 * cos2;

		if denom.abs() < INTERSECTION_EPSILON {
			return None;
		}

		let x = (self.rho * sin2 - other.rho * sin1) / denom;
		let y = (other.rho * cos1 - self.rho * cos2) / denom;

		Some((x, y))
	}
}

pub struct LineDetector {
	width: usize,
	height: usize,
	rho_resolution: f32,
	theta_resolution: f32,
	max_rho: f32,
	sin_table: Vec<f32>,
	cos_table: Vec<f32>,
}

impl LineDetector {
	pub fn new(width: usize, height: usize) -> Self {
		let max_rho = ((width.pow(2) + height.pow(2)) as f32).sqrt();
		let theta_resolution = PI / 180.0;
		let num_thetas = (PI / theta_resolution) as usize;

		let sin_table: Vec<f32> = (0..num_thetas)
			.map(|i| (i as f32 * theta_resolution).sin())
			.collect();
		let cos_table: Vec<f32> = (0..num_thetas)
			.map(|i| (i as f32 * theta_resolution).cos())
			.collect();

		Self {
			width,
			height,
			rho_resolution: 1.0,
			theta_resolution,
			max_rho,
			sin_table,
			cos_table,
		}
	}

	pub fn detect_lines(&self, edges: &[bool], threshold: usize) -> Vec<HoughLine> {
		let num_thetas = self.sin_table.len();
		let num_rhos = (2.0 * self.max_rho / self.rho_resolution) as usize;

		tracing::debug!(
			"Hough transform: {} theta bins, {} rho bins, max_rho={:.1}",
			num_thetas,
			num_rhos,
			self.max_rho
		);

		let mut accumulator = vec![0usize; num_thetas * num_rhos];

		let edge_count = edges.iter().filter(|&&e| e).count();
		tracing::debug!("Processing {} edge pixels", edge_count);

		for y in 0..self.height {
			for x in 0..self.width {
				let idx = y * self.width + x;
				if !edges[idx] {
					continue;
				}

				let xf = x as f32;
				let yf = y as f32;

				for theta_idx in 0..num_thetas {
					let rho = xf * self.cos_table[theta_idx] + yf * self.sin_table[theta_idx];
					let rho_idx = ((rho + self.max_rho) / self.rho_resolution) as usize;

					if rho_idx < num_rhos {
						accumulator[theta_idx * num_rhos + rho_idx] += 1;
					}
				}
			}
		}

		let mut lines = Vec::new();
		for theta_idx in 0..num_thetas {
			for rho_idx in 0..num_rhos {
				let votes = accumulator[theta_idx * num_rhos + rho_idx];

				if votes >= threshold
					&& self.is_local_maximum(&accumulator, theta_idx, rho_idx, num_thetas, num_rhos)
				{
					let theta = theta_idx as f32 * self.theta_resolution;
					let rho = rho_idx as f32 * self.rho_resolution - self.max_rho;

					lines.push(HoughLine { rho, theta, votes });
				}
			}
		}

		lines.sort_unstable_by(|a, b| b.votes.cmp(&a.votes));

		tracing::debug!(
			"Detected {} lines above threshold {}",
			lines.len(),
			threshold
		);

		lines
	}

	fn is_local_maximum(
		&self, accumulator: &[usize], theta_idx: usize, rho_idx: usize, num_thetas: usize,
		num_rhos: usize,
	) -> bool {
		let center_votes = accumulator[theta_idx * num_rhos + rho_idx];

		for dt in -1..=1 {
			for dr in -1..=1 {
				if dt == 0 && dr == 0 {
					continue;
				}

				let t = theta_idx as i32 + dt;
				let r = rho_idx as i32 + dr;

				if t < 0 || r < 0 || t >= num_thetas as i32 || r >= num_rhos as i32 {
					continue;
				}

				let neighbor_votes = accumulator[t as usize * num_rhos + r as usize];
				if neighbor_votes > center_votes {
					return false;
				}
			}
		}

		true
	}

	pub fn detect_orthogonal_lines(
		&self, edges: &[bool], threshold: usize,
	) -> (Vec<HoughLine>, Vec<HoughLine>) {
		let all_lines = self.detect_lines(edges, threshold);

		let mut horizontal = Vec::new();
		let mut vertical = Vec::new();

		for line in all_lines {
			if line.is_horizontal() {
				horizontal.push(line);
			} else if line.is_vertical() {
				vertical.push(line);
			}
		}

		(horizontal, vertical)
	}

	pub fn cluster_lines(&self, lines: &[HoughLine], distance_threshold: f32) -> Vec<HoughLine> {
		if lines.is_empty() {
			return Vec::new();
		}

		let mut clusters: Vec<Vec<HoughLine>> = Vec::new();
		let theta_threshold = PI / 90.0;

		for line in lines {
			let mut found_cluster = false;

			for cluster in &mut clusters {
				let representative = &cluster[0];
				let rho_diff = (line.rho - representative.rho).abs();
				let theta_diff = (line.theta - representative.theta).abs();

				if rho_diff < distance_threshold && theta_diff < theta_threshold {
					cluster.push(*line);
					found_cluster = true;
					break;
				}
			}

			if !found_cluster {
				clusters.push(vec![*line]);
			}
		}

		tracing::debug!(
			"Clustered {} lines into {} clusters (threshold: {:.1})",
			lines.len(),
			clusters.len(),
			distance_threshold
		);

		clusters
			.into_iter()
			.filter_map(|cluster| cluster.into_iter().max_by_key(|l| l.votes))
			.collect()
	}
}
