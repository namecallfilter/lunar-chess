use rayon::prelude::*;

pub struct EdgeDetector {
	width: usize,
	height: usize,
}

impl EdgeDetector {
	pub const fn new(width: usize, height: usize) -> Self {
		Self { width, height }
	}

	pub fn sobel_magnitude(&self, gray: &[u8]) -> Vec<f32> {
		let len = gray.len();
		let mut magnitude = vec![0.0f32; len];

		const GX: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
		const GY: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

		let width = self.width;
		let height = self.height;

		magnitude
			.par_chunks_mut(width)
			.enumerate()
			.for_each(|(y, row_out)| {
				if y == 0 || y == height - 1 {
					return;
				}

				let row_base = y * width;
				let prev_row = (y - 1) * width;
				let next_row = (y + 1) * width;

				for x in 1..(width - 1) {
					let px_prev = x - 1;
					let px_next = x + 1;

					let p0 = gray[prev_row + px_prev] as f32;
					let p1 = gray[prev_row + x] as f32;
					let p2 = gray[prev_row + px_next] as f32;
					let p3 = gray[row_base + px_prev] as f32;
					let p5 = gray[row_base + px_next] as f32;
					let p6 = gray[next_row + px_prev] as f32;
					let p7 = gray[next_row + x] as f32;
					let p8 = gray[next_row + px_next] as f32;

					let sum_x =
						p0 * GX[0] + p2 * GX[2] + p3 * GX[3] + p5 * GX[5] + p6 * GX[6] + p8 * GX[8];
					let sum_y =
						p0 * GY[0] + p1 * GY[1] + p2 * GY[2] + p6 * GY[6] + p7 * GY[7] + p8 * GY[8];

					row_out[x] = (sum_x * sum_x + sum_y * sum_y).sqrt();
				}
			});

		magnitude
	}

	pub fn simple_edges(&self, gray: &[u8], threshold: f32) -> Vec<bool> {
		let magnitude = self.sobel_magnitude(gray);
		magnitude.iter().map(|&m| m > threshold).collect()
	}
}
