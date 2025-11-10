pub struct EdgeDetector {
	width: usize,
	height: usize,
}

impl EdgeDetector {
	pub fn new(width: usize, height: usize) -> Self {
		Self { width, height }
	}

	pub fn sobel(&self, gray: &[u8]) -> (Vec<f32>, Vec<f32>) {
		let mut magnitude = vec![0.0f32; gray.len()];
		let mut direction = vec![0.0f32; gray.len()];

		let gx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
		let gy = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

		for y in 1..(self.height - 1) {
			for x in 1..(self.width - 1) {
				let mut sum_x = 0.0;
				let mut sum_y = 0.0;

				for ky in 0..3 {
					for kx in 0..3 {
						let py = y + ky - 1;
						let px = x + kx - 1;
						let pixel_value = gray[py * self.width + px] as f32;

						let kernel_idx = ky * 3 + kx;
						sum_x += pixel_value * gx[kernel_idx];
						sum_y += pixel_value * gy[kernel_idx];
					}
				}

				let idx = y * self.width + x;
				magnitude[idx] = (sum_x * sum_x + sum_y * sum_y).sqrt();
				direction[idx] = sum_y.atan2(sum_x);
			}
		}

		(magnitude, direction)
	}

	pub fn simple_edges(&self, gray: &[u8], threshold: f32) -> Vec<bool> {
		let (magnitude, _) = self.sobel(gray);
		magnitude.iter().map(|&m| m > threshold).collect()
	}
}
