use image::RgbImage;

#[derive(Debug)]
pub struct EdgeMap {
	pub data: Vec<u8>, // 0 or 1
	pub width: usize,
	pub height: usize,
}

impl EdgeMap {
	pub fn new(width: usize, height: usize) -> Self {
		Self {
			data: vec![0; width * height],
			width,
			height,
		}
	}

	#[inline]
	pub fn idx(&self, x: usize, y: usize) -> usize {
		y * self.width + x
	}

	#[inline]
	pub fn set_edge(&mut self, x: usize, y: usize) {
		if x < self.width && y < self.height {
			let idx = self.idx(x, y);
			self.data[idx] = 1;
		}
	}

	#[inline]
	pub fn is_edge(&self, x: usize, y: usize) -> bool {
		if x < self.width && y < self.height {
			self.data[self.idx(x, y)] != 0
		} else {
			false
		}
	}

	pub fn width(&self) -> usize {
		self.width
	}

	pub fn height(&self) -> usize {
		self.height
	}
}

const SOBEL_GX: [[i16; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
const SOBEL_GY: [[i16; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

pub fn sobel_edge_detection(image: &RgbImage, threshold: u16) -> EdgeMap {
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
