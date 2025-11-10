use image::RgbaImage;

pub fn to_grayscale(image: &RgbaImage) -> Vec<u8> {
	let (width, height) = (image.width(), image.height());
	let mut gray = Vec::with_capacity((width * height) as usize);

	for pixel in image.pixels() {
		let r = pixel[0] as f32;
		let g = pixel[1] as f32;
		let b = pixel[2] as f32;

		let luminosity = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
		gray.push(luminosity);
	}

	gray
}
