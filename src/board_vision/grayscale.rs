use image::RgbaImage;

const R_WEIGHT: f32 = 0.299;
const G_WEIGHT: f32 = 0.587;
const B_WEIGHT: f32 = 0.114;

pub fn to_grayscale(image: &RgbaImage) -> Vec<u8> {
	let (width, height) = (image.width(), image.height());
	let capacity = (width * height) as usize;
	let mut gray = Vec::with_capacity(capacity);

	for pixel in image.pixels() {
		let luminosity = (R_WEIGHT * pixel[0] as f32
			+ G_WEIGHT * pixel[1] as f32
			+ B_WEIGHT * pixel[2] as f32) as u8;
		gray.push(luminosity);
	}

	gray
}
