#[derive(Clone, Debug)]
pub struct DetectedBoard {
	pub x: f32,
	pub y: f32,
	pub width: f32,
	pub height: f32,
	pub playing_as_white: bool,
}

impl DetectedBoard {
	#[inline]
	pub fn cell_size(&self) -> (f32, f32) {
		(self.width / 8.0, self.height / 8.0)
	}
}

#[derive(Clone, Debug)]
pub struct DetectedPiece {
	pub x: f32,
	pub y: f32,
	pub width: f32,
	pub height: f32,
	pub piece_type: char,
	pub confidence: f32,
}
