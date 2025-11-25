#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenDimensions {
	pub width: u32,
	pub height: u32,
}

impl ScreenDimensions {
	pub const fn new(width: u32, height: u32) -> Self {
		Self { width, height }
	}
}

impl Default for ScreenDimensions {
	fn default() -> Self {
		Self {
			width: 1920,
			height: 1080,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSize {
	pub width: f32,
	pub height: f32,
}

impl CellSize {
	pub const fn new(width: f32, height: f32) -> Self {
		Self { width, height }
	}
}
