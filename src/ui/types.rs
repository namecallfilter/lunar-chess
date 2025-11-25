use raqote::SolidSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
	pub a: u8,
	pub r: u8,
	pub g: u8,
	pub b: u8,
}

impl Color {
	pub const fn new(a: u8, r: u8, g: u8, b: u8) -> Self {
		Self { a, r, g, b }
	}

	pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
		Self { a: 255, r, g, b }
	}

	pub const fn with_alpha(self, a: u8) -> Self {
		Self { a, ..self }
	}

	pub fn to_solid_source(self) -> SolidSource {
		SolidSource::from_unpremultiplied_argb(self.a, self.r, self.g, self.b)
	}
}

#[derive(Debug, Clone, Copy)]
pub struct Point2D {
	pub x: f32,
	pub y: f32,
}

impl Point2D {
	pub const fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}
}

#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
	pub x: f32,
	pub y: f32,
}

impl Vec2 {
	pub const fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}

	pub fn perpendicular(self) -> Self {
		Self {
			x: -self.y,
			y: self.x,
		}
	}

	pub fn negate(self) -> Self {
		Self {
			x: -self.x,
			y: -self.y,
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub struct ArrowStyle {
	pub shaft_width: f32,
	pub head_length: f32,
	pub head_width: f32,
	pub margin: f32,
}

impl ArrowStyle {
	pub fn from_cell_size(cell_width: f32, cell_height: f32) -> Self {
		const SHAFT_WIDTH_FRACTION: f32 = 0.25;
		const HEAD_LENGTH_FRACTION: f32 = 0.5;
		const HEAD_WIDTH_MULTIPLIER: f32 = 2.5;
		const MARGIN_FRACTION: f32 = 0.3;

		let cell_min = cell_width.min(cell_height);
		let shaft_width = cell_min * SHAFT_WIDTH_FRACTION;

		Self {
			shaft_width,
			head_length: cell_min * HEAD_LENGTH_FRACTION,
			head_width: shaft_width * HEAD_WIDTH_MULTIPLIER,
			margin: cell_min * MARGIN_FRACTION,
		}
	}
}
