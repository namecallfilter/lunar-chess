#[derive(Debug, Clone, Copy)]
pub struct Point {
	pub x: f32,
	pub y: f32,
}

impl Point {
	pub fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}
}

pub const EPSILON: f32 = 1e-6;

pub fn find_most_distant_points(points: &[Point]) -> (Point, Point) {
	debug_assert!(points.len() >= 2);

	let mut max_dist_sq = 0.0f32;
	let mut best_pair = (points[0], points[1]);

	for i in 0..points.len() {
		for j in (i + 1)..points.len() {
			let dx = points[j].x - points[i].x;
			let dy = points[j].y - points[i].y;
			let dist_sq = dx * dx + dy * dy;
			if dist_sq > max_dist_sq {
				max_dist_sq = dist_sq;
				best_pair = (points[i], points[j]);
			}
		}
	}

	best_pair
}
