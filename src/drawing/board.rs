use font_kit::{family_name::FamilyName, properties::Properties, source::SystemSource};
use raqote::{
	DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};

#[derive(Clone, Debug)]
pub struct DetectedBoard {
	pub x: f32,
	pub y: f32,
	pub width: f32,
	pub height: f32,
	pub confidence: f32,
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

impl DetectedBoard {
	pub fn from_yolo_output(
		predictions: &ndarray::ArrayViewD<f32>, img_width: u32, img_height: u32,
		confidence_threshold: f32,
	) -> Option<Self> {
		let shape = predictions.shape();
		if shape.len() < 3 {
			tracing::warn!("Invalid YOLO output shape for board detection");
			return None;
		}

		let num_detections = shape[2];
		let scale_factor = 640.0;
		let scale_x = img_width as f32 / scale_factor;
		let scale_y = img_height as f32 / scale_factor;

		let mut best_detection: Option<DetectedBoard> = None;
		let mut max_confidence = confidence_threshold;

		for i in 0..num_detections {
			let confidence = predictions[[0, 4, i]];

			if confidence > max_confidence {
				let x_center = predictions[[0, 0, i]];
				let y_center = predictions[[0, 1, i]];
				let width = predictions[[0, 2, i]];
				let height = predictions[[0, 3, i]];

				let x = (x_center - width / 2.0) * scale_x;
				let y = (y_center - height / 2.0) * scale_y;
				let w = width * scale_x;
				let h = height * scale_y;

				best_detection = Some(DetectedBoard {
					x,
					y,
					width: w,
					height: h,
					confidence,
				});
				max_confidence = confidence;
			}
		}

		best_detection
	}
}

pub fn draw_board_outline(dt: &mut DrawTarget, x: f32, y: f32, width: f32, height: f32) {
	let mut pb = PathBuilder::new();
	pb.rect(x, y, width, height);
	let path = pb.finish();

	let stroke_style = StrokeStyle {
		width: 4.0,
		..Default::default()
	};

	dt.stroke(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(255, 0, 255, 0)),
		&stroke_style,
		&DrawOptions::new(),
	);
}

pub fn draw_chess_grid(dt: &mut DrawTarget, x: f32, y: f32, width: f32, height: f32) {
	let cell_width = width / 8.0;
	let cell_height = height / 8.0;

	let stroke_style = StrokeStyle {
		width: 1.5,
		..Default::default()
	};

	let grid_color = Source::Solid(SolidSource::from_unpremultiplied_argb(200, 255, 255, 0));

	for i in 1..8 {
		let mut pb = PathBuilder::new();
		let x_pos = x + (i as f32 * cell_width);
		pb.move_to(x_pos, y);
		pb.line_to(x_pos, y + height);
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}

	for i in 1..8 {
		let mut pb = PathBuilder::new();
		let y_pos = y + (i as f32 * cell_height);
		pb.move_to(x, y_pos);
		pb.line_to(x + width, y_pos);
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}
}

pub fn draw_move_arrow(
	dt: &mut DrawTarget, board: &DetectedBoard, from_file: u8, from_rank: u8, to_file: u8,
	to_rank: u8, color: (u8, u8, u8, u8),
) {
	if from_file >= 8 || from_rank >= 8 || to_file >= 8 || to_rank >= 8 {
		return;
	}

	let cell_width = board.width / 8.0;
	let cell_height = board.height / 8.0;

	let from_x = board.x + (from_file as f32 + 0.5) * cell_width;
	let from_y = board.y + (from_rank as f32 + 0.5) * cell_height;
	let to_x = board.x + (to_file as f32 + 0.5) * cell_width;
	let to_y = board.y + (to_rank as f32 + 0.5) * cell_height;

	let file_diff = (to_file as i8 - from_file as i8).abs();
	let rank_diff = (to_rank as i8 - from_rank as i8).abs();
	let is_knight_move = (file_diff == 2 && rank_diff == 1) || (file_diff == 1 && rank_diff == 2);

	if is_knight_move {
		draw_knight_arrow(dt, board, from_file, from_rank, to_file, to_rank, color);
	} else {
		draw_straight_arrow(
			dt,
			from_x,
			from_y,
			to_x,
			to_y,
			cell_width,
			cell_height,
			color,
		);
	}
}

fn draw_straight_arrow(
	dt: &mut DrawTarget, from_x: f32, from_y: f32, to_x: f32, to_y: f32, cell_width: f32,
	cell_height: f32, color: (u8, u8, u8, u8),
) {
	let dx = to_x - from_x;
	let dy = to_y - from_y;
	let length = (dx * dx + dy * dy).sqrt();

	if length < 1.0 {
		return;
	}

	let norm_dx = dx / length;
	let norm_dy = dy / length;

	let arrow_width = cell_width.min(cell_height) * 0.25;
	let arrow_head_length = cell_width.min(cell_height) * 0.5;
	let arrow_head_width = arrow_width * 2.5;

	let margin = cell_width.min(cell_height) * 0.3;
	let shaft_start_x = from_x + norm_dx * margin;
	let shaft_start_y = from_y + norm_dy * margin;

	let shaft_end_x = to_x - norm_dx * arrow_head_length;
	let shaft_end_y = to_y - norm_dy * arrow_head_length;

	let perp_x = -norm_dy;
	let perp_y = norm_dx;

	let mut pb = PathBuilder::new();
	pb.move_to(
		shaft_start_x + perp_x * arrow_width / 2.0,
		shaft_start_y + perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_end_x + perp_x * arrow_width / 2.0,
		shaft_end_y + perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_end_x - perp_x * arrow_width / 2.0,
		shaft_end_y - perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_start_x - perp_x * arrow_width / 2.0,
		shaft_start_y - perp_y * arrow_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);

	let mut pb = PathBuilder::new();
	pb.move_to(to_x, to_y);
	pb.line_to(
		shaft_end_x + perp_x * arrow_head_width / 2.0,
		shaft_end_y + perp_y * arrow_head_width / 2.0,
	);
	pb.line_to(
		shaft_end_x - perp_x * arrow_head_width / 2.0,
		shaft_end_y - perp_y * arrow_head_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);
}

fn draw_knight_arrow(
	dt: &mut DrawTarget, board: &DetectedBoard, from_file: u8, from_rank: u8, to_file: u8,
	to_rank: u8, color: (u8, u8, u8, u8),
) {
	let cell_width = board.width / 8.0;
	let cell_height = board.height / 8.0;

	let from_x = board.x + (from_file as f32 + 0.5) * cell_width;
	let from_y = board.y + (from_rank as f32 + 0.5) * cell_height;
	let to_x = board.x + (to_file as f32 + 0.5) * cell_width;
	let to_y = board.y + (to_rank as f32 + 0.5) * cell_height;

	let file_diff = (to_file as i8 - from_file as i8).abs();
	let rank_diff = (to_rank as i8 - from_rank as i8).abs();

	let mid_x = if file_diff == 2 { to_x } else { from_x };

	let mid_y = if rank_diff == 2 { to_y } else { from_y };

	let arrow_width = cell_width.min(cell_height) * 0.25;
	let arrow_head_length = cell_width.min(cell_height) * 0.5;
	let arrow_head_width = arrow_width * 2.5;
	let margin = cell_width.min(cell_height) * 0.3;

	let dx1 = mid_x - from_x;
	let dy1 = mid_y - from_y;
	let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();

	let dx2 = to_x - mid_x;
	let dy2 = to_y - mid_y;
	let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

	if len1 < 1.0 || len2 < 1.0 {
		return;
	}

	let norm_dx1 = dx1 / len1;
	let norm_dy1 = dy1 / len1;
	let perp_x1 = -norm_dy1;
	let perp_y1 = norm_dx1;

	let norm_dx2 = dx2 / len2;
	let norm_dy2 = dy2 / len2;
	let perp_x2 = -norm_dy2;
	let perp_y2 = norm_dx2;

	let seg1_start_x = from_x + norm_dx1 * margin;
	let seg1_start_y = from_y + norm_dy1 * margin;

	let seg2_end_x = to_x - norm_dx2 * arrow_head_length;
	let seg2_end_y = to_y - norm_dy2 * arrow_head_length;

	fn line_intersection(
		x1: f32, y1: f32, dx1: f32, dy1: f32, x2: f32, y2: f32, dx2: f32, dy2: f32,
	) -> Option<(f32, f32)> {
		let denom = dx1 * dy2 - dy1 * dx2;
		if denom.abs() < 0.0001 {
			return None;
		}
		let t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom;
		Some((x1 + t * dx1, y1 + t * dy1))
	}

	let outer_miter = line_intersection(
		seg1_start_x + perp_x1 * arrow_width / 2.0,
		seg1_start_y + perp_y1 * arrow_width / 2.0,
		norm_dx1,
		norm_dy1,
		seg2_end_x + perp_x2 * arrow_width / 2.0,
		seg2_end_y + perp_y2 * arrow_width / 2.0,
		-norm_dx2,
		-norm_dy2,
	);

	let inner_miter = line_intersection(
		seg1_start_x - perp_x1 * arrow_width / 2.0,
		seg1_start_y - perp_y1 * arrow_width / 2.0,
		norm_dx1,
		norm_dy1,
		seg2_end_x - perp_x2 * arrow_width / 2.0,
		seg2_end_y - perp_y2 * arrow_width / 2.0,
		-norm_dx2,
		-norm_dy2,
	);

	let mut pb = PathBuilder::new();

	pb.move_to(
		seg1_start_x + perp_x1 * arrow_width / 2.0,
		seg1_start_y + perp_y1 * arrow_width / 2.0,
	);

	if let Some((mx, my)) = outer_miter {
		pb.line_to(mx, my);
	} else {
		pb.line_to(
			mid_x + perp_x1 * arrow_width / 2.0,
			mid_y + perp_y1 * arrow_width / 2.0,
		);
	}

	pb.line_to(
		seg2_end_x + perp_x2 * arrow_width / 2.0,
		seg2_end_y + perp_y2 * arrow_width / 2.0,
	);

	pb.line_to(
		seg2_end_x - perp_x2 * arrow_width / 2.0,
		seg2_end_y - perp_y2 * arrow_width / 2.0,
	);

	if let Some((mx, my)) = inner_miter {
		pb.line_to(mx, my);
	} else {
		pb.line_to(
			mid_x - perp_x1 * arrow_width / 2.0,
			mid_y - perp_y1 * arrow_width / 2.0,
		);
	}

	pb.line_to(
		seg1_start_x - perp_x1 * arrow_width / 2.0,
		seg1_start_y - perp_y1 * arrow_width / 2.0,
	);

	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);

	let mut pb = PathBuilder::new();
	pb.move_to(to_x, to_y);
	pb.line_to(
		seg2_end_x + perp_x2 * arrow_head_width / 2.0,
		seg2_end_y + perp_y2 * arrow_head_width / 2.0,
	);
	pb.line_to(
		seg2_end_x - perp_x2 * arrow_head_width / 2.0,
		seg2_end_y - perp_y2 * arrow_head_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);
}

impl DetectedPiece {
	pub fn from_yolo_output(
		predictions: &ndarray::ArrayViewD<f32>, img_width: u32, img_height: u32,
		confidence_threshold: f32,
	) -> Vec<Self> {
		let shape = predictions.shape();
		if shape.len() < 3 {
			tracing::warn!("Invalid YOLO output shape for piece detection");
			return Vec::new();
		}

		let num_detections = shape[2];
		let num_classes = shape[1] - 4;
		let scale_factor = 640.0;
		let scale_x = img_width as f32 / scale_factor;
		let scale_y = img_height as f32 / scale_factor;

		let piece_chars = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P'];
		let mut pieces = Vec::new();

		for i in 0..num_detections {
			let x_center = predictions[[0, 0, i]];
			let y_center = predictions[[0, 1, i]];
			let width = predictions[[0, 2, i]];
			let height = predictions[[0, 3, i]];

			let mut max_class_idx = 0;
			let mut max_class_conf = 0.0f32;

			for class_idx in 0..num_classes.min(12) {
				let class_conf = predictions[[0, 4 + class_idx, i]];
				if class_conf > max_class_conf {
					max_class_conf = class_conf;
					max_class_idx = class_idx;
				}
			}

			if max_class_conf > confidence_threshold {
				let x = (x_center - width / 2.0) * scale_x;
				let y = (y_center - height / 2.0) * scale_y;
				let w = width * scale_x;
				let h = height * scale_y;

				pieces.push(DetectedPiece {
					x,
					y,
					width: w,
					height: h,
					piece_type: piece_chars[max_class_idx],
					confidence: max_class_conf,
				});
			}
		}

		nms_pieces(pieces, 0.45)
	}
}

fn nms_pieces(mut pieces: Vec<DetectedPiece>, iou_threshold: f32) -> Vec<DetectedPiece> {
	if pieces.is_empty() {
		return pieces;
	}

	pieces.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

	let mut keep = Vec::new();
	let mut suppressed = vec![false; pieces.len()];

	for i in 0..pieces.len() {
		if suppressed[i] {
			continue;
		}

		keep.push(pieces[i].clone());

		for j in (i + 1)..pieces.len() {
			if suppressed[j] {
				continue;
			}

			let iou = calculate_iou(&pieces[i], &pieces[j]);
			if iou > iou_threshold {
				suppressed[j] = true;
			}
		}
	}

	keep
}

fn calculate_iou(a: &DetectedPiece, b: &DetectedPiece) -> f32 {
	let x1_inter = a.x.max(b.x);
	let y1_inter = a.y.max(b.y);
	let x2_inter = (a.x + a.width).min(b.x + b.width);
	let y2_inter = (a.y + a.height).min(b.y + b.height);

	let inter_width = (x2_inter - x1_inter).max(0.0);
	let inter_height = (y2_inter - y1_inter).max(0.0);
	let inter_area = inter_width * inter_height;

	let area_a = a.width * a.height;
	let area_b = b.width * b.height;
	let union_area = area_a + area_b - inter_area;

	if union_area > 0.0 {
		inter_area / union_area
	} else {
		0.0
	}
}

pub fn draw_piece_labels(dt: &mut DrawTarget, pieces: &[DetectedPiece]) {
	let font = SystemSource::new()
		.select_best_match(
			&[FamilyName::Title("Arial".into()), FamilyName::SansSerif],
			&Properties::new(),
		)
		.map(|handle| handle.load())
		.ok();

	let Some(Ok(font)) = font else {
		tracing::warn!("Could not load system font for piece labels");
		return;
	};

	for piece in pieces {
		let mut pb = PathBuilder::new();
		let label_size = 30.0;
		pb.rect(piece.x, piece.y, label_size, label_size);
		let path = pb.finish();

		dt.fill(
			&path,
			&Source::Solid(SolidSource::from_unpremultiplied_argb(180, 50, 50, 50)),
			&DrawOptions::new(),
		);

		let text = piece.piece_type.to_string();
		let color = if piece.piece_type.is_uppercase() {
			SolidSource::from_unpremultiplied_argb(255, 255, 255, 255)
		} else {
			SolidSource::from_unpremultiplied_argb(255, 200, 200, 200)
		};

		dt.set_transform(&Transform::identity());
		dt.draw_text(
			&font,
			24.0,
			&text,
			Point::new(piece.x + 5.0, piece.y + 23.0),
			&Source::Solid(color),
			&DrawOptions::new(),
		);
	}

	dt.set_transform(&Transform::identity());
}
