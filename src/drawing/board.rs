use font_kit::{family_name::FamilyName, properties::Properties, source::SystemSource};
use raqote::{
	DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};

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

pub fn draw_detected_board(dt: &mut DrawTarget, board: &DetectedBoard) {
	draw_board_outline(dt, board.x, board.y, board.width, board.height);
	draw_chess_grid(dt, board.x, board.y, board.width, board.height);

	// draw_confidence_label(dt, board);
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

// fn draw_confidence_label(dt: &mut DrawTarget, board: &DetectedBoard) {
// 	let label_height = 25.0;
// 	let label_width = 120.0;

// 	let mut pb = PathBuilder::new();
// 	pb.rect(
// 		board.x,
// 		board.y - label_height - 5.0,
// 		label_width,
// 		label_height,
// 	);
// 	let path = pb.finish();

// 	dt.fill(
// 		&path,
// 		&Source::Solid(SolidSource::from_unpremultiplied_argb(200, 0, 0, 0)),
// 		&DrawOptions::new(),
// 	);

// 	// Note: add text rendering
// }

pub fn draw_square_highlight(
	dt: &mut DrawTarget, board: &DetectedBoard, file: u8, rank: u8, color: (u8, u8, u8, u8),
) {
	if file >= 8 || rank >= 8 {
		return;
	}

	let cell_width = board.width / 8.0;
	let cell_height = board.height / 8.0;

	let x = board.x + (file as f32 * cell_width);
	let y = board.y + (rank as f32 * cell_height);

	let mut pb = PathBuilder::new();
	pb.rect(x, y, cell_width, cell_height);
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
			return Vec::new();
		}

		let num_detections = shape[2];
		let num_classes = shape[1] - 4;
		let scale_factor = 640.0;
		let scale_x = img_width as f32 / scale_factor;
		let scale_y = img_height as f32 / scale_factor;

		// 0: r, 1: n, 2: b, 3: q, 4: k, 5: p, 6: R, 7: N, 8: B, 9: Q, 10: K, 11: P
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
		eprintln!("Warning: Could not load system font for piece labels");
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
