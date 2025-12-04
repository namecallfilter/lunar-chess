use crate::{
	chess::PieceType,
	model::{
		Confidence,
		detected::{DetectedPiece, Rect},
	},
	vision::detector::YOLO_INPUT_SIZE,
};

const NMS_IOU_THRESHOLD: f32 = 0.7;

pub fn from_yolo_output(
	predictions: &ndarray::ArrayViewD<f32>, img_width: u32, img_height: u32,
	confidence_threshold: Confidence,
) -> Vec<DetectedPiece> {
	let threshold = confidence_threshold.value();
	let shape = predictions.shape();
	if shape.len() < 3 {
		tracing::warn!("Invalid YOLO output shape for piece detection");
		return Vec::new();
	}

	let num_detections = shape[2];
	let num_classes = shape[1] - 4;
	let scale_x = img_width as f32 / YOLO_INPUT_SIZE as f32;
	let scale_y = img_height as f32 / YOLO_INPUT_SIZE as f32;

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

		if max_class_conf > threshold {
			let Some(piece_type) = PieceType::from_class_index(max_class_idx) else {
				continue;
			};

			let x = (x_center - width / 2.0) * scale_x;
			let y = (y_center - height / 2.0) * scale_y;
			let w = width * scale_x;
			let h = height * scale_y;

			pieces.push(DetectedPiece::new(
				Rect::new(x, y, w, h),
				piece_type,
				Confidence::new(max_class_conf),
			));
		}
	}

	nms_pieces(pieces, NMS_IOU_THRESHOLD)
}

fn nms_pieces(mut pieces: Vec<DetectedPiece>, iou_threshold: f32) -> Vec<DetectedPiece> {
	if pieces.is_empty() {
		return pieces;
	}

	pieces.sort_unstable_by(|a, b| {
		let ca = a.confidence.value();
		let cb = b.confidence.value();
		match (ca.is_nan(), cb.is_nan()) {
			(true, true) => std::cmp::Ordering::Equal,
			(true, false) => std::cmp::Ordering::Greater,
			(false, true) => std::cmp::Ordering::Less,
			(false, false) => cb.partial_cmp(&ca).unwrap(),
		}
	});

	let mut keep = Vec::with_capacity(pieces.len());
	let mut suppressed = vec![false; pieces.len()];

	for i in 0..pieces.len() {
		if suppressed[i] {
			continue;
		}

		for j in (i + 1)..pieces.len() {
			if suppressed[j] {
				continue;
			}

			let iou = calculate_iou(&pieces[i], &pieces[j]);
			if iou > iou_threshold {
				suppressed[j] = true;
			}
		}

		keep.push(pieces[i]);
	}

	keep
}

fn calculate_iou(a: &DetectedPiece, b: &DetectedPiece) -> f32 {
	let x1_inter = a.x().max(b.x());
	let y1_inter = a.y().max(b.y());
	let x2_inter = (a.x() + a.width()).min(b.x() + b.width());
	let y2_inter = (a.y() + a.height()).min(b.y() + b.height());

	let inter_width = (x2_inter - x1_inter).max(0.0);
	let inter_height = (y2_inter - y1_inter).max(0.0);
	let inter_area = inter_width * inter_height;

	let area_a = a.width() * a.height();
	let area_b = b.width() * b.height();
	let union_area = area_a + area_b - inter_area;

	if union_area > 0.0 {
		inter_area / union_area
	} else {
		0.0
	}
}
