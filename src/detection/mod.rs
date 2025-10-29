use anyhow::Result;
use image::RgbaImage;
use ndarray::Array4;
use ort::{
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef,
};

use crate::drawing::{DetectedBoard, DetectedPiece};

pub struct ChessDetector {
	board_model: Session,
	piece_model: Session,
}

impl ChessDetector {
	pub fn new() -> Result<Self> {
		let board_model = Session::builder()?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.with_intra_threads(4)?
			.commit_from_file("models/board.onnx")?;

		let piece_model = Session::builder()?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.with_intra_threads(4)?
			.commit_from_file("models/piece.onnx")?;

		Ok(Self {
			board_model,
			piece_model,
		})
	}

	pub fn detect_board(&mut self, image: &RgbaImage) -> Result<Option<DetectedBoard>> {
		let original_width = image.width();
		let original_height = image.height();

		let target_size = 640u32;
		let resized = image::imageops::resize(
			image,
			target_size,
			target_size,
			image::imageops::FilterType::Lanczos3,
		);

		let array = image_to_tensor(&resized);

		let board_outputs = self
			.board_model
			.run(ort::inputs!["images" => TensorRef::from_array_view(&array)?])?;
		let board_predictions = board_outputs["output0"].try_extract_array::<f32>()?;

		let predictions_view = board_predictions.view();
		let detected_board = DetectedBoard::from_yolo_output(
			&predictions_view,
			original_width,
			original_height,
			0.5,
		);

		Ok(detected_board)
	}

	pub fn detect_pieces(
		&mut self, image: &RgbaImage, board: &DetectedBoard,
	) -> Result<Vec<DetectedPiece>> {
		let board_cropped = image::imageops::crop_imm(
			image,
			board.x as u32,
			board.y as u32,
			board.width as u32,
			board.height as u32,
		);

		let warped_board = image::imageops::resize(
			&board_cropped.to_image(),
			640,
			640,
			image::imageops::FilterType::Lanczos3,
		);

		warped_board.save("debug_warped.png")?;

		let warped_array = image_to_tensor(&warped_board);

		let piece_outputs = self
			.piece_model
			.run(ort::inputs!["images" => TensorRef::from_array_view(&warped_array)?])?;
		let piece_predictions = piece_outputs["output0"].try_extract_array::<f32>()?;

		let piece_predictions_view = piece_predictions.view();
		let detected_pieces_warped =
			DetectedPiece::from_yolo_output(&piece_predictions_view, 640, 640, 0.5);

		let scale_x = board.width / 640.0;
		let scale_y = board.height / 640.0;

		let detected_pieces: Vec<DetectedPiece> = detected_pieces_warped
			.into_iter()
			.map(|mut piece| {
				piece.x = board.x + (piece.x * scale_x);
				piece.y = board.y + (piece.y * scale_y);
				piece.width *= scale_x;
				piece.height *= scale_y;
				piece
			})
			.collect();

		Ok(detected_pieces)
	}
}

fn image_to_tensor(image: &RgbaImage) -> Array4<f32> {
	let (width, height) = (image.width() as usize, image.height() as usize);
	let mut array = Array4::<f32>::zeros((1, 3, height, width));

	for (x, y, pixel) in image.enumerate_pixels() {
		let xi = x as usize;
		let yi = y as usize;
		let channels = pixel.0;
		array[[0, 0, yi, xi]] = channels[0] as f32 / 255.0;
		array[[0, 1, yi, xi]] = channels[1] as f32 / 255.0;
		array[[0, 2, yi, xi]] = channels[2] as f32 / 255.0;
	}

	array
}
