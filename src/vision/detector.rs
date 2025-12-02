use anyhow::{Context, Result};
use fast_image_resize::{PixelType, Resizer, images::Image};
use image::RgbaImage;
use ndarray::Array4;
use ort::{
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef,
};

use crate::{
	config::CONFIG,
	model::detected::{DetectedBoard, DetectedPiece, Rect},
	vision::{
		board_detection::{BoardStabilizer, detect_board_hough},
		postprocess,
	},
};

pub const YOLO_INPUT_SIZE: u32 = 640;

pub const MAX_PIECES: usize = 32;

struct ImageBuffers {
	tensor_array: Array4<f32>,
}

impl ImageBuffers {
	fn new() -> Self {
		Self {
			tensor_array: Array4::<f32>::zeros((
				1,
				3,
				YOLO_INPUT_SIZE as usize,
				YOLO_INPUT_SIZE as usize,
			)),
		}
	}
}

pub struct ChessDetector {
	piece_model: Session,
	resizer: Resizer,
	buffers: ImageBuffers,
	stabilizer: BoardStabilizer,
}

impl ChessDetector {
	pub fn new() -> Result<Self> {
		tracing::debug!("Initializing piece detection model with CUDA...");

		let piece_model = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.commit_from_file(&CONFIG.detection.path)?;

		tracing::debug!("Piece detection model loaded");

		Ok(Self {
			piece_model,
			resizer: Resizer::new(),
			buffers: ImageBuffers::new(),
			stabilizer: BoardStabilizer::new(0.15),
		})
	}

	pub fn detect_board(&mut self, image: &RgbaImage) -> Result<Option<DetectedBoard>> {
		let raw_detection = detect_board_hough(image);

		Ok(self.stabilizer.update(raw_detection))
	}

	pub fn detect_pieces(
		&mut self, image: &RgbaImage, board: &DetectedBoard,
	) -> Result<Vec<DetectedPiece>> {
		let img_w = image.width();
		let img_h = image.height();

		let x = board.x().max(0.0).floor() as u32;
		let y = board.y().max(0.0).floor() as u32;

		if x >= img_w || y >= img_h {
			return Ok(Vec::new());
		}

		let remaining_w = (img_w - x) as f32;
		let remaining_h = (img_h - y) as f32;

		let w = board.width().min(remaining_w).floor() as u32;
		let h = board.height().min(remaining_h).floor() as u32;

		if w == 0 || h == 0 {
			tracing::debug!(
				"Invalid board crop area: x={}, y={}, w={}, h={}",
				x,
				y,
				w,
				h
			);
			return Ok(Vec::new());
		}

		let board_cropped = image::imageops::crop_imm(image, x, y, w, h);

		let board_cropped_img = board_cropped.to_image();
		let (width, height) = (board_cropped_img.width(), board_cropped_img.height());
		let mut raw_buf = board_cropped_img.into_raw();
		let src_image =
			Image::from_slice_u8(width, height, raw_buf.as_mut_slice(), PixelType::U8x4)?;

		let mut dst_image = Image::new(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, PixelType::U8x4);
		self.resizer.resize(&src_image, &mut dst_image, None)?;

		let warped_board =
			RgbaImage::from_raw(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, dst_image.into_vec())
				.context("Failed to create warped board image")?;

		image_to_tensor_inplace(&warped_board, &mut self.buffers.tensor_array);

		let piece_outputs = self.piece_model.run(
			ort::inputs!["images" => TensorRef::from_array_view(self.buffers.tensor_array.view())?],
		)?;

		let piece_predictions = piece_outputs["output0"].try_extract_array::<f32>()?;

		let piece_predictions_view = piece_predictions.view();
		let detected_pieces_warped = postprocess::from_yolo_output(
			&piece_predictions_view,
			YOLO_INPUT_SIZE,
			YOLO_INPUT_SIZE,
			CONFIG.detection.piece_confidence_threshold,
		);

		let scale_x = board.width() / YOLO_INPUT_SIZE as f32;
		let scale_y = board.height() / YOLO_INPUT_SIZE as f32;

		let detected_pieces: Vec<DetectedPiece> = detected_pieces_warped
			.into_iter()
			.map(|piece| {
				DetectedPiece::new(
					Rect::new(
						board.x() + (piece.x() * scale_x),
						board.y() + (piece.y() * scale_y),
						piece.width() * scale_x,
						piece.height() * scale_y,
					),
					piece.piece_type,
					piece.confidence,
				)
			})
			.collect();

		Ok(detected_pieces)
	}
}

fn image_to_tensor_inplace(image: &RgbaImage, array: &mut Array4<f32>) {
	for (x, y, pixel) in image.enumerate_pixels() {
		let xi = x as usize;
		let yi = y as usize;
		let channels = pixel.0;
		array[[0, 0, yi, xi]] = channels[0] as f32 / 255.0;
		array[[0, 1, yi, xi]] = channels[1] as f32 / 255.0;
		array[[0, 2, yi, xi]] = channels[2] as f32 / 255.0;
	}
}
