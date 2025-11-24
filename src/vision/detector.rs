use anyhow::{Context as _, Result};
use fast_image_resize::{PixelType, Resizer, images::Image};
use image::{RgbaImage, imageops::grayscale};
use ndarray::Array4;
use ort::{
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef,
};

use crate::{
	model::detected::{DetectedBoard, DetectedPiece},
	vision::{board_detection::detect_board_scanline, postprocess},
};

const PIECE_MODEL_PATH: &str = "models/piece.onnx";
const YOLO_TARGET_SIZE: u32 = 640;
const PIECE_CONFIDENCE_THRESHOLD: f32 = 0.75;

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
				YOLO_TARGET_SIZE as usize,
				YOLO_TARGET_SIZE as usize,
			)),
		}
	}
}

pub struct ChessDetector {
	piece_model: Session,
	resizer: Resizer,
	buffers: ImageBuffers,
}

impl ChessDetector {
	pub fn new() -> Result<Self> {
		tracing::debug!("Initializing piece detection model with CUDA...");

		let piece_model = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.commit_from_file(PIECE_MODEL_PATH)?;

		tracing::debug!("Piece detection model loaded");

		Ok(Self {
			piece_model,
			resizer: Resizer::new(),
			buffers: ImageBuffers::new(),
		})
	}

	pub fn detect_board(&self, image: &RgbaImage) -> Result<Option<DetectedBoard>> {
		Ok(detect_board_scanline(&grayscale::<RgbaImage>(image)))
	}

	pub fn detect_pieces(
		&mut self, image: &RgbaImage, board: &DetectedBoard,
	) -> Result<Vec<DetectedPiece>> {
		if board.x < 0.0 || board.y < 0.0 || board.width <= 0.0 || board.height <= 0.0 {
			tracing::debug!(
				"Invalid board bounds: x={}, y={}, w={}, h={}",
				board.x,
				board.y,
				board.width,
				board.height
			);
			return Ok(Vec::new());
		}

		let board_cropped = image::imageops::crop_imm(
			image,
			board.x as u32,
			board.y as u32,
			board.width as u32,
			board.height as u32,
		);

		let board_cropped_img = board_cropped.to_image();
		let (width, height) = (board_cropped_img.width(), board_cropped_img.height());
		let mut raw_buf = board_cropped_img.into_raw();
		let src_image =
			Image::from_slice_u8(width, height, raw_buf.as_mut_slice(), PixelType::U8x4)?;

		let mut dst_image = Image::new(YOLO_TARGET_SIZE, YOLO_TARGET_SIZE, PixelType::U8x4);
		self.resizer.resize(&src_image, &mut dst_image, None)?;

		let warped_board =
			RgbaImage::from_raw(YOLO_TARGET_SIZE, YOLO_TARGET_SIZE, dst_image.into_vec())
				.context("Failed to create warped board image")?;

		image_to_tensor_inplace(&warped_board, &mut self.buffers.tensor_array);

		let piece_outputs = self.piece_model.run(
			ort::inputs!["images" => TensorRef::from_array_view(self.buffers.tensor_array.view())?],
		)?;

		let piece_predictions = piece_outputs["output0"].try_extract_array::<f32>()?;

		let piece_predictions_view = piece_predictions.view();
		let detected_pieces_warped = postprocess::from_yolo_output(
			&piece_predictions_view,
			YOLO_TARGET_SIZE,
			YOLO_TARGET_SIZE,
			PIECE_CONFIDENCE_THRESHOLD,
		);

		let scale_x = board.width / YOLO_TARGET_SIZE as f32;
		let scale_y = board.height / YOLO_TARGET_SIZE as f32;

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
