use anyhow::{Context, Result};
use fast_image_resize::{PixelType, Resizer, images::Image};
use image::RgbaImage;
use ndarray::Array4;
use ort::{
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef,
};

use crate::drawing::{DetectedBoard, DetectedPiece};

pub struct ChessDetector {
	board_model: Session,
	piece_model: Session,
	resizer: Resizer,
}

impl ChessDetector {
	pub fn new() -> Result<Self> {
		tracing::info!("Initializing ONNX models with CUDA acceleration...");

		let board_model = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.with_intra_threads(4)?
			.commit_from_file("models/board.onnx")
			.context("Failed to load board detection model")?;

		let piece_model = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.with_optimization_level(GraphOptimizationLevel::Level3)?
			.with_intra_threads(4)?
			.commit_from_file("models/piece.onnx")
			.context("Failed to load piece detection model")?;

		tracing::info!("Chess detection models loaded successfully");

		Ok(Self {
			board_model,
			piece_model,
			resizer: Resizer::new(),
		})
	}

	pub fn detect_board(&mut self, image: &RgbaImage) -> Result<Option<DetectedBoard>> {
		let total_start = std::time::Instant::now();

		let original_width = image.width();
		let original_height = image.height();
		let target_size = 640u32;

		let resize_start = std::time::Instant::now();
		let mut raw_buf = image.as_raw().clone();
		let src_image = Image::from_slice_u8(
			original_width,
			original_height,
			raw_buf.as_mut_slice(),
			PixelType::U8x4,
		)?;

		let mut dst_image = Image::new(target_size, target_size, PixelType::U8x4);
		self.resizer.resize(&src_image, &mut dst_image, None)?;

		let resized = RgbaImage::from_raw(target_size, target_size, dst_image.into_vec())
			.context("Failed to create resized image")?;
		tracing::trace!("Board image resize took {:?}", resize_start.elapsed());

		let tensor_start = std::time::Instant::now();
		let array = image_to_tensor(&resized);
		tracing::trace!("Board tensor conversion took {:?}", tensor_start.elapsed());

		let inference_start = std::time::Instant::now();
		let board_outputs = self
			.board_model
			.run(ort::inputs!["images" => TensorRef::from_array_view(&array)?])?;
		tracing::trace!("Board inference took {:?}", inference_start.elapsed());

		let extract_start = std::time::Instant::now();
		let board_predictions = board_outputs["output0"].try_extract_array::<f32>()?;
		tracing::trace!("Board result extraction took {:?}", extract_start.elapsed());

		let parse_start = std::time::Instant::now();
		let predictions_view = board_predictions.view();
		let detected_board = DetectedBoard::from_yolo_output(
			&predictions_view,
			original_width,
			original_height,
			0.5,
		);
		tracing::trace!("Board result parsing took {:?}", parse_start.elapsed());

		tracing::trace!("Board detection completed in {:?}", total_start.elapsed());

		Ok(detected_board)
	}

	pub fn detect_pieces(
		&mut self, image: &RgbaImage, board: &DetectedBoard,
	) -> Result<Vec<DetectedPiece>> {
		let total_start = std::time::Instant::now();

		let crop_start = std::time::Instant::now();
		let board_cropped = image::imageops::crop_imm(
			image,
			board.x as u32,
			board.y as u32,
			board.width as u32,
			board.height as u32,
		);
		tracing::trace!("Board crop took {:?}", crop_start.elapsed());

		let resize_start = std::time::Instant::now();
		let board_cropped_img = board_cropped.to_image();
		let (width, height) = (board_cropped_img.width(), board_cropped_img.height());
		let mut raw_buf = board_cropped_img.into_raw();
		let src_image =
			Image::from_slice_u8(width, height, raw_buf.as_mut_slice(), PixelType::U8x4)?;

		let mut dst_image = Image::new(640, 640, PixelType::U8x4);
		self.resizer.resize(&src_image, &mut dst_image, None)?;

		let warped_board = RgbaImage::from_raw(640, 640, dst_image.into_vec())
			.context("Failed to create warped board image")?;
		tracing::trace!("Piece image resize took {:?}", resize_start.elapsed());

		let tensor_start = std::time::Instant::now();
		let warped_array = image_to_tensor(&warped_board);
		tracing::trace!("Piece tensor conversion took {:?}", tensor_start.elapsed());

		let inference_start = std::time::Instant::now();
		let piece_outputs = self
			.piece_model
			.run(ort::inputs!["images" => TensorRef::from_array_view(&warped_array)?])?;
		tracing::trace!("Piece inference took {:?}", inference_start.elapsed());

		let extract_start = std::time::Instant::now();
		let piece_predictions = piece_outputs["output0"].try_extract_array::<f32>()?;
		tracing::trace!("Piece result extraction took {:?}", extract_start.elapsed());

		let parse_start = std::time::Instant::now();
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
		tracing::trace!("Piece result parsing took {:?}", parse_start.elapsed());

		tracing::trace!("Piece detection completed in {:?}", total_start.elapsed());

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
