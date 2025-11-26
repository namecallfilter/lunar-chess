use super::Confidence;
use crate::{
	chess::{BOARD_SIZE, PieceType, PlayerColor, Square},
	ui::{CellSize, Point2D},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
	x: f32,
	y: f32,
	width: f32,
	height: f32,
}

impl Rect {
	pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
		Self {
			x,
			y,
			width: width.max(0.0),
			height: height.max(0.0),
		}
	}

	#[inline]
	pub const fn x(&self) -> f32 {
		self.x
	}

	#[inline]
	pub const fn y(&self) -> f32 {
		self.y
	}

	#[inline]
	pub const fn width(&self) -> f32 {
		self.width
	}

	#[inline]
	pub const fn height(&self) -> f32 {
		self.height
	}

	#[inline]
	pub fn center(&self) -> Point2D {
		Point2D::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
	}

	#[inline]
	pub fn is_empty(&self) -> bool {
		self.width == 0.0 || self.height == 0.0
	}
}

#[derive(Clone, Copy, Debug)]
pub struct DetectedBoard {
	pub rect: Rect,
	pub player_color: PlayerColor,
}

impl DetectedBoard {
	pub fn new(rect: Rect) -> Self {
		Self {
			rect,
			player_color: PlayerColor::White,
		}
	}

	#[inline]
	pub fn cell_size(&self) -> CellSize {
		CellSize::new(
			self.rect.width() / BOARD_SIZE as f32,
			self.rect.height() / BOARD_SIZE as f32,
		)
	}

	#[inline]
	pub fn cell_center(&self, file: u8, rank: u8) -> Point2D {
		let cell_size = self.cell_size();
		Point2D::new(
			self.x() + (file as f32 + 0.5) * cell_size.width,
			self.y() + (rank as f32 + 0.5) * cell_size.height,
		)
	}

	#[inline]
	pub fn square_center(&self, square: &Square) -> Point2D {
		self.cell_center(square.file.index(), square.rank.index())
	}

	#[inline]
	pub fn x(&self) -> f32 {
		self.rect.x()
	}

	#[inline]
	pub fn y(&self) -> f32 {
		self.rect.y()
	}

	#[inline]
	pub fn width(&self) -> f32 {
		self.rect.width()
	}

	#[inline]
	pub fn height(&self) -> f32 {
		self.rect.height()
	}
}

#[derive(Clone, Copy, Debug)]
pub struct DetectedPiece {
	pub rect: Rect,
	pub piece_type: PieceType,
	pub confidence: Confidence,
}

impl DetectedPiece {
	pub fn new(rect: Rect, piece_type: PieceType, confidence: Confidence) -> Self {
		Self {
			rect,
			piece_type,
			confidence,
		}
	}

	#[inline]
	pub fn x(&self) -> f32 {
		self.rect.x()
	}

	#[inline]
	pub fn y(&self) -> f32 {
		self.rect.y()
	}

	#[inline]
	pub fn width(&self) -> f32 {
		self.rect.width()
	}

	#[inline]
	pub fn height(&self) -> f32 {
		self.rect.height()
	}

	#[inline]
	pub fn center(&self) -> Point2D {
		self.rect.center()
	}
}

#[derive(Clone, Debug)]
pub struct BoardState {
	pub board: DetectedBoard,
	pub pieces: Vec<DetectedPiece>,
}

impl BoardState {
	pub fn new(board: DetectedBoard, pieces: Vec<DetectedPiece>) -> Self {
		Self { board, pieces }
	}

	pub fn detect_orientation(&self) -> PlayerColor {
		let mut white_bottom = 0;
		let mut white_top = 0;

		let mid_y = self.board.y() + self.board.height() / 2.0;

		for piece in &self.pieces {
			if piece.piece_type.is_white() {
				let piece_center = piece.center();

				if piece_center.y > mid_y {
					white_bottom += 1;
				} else {
					white_top += 1;
				}
			}
		}

		if white_bottom > white_top {
			PlayerColor::White
		} else {
			PlayerColor::Black
		}
	}

	pub fn to_fen(&self) -> String {
		let mut chess_board: [[Option<PieceType>; BOARD_SIZE]; BOARD_SIZE] =
			[[None; BOARD_SIZE]; BOARD_SIZE];

		for piece in &self.pieces {
			if let Some(square) = self.piece_to_square(piece) {
				let file = square.file.index() as usize;
				let rank = square.rank.index() as usize;

				if chess_board[rank][file].is_some() {
					continue;
				}

				chess_board[rank][file] = Some(piece.piece_type);
			}
		}

		if self.board.player_color.is_black() {
			chess_board.reverse();

			for row in chess_board.iter_mut() {
				row.reverse();
			}
		}

		let mut fen = String::new();

		for (rank, row) in chess_board.iter().enumerate() {
			let mut empty_count = 0;

			for cell in row.iter() {
				match *cell {
					Some(piece) => {
						if empty_count > 0 {
							fen.push_str(&empty_count.to_string());

							empty_count = 0;
						}

						fen.push(piece.to_fen_char());
					}
					None => {
						empty_count += 1;
					}
				}
			}

			if empty_count > 0 {
				fen.push_str(&empty_count.to_string());
			}

			if rank < BOARD_SIZE - 1 {
				fen.push('/');
			}
		}

		fen.push(' ');
		fen.push(self.board.player_color.to_fen_char());
		fen.push_str(" - - 0 1");

		fen
	}

	pub fn piece_to_square(&self, piece: &DetectedPiece) -> Option<Square> {
		let cell_size = self.board.cell_size();
		let piece_center = piece.center();

		let rel_x = piece_center.x - self.board.x();
		let rel_y = piece_center.y - self.board.y();

		let file = (rel_x / cell_size.width) as i32;
		let rank = (rel_y / cell_size.height) as i32;

		if (0..BOARD_SIZE as i32).contains(&file) && (0..BOARD_SIZE as i32).contains(&rank) {
			Square::from_indices(file as u8, rank as u8)
		} else {
			None
		}
	}
}
