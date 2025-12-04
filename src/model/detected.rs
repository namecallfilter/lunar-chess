use super::Confidence;
use crate::{
	chess::{BOARD_SIZE, PieceType, PlayerColor, Square},
	errors::{BoardToFenError, Fen},
	ui::{CellSize, Point2D},
};

#[derive(Clone, Copy, Debug)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoardOrientation {
	Unknown,
	WhiteAtBottom,
	WhiteAtTop,
}

impl BoardOrientation {
	pub fn to_player_color(self) -> Option<PlayerColor> {
		match self {
			BoardOrientation::Unknown => None,
			BoardOrientation::WhiteAtBottom => Some(PlayerColor::White),
			BoardOrientation::WhiteAtTop => Some(PlayerColor::Black),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct DetectedBoard {
	pub rect: Rect,
	pub orientation: BoardOrientation,
}

impl DetectedBoard {
	pub fn new(rect: Rect) -> Self {
		Self {
			rect,
			orientation: BoardOrientation::Unknown,
		}
	}

	pub fn player_color(&self) -> Option<PlayerColor> {
		self.orientation.to_player_color()
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

	pub fn get_square_for_piece(&self, piece: &DetectedPiece) -> Option<Square> {
		let cell_size = self.cell_size();
		let piece_center = piece.center();

		let rel_x = piece_center.x - self.x();
		let rel_y = piece_center.y - self.y();

		let file = (rel_x / cell_size.width).floor() as i32;
		let rank = (rel_y / cell_size.height).floor() as i32;

		if (0..BOARD_SIZE as i32).contains(&file) && (0..BOARD_SIZE as i32).contains(&rank) {
			Square::from_indices(file as u8, rank as u8)
		} else {
			None
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
	pub version: u64,
}

impl BoardState {
	pub fn new(board: DetectedBoard, pieces: Vec<DetectedPiece>, version: u64) -> Self {
		Self {
			board,
			pieces,
			version,
		}
	}

	pub fn to_fen(&self) -> Result<Fen, BoardToFenError> {
		Self::calculate_fen(&self.board, &self.pieces)
	}

	pub fn calculate_orientation(
		board: &DetectedBoard, pieces: &[DetectedPiece],
	) -> BoardOrientation {
		let mut white_bottom = 0;
		let mut white_top = 0;

		let mid_y = board.y() + board.height() / 2.0;

		for piece in pieces {
			if piece.piece_type.is_white() {
				let piece_center = piece.center();

				if piece_center.y > mid_y {
					white_bottom += 1;
				} else {
					white_top += 1;
				}
			}
		}

		if white_top == 0 && white_bottom == 0 {
			BoardOrientation::Unknown
		} else if white_bottom > white_top {
			BoardOrientation::WhiteAtBottom
		} else if white_top > white_bottom {
			BoardOrientation::WhiteAtTop
		} else {
			BoardOrientation::Unknown
		}
	}

	pub fn calculate_fen(
		board: &DetectedBoard, pieces: &[DetectedPiece],
	) -> Result<Fen, BoardToFenError> {
		let player_color = board
			.orientation
			.to_player_color()
			.ok_or(BoardToFenError::UnknownOrientation)?;

		let mut chess_board: [[Option<PieceType>; BOARD_SIZE]; BOARD_SIZE] =
			[[None; BOARD_SIZE]; BOARD_SIZE];

		for piece in pieces {
			if let Some(square) = board.get_square_for_piece(piece) {
				let file = square.file.index() as usize;
				let rank = square.rank.index() as usize;

				if chess_board[rank][file].is_some() {
					continue;
				}

				chess_board[rank][file] = Some(piece.piece_type);
			}
		}

		if player_color.is_black() {
			chess_board.reverse();

			for row in chess_board.iter_mut() {
				row.reverse();
			}
		}

		let has_white_king = chess_board
			.iter()
			.flatten()
			.any(|p| matches!(p, Some(PieceType::WHITE_KING)));
		let has_black_king = chess_board
			.iter()
			.flatten()
			.any(|p| matches!(p, Some(PieceType::BLACK_KING)));

		if !has_white_king {
			return Err(BoardToFenError::MissingKing { color: "white" });
		}
		if !has_black_king {
			return Err(BoardToFenError::MissingKing { color: "black" });
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
		fen.push(player_color.to_fen_char());
		fen.push(' ');

		let castling = Self::infer_castling_rights(&chess_board);
		if castling.is_empty() {
			fen.push('-');
		} else {
			fen.push_str(&castling);
		}

		fen.push_str(" - 0 1");

		Ok(Fen::from_validated(fen))
	}

	fn infer_castling_rights(board: &[[Option<PieceType>; BOARD_SIZE]; BOARD_SIZE]) -> String {
		let mut castling = String::new();

		// Board is normalized so white is at the bottom (rank 7) and black at the top (rank 0)
		// White back rank: row 7
		// Black back rank: row 0

		if matches!(board[7][4], Some(PieceType::WHITE_KING)) {
			if matches!(board[7][7], Some(PieceType::WHITE_ROOK)) {
				castling.push('K');
			}

			if matches!(board[7][0], Some(PieceType::WHITE_ROOK)) {
				castling.push('Q');
			}
		}

		if matches!(board[0][4], Some(PieceType::BLACK_KING)) {
			if matches!(board[0][7], Some(PieceType::BLACK_ROOK)) {
				castling.push('k');
			}

			if matches!(board[0][0], Some(PieceType::BLACK_ROOK)) {
				castling.push('q');
			}
		}

		castling
	}
}
