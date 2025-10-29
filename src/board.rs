use crate::drawing::{DetectedBoard, DetectedPiece};
// use shakmaty

pub fn to_fen(board: &DetectedBoard, pieces: &[DetectedPiece]) -> String {
	let mut chess_board = [[None; 8]; 8];

	for piece in pieces {
		if let Some((file, rank)) = piece_to_square(piece, board)
			&& file < 8
			&& rank < 8
		{
			chess_board[rank as usize][file as usize] = Some(piece.piece_type);
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
					fen.push(piece);
				}
				None => {
					empty_count += 1;
				}
			}
		}

		if empty_count > 0 {
			fen.push_str(&empty_count.to_string());
		}

		if rank < 7 {
			fen.push('/');
		}
	}

	tracing::debug!("Generated FEN: {}", fen);
	fen
}

pub fn piece_to_square(piece: &DetectedPiece, board: &DetectedBoard) -> Option<(u8, u8)> {
	let cell_width = board.width / 8.0;
	let cell_height = board.height / 8.0;

	let piece_center_x = piece.x + (piece.width / 2.0);
	let piece_center_y = piece.y + (piece.height / 2.0);

	let rel_x = piece_center_x - board.x;
	let rel_y = piece_center_y - board.y;

	let file = (rel_x / cell_width) as i32;
	let rank = (rel_y / cell_height) as i32;

	if (0..8).contains(&file) && (0..8).contains(&rank) {
		Some((file as u8, rank as u8))
	} else {
		None
	}
}

pub fn square_to_algebraic(file: u8, rank: u8) -> Option<String> {
	if file >= 8 || rank >= 8 {
		return None;
	}

	let file_char = (b'a' + file) as char;
	let rank_char = (b'1' + rank) as char;

	Some(format!("{}{}", file_char, rank_char))
}
